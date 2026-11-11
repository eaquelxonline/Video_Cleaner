"""Video_Flow (VC-Flow) — maskeli karelerden çift yönlü optik akış ve güven tahmini.

Görev: komşu kareler arasında ARKA PLANIN hareketini kestirmek. Kaplama (altyazı)
ekranda sabit durduğu için ham eşleştirme maske içinde yanlış "hareketsiz" akış
verir; model maskeyi girdi olarak alır ve maske altındaki akışı çevreden
tamamlamayı öğrenir (akış tamamlama). Çıktı, piksel çekme aşamasında "bu piksel
başka hangi karede temiz görünüyor" sorusunu yanıtlamak için kullanılır.

Yapı (RAFT ailesinden, 720p × 30 kare için hafifletilmiş)
    * Ortak özellik kodlayıcı (1/8 ölçek) — her kare için BİR kez çalışır.
    * Yerel korelasyon piramidi: tüm-çiftler hacmi yerine, her yinelemede güncel
      akışla çarpıtılmış hedef özelliklerle ±r komşulukta korelasyon. 720p'de
      tüm-çiftler hacmi çift başına ~830 MB tutardı; bu yol sabit ve küçük bellek ister.
    * Ayrılabilir ConvGRU ile yinelemeli iyileştirme, dışbükey 8× büyütme.

Girdi
    frames [B, T, 3, H, W], mask [B, T, 1, H, W] (0..1 yumuşak silme maskesi), valid [B, T]

Çıktı (sözlük)
    flow_fw [B, T-1, 2, H, W]  t → t+1 (piksel; kanal 0 = x, 1 = y)
    flow_bw [B, T-1, 2, H, W]  t+1 → t
    conf_fw, conf_bw [B, T-1, 1, H, W]  güven logiti
    iters_fw, iters_bw  eğitimde: her yinelemenin tam çözünürlüklü akışı (dizi kaybı için)

Başlangıç bütçesi: 2²³ = 8.388.608 parametre (üst sınır).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Core import check_window, run_block, warp


@dataclass
class FlowConfig:
    encoder_widths: Tuple[int, int, int] = (64, 128, 216)   # 1/2, 1/4, 1/8
    feature_dim: int = 192          # korelasyon özellikleri
    context_dim: int = 128          # GRU'ya sabit bağlam
    hidden_dim: int = 176           # GRU gizli durumu
    corr_radius: int = 4            # ±4 → seviye başına 81 kanal
    corr_levels: int = 3            # 1/8, 1/16, 1/32: yineleme başına ±128 piksele kadar arama
    iters: int = 6
    use_checkpoint: bool = False
    infer_chunk: int = 4            # çıkarımda kodlayıcı ve büyütme kaçar kare/çift işlesin (0 = hepsi)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, channels)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1, self.norm2 = _norm(cout), _norm(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride), _norm(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class Encoder(nn.Module):
    """4 kanal (RGB + maske) → 1/8 ölçekte ``out_dim`` kanal."""

    def __init__(self, widths: Tuple[int, int, int], out_dim: int):
        super().__init__()
        w0, w1, w2 = widths
        self.stem = nn.Sequential(nn.Conv2d(4, w0, 7, stride=2, padding=3), _norm(w0), nn.ReLU())
        self.layers = nn.Sequential(
            ResBlock(w0, w0), ResBlock(w0, w0),
            ResBlock(w0, w1, stride=2), ResBlock(w1, w1),
            ResBlock(w1, w2, stride=2), ResBlock(w2, w2),
        )
        self.out = nn.Conv2d(w2, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.layers(self.stem(x)))


def local_correlation(f1: torch.Tensor, f2: torch.Tensor, radius: int) -> torch.Tensor:
    """f1 ile f2'nin ±radius komşuluğundaki korelasyonu: [N, (2r+1)², h, w]."""
    n, c, h, w = f1.shape
    padded = F.pad(f2, (radius, radius, radius, radius))
    scale = c ** -0.5
    maps = []
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            maps.append((f1 * padded[:, :, dy: dy + h, dx: dx + w]).sum(1) * scale)
    return torch.stack(maps, dim=1)


class SepConvGRU(nn.Module):
    """Ayrılabilir (1×5 ardından 5×1) evrişimli GRU."""

    def __init__(self, hidden: int, inputs: int):
        super().__init__()
        cin = hidden + inputs
        self.z1, self.r1, self.q1 = (nn.Conv2d(cin, hidden, (1, 5), padding=(0, 2)) for _ in range(3))
        self.z2, self.r2, self.q2 = (nn.Conv2d(cin, hidden, (5, 1), padding=(2, 0)) for _ in range(3))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for z_conv, r_conv, q_conv in ((self.z1, self.r1, self.q1), (self.z2, self.r2, self.q2)):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(z_conv(hx))
            r = torch.sigmoid(r_conv(hx))
            q = torch.tanh(q_conv(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
        return h


class UpdateBlock(nn.Module):
    """Korelasyon + güncel akış + bağlam + maske → gizli durum güncellemesi ve akış artışı."""

    def __init__(self, corr_channels: int, hidden: int, context: int):
        super().__init__()
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_channels, 192, 1), nn.ReLU(), nn.Conv2d(192, 128, 3, padding=1), nn.ReLU()
        )
        self.flow_enc = nn.Sequential(nn.Conv2d(2, 64, 7, padding=3), nn.ReLU(), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.merge = nn.Conv2d(128 + 32, 126, 3, padding=1)
        self.gru = SepConvGRU(hidden, 126 + 2 + context + 1)
        self.flow_head = nn.Sequential(nn.Conv2d(hidden, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 2, 3, padding=1))

    def forward(self, net: torch.Tensor, context: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor):
        motion = F.relu(self.merge(torch.cat([self.corr_enc(corr), self.flow_enc(flow)], dim=1)))
        net = self.gru(net, torch.cat([motion, flow, context, mask], dim=1))
        return net, self.flow_head(net)


def convex_upsample(flow: torch.Tensor, weights: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Kaba akışı, 3×3 komşuların öğrenilmiş dışbükey birleşimiyle ``factor`` kat büyütür."""
    n, _, h, w = flow.shape
    weights = torch.softmax(weights.view(n, 1, 9, factor, factor, h, w), dim=2)
    patches = F.unfold(factor * flow, 3, padding=1).view(n, 2, 9, 1, 1, h, w)
    up = (weights * patches).sum(dim=2)                                  # [N, 2, f, f, h, w]
    return up.permute(0, 1, 4, 2, 5, 3).reshape(n, 2, factor * h, factor * w)


class VideoFlow(nn.Module):
    def __init__(self, config: Optional[FlowConfig] = None):
        super().__init__()
        cfg = self.config = config or FlowConfig()
        self.feature_net = Encoder(cfg.encoder_widths, cfg.feature_dim)
        self.context_net = Encoder(cfg.encoder_widths, cfg.hidden_dim + cfg.context_dim)
        corr_channels = cfg.corr_levels * (2 * cfg.corr_radius + 1) ** 2
        self.update = UpdateBlock(corr_channels, cfg.hidden_dim, cfg.context_dim)
        self.up_mask = nn.Sequential(nn.Conv2d(cfg.hidden_dim, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 64 * 9, 1))
        self.conf_head = nn.Sequential(nn.Conv2d(cfg.hidden_dim, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def _correlation(self, f1: List[torch.Tensor], f2: List[torch.Tensor], flow: torch.Tensor) -> torch.Tensor:
        """Piramidin her seviyesinde, güncel akışla hizalanmış hedefle yerel korelasyon."""
        size = flow.shape[-2:]
        out = []
        for level, (a, b) in enumerate(zip(f1, f2)):
            if level == 0:
                level_flow = flow
            else:
                level_flow = F.interpolate(flow, size=a.shape[-2:], mode="bilinear", align_corners=False) / (2 ** level)
            corr = local_correlation(a, warp(b, level_flow), self.config.corr_radius)
            if level > 0:
                corr = F.interpolate(corr, size=size, mode="bilinear", align_corners=False)
            out.append(corr)
        return torch.cat(out, dim=1)

    def _upsample(self, flow: torch.Tensor, net: torch.Tensor) -> torch.Tensor:
        """Dışbükey 8× büyütme. Çıkarımda çiftler dilimlenir: maske tensörü çift başına 576 kanaldır."""
        chunk = self.config.infer_chunk
        if torch.is_grad_enabled() or chunk <= 0 or flow.shape[0] <= chunk:
            return convex_upsample(flow, self.up_mask(net))
        return torch.cat([convex_upsample(f, self.up_mask(n)) for f, n in zip(flow.split(chunk), net.split(chunk))], dim=0)

    def _step(self, net, context, flow, mask, *pyramids):
        levels = self.config.corr_levels
        corr = self._correlation(list(pyramids[:levels]), list(pyramids[levels:]), flow)
        net, delta = self.update(net, context, corr, flow, mask)
        return net, flow + delta

    def _refine(self, fmap, cmap, mask_lr, src, dst, h: int, w: int):
        """Verilen (kaynak, hedef) kare çiftleri için yinelemeli akış iyileştirmesi.

        Dönen: (tam çözünürlüklü akışların listesi — eğitimde her yineleme, çıkarımda yalnızca
        sonuncusu —, tam çözünürlüklü güven logiti).
        """
        cfg = self.config
        f1, f2 = fmap[src], fmap[dst]
        net, context = torch.split(cmap[src], [cfg.hidden_dim, cfg.context_dim], dim=1)
        net, context = torch.tanh(net), F.relu(context)
        mask_src = mask_lr[src]

        pyr1, pyr2 = [f1], [f2]
        for _ in range(1, cfg.corr_levels):
            pyr1.append(F.avg_pool2d(pyr1[-1], 2))
            pyr2.append(F.avg_pool2d(pyr2[-1], 2))

        flow = torch.zeros(src.numel(), 2, h // 8, w // 8, device=fmap.device, dtype=fmap.dtype)
        history: List[torch.Tensor] = []
        for it in range(cfg.iters):
            flow = flow.detach()                       # RAFT: yinelemeler arası gradyan kesilir
            net, flow = run_block(self._step, net, context, flow, mask_src, *pyr1, *pyr2, use_checkpoint=cfg.use_checkpoint)
            if self.training or it == cfg.iters - 1:
                history.append(self._upsample(flow, net))
        conf = F.interpolate(self.conf_head(net), size=(h, w), mode="bilinear", align_corners=False)
        return history, conf

    def forward(self, frames: torch.Tensor, mask: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        valid = check_window(frames, valid)
        cfg = self.config
        b, t, _, h, w = frames.shape
        if t < 2:
            raise ValueError("akış için en az 2 kare gerekir")
        x = torch.cat([frames, mask.to(frames.dtype)], dim=2).flatten(0, 1)       # [B·T, 4, H, W]
        fmap = run_block(self.feature_net, x, chunk=cfg.infer_chunk)
        cmap = run_block(self.context_net, x, chunk=cfg.infer_chunk)
        mask_lr = F.avg_pool2d(mask.flatten(0, 1).to(frames.dtype), 8)

        # Çiftler: önce tüm ileri (t → t+1), sonra tüm geri (t+1 → t) yönler tek yığında.
        idx = torch.arange(b * t, device=frames.device).view(b, t)
        src = torch.cat([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)])
        dst = torch.cat([idx[:, 1:].reshape(-1), idx[:, :-1].reshape(-1)])
        # Çıkarımda çiftler dilim dilim iyileştirilir: 720p × 30 karede 58 çiftin tamamı tek
        # yığında işlenirse yalnızca GRU girdisi ~1,5 GiB (fp32) tutar. Çiftler birbirinden
        # bağımsız olduğundan sonuç birebir aynıdır.
        count = src.numel()
        step = 2 * cfg.infer_chunk
        if not torch.is_grad_enabled() and 0 < step < count:
            parts = [self._refine(fmap, cmap, mask_lr, src[i: i + step], dst[i: i + step], h, w) for i in range(0, count, step)]
            history = [torch.cat([part[0][-1] for part in parts])]
            conf = torch.cat([part[1] for part in parts])
        else:
            history, conf = self._refine(fmap, cmap, mask_lr, src, dst, h, w)

        pairs = b * (t - 1)
        pair_valid = (valid[:, :-1] & valid[:, 1:]).view(b, t - 1, 1, 1, 1)
        shape = lambda x: x.view(b, t - 1, x.shape[1], h, w)
        final = history[-1]
        out = {
            "flow_fw": shape(final[:pairs]) * pair_valid,
            "flow_bw": shape(final[pairs:]) * pair_valid,
            "conf_fw": shape(conf[:pairs]).masked_fill(~pair_valid, -20.0),
            "conf_bw": shape(conf[pairs:]).masked_fill(~pair_valid, -20.0),
        }
        if self.training:
            out["iters_fw"] = [shape(f[:pairs]) * pair_valid for f in history]
            out["iters_bw"] = [shape(f[pairs:]) * pair_valid for f in history]
        return out
