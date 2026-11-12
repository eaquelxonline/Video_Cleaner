"""Video_Detector (VC-Detector) — kaplama (altyazı/filigran) algılama ve ayrıştırma modeli.

Görev: görüntüye SONRADAN eklenmiş yazıyı bulmak, saydamlığını ve rengini
kestirmek ve onu sahneye ait gerçek yazıdan (tabela, plaka, forma numarası,
ekran içi metin) ayırmak. Klasik OCR değildir; "bu yazı görüntünün parçası mı,
üstüne mi bindirilmiş" sorusunu öğrenir. Bunun en güçlü ipucu zamansaldır:
bindirilmiş yazı ekran uzayında sabit dururken arka plan hareket eder. Bu yüzden
1/8 ve 1/16 ölçeklerinde kareler arası (zamansal) karıştırma katmanları vardır.

Girdi
    frames [B, T, 3, H, W]  RGB 0..1
    valid  [B, T] bool      dolgu konumları False

Çıktı (sözlük; hepsi tam çözünürlükte, [B, T, C, H, W])
    overlay_logit    C=1  silinecek kaplama (yazı gövdesi + kontur)
    shadow_logit     C=1  gölge / parıltı / kenar efekti
    alpha            C=1  0..1 birleşik saydamlık (yazı + gölge katmanı)
    fg_rgb           C=3  0..1 birleşik kaplama rengi F  (B = (I − αF)/(1 − α) için gerekli)
    scene_text_logit C=1  korunacak sahne yazısı
    confidence_logit C=1  tahminin güvenilirliği

Not: ``fg_rgb`` özgün planın çıktı listesinde yoktu; analitik alfa çözümü F
bilinmeden yapılamadığı için eklendi.

Başlangıç bütçesi: 2²² = 4.194.304 parametre (üst sınır; doldurma parametresi yok).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Core import ConvBlock, Downsample, LayerNorm2d, Upsample, check_window, flatten_time, run_block, unflatten_time

HEADS: Tuple[Tuple[str, int], ...] = (
    ("overlay_logit", 1),
    ("shadow_logit", 1),
    ("alpha", 1),
    ("fg_rgb", 3),
    ("scene_text_logit", 1),
    ("confidence_logit", 1),
)
OUT_CHANNELS = sum(c for _, c in HEADS)


@dataclass
class DetectorConfig:
    widths: Tuple[int, int, int, int] = (32, 64, 144, 272)     # 1/2, 1/4, 1/8, 1/16 ölçek kanal sayıları
    depths: Tuple[int, int, int, int] = (2, 2, 4, 5)           # ölçek başına blok sayısı
    decoder_depths: Tuple[int, int, int] = (2, 1, 1)           # 1/8, 1/4, 1/2 çözücü blokları
    temporal_depths: Tuple[int, int] = (1, 2)                  # 1/8 ve 1/16 ölçeğinde zamansal blok
    temporal_kernel: int = 5
    expansion: int = 3
    use_checkpoint: bool = False
    infer_chunk: int = 4                                       # çıkarımda kare başına bloklar kaçar kare işlesin (0 = hepsi)


class TemporalMix(nn.Module):
    """Kareler arası karıştırma: zaman ekseninde derinlik-yönlü evrişim + nokta evrişim.

    Dolgu kareleri (valid=False) karıştırmaya sıfır olarak girer.
    """

    def __init__(self, dim: int, kernel: int = 5):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.dw = nn.Conv3d(dim, dim, (kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: [B·T, C, h, w], valid: [B, T] (float veya bool)."""
        b, t = valid.shape
        y = unflatten_time(self.norm(x), b) * valid.view(b, t, 1, 1, 1).to(x.dtype)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)                  # [B, T, C, h, w]
        y = self.pw(F.gelu(y.reshape(b * t, *y.shape[2:])))
        return x + self.gamma * y


class VideoDetector(nn.Module):
    """Hafif U-Net + zamansal karıştırma. Çıkış başlığı 1/2 ölçekten PixelShuffle ile tam çözünürlüğe çıkar."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__()
        cfg = self.config = config or DetectorConfig()
        w, d = cfg.widths, cfg.depths
        block = lambda dim: ConvBlock(dim, cfg.expansion, kernel=5)

        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, stride=2, padding=1), LayerNorm2d(w[0]))
        self.enc = nn.ModuleList([nn.ModuleList([block(w[i]) for _ in range(d[i])]) for i in range(4)])
        self.down = nn.ModuleList([Downsample(w[i], w[i + 1]) for i in range(3)])
        self.temporal8 = nn.ModuleList([TemporalMix(w[2], cfg.temporal_kernel) for _ in range(cfg.temporal_depths[0])])
        self.temporal16 = nn.ModuleList([TemporalMix(w[3], cfg.temporal_kernel) for _ in range(cfg.temporal_depths[1])])

        self.up = nn.ModuleList([Upsample(w[3 - i], w[2 - i]) for i in range(3)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * w[2 - i], w[2 - i], 1) for i in range(3)])
        self.dec = nn.ModuleList(
            [nn.ModuleList([block(w[2 - i]) for _ in range(cfg.decoder_depths[i])]) for i in range(3)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(w[0], w[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(w[0], OUT_CHANNELS * 4, 1),
            nn.PixelShuffle(2),
        )
        self._init_head()

    def _init_head(self) -> None:
        """Başlangıç önyargıları: piksellerin büyük çoğunluğu kaplama DEĞİLDİR."""
        bias = {"overlay_logit": -2.0, "shadow_logit": -3.0, "scene_text_logit": -3.0}
        last = self.head[2]
        offset = 0
        with torch.no_grad():
            last.bias.zero_()
            for name, channels in HEADS:
                if name in bias:
                    # PixelShuffle(2): çıkış kanalı c, giriş kanalları 4c..4c+3'ten gelir.
                    last.bias[4 * offset: 4 * (offset + channels)] = bias[name]
                offset += channels

    def _decode(self, x: torch.Tensor, skips) -> torch.Tensor:
        """1/16 ölçekli özelliklerden tam çözünürlüklü 8 kanallı çıkışa; kare başına."""
        for i in range(3):
            x = self.fuse[i](torch.cat([self.up[i](x), skips[2 - i]], dim=1))
            for blk in self.dec[i]:
                x = run_block(blk, x, use_checkpoint=self.config.use_checkpoint)
        return self.head(x)

    # ---- Hibrit çıkarım (telefon): kare başına kodlayıcı / 30 karelik zamansal çekirdek / kare başına çözücü.
    # Ara özellikler grafiğin dışında (Rust halka tamponu) tutulur; her kare BİR kez kodlanır. Üçünün
    # bileşimi ``forward`` ile aynı sonucu verir (selftest denetler).

    def encode_frame(self, frame: torch.Tensor):
        """frame [N, 3, H, W] → (1/2 atlama, 1/4 atlama, 1/8 zamansal öncesi özellik)."""
        x = self.stem(frame)
        skips = []
        for i in range(3):
            for blk in self.enc[i]:
                x = blk(x)
            if i < 2:
                skips.append(x)
                x = self.down[i](x)
        return skips[0], skips[1], x

    def temporal_core(self, x8: torch.Tensor, valid: torch.Tensor):
        """x8 [T, C₂, h, w] (bir pencere), valid [1, T] → (1/8 atlama, 1/16 özellik); kareler arası tek aşama."""
        vf = valid.to(x8.dtype)
        for mix in self.temporal8:
            x8 = mix(x8, vf)
        x = self.down[2](x8)
        for blk in self.enc[3]:
            x = blk(x)
        for mix in self.temporal16:
            x = mix(x, vf)
        return x8, x

    def decode_frame(self, x16: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Kare başına çözücü → [N, 8, H, W] (alfa ve kaplama rengi sigmoid sonrası, diğerleri logit)."""
        out = self._decode(x16, [s0, s1, s2])
        offset = 0
        parts = []
        for name, channels in HEADS:
            part = out[:, offset: offset + channels]
            parts.append(torch.sigmoid(part) if name in ("alpha", "fg_rgb") else part)
            offset += channels
        return torch.cat(parts, dim=1)

    def forward(self, frames: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        valid = check_window(frames, valid)
        ckpt, chunk = self.config.use_checkpoint, self.config.infer_chunk
        vf = valid.to(frames.dtype)
        x, b = flatten_time(frames)
        x = run_block(self.stem, x, chunk=chunk)

        skips = []
        for i in range(4):
            for blk in self.enc[i]:
                x = run_block(blk, x, use_checkpoint=ckpt, chunk=chunk)
            if i == 2:
                for mix in self.temporal8:
                    x = run_block(mix, x, vf, use_checkpoint=ckpt)
            if i == 3:
                for mix in self.temporal16:
                    x = run_block(mix, x, vf, use_checkpoint=ckpt)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)

        # Çözücü kare başınadır; çıkarımda dilim dilim çalışır (1/2 ölçekteki birleştirme
        # tüm kareler için aynı anda bellekte durmaz).
        count = x.shape[0]
        if not torch.is_grad_enabled() and 0 < chunk < count:
            x = torch.cat([self._decode(x[i: i + chunk], [s[i: i + chunk] for s in skips]) for i in range(0, count, chunk)])
        else:
            x = self._decode(x, skips)
        out = unflatten_time(x, b)                                       # [B, T, 8, H, W]
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, channels in HEADS:
            result[name] = out[:, :, offset: offset + channels]
            offset += channels
        result["alpha"] = torch.sigmoid(result["alpha"])
        result["fg_rgb"] = torch.sigmoid(result["fg_rgb"])
        return result
