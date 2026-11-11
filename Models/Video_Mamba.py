"""Video_Mamba — çok ölçekli, çift yönlü durum uzayı (Bi-SSM) restorasyon omurgası.

Görev: deterministik aşamaların (analitik alfa çözümü, piksel çekme) kurtaramadığı
bölgeyi 30 karelik zamansal bağlamla tamamlamak ve her piksel için belirsizlik
üretmek. Model her pikseli sıfırdan üretmez: önsellerin güvenle harmanlandığı bir
taban görüntünün üzerine ARTIK (residual) tahmin eder.

    Uzamsal kodlayıcı (1/2 → 1/4)        yerel kenar, doku, geometri
    Bi-SSM zamansal çekirdek (1/8, 1/16) 30 kare boyunca ileri + geri bilgi aktarımı
    Yerel dikkat (1/16)                  pencere içi uzun menzilli uzamsal ilişki
    Kare başına çözücü (1/8 → tam)       atlama bağlantılarıyla ince detay

Tasarım kararları
    * Mamba taraması milyonlarca tam çözünürlüklü pikselde değil, 1/8 ve 1/16
      ölçekli özellik haritalarında, YALNIZCA zaman ekseninde yapılır: dizi uzunluğu
      30, yığın = B·h·w. 30 adımlık döngü düz PyTorch ile yeterince hızlıdır; özel
      CUDA çekirdeği veya ``mamba-ssm`` paketi gerekmez.
    * Zamansal katmanlar yalnızca kodlayıcıdadır; çözücü tümüyle kare başınadır. Bu
      sayede Mod A (14+1+15) yalnızca hedef kareyi çözerek çözücü maliyetinin ~29/30'unu
      atlar ve aynı pencere için Mod B'nin 14. karesiyle AYNI sonucu verir.
    * Tarama fp32'de yapılır (BF16'nın 8 bitlik mantisi yinelemeli durumda hata
      biriktirir); geri kalan her şey autocast altındadır.

Girdi
    frames [B, T, 3, H, W], mask [B, T, 1, H, W] (yumuşak silme maskesi)
    priors: bg_gamma [.., 3], bg_linear [.., 3], analytic_conf [.., 1],
            pulled [.., 3], pulled_conf [.., 1]      (hepsi [B, T, C, H, W])
    valid [B, T], target: None (tüm kareler) veya çözülecek tek konum

Çıktı (T' = T veya 1)
    rgb        [B, T', 3, H, W]  0..1 onarılmış görüntü (kapı ÖNCESİ)
    log_b      [B, T', 1, H, W]  Laplace ölçeğinin logaritması
    confidence [B, T', 1, H, W]  exp(−b / b₀) ∈ (0, 1]

Başlangıç bütçesi: 2²⁵ = 33.554.432 parametre (üst sınır).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Core import ConvBlock, Downsample, LayerNorm2d, Upsample, check_window, run_block

IN_CHANNELS = 3 + 1 + 3 + 3 + 1 + 3 + 1     # kare, maske, 2 analitik çözüm + güven, çekilen + güven
CONFIDENCE_SCALE = 0.05                     # b₀: güvenin 1/e'ye düştüğü Laplace ölçeği (0..1 yoğunluk biriminde)


@dataclass
class MambaConfig:
    widths: Tuple[int, int, int, int] = (64, 128, 256, 456)   # 1/2, 1/4, 1/8, 1/16
    enc_depths: Tuple[int, int, int, int] = (2, 2, 4, 6)
    dec_depths: Tuple[int, int, int] = (3, 2, 2)              # 1/8, 1/4, 1/2 çözücü blokları
    conv_expansion: int = 4
    ssm_expand: int = 2
    ssm_state: int = 8
    ssm_conv: int = 3
    attn_window: int = 8
    attn_heads: int = 8
    ssm_fp32: bool = True
    use_checkpoint: bool = False
    infer_chunk: int = 4            # çıkarımda kare başına bloklar kaçar kare işlesin (0 = hepsi)
    ssm_token_chunk: int = 2048     # çıkarımda zamansal tarama kaçar uzamsal konum işlesin (0 = hepsi)


# =============================================================================
# Seçici durum uzayı (Mamba) — zaman ekseninde
# =============================================================================

class SelectiveSSM(nn.Module):
    """Tek yönlü seçici durum uzayı çekirdeği.

        h_t = exp(Δ_t A) · h_{t-1} + Δ_t B_t x_t          y_t = C_t · h_t + D x_t

    Δ, B, C girdiye bağlıdır ("seçicilik"): model hangi karenin bilgisinin duruma
    yazılacağına ve ne kadar süre tutulacağına içerikten karar verir.
    Girdi/çıktı: [N, T, D] (N = B·h·w uzamsal konum).
    """

    def __init__(self, dim: int, state: int = 8, conv: int = 3, fp32: bool = True):
        super().__init__()
        self.dim, self.state, self.fp32 = dim, state, fp32
        self.rank = max(1, math.ceil(dim / 16))
        self.conv = nn.Conv1d(dim, dim, conv, padding=conv - 1, groups=dim)
        self.x_proj = nn.Linear(dim, self.rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(self.rank, dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)).repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self._init_dt()

    def _init_dt(self, dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        """Δ başlangıçta [dt_min, dt_max] aralığına düşecek şekilde önyargı ayarı (Mamba ile aynı)."""
        std = self.rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))       # softplus'ın tersi

    def _scan(self, x: torch.Tensor, dt: torch.Tensor, b: torch.Tensor, c: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        n, t, d = x.shape
        a = -torch.exp(self.A_log.to(x.dtype))                               # [D, S], hep negatif → kararlı
        h = x.new_zeros(n, d, self.state)
        ys = []
        for i in range(t):
            decay = torch.exp(dt[:, i, :, None] * a)                         # [N, D, S]
            write = (dt[:, i] * x[:, i])[:, :, None] * b[:, i, None, :]      # [N, D, S]
            h_new = decay * h + write
            # Dolgu karesi duruma yazılmaz; durum bir sonraki geçerli kareye aynen taşınır.
            h = h_new if valid is None else torch.where(valid[:, i, None, None], h_new, h)
            ys.append((h * c[:, i, None, :]).sum(-1))                        # [N, D]
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = x.shape[1]
        x = F.silu(self.conv(x.transpose(1, 2))[..., :t].transpose(1, 2))    # nedensel 1B evrişim
        dt_low, b, c = self.x_proj(x).split([self.rank, self.state, self.state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_low))
        if self.fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                y = self._scan(x.float(), dt.float(), b.float(), c.float(), valid)
            y = y.to(x.dtype)
        else:
            y = self._scan(x, dt, b, c, valid)
        return y + x * self.D


class BiMamba(nn.Module):
    """Çift yönlü zamansal Mamba bloğu: ileri ve geri tarama, ortak giriş/çıkış izdüşümü.

    Girdi/çıktı: [B·T, C, h, w] (kare başına harita); içeride [B·h·w, T, C] dizisine açılır.
    """

    def __init__(self, dim: int, expand: int = 2, state: int = 8, conv: int = 3, fp32: bool = True, token_chunk: int = 0):
        super().__init__()
        inner = dim * expand
        self.token_chunk = token_chunk
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.fwd = SelectiveSSM(inner, state, conv, fp32)
        self.bwd = SelectiveSSM(inner, state, conv, fp32)
        self.out_proj = nn.Linear(inner, dim)
        nn.init.zeros_(self.out_proj.weight)        # blok özdeşlik olarak başlar
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: [B·T, C, h, w]; valid: [B, T] (bool veya 0/1 float)."""
        b, t = valid.shape
        _, c, h, w = x.shape
        seq = x.view(b, t, c, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        ok = (valid > 0.5).repeat_interleave(h * w, dim=0)                   # [B·h·w, T]
        # Her uzamsal konum bağımsız bir zaman dizisidir. Çıkarımda konumlar dilim dilim
        # taranır: 720p'de 1/8 ölçekte 14.400 konumun genişletilmiş (2×) dizileri tek seferde
        # gigabaytlarca geçici bellek ister. Sonuç birebir aynıdır.
        count, step = seq.shape[0], self.token_chunk
        if not torch.is_grad_enabled() and 0 < step < count:
            seq = torch.cat([self._mix(seq[i: i + step], ok[i: i + step]) for i in range(0, count, step)])
        else:
            seq = self._mix(seq, ok)
        return seq.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).reshape(b * t, c, h, w)

    def _mix(self, seq: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """[N, T, C] dizilerinde ileri + geri tarama ve artık bağlantı."""
        u, gate = self.in_proj(self.norm(seq)).chunk(2, dim=-1)
        u = u * ok.unsqueeze(-1).to(u.dtype)
        y = self.fwd(u, ok) + self.bwd(u.flip(1), ok.flip(1)).flip(1)
        return seq + self.out_proj(y * F.silu(gate))


# =============================================================================
# Yerel (pencere içi) uzamsal dikkat
# =============================================================================

class WindowAttention(nn.Module):
    """Örtüşmeyen ws×ws pencerelerde çok başlı öz-dikkat + göreli konum önyargısı.

    Pencereler arası iletişimi komşu evrişim blokları (7×7) sağlar. Pencere katına
    tamamlamak için eklenen dolgu konumlarına dikkat edilmez.
    """

    def __init__(self, dim: int, window: int = 8, heads: int = 8):
        super().__init__()
        self.window, self.heads = window, heads
        self.norm = LayerNorm2d(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * window - 1) + rel[..., 1], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        ws = self.window
        ph, pw = (-h) % ws, (-w) % ws
        y = F.pad(self.norm(x), (0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        # [n, c, hp, wp] → [n·pencere, ws², c]
        tokens = y.view(n, c, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        bias = self.bias_table[self.rel_index].permute(2, 0, 1).unsqueeze(0).to(tokens.dtype)   # [1, başlık, L, L]
        if ph or pw:
            real = F.pad(x.new_ones(1, 1, h, w), (0, pw, 0, ph))
            real = real.view(1, 1, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, 1, 1, ws * ws)
            bias = bias + (1.0 - real.repeat(n, 1, 1, 1)).to(tokens.dtype) * -1e4
        q, k, v = self.qkv(tokens).view(-1, ws * ws, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = self.proj(out.transpose(1, 2).reshape(-1, ws * ws, c))
        out = out.view(n, hp // ws, wp // ws, ws, ws, c).permute(0, 5, 1, 3, 2, 4).reshape(n, c, hp, wp)
        return x + out[:, :, :h, :w]


class SpatioTemporalBlock(nn.Module):
    """Uzamsal evrişim → zamansal Bi-SSM → (isteğe bağlı) yerel dikkat."""

    def __init__(self, dim: int, cfg: MambaConfig, attention: bool):
        super().__init__()
        self.conv = ConvBlock(dim, cfg.conv_expansion, kernel=7)
        self.ssm = BiMamba(dim, cfg.ssm_expand, cfg.ssm_state, cfg.ssm_conv, cfg.ssm_fp32, cfg.ssm_token_chunk)
        self.attn = WindowAttention(dim, cfg.attn_window, cfg.attn_heads) if attention else None
        self.chunk = cfg.infer_chunk

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Evrişim ve dikkat kare başınadır (parçalanabilir); Bi-SSM tüm kareleri birlikte görür.
        x = self.ssm(run_block(self.conv, x, chunk=self.chunk), valid)
        return x if self.attn is None else run_block(self.attn, x, chunk=self.chunk)


# =============================================================================
# Model
# =============================================================================

class VideoMamba(nn.Module):
    def __init__(self, config: Optional[MambaConfig] = None):
        super().__init__()
        cfg = self.config = config or MambaConfig()
        w, d = cfg.widths, cfg.enc_depths
        conv = lambda dim: ConvBlock(dim, cfg.conv_expansion, kernel=7)

        # PixelUnshuffle kayıpsız bir yeniden dizilimdir: ince yazı çizgileri adımlı evrişimde kaybolmaz.
        self.embed = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(IN_CHANNELS * 4, w[0], 3, padding=1), LayerNorm2d(w[0]))
        self.enc1 = nn.ModuleList([conv(w[0]) for _ in range(d[0])])
        self.enc2 = nn.ModuleList([conv(w[1]) for _ in range(d[1])])
        self.enc3 = nn.ModuleList([SpatioTemporalBlock(w[2], cfg, attention=False) for _ in range(d[2])])
        self.enc4 = nn.ModuleList([SpatioTemporalBlock(w[3], cfg, attention=True) for _ in range(d[3])])
        self.down = nn.ModuleList([Downsample(w[i], w[i + 1]) for i in range(3)])

        self.up = nn.ModuleList([Upsample(w[3 - i], w[2 - i]) for i in range(3)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * w[2 - i], w[2 - i], 1) for i in range(3)])
        self.dec = nn.ModuleList([nn.ModuleList([conv(w[2 - i]) for _ in range(cfg.dec_depths[i])]) for i in range(3)])
        self.head = nn.Sequential(nn.Conv2d(w[0], w[0], 3, padding=1), nn.GELU(), nn.Conv2d(w[0], 4 * 4, 1), nn.PixelShuffle(2))
        with torch.no_grad():
            # Başlangıçta artık = 0 (çıktı = taban görüntü) ve b ≈ e⁻³ ≈ 0,05.
            self.head[2].weight.zero_()
            self.head[2].bias.zero_()
            self.head[2].bias[12:16] = -3.0

    def _decode(self, x: torch.Tensor, s3: torch.Tensor, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """1/16 ölçekli özelliklerden tam çözünürlüklü çıktıya (artık RGB + log b); kare başına."""
        for i, skip in enumerate((s3, s2, s1)):
            x = self.fuse[i](torch.cat([self.up[i](x), skip], dim=1))
            for blk in self.dec[i]:
                x = run_block(blk, x, use_checkpoint=self.config.use_checkpoint)
        return self.head(x)

    @staticmethod
    def base_image(frames: torch.Tensor, priors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Önsellerin güvenle harmanlandığı taban: çekilen piksel > analitik çözüm > orijinal."""
        ac = priors["analytic_conf"].to(frames.dtype)
        pc = priors["pulled_conf"].to(frames.dtype)
        analytic = ac * priors["bg_gamma"].to(frames.dtype) + (1.0 - ac) * frames
        return pc * priors["pulled"].to(frames.dtype) + (1.0 - pc) * analytic

    def forward(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        priors: Dict[str, torch.Tensor],
        valid: Optional[torch.Tensor] = None,
        target: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        valid = check_window(frames, valid)
        cfg = self.config
        ckpt, chunk = cfg.use_checkpoint, cfg.infer_chunk
        b, t, _, h, w = frames.shape
        dtype = frames.dtype
        vf = valid.to(dtype)

        # 15 kanallı girdi tüm kareler için tek seferde kurulmaz: 720p × 30 karede yalnızca
        # bu tensör ~1,7 GiB (fp32) tutar. Çıkarımda dilim dilim kurulup gömülür.
        parts = [
            p.flatten(0, 1)
            for p in (
                frames, mask, priors["bg_gamma"], priors["bg_linear"], priors["analytic_conf"],
                priors["pulled"], priors["pulled_conf"],
            )
        ]
        keep = vf.reshape(b * t, 1, 1, 1)                                    # dolgu kareleri sıfır

        def embed(lo: int, hi: int) -> torch.Tensor:
            return self.embed(torch.cat([p[lo:hi].to(dtype) for p in parts], dim=1) * keep[lo:hi])

        total = b * t
        sliced = not torch.is_grad_enabled() and 0 < chunk < total
        x = torch.cat([embed(i, min(total, i + chunk)) for i in range(0, total, chunk)]) if sliced else embed(0, total)

        # ---- kodlayıcı (tüm kareler; zamansal bilgi burada karışır) ----
        for blk in self.enc1:
            x = run_block(blk, x, use_checkpoint=ckpt, chunk=chunk)
        s1 = x
        x = self.down[0](x)
        for blk in self.enc2:
            x = run_block(blk, x, use_checkpoint=ckpt, chunk=chunk)
        s2 = x
        x = self.down[1](x)
        for blk in self.enc3:
            x = run_block(blk, x, vf, use_checkpoint=ckpt)
        s3 = x
        x = self.down[2](x)
        for blk in self.enc4:
            x = run_block(blk, x, vf, use_checkpoint=ckpt)

        # ---- çözücü (kare başına; Mod A'da yalnızca hedef kare) ----
        if target is not None:
            if not 0 <= target < t:
                raise ValueError(f"target 0..{t - 1} aralığında olmalı")
            pick = lambda z: z.view(b, t, *z.shape[1:])[:, target]
            x, s1, s2, s3 = pick(x), pick(s1), pick(s2), pick(s3)
            index = [target]
        else:
            index = list(range(t))
        # Çözücü kare başınadır; çıkarımda dilim dilim çalışır ki 1/2 ölçekteki birleştirme
        # (atlama bağlantısı + büyütülmüş harita) 30 kare için aynı anda bellekte durmasın.
        count = x.shape[0]
        if not torch.is_grad_enabled() and 0 < chunk < count:
            out = torch.cat(
                [self._decode(x[i: i + chunk], s3[i: i + chunk], s2[i: i + chunk], s1[i: i + chunk]) for i in range(0, count, chunk)]
            )
        else:
            out = self._decode(x, s3, s2, s1)
        out = out.view(b, len(index), 4, h, w)

        base = self.base_image(frames[:, index], {k: v[:, index] for k, v in priors.items()})
        rgb = (base + out[:, :, :3]).clamp(0.0, 1.0)
        log_b = out[:, :, 3:4].float().clamp(-7.0, 2.0)
        confidence = torch.exp(-torch.exp(log_b) / CONFIDENCE_SCALE)
        return {"rgb": rgb, "log_b": log_b, "confidence": confidence}
