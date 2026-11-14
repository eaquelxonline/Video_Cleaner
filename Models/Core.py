"""VC-Core (Python) — Video_Cleaner v2.0 ortak çekirdeği ve iki modlu birleşik motor.

Bu dosya üç modelin (VC-Detector = Video_Detector, VC-Flow = Video_Flow, VC-Mamba =
Video_Mamba) paylaştığı yapı taşlarını ve onları tek üretim hattında birleştiren
``VideoCleaner`` motorunu içerir. Android tarafındaki karşılığı Rust/Core.rs (halka, pencere
zamanlayıcı, bit-exact kapı) ve Kotlin/Core.kt'dir (paket com.video.cleaner).

    kareler ─► Detector ─► maske, alfa, kaplama rengi, sahne yazısı, güven
            ─► Flow     ─► ileri/geri akış + güven            (maskeli karelerden)
            ─► Deterministik kurtarma
                   1) analitik alfa çözümü      B = (I − αF) / (1 − α)
                   2) tek seferlik piksel çekme (akış haritaları taşınır,
                      renk kaynak kareden yalnızca BİR kez örneklenir)
            ─► Mamba    ─► kalan belirsiz bölge için onarım + belirsizlik
                           (çıkarımda yalnızca dinamik ROI tüpünde; maske yoksa hiç çalışmaz)
            ─► Bit-exact kapı: maskesiz pikseller ``torch.where`` ile aynen geçer

Tensör sözleşmesi (tüm modeller için ortak)
-------------------------------------------
    frames : float [B, T, 3, H, W]   RGB, 0..1, gama kodlu (BT.709/sRGB)
    valid  : bool  [B, T]            False = dolgu konumu (sahne/akış sınırı)
    T      : 15 (WINDOW); H ve W 16'nın katı olmalı (720p: 1280×720 ✓)

Pencere modları (Rust/Core.rs içindeki Scheduler ile aynı tanım)
----------------------------------------------------------------
    Mod B (11+4)  : 15 kare girer, 15 kare çıkar; pencere 11 kare kayar, ardışık iki pencere 4 karede
                    örtüşür ve örtüşen tahminler ``öncelik × güven`` ile harmanlanır (sınır sıçraması yok).
    Mod A (7+1+7) : aynı 15 kare girer, yalnızca 7. konum çözülür.
    İki mod AYNI ağırlıkları ve AYNI kodlayıcıyı kullanır; aynı pencere için
    Mod A çıktısı Mod B çıktısının 7. karesine eşittir (test_engine ile denetlenir).
    14 Kasım telefon ölçümünden sonra 30 → 15: 30 karelik ara bellek halkası 526 MB'tı.

İçe aktarma: dosyalar ``Models/`` dizininden düz modül olarak yüklenir
(``python Models/Trainer.py`` veya ``sys.path``'e ``Models`` eklenerek).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

WINDOW = 15          # modelin tek geçişte gördüğü kare sayısı
STRIDE = 11          # Mod B'nin kayması
OVERLAP = WINDOW - STRIDE   # ardışık pencerelerin ortak kare sayısı (4; harmanlanır)
TARGET_A = 7         # Mod A'da hedef karenin konumu (7 geçmiş + hedef + 7 gelecek)

# Başlangıç parametre bütçeleri. Bunlar ÜST SINIRDIR: modeller bu sayıları
# doldurmak için kullanılmayan parametre içermez (bkz. parameter_report).
PARAM_BUDGET = {"detector": 1 << 22, "flow": 1 << 23, "mamba": 1 << 25}
# Resmi sistem adları (rapor ve kayıtlarda kullanılır).
SYSTEM_NAMES = {"detector": "VC-Detector", "flow": "VC-Flow", "mamba": "VC-Mamba", "total": "toplam"}


# =============================================================================
# 1. Ortak katmanlar
# =============================================================================

class Norm2d(nn.BatchNorm2d):
    """Evrişim bloklarının normalizasyonu: BatchNorm2d, her zaman bir evrişimin ARDINDAN (son-norm).

    Çıkarımda BN sabit bir kanal başına ölçek + kaydırmadır ve önündeki evrişime katlanır (ORT
    Conv+BN birleştirmesi): grafikte ayrı düğüm kalmaz, QDQ'da evrişim + ReLU tek QLinearConv olur.
    14 Kasım telefon ölçümü: LayerNorm2d (GroupNorm(1)) birleşmeyi kırıyordu; x86'da INT8 det_enc
    92,1 → 35,5 ms. Eğitimde istatistik B·T kare üzerinden toplanır (pencere başına 15 kare).
    Dizi normalizasyonu (Bi-SSM, pencere dikkati, bağlam köprüsü) zamansal çekirdekte değişmeden kalır.
    """

    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-5, momentum=0.01)


class ConvBlock(nn.Module):
    """ConvNeXt tarzı artık blok (mobil): derinlik-yönlü k×k → BN → 1×1 genişlet → ReLU → 1×1.

    ReLU, GELU'nun aksine QDQ'da evrişimle tek QLinearConv'a kaynaşır. Katman ölçeği (``gamma``)
    eğitimde ayrı parametredir; dışa aktarımda ``fold_layer_scale`` ile ``pw2``'ye katlanır.
    """

    def __init__(self, dim: int, expansion: int = 4, kernel: int = 7, layer_scale: float = 1e-2):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = Norm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, 1)
        self.pw2 = nn.Conv2d(dim * expansion, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), layer_scale))
        self.scale_folded = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(F.relu(self.pw1(self.norm(self.dw(x)))))
        return x + (y if self.scale_folded else self.gamma * y)


def fold_layer_scale(model: nn.Module) -> nn.Module:
    """Çıkarım için: ``x + γ·conv(z)`` → ``x + conv'(z)`` (γ son evrişimin ağırlık ve önyargısına katlanır).

    Sonuç birebir aynıdır; grafikte γ çarpımı (QDQ'da evrişim ile toplama arasındaki ayrı Mul) kalmaz.
    ``scale_folded`` ve ``gamma`` taşıyan her modülün katlanacak evrişimi ``scale_target`` ile adlandırılır.
    """
    with torch.no_grad():
        for m in model.modules():
            if getattr(m, "scale_folded", None) is False and isinstance(getattr(m, "gamma", None), torch.Tensor):
                conv = getattr(m, getattr(m, "scale_target", "pw2"))
                g = m.gamma.detach().reshape(-1)
                conv.weight.mul_(g.view(-1, *([1] * (conv.weight.dim() - 1))))
                if conv.bias is not None:
                    conv.bias.mul_(g)
                m.scale_folded = True
    return model


class Downsample(nn.Module):
    """2× küçültme: adım-2 evrişim → BN (son-norm; BN evrişime katlanır)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 2, stride=2)
        self.norm = Norm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class Upsample(nn.Module):
    """2× büyütme: çift doğrusal örnekleme + 3×3 evrişim (dama tahtası izi bırakmaz)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


def run_block(block: nn.Module, *args: torch.Tensor, use_checkpoint: bool = False, chunk: int = 0) -> torch.Tensor:
    """Bloğu isteğe bağlı aktivasyon denetim noktasıyla veya parçalı olarak çalıştırır.

    * Eğitim (gradyan açık) + ``use_checkpoint``: ara aktivasyonlar saklanmaz, geri
      yayılımda yeniden hesaplanır — VRAM karşılığında süre.
    * Çıkarım (gradyan kapalı) + ``chunk`` > 0: KARE BAŞINA çalışan bloklar yığın
      boyutunda ``chunk``'lık dilimlerle işlenir. Sonuç birebir aynıdır (kareler bu
      bloklarda birbirinden bağımsızdır); yalnızca blok içi geçici bellek (ör. 1/2
      ölçekte 4× genişletme) 15 kare yerine ``chunk`` kare kadar yer tutar.
    """
    if torch.is_grad_enabled():
        if use_checkpoint and any(a.requires_grad for a in args if torch.is_tensor(a)):
            return _checkpoint(block, *args, use_reentrant=False)
    elif chunk > 0 and len(args) == 1 and args[0].shape[0] > chunk:
        return torch.cat([block(part) for part in args[0].split(chunk)], dim=0)
    return block(*args)


def flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """[B, T, C, H, W] → [B·T, C, H, W]."""
    b, t = x.shape[:2]
    return x.reshape(b * t, *x.shape[2:]), b


def unflatten_time(x: torch.Tensor, batch: int) -> torch.Tensor:
    """[B·T, C, H, W] → [B, T, C, H, W]."""
    return x.reshape(batch, -1, *x.shape[1:])


def count_parameters(module: nn.Module) -> int:
    """Eğitilebilir parametre sayısı."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_window(frames: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Girdi sözleşmesini doğrular ve geçerlilik maskesini döndürür."""
    if frames.dim() != 5 or frames.shape[2] != 3:
        raise ValueError(f"frames [B, T, 3, H, W] olmalı, gelen: {tuple(frames.shape)}")
    b, t, _, h, w = frames.shape
    if h % 16 or w % 16:
        raise ValueError(f"H ve W 16'nın katı olmalı, gelen: {h}×{w}")
    if valid is None:
        valid = torch.ones(b, t, dtype=torch.bool, device=frames.device)
    if valid.shape != (b, t):
        raise ValueError(f"valid [B, T] olmalı, gelen: {tuple(valid.shape)}")
    return valid.bool()


# =============================================================================
# 2. Renk uzayı ve analitik alfa çözümü
# =============================================================================

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Gama kodlu 0..1 değerleri doğrusal ışığa çevirir (sRGB parçalı eğrisi)."""
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Doğrusal ışığı gama kodlu 0..1 değerlere çevirir."""
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-8) ** (1.0 / 2.4) - 0.055)


def composite(background: torch.Tensor, fg: torch.Tensor, alpha: torch.Tensor, space: str = "gamma") -> torch.Tensor:
    """I = αF + (1 − α)B. ``space``: harmanlamanın yapıldığı uzay ("gamma" | "linear")."""
    if space == "gamma":
        return alpha * fg + (1.0 - alpha) * background
    if space == "linear":
        return linear_to_srgb(alpha * srgb_to_linear(fg) + (1.0 - alpha) * srgb_to_linear(background))
    raise ValueError(f"bilinmeyen harmanlama uzayı: {space}")


def analytic_background(
    image: torch.Tensor, alpha: torch.Tensor, fg: torch.Tensor, alpha_max: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Yarı saydam kaplamanın altındaki arka planı kapalı biçimde çözer.

        B = (I − αF) / (1 − α)

    Kaplamanın hangi uzayda harmanlandığı videodan bilinemez: gömülü altyazıların
    çoğu (ffmpeg/libass, mobil düzenleyiciler) gama kodlu değerler üzerinde,
    profesyonel kompozitleme ise doğrusal ışıkta harmanlar. Bu yüzden çözüm HER İKİ
    uzayda da üretilir; Mamba ikisini de önsel olarak alır ve uygun olanı seçmeyi
    öğrenir.

    Dönen ``reliability`` = (1 − α): I'daki gürültü 1/(1 − α) ile büyüdüğünden α → 1
    iken çözüm kullanılamaz; α > ``alpha_max`` için güven 0'dır (opak kaplamanın
    altı bu denklemle geri getirilemez).
    """
    a = alpha.clamp(0.0, alpha_max)
    inv = 1.0 / (1.0 - a)
    bg_gamma = ((image - a * fg) * inv).clamp(0.0, 1.0)
    bg_linear = linear_to_srgb(((srgb_to_linear(image) - a * srgb_to_linear(fg)) * inv).clamp(0.0, 1.0))
    reliability = (1.0 - alpha).clamp(0.0, 1.0) * (alpha <= alpha_max).to(image.dtype)
    return bg_gamma, bg_linear, reliability


# =============================================================================
# 3. Optik akış yardımcıları ve tek seferlik piksel çekme
# =============================================================================

def coords_grid(h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Piksel koordinat ızgarası [1, 2, H, W]; kanal 0 = x, kanal 1 = y."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij"
    )
    return torch.stack([xs, ys], dim=0).unsqueeze(0)


def sample_at(x: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """``x`` [N, C, H, W] haritasını mutlak piksel koordinatlarında [N, 2, H', W'] örnekler."""
    h, w = x.shape[-2:]
    # Koordinat aritmetiği her zaman fp32'de yapılır: BF16'nın 8 bitlik mantisi 1280
    # piksellik bir eksende alt-piksel konumu taşıyamaz.
    with torch.autocast(device_type=x.device.type, enabled=False):
        coords = coords.float()
        gx = 2.0 * coords[:, 0] / max(w - 1, 1) - 1.0
        gy = 2.0 * coords[:, 1] / max(h - 1, 1) - 1.0
        out = F.grid_sample(x.float(), torch.stack([gx, gy], dim=-1), mode=mode, padding_mode="border", align_corners=True)
    return out.to(x.dtype)


def warp(x: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """``x``'i geriye doğru çarpıtır: çıktı(p) = x(p + flow(p))."""
    grid = coords_grid(x.shape[-2], x.shape[-1], x.device, flow.dtype)
    return sample_at(x, grid + flow, mode)


def in_bounds(flow: torch.Tensor) -> torch.Tensor:
    """p + flow(p) kare içinde mi? [N, 1, H, W] bool."""
    h, w = flow.shape[-2:]
    q = coords_grid(h, w, flow.device, flow.dtype) + flow
    return ((q[:, 0:1] >= 0) & (q[:, 0:1] <= w - 1) & (q[:, 1:2] >= 0) & (q[:, 1:2] <= h - 1))


def occlusion_mask(flow_ab: torch.Tensor, flow_ba: torch.Tensor, a1: float = 0.01, a2: float = 0.5) -> torch.Tensor:
    """İleri-geri tutarlılık denetimi. True = a'daki piksel b'de örtülü/güvenilmez.

    Tutarlı bir eşleşmede ``flow_ab(p) + flow_ba(p + flow_ab(p)) ≈ 0`` olmalıdır.
    """
    back = warp(flow_ba, flow_ab)
    diff = (flow_ab + back).pow(2).sum(1, keepdim=True)
    mag = flow_ab.pow(2).sum(1, keepdim=True) + back.pow(2).sum(1, keepdim=True)
    return (diff > a1 * mag + a2) | ~in_bounds(flow_ab)


def pull_pixels(
    frames: torch.Tensor,
    clean: torch.Tensor,
    flow_fw: torch.Tensor,
    flow_bw: torch.Tensor,
    reliable_fw: torch.Tensor,
    reliable_bw: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    decay: float = 0.92,
    select: str = "block",
    block: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maskeli pikselleri, başka karelerdeki temiz karşılıklarından TEK örneklemeyle çeker.

    Klasik yayılımda renk kareden kareye defalarca çarpıtılır ve her adımda keskinlik
    kaybolur. Burada renk yerine *kaynak haritası* taşınır: her piksel "hangi karenin
    hangi koordinatından geleceğini" (kaynak kare no, x, y) tutar. Zincir tamamlanınca
    renk, kaynak kareden yalnızca bir kez (5 boyutlu ``grid_sample``) okunur.

    Args:
        frames      [B, T, 3, H, W]  kaplamalı kareler
        clean       [B, T, 1, H, W]  bool — güvenilir kaynak pikseller (maskesiz)
        flow_fw     [B, T-1, 2, H, W] t → t+1 akışı (maske altında tamamlanmış)
        flow_bw     [B, T-1, 2, H, W] t+1 → t akışı
        reliable_fw [B, T-1, 1, H, W] bool — flow_fw güvenilir mi (örtülme yok, güven yüksek)
        reliable_bw [B, T-1, 1, H, W] bool
        valid       [B, T] bool — dolgu kareleri kaynak olamaz, zincir onlardan geçemez
        decay       her sıçramada güvenin çarpıldığı katsayı
        select      ileri (gelecek) ve geri (geçmiş) zincirden gelen iki adayın birleşimi:
                    "blend": güvenle ağırlıklı ortalama;
                    "block": ``block``×``block`` alt blok başına ortalama güveni yüksek olan YÖN
                    seçilir (anizotropik komşu seçimi): blok içinde tek kaynak yönü kalır, iki
                    farklı kaynağın ortalamasından doğan hayalet/çift kenar oluşmaz. Seçilen yönün
                    çözemediği pikselde diğer yön kullanılır.

    Returns:
        pulled [B, T, 3, H, W] — çekilen renk (çözülemeyen piksellerde orijinal)
        conf   [B, T, 1, H, W] — 0..1; temiz pikselde 1, çözülemeyende 0
    """
    b, t, _, h, w = frames.shape
    dtype = flow_fw.dtype
    grid = coords_grid(h, w, frames.device, dtype)
    clean = clean.bool()
    if valid is not None:
        clean = clean & valid.view(b, t, 1, 1, 1)

    def sweep(direction: int):
        """direction=+1: kaynaklar gelecek karelerde; -1: geçmiş karelerde."""
        xy = [grid.expand(b, 2, h, w) for _ in range(t)]
        src = [torch.full((b, 1, h, w), float(i), device=frames.device, dtype=dtype) for i in range(t)]
        hops = [torch.zeros(b, 1, h, w, device=frames.device, dtype=dtype) for _ in range(t)]
        ok = [clean[:, i] for i in range(t)]
        order = range(t - 2, -1, -1) if direction > 0 else range(1, t)
        for i in order:
            nb = i + direction
            flow = flow_fw[:, i] if direction > 0 else flow_bw[:, i - 1]
            reliable = reliable_fw[:, i] if direction > 0 else reliable_bw[:, i - 1]
            q = grid + flow
            state = torch.cat([xy[nb], src[nb], hops[nb], ok[nb].to(dtype)], dim=1)
            lin = sample_at(state, q, "bilinear")
            src_near = sample_at(src[nb], q, "nearest")
            # Dört komşunun hepsi çözülmüş ve AYNI kaynak kareyi gösteriyor olmalı;
            # aksi hâlde koordinatların ara değeri anlamsızdır.
            consistent = (lin[:, 4:5] > 0.999) & ((lin[:, 2:3] - src_near).abs() < 1e-3)
            take = (~ok[i]) & reliable.bool() & in_bounds(flow) & consistent
            xy[i] = torch.where(take, lin[:, 0:2], xy[i])
            src[i] = torch.where(take, src_near, src[i])
            hops[i] = torch.where(take, lin[:, 3:4] + 1.0, hops[i])
            ok[i] = ok[i] | take
        return torch.stack(xy, 1), torch.stack(src, 1), torch.stack(hops, 1), torch.stack(ok, 1)

    def gather(xy: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        gx = 2.0 * xy[:, :, 0] / max(w - 1, 1) - 1.0
        gy = 2.0 * xy[:, :, 1] / max(h - 1, 1) - 1.0
        gt = 2.0 * src[:, :, 0] / max(t - 1, 1) - 1.0
        grid5 = torch.stack([gx, gy, gt], dim=-1).to(frames.dtype)          # [B, T, H, W, 3]
        volume = frames.permute(0, 2, 1, 3, 4)                              # [B, 3, T, H, W]
        out = F.grid_sample(volume, grid5, mode="bilinear", padding_mode="border", align_corners=True)
        return out.permute(0, 2, 1, 3, 4)

    xy_f, src_f, hops_f, ok_f = sweep(+1)
    xy_p, src_p, hops_p, ok_p = sweep(-1)
    w_f = ok_f.to(frames.dtype) * decay ** hops_f.to(frames.dtype)
    w_p = ok_p.to(frames.dtype) * decay ** hops_p.to(frames.dtype)
    if select == "block" and h % block == 0 and w % block == 0:
        mf = F.avg_pool2d(w_f.flatten(0, 1), block)
        mp = F.avg_pool2d(w_p.flatten(0, 1), block)
        prefer_f = F.interpolate((mf >= mp).to(frames.dtype), scale_factor=block, mode="nearest").view_as(w_f) > 0.5
        use_f = torch.where(prefer_f, w_f > 0, w_p <= 0)
        w_f = torch.where(use_f, w_f, torch.zeros_like(w_f))
        w_p = torch.where(use_f, torch.zeros_like(w_p), w_p)
    elif select not in ("blend", "block"):
        raise ValueError("select 'blend' veya 'block' olmalı")
    total = w_f + w_p
    mixed = (w_f * gather(xy_f, src_f) + w_p * gather(xy_p, src_p)) / total.clamp_min(1e-6)
    solved = total > 0
    # Temiz pikseller hiçbir aradeğerlemeden geçmez: orijinal değer aynen kalır.
    pulled = torch.where(solved & ~clean, mixed, frames)
    conf = torch.where(clean, torch.ones_like(total), torch.maximum(w_f, w_p))
    return pulled, conf


# =============================================================================
# 4. Haar dalgacık dönüşümü (kenar/frekans kaybı için)
# =============================================================================

def haar_dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tek seviyeli 2×2 Haar. Dönen: (LL [.., C, H/2, W/2], detay [.., 3C, H/2, W/2]).

    Assembly çekirdeğiyle aynı tanım (4'e bölünmüş): LL = (a+b+c+d)/4 vb. Dört bant
    toplamda girişle aynı sayıda değer taşır; dönüşüm belleği küçültmez.
    """
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.25
    hl = (a - b + c - d) * 0.25
    lh = (a + b - c - d) * 0.25
    hh = (a - b - c + d) * 0.25
    return ll, torch.cat([hl, lh, hh], dim=-3)


def haar_idwt(ll: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """``haar_dwt`` dönüşümünün tersi."""
    hl, lh, hh = detail.chunk(3, dim=-3)
    out = ll.new_zeros(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2)
    out[..., 0::2, 0::2] = ll + hl + lh + hh
    out[..., 0::2, 1::2] = ll - hl + lh - hh
    out[..., 1::2, 0::2] = ll + hl - lh - hh
    out[..., 1::2, 1::2] = ll - hl - lh + hh
    return out


# =============================================================================
# 5. Birleşik motor
# =============================================================================

@dataclass
class EngineConfig:
    """Motor ayarları. Model yapılandırmaları ilgili dosyalardaki varsayılanlardır."""

    detector: object = None
    flow: object = None
    mamba: object = None
    mask_threshold: float = 0.5        # silme olasılığı eşiği
    mask_dilation: int = 2             # kenar yumuşatması ve sıkıştırma halesi için genişletme (piksel)
    flow_confidence: float = 0.5       # akışın güvenilir sayılması için alt sınır
    alpha_max: float = 0.95            # bunun üstünde analitik çözüm kullanılmaz
    pull_decay: float = 0.92           # piksel çekmede sıçrama başına güven çarpanı
    pull_select: str = "block"         # ileri/geri kaynak birleşimi: "block" (anizotropik seçim) | "blend"
    pull_block: int = 8                # anizotropik seçim alt blok boyu (piksel)
    roi_tube: bool = True              # çıkarımda VC-Mamba yalnızca maskenin zamansal sınır kutusunda çalışsın
    roi_margin: int = 32               # ROI çevresinde bırakılan bağlam payı (piksel)
    roi_max_fraction: float = 0.6      # ROI karenin bu oranından büyükse tam kare işlenir
    # Geri yazma kenarı alfa yumuşatması (piksel; 0 = sert geçiş). Onarım bölgesi silme maskesinden bu kadar
    # daha genişletilir; dış kenardan içe kosinüs rampayla orijinale geçilir, bölge dışı bit-exact kalır.
    # Rust hattındaki Pipeline::set_feather ile aynı tanım (orada 3-4 chamfer, burada satranç tahtası mesafe).
    feather_px: int = 4


def roi_tube(
    mask: torch.Tensor, valid: torch.Tensor, margin: int, max_fraction: float, multiple: int = 16
) -> Optional[Tuple[int, int, int, int]]:
    """Pencere boyunca silinecek bölgenin (tüm karelerin birleşimi) sınır kutusu: (y0, y1, x0, x1).

    Kutu ``margin`` kadar genişletilir ve kenarları ``multiple``'ın katına oturtulur (model
    sözleşmesi). Maske boşsa ``(0, 0, 0, 0)`` (hiçbir şey çalıştırma); kutu karenin
    ``max_fraction`` oranından büyükse ``None`` (tam kare).
    """
    h, w = mask.shape[-2:]
    m = (mask.bool() & valid.view(*valid.shape, 1, 1, 1)).flatten(0, 2).any(0)       # [H, W]
    if not bool(m.any()):
        return (0, 0, 0, 0)
    ys = torch.nonzero(m.any(1)).flatten()
    xs = torch.nonzero(m.any(0)).flatten()

    def span(lo: int, hi: int, size: int) -> Tuple[int, int]:
        lo = max(0, lo - margin) // multiple * multiple
        hi = min(size, -(-(hi + margin) // multiple) * multiple)
        if (hi - lo) % multiple:
            hi = min(size, lo + -(-(hi - lo) // multiple) * multiple)
            lo = max(0, hi - -(-(hi - lo) // multiple) * multiple)
        return lo, hi

    y0, y1 = span(int(ys[0]), int(ys[-1]) + 1, h)
    x0, x1 = span(int(xs[0]), int(xs[-1]) + 1, w)
    if (y1 - y0) % multiple or (x1 - x0) % multiple or (y1 - y0) * (x1 - x0) > max_fraction * h * w:
        return None
    return (y0, y1, x0, x1)


def feather_blend(src: torch.Tensor, est: torch.Tensor, gate: torch.Tensor, feather: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact kapı + kenar yumuşatması. gate [.., 1, H, W] bool (silme maskesi).

    Bölge = gate'in ``feather`` piksel genişletilmişi. Bölge içindeki piksel, dış kenara satranç tahtası
    mesafesi d (1 = kenar komşusu) ile α = ½(1 − cos(π·d/(F+1))) ağırlık alır; d > F ise α = 1 (gate'in
    tamamı bu alandadır). Bölge dışı ``torch.where`` ile girdiden AYNEN seçilir. Dönen: (çıktı, bölge).
    """
    if feather <= 0:
        return torch.where(gate, est.to(src.dtype), src), gate
    region = dilate(gate, feather)
    flat = region.reshape(-1, 1, *region.shape[-2:]).float()
    depth = flat.clone()
    eroded = flat
    for _ in range(feather):
        # Kare kenarının ötesi bölge İÇİ sayılır (köşedeki filigran kenara doğru yarım kalmasın).
        eroded = -F.max_pool2d(-F.pad(eroded, (1, 1, 1, 1), value=1.0), 3, stride=1)
        depth = depth + eroded
    alpha = torch.where(depth > feather, torch.ones_like(depth), 0.5 * (1.0 - torch.cos(math.pi * depth / (feather + 1))))
    alpha = alpha.reshape(region.shape).to(src.dtype)
    blended = alpha * est.to(src.dtype) + (1.0 - alpha) * src
    return torch.where(region, blended, src), region


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """İkili/olasılık maskesini ``radius`` piksel genişletir. [..., 1, H, W]."""
    if radius <= 0:
        return mask
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    out = F.max_pool2d(flat, 2 * radius + 1, stride=1, padding=radius)
    return out.reshape(mask.shape).to(mask.dtype)


class VideoCleaner(nn.Module):
    """Detector + Flow + deterministik kurtarma + Mamba + bit-exact kapı."""

    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__()
        # Döngüsel içe aktarmayı önlemek için modeller burada yüklenir.
        from Video_Detector import DetectorConfig, VideoDetector
        from Video_Flow import FlowConfig, VideoFlow
        from Video_Mamba import MambaConfig, VideoMamba

        self.config = config or EngineConfig()
        self.detector = VideoDetector(self.config.detector or DetectorConfig())
        self.flow = VideoFlow(self.config.flow or FlowConfig())
        self.mamba = VideoMamba(self.config.mamba or MambaConfig())

    # ---- parametre bütçesi ---------------------------------------------------

    def parameter_report(self) -> Dict[str, Dict[str, float]]:
        """Model başına gerçek parametre sayısı, bütçe ve BF16 ağırlık belleği (MiB)."""
        report = {}
        for name in ("detector", "flow", "mamba"):
            n = count_parameters(getattr(self, name))
            report[name] = {
                "params": n,
                "budget": PARAM_BUDGET[name],
                "usage": n / PARAM_BUDGET[name],
                "bf16_mib": n * 2 / (1024 ** 2),
            }
        total = sum(r["params"] for r in report.values())
        budget = sum(PARAM_BUDGET.values())
        report["total"] = {"params": total, "budget": budget, "usage": total / budget, "bf16_mib": total * 2 / (1024 ** 2)}
        return report

    # ---- aşamalar ------------------------------------------------------------

    def erase_mask(self, det: Dict[str, torch.Tensor], valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detector çıktısından (yumuşak, sert) silme maskesi üretir.

        Sahne yazısı (tabela, plaka, ekran içi metin) olasılığı yüksek pikseller
        maskeden düşülür: bunlar görüntüye aittir ve korunur.
        """
        cfg = self.config
        overlay = torch.sigmoid(det["overlay_logit"])
        shadow = torch.sigmoid(det["shadow_logit"])
        keep = 1.0 - torch.sigmoid(det["scene_text_logit"])
        soft = torch.maximum(overlay, shadow) * keep
        soft = soft * valid.view(*valid.shape, 1, 1, 1).to(soft.dtype)
        hard = dilate((soft > cfg.mask_threshold).to(soft.dtype), cfg.mask_dilation) > 0.5
        hard = hard & valid.view(*valid.shape, 1, 1, 1)
        return soft, hard

    def flow_reliability(self, flows: Dict[str, torch.Tensor], valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Akışın güvenilir olduğu pikseller: güven yüksek, ileri-geri tutarlı, iki kare de geçerli."""
        b, pairs = flows["flow_fw"].shape[:2]
        fw = flows["flow_fw"].flatten(0, 1).float()
        bw = flows["flow_bw"].flatten(0, 1).float()
        thr = self.config.flow_confidence
        ok_fw = (torch.sigmoid(flows["conf_fw"].flatten(0, 1).float()) > thr) & ~occlusion_mask(fw, bw)
        ok_bw = (torch.sigmoid(flows["conf_bw"].flatten(0, 1).float()) > thr) & ~occlusion_mask(bw, fw)
        pair_valid = (valid[:, :-1] & valid[:, 1:]).view(b, pairs, 1, 1, 1)
        return ok_fw.view(b, pairs, 1, *fw.shape[-2:]) & pair_valid, ok_bw.view(b, pairs, 1, *bw.shape[-2:]) & pair_valid

    def forward(
        self,
        frames: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        mode: str = "B",
        override: Optional[Dict[str, torch.Tensor]] = None,
        detach_priors: bool = True,
        context_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Bir 15 karelik pencereyi onarır.

        Args:
            frames   [B, T, 3, H, W]
            valid    [B, T] bool (None = tümü geçerli)
            mode     "B": tüm kareler çözülür; "A": yalnızca TARGET_A konumu çözülür
            override aşamalı eğitim için: Detector/Flow çıktılarının yerine konacak
                     gerçek (GT) değerler — anahtarlar: ``mask_soft``, ``mask_hard``,
                     ``alpha``, ``fg_rgb``, ``flow_fw``, ``flow_bw``,
                     ``reliable_fw``, ``reliable_bw``. Verilenler için ilgili model
                     hiç çalıştırılmaz.
            detach_priors True ise önseller (analitik çözüm, çekilen pikseller) Mamba'ya
                     gradyansız verilir (aşamalı eğitim). Ortak ince ayarda False.
            context_state önceki Mod B penceresinin VC-Mamba özeti (sahne başında None).

        Returns (sözlük):
            restored [B, T', 3, H, W]  kapıdan geçmiş çıktı (maskesiz pikseller girdiyle birebir aynı)
            raw      [B, T', 3, H, W]  Mamba'nın kapı öncesi tahmini
            log_b    [B, T', 1, H, W]  Laplace ölçeğinin logaritması (belirsizlik)
            confidence [B, T', 1, H, W] 0..1 güven (Rust çekirdeğine harmanlama ağırlığı olarak gider)
            mask     [B, T', 1, H, W]  bool sert maske
            index    çözülen pencere konumları (liste)
            context_state bir sonraki pencereye aktarılacak özet (ROI/bypass'ta önceki aynen)
            roi      (y0, y1, x0, x1) VC-Mamba'nın çalıştığı bölge, None = tam kare, boş = atlandı
            + ara çıktılar: det, flows, mask_soft, priors
        T' = 15 (Mod B) veya 1 (Mod A).
        """
        if mode not in ("A", "B"):
            raise ValueError("mode 'A' veya 'B' olmalı")
        valid = check_window(frames, valid)
        override = override or {}
        b, t = frames.shape[:2]
        if mode == "A" and t <= TARGET_A:
            raise ValueError(f"Mod A en az {TARGET_A + 1} kare ister")
        vmask = valid.view(b, t, 1, 1, 1)

        # 1) Detector
        det: Dict[str, torch.Tensor] = {}
        need_det = not {"mask_soft", "mask_hard", "alpha", "fg_rgb"} <= set(override)
        if need_det:
            det = self.detector(frames, valid)
            mask_soft, mask_hard = self.erase_mask(det, valid)
        mask_soft = override.get("mask_soft", mask_soft if need_det else None)
        mask_hard = override.get("mask_hard", mask_hard if need_det else None).bool() & vmask
        alpha = override.get("alpha", det.get("alpha"))
        fg_rgb = override.get("fg_rgb", det.get("fg_rgb"))
        det_conf = torch.sigmoid(det["confidence_logit"]) if "confidence_logit" in det else torch.ones_like(alpha)

        # 2) Flow
        flows: Dict[str, torch.Tensor] = {}
        if "flow_fw" in override and "flow_bw" in override:
            flows = {"flow_fw": override["flow_fw"], "flow_bw": override["flow_bw"]}
            ones = torch.ones(b, t - 1, 1, *frames.shape[-2:], dtype=torch.bool, device=frames.device)
            pair_valid = (valid[:, :-1] & valid[:, 1:]).view(b, t - 1, 1, 1, 1)
            rel_fw = override.get("reliable_fw", ones).bool() & pair_valid
            rel_bw = override.get("reliable_bw", ones).bool() & pair_valid
        else:
            flows = self.flow(frames, mask_soft, valid)
            rel_fw, rel_bw = self.flow_reliability(flows, valid)

        # 3) Deterministik kurtarma (önseller)
        with torch.set_grad_enabled(not detach_priors and torch.is_grad_enabled()):
            f32 = frames.float()
            bg_gamma, bg_linear, reliability = analytic_background(f32, alpha.float(), fg_rgb.float(), self.config.alpha_max)
            analytic_conf = reliability * det_conf.float() * mask_hard.to(f32.dtype)
            pulled, pulled_conf = pull_pixels(
                f32, ~mask_hard, flows["flow_fw"].float(), flows["flow_bw"].float(), rel_fw, rel_bw, valid,
                self.config.pull_decay, self.config.pull_select, self.config.pull_block,
            )
        priors = {
            "bg_gamma": bg_gamma, "bg_linear": bg_linear, "analytic_conf": analytic_conf,
            "pulled": pulled, "pulled_conf": pulled_conf,
        }
        if detach_priors:
            priors = {k: v.detach() for k, v in priors.items()}

        # 4) Mamba — çıkarımda dinamik ROI tüpü: yalnızca maskenin pencere boyunca kapladığı kutu
        #    (+ bağlam payı) işlenir; maske hiç yoksa model atlanır (akıllı bypass). Eğitimde tam kare.
        target = TARGET_A if mode == "A" else None
        index = [TARGET_A] if mode == "A" else list(range(t))
        roi = None
        if self.config.roi_tube and not torch.is_grad_enabled():
            roi = roi_tube(mask_hard, valid, self.config.roi_margin, self.config.roi_max_fraction)
        if roi == (0, 0, 0, 0):
            src_frames = frames[:, index]
            out = {
                "rgb": src_frames,
                "log_b": torch.full_like(src_frames[:, :, :1], -7.0, dtype=torch.float32),
                "confidence": torch.ones_like(src_frames[:, :, :1], dtype=torch.float32),
                "context_state": context_state,
            }
        elif roi is not None:
            y0, y1, x0, x1 = roi
            crop = lambda z: z[..., y0:y1, x0:x1]
            part = self.mamba(
                crop(frames), crop(mask_soft), {k: crop(v) for k, v in priors.items()}, valid,
                target=target, context_state=context_state,
            )
            src_frames = frames[:, index]
            rgb = src_frames.clone()
            rgb[..., y0:y1, x0:x1] = part["rgb"].to(rgb.dtype)
            log_b = torch.full_like(src_frames[:, :, :1], -7.0, dtype=torch.float32)
            log_b[..., y0:y1, x0:x1] = part["log_b"]
            conf = torch.ones_like(log_b)
            conf[..., y0:y1, x0:x1] = part["confidence"]
            out = {"rgb": rgb, "log_b": log_b, "confidence": conf, "context_state": part["context_state"]}
        else:
            out = self.mamba(frames, mask_soft, priors, valid, target=target, context_state=context_state)

        # 5) Bit-exact koruma kapısı: maskesiz piksel girdiden AYNEN seçilir (çarpma/yuvarlama yok).
        src = frames[:, index]
        restored, gate = feather_blend(src, out["rgb"], mask_hard[:, index], self.config.feather_px)
        return {
            "restored": restored,
            "raw": out["rgb"],
            "log_b": out["log_b"],
            "confidence": out["confidence"],
            "mask": gate,
            "index": index,
            "mask_soft": mask_soft,
            "mask_hard": mask_hard,
            "det": det,
            "flows": flows,
            "reliable_fw": rel_fw,
            "reliable_bw": rel_bw,
            "priors": priors,
            "context_state": out.get("context_state"),
            "roi": roi,
        }

    # ---- uzun video: kayan pencere -------------------------------------------

    @torch.no_grad()
    def restore_video(self, video: torch.Tensor, cuts: Tuple[int, ...] = (), precision: Tuple[int, ...] = ()) -> Dict[str, torch.Tensor]:
        """Uzun bir klibi Rust zamanlayıcısıyla AYNI pencere düzeninde onarır (değerlendirme için).

        Olağan akış Mod B'dir: pencereler STRIDE (11) kare kayar, ardışık iki pencere OVERLAP (4)
        karede örtüşür; örtüşen karelerin iki tahmini ``öncelik × güven`` ile harmanlanır; öncelik =
        1 + min(geçerli geçmiş, geçerli gelecek) — pencere kenarındaki tahmin az, ortadaki çok ağırlık
        alır. ``precision`` içindeki karelerin sonucu Mod A (7+1+7) tahminiyle değiştirilir.
        Pencereler ``cuts`` ile verilen sahne başlangıçlarını aşmaz; eksik konumlar
        dolgudur (valid=False).

        Args:
            video [N, 3, H, W] 0..1
        Returns:
            restored [N, 3, H, W], mask [N, 1, H, W] bool, confidence [N, 1, H, W]
        """
        n = video.shape[0]
        device = video.device
        acc = torch.zeros_like(video, dtype=torch.float32)
        weight = torch.zeros(n, 1, *video.shape[-2:], device=device)
        mask_any = torch.zeros(n, 1, *video.shape[-2:], dtype=torch.bool, device=device)
        best_conf = torch.zeros_like(weight)
        bounds = sorted({0, n, *[c for c in cuts if 0 < c < n]})

        def window(base: int, seg_lo: int, seg_hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
            idx = torch.arange(base, base + WINDOW, device=device)
            ok = (idx >= seg_lo) & (idx < seg_hi)
            frames = video[idx.clamp(seg_lo, seg_hi - 1)] * ok.view(-1, 1, 1, 1)
            return frames.unsqueeze(0), ok.unsqueeze(0)

        for seg_lo, seg_hi in zip(bounds[:-1], bounds[1:]):
            # Rust Scheduler'daki B(j) adımı: e = sıradaki kesinleşecek kare, j = min(11, kalan).
            # Pencere [e+j-11, e+j+4); konum [11-j, 11) kesinleşir, [11, 15) bir sonraki pencereyle
            # harmanlanır. Olağan akışta pencereler [0,15) → [11,26) → … ; kısa sahnede ve sahne sonunda
            # pencere geçmiş tarafına dolgulu kayar (Rust ile birebir aynı kareler, aynı konumlar).
            e = seg_lo
            context = None          # sahne başında bloklar arası bağlam sıfırlanır
            while e < seg_hi:
                j = min(STRIDE, seg_hi - e)
                base = e + j - STRIDE
                frames, ok = window(base, seg_lo, seg_hi)
                out = self(frames, ok, mode="B", context_state=context)
                context = out["context_state"]
                okv = ok[0]
                past = torch.cumsum(okv.int(), 0) - okv.int()
                future = okv.int().sum() - past - okv.int()
                prior = (1 + torch.minimum(past, future)).float()
                for pos in range(STRIDE - j, WINDOW):
                    if not bool(okv[pos]):
                        continue
                    i = base + pos
                    m = out["mask"][0, pos]
                    wgt = prior[pos] * out["confidence"][0, pos].float().clamp_min(1e-3) * m
                    acc[i] += wgt * out["raw"][0, pos].float()
                    weight[i] += wgt
                    mask_any[i] |= m
                    best_conf[i] = torch.maximum(best_conf[i], out["confidence"][0, pos].float() * m)
                e += j
            for i in precision:
                if seg_lo <= i < seg_hi:
                    frames, ok = window(i - TARGET_A, seg_lo, seg_hi)
                    out = self(frames, ok, mode="A")
                    m = out["mask"][0, 0]
                    acc[i] = out["raw"][0, 0].float() * m
                    weight[i] = m.float()
                    mask_any[i] = m
                    best_conf[i] = out["confidence"][0, 0].float() * m
        fused = acc / weight.clamp_min(1e-6)
        # mask_any, her pencerenin (yumuşatma bölgesi dahil) birleşimidir; yumuşatma burada çekirdek
        # maskeye göre yeniden hesaplanır ki örtüşen pencereler kenarda tutarlı olsun.
        core = mask_any
        if self.config.feather_px > 0:
            core = -F.max_pool2d(-mask_any.float(), 2 * self.config.feather_px + 1, stride=1, padding=self.config.feather_px) > 0.5
        restored, mask_any = feather_blend(video, fused, core, self.config.feather_px)
        return {"restored": restored, "mask": mask_any, "confidence": best_conf}


def build_engine(config: Optional[EngineConfig] = None) -> VideoCleaner:
    """Varsayılan yapılandırmayla motoru kurar."""
    return VideoCleaner(config)


def format_parameter_report(report: Dict[str, Dict[str, float]]) -> str:
    """``parameter_report`` çıktısını okunur tabloya çevirir."""
    lines = [f"{'model':<14}{'parametre':>14}{'bütçe':>14}{'kullanım':>10}{'BF16 MiB':>10}"]
    for name, r in report.items():
        label = SYSTEM_NAMES.get(name, name)
        lines.append(f"{label:<14}{int(r['params']):>14,}{int(r['budget']):>14,}{r['usage']:>9.1%}{r['bf16_mib']:>10.1f}")
    return "\n".join(lines)


__all__ = [
    "WINDOW", "STRIDE", "OVERLAP", "TARGET_A", "PARAM_BUDGET", "SYSTEM_NAMES", "roi_tube",
    "Norm2d", "fold_layer_scale", "ConvBlock", "Downsample", "Upsample", "run_block",
    "flatten_time", "unflatten_time", "count_parameters", "check_window",
    "srgb_to_linear", "linear_to_srgb", "composite", "analytic_background",
    "coords_grid", "sample_at", "warp", "in_bounds", "occlusion_mask", "pull_pixels",
    "haar_dwt", "haar_idwt", "dilate",
    "EngineConfig", "VideoCleaner", "build_engine", "format_parameter_report",
]
