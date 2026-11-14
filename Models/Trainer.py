"""Video_Cleaner v2.0 — VC-Trainer ve VC-Tester: eğitim, ölçüm, sürüm kapıları ve dışa aktarma.

Kullanım (proje kökünden):

    python Models/Trainer.py --selftest                       # kurulum denetimi (CPU'da ~1-2 dk)
    python Models/Trainer.py --stage calibrate                # veri hattı + bellek kalibrasyonu
    python Models/Trainer.py --stage mamba --benchmark 1000   # GERÇEK adım süresi ve VRAM ölçümü
    python Models/Trainer.py --stage detector --data /veri/klipler --steps 60000
    python Models/Trainer.py --stage flow     --resume runs/detector/last.pt
    python Models/Trainer.py --stage mamba    --resume runs/flow/last.pt
    python Models/Trainer.py --stage joint    --resume runs/mamba/last.pt
    python Models/Trainer.py --export out/ --resume runs/joint/last.pt
    python Models/Trainer.py --audit --resume runs/joint/last.pt   # VC-Tester sürüm kapıları (JSON)

Eğitim sırası: kalibrasyon → Detector → Flow → Mamba → ortak ince ayar → dışa aktarma.
Toplam süre önceden VARSAYILMAZ: ``--benchmark`` ölçülen adım süresinden hesaplar.

Veri
    ``--data`` verilirse her alt klasör bir klip sayılır (sıralı JPG/PNG kareler,
    en az 30 adet). Verilmezse (veya ``--procedural-ratio`` oranında) bilinen
    hareketli yordamsal klipler üretilir; bunlarda gerçek optik akış da bilinir.
    Kaplamalar (altyazı, kontur, gölge, parıltı, kutu, filigran) sentetik eklenir;
    böylece temiz arka plan, maske, alfa ve kaplama rengi kesin olarak bilinir.
    Sahneye ait yazılar (korunacak negatif örnekler) arka plana gömülür ve onunla
    birlikte hareket eder.
    Eğitim ve doğrulama klipleri KESİN olarak ayrıdır: ``--val-data`` ile ayrı bir
    klasör verilir; verilmezse ``--data`` altındaki klip klasörlerinin
    ``--val-fraction`` kadarı (klasör adının sağlamasına göre, her çalıştırmada aynı
    kümeler) doğrulamaya ayrılır ve eğitimde hiç kullanılmaz. Ayrım klip (klasör)
    düzeyindedir; aynı videonun kareleri iki kümeye dağılmaz. Aynı videoyu birden
    çok klasöre böldüyseniz hepsini aynı tarafa koymak için ``--val-data`` kullanın.
    Dış bir ölçüt kümesi (ör. MFVSR) eğitim verisine KARIŞTIRILMAMALIDIR.

Karma hassasiyet: CUDA'da BF16 autocast. BF16'da GradScaler (kayıp ölçekleme)
kullanılmaz; o yalnızca FP16 içindir.

Gereksinimler: torch (2.15 ile denendi), numpy, pillow. İsteğe bağlı: torchvision
(algısal kayıp için, ``--vgg-weights`` ile).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Core import (  # noqa: E402
    PARAM_BUDGET, STRIDE, TARGET_A, WINDOW, EngineConfig, VideoCleaner, composite, coords_grid,
    count_parameters, dilate, format_parameter_report, haar_dwt, linear_to_srgb, srgb_to_linear, warp,
)
from Video_Detector import HEADS as DETECTOR_HEADS, DetectorConfig, VideoDetector  # noqa: E402
from Video_Flow import FlowConfig, VideoFlow  # noqa: E402
from Video_Mamba import MambaConfig, VideoMamba  # noqa: E402

Batch = Dict[str, torch.Tensor]


# =============================================================================
# 1. Yapılandırma
# =============================================================================

@dataclass
class LossWeights:
    """Birleşik kaybın yedi terimi (λ₁..λ₇) ve yardımcı ağırlıklar."""

    mask_reconstruction: float = 1.0     # λ₁ silinen yazının arkasındaki görüntü
    preservation: float = 0.5            # λ₂ maske çevresinde orijinalle tutarlılık
    temporal_warp: float = 0.5           # λ₃ hareket varken kareler arası uyum (fazlalık hatası)
    perceptual: float = 0.05             # λ₄ algısal doku kalitesi (VGG verilmişse)
    edge_frequency: float = 0.25         # λ₅ kenar/keskinlik (Haar detay bantları)
    scene_text: float = 1.0              # λ₆ sahne yazısına dokunmama
    alpha: float = 1.0                   # λ₇ alfa/renk ayrıştırmasının tutarlılığı
    nll: float = 0.1                     # belirsizlik (Laplace NLL) katkısı
    flow: float = 1.0                    # ortak eğitimde akış kaybı
    detector: float = 1.0                # ortak eğitimde algılama kaybı
    # Zamansal PatchGAN (yalnızca eğitim; telefonda 0 ms): L1 + DWT + algısal kayıpla eğitilen 20–40M
    # parametreli onarıcılar kalın yazının altında "plastik" (fazla pürüzsüz) doku üretir. Ayrıştırıcı
    # onarılan bölgeyi gerçek dokudan ayırt etmeye çalışır; onarıcı onu kandıracak dokuyu öğrenir.
    adversarial: float = 0.01


@dataclass
class TrainConfig:
    stage: str = "detector"
    steps: int = 0                       # 0 = aşamanın varsayılanı
    crop: Tuple[int, int] = (256, 448)   # (yükseklik, genişlik); 16'nın katı
    frames: int = WINDOW
    batch: int = 1                       # mikro-yığın
    accum: int = 4                       # gradyan biriktirme → etkin yığın = batch × accum
    lr: float = 0.0                      # 0 = aşamanın varsayılanı
    weight_decay: float = 0.01
    warmup: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    mode_a_prob: float = 0.25            # Mamba/ortak aşamada Mod A (yalnızca 7. kare) adımı oranı
    truncate_prob: float = 0.2           # dolgulu (kısa sahne) pencere oranı
    amp: bool = True
    checkpointing: bool = True           # aktivasyon denetim noktası (VRAM ↓, süre ↑)
    compile: bool = False                # torch.compile (kararlılık sağlandıktan sonra)
    workers: int = 4
    seed: int = 1234
    data: str = ""
    val_data: str = ""                   # ayrı doğrulama klipleri (yoksa data'nın val_fraction'ı ayrılır)
    val_fraction: float = 0.1
    fonts: str = ""
    text_file: str = ""
    procedural_ratio: float = 0.25
    out: str = "runs"
    resume: str = ""
    vgg_weights: str = ""
    log_every: int = 50
    save_every: int = 2000
    val_every: int = 2000
    val_clips: int = 8
    tiny: bool = False                   # küçük modeller (yalnızca duman testi)
    gan_start: int = 20_000              # ayrıştırıcı kaybının devreye girdiği adım (önce yapı oturur)
    gan_lr: float = 1e-4
    weights: LossWeights = field(default_factory=LossWeights)


STAGE_DEFAULTS = {            # (adım sayısı, öğrenme oranı)
    "detector": (60_000, 4e-4),
    "flow": (100_000, 2e-4),
    "mamba": (500_000, 2e-4),
    "joint": (40_000, 5e-5),
}


def tiny_engine_config(checkpoint: bool = False) -> EngineConfig:
    """Hat denemeleri için küçük modeller (gerçek eğitimde KULLANILMAZ)."""
    return EngineConfig(
        detector=DetectorConfig(widths=(8, 16, 24, 32), depths=(1, 1, 1, 1), decoder_depths=(1, 1, 1), temporal_depths=(1, 1), use_checkpoint=checkpoint),
        flow=FlowConfig(encoder_widths=(16, 24, 32), feature_dim=32, context_dim=16, hidden_dim=24, corr_radius=2, corr_levels=2, iters=3, use_checkpoint=checkpoint),
        mamba=MambaConfig(widths=(8, 16, 24, 32), dec_widths=(12, 8, 4), enc_depths=(1, 1, 1, 1), dec_depths=(1, 1, 1), ssm_state=4, attn_window=4, attn_heads=4, use_checkpoint=checkpoint),
    )


def make_engine(cfg: TrainConfig) -> VideoCleaner:
    if cfg.tiny:
        return VideoCleaner(tiny_engine_config(cfg.checkpointing))
    return VideoCleaner(
        EngineConfig(
            detector=DetectorConfig(use_checkpoint=cfg.checkpointing),
            flow=FlowConfig(use_checkpoint=cfg.checkpointing),
            mamba=MambaConfig(use_checkpoint=cfg.checkpointing),
        )
    )


# =============================================================================
# 2. Sentetik veri
# =============================================================================

WORDS = (
    "merhaba dünya bugün hava çok güzel şimdi nereye gidiyoruz teşekkür ederim lütfen bekle "
    "gerçekten inanamıyorum öyle değil mi hadi başlayalım yarın görüşürüz ışık gölge çığlık "
    "the quick brown fox jumps over lazy dog where are you going right now please wait here "
    "I can not believe this subscribe episode chapter warning breaking news live 2026 1080p"
).split()

FONT_DIRS = (
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
    "/System/Library/Fonts", "/Library/Fonts", "C:/Windows/Fonts", "/system/fonts",
)


def find_fonts(extra: str = "", limit: int = 400) -> List[str]:
    """Kullanılabilir TrueType/OpenType yazı tiplerini bulur. Hiç yoksa Pillow'un gömülü yazı tipi kullanılır."""
    found: List[str] = []
    for root in ([extra] if extra else []) + list(FONT_DIRS):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for base, _, files in os.walk(root):
            for name in sorted(files):
                if name.lower().endswith((".ttf", ".otf", ".ttc")):
                    found.append(os.path.join(base, name))
                    if len(found) >= limit:
                        return found
    return found


def load_font(fonts: List[str], size: int, rng: random.Random) -> ImageFont.ImageFont:
    for _ in range(4):
        if not fonts:
            break
        try:
            return ImageFont.truetype(rng.choice(fonts), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def random_text(rng: random.Random, lines: Optional[List[str]]) -> str:
    if lines and rng.random() < 0.8:
        return rng.choice(lines)
    text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 6)))
    roll = rng.random()
    if roll < 0.2:
        text = text.upper()
    elif roll < 0.35:
        text = text.capitalize() + rng.choice(".!?…")
    if rng.random() < 0.25:
        words = text.split()
        text = " ".join(words[: len(words) // 2]) + "\n" + " ".join(words[len(words) // 2:])
    return text.strip() or "…"


def smooth_noise(gen: torch.Generator, channels: int, h: int, w: int) -> torch.Tensor:
    """Çok ölçekli yumuşak doku [C, h, w], 0..1."""
    out = torch.zeros(1, channels, h, w)
    for cell in (96, 48, 24, 12, 6, 3):
        gh, gw = max(2, h // cell + 1), max(2, w // cell + 1)
        layer = torch.rand(1, channels, gh, gw, generator=gen)
        out += F.interpolate(layer, size=(h, w), mode="bicubic", align_corners=True) * cell ** 0.7
    out = out - out.amin(dim=(2, 3), keepdim=True)
    return (out / out.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))[0]


def affine_points(points: torch.Tensor, center: torch.Tensor, scale: float, theta: float, origin: torch.Tensor) -> torch.Tensor:
    """Kare koordinatlarını tuval koordinatına taşır: q = c + s·R(θ)·(p − p₀). points: [2, H, W]."""
    cos, sin = math.cos(theta), math.sin(theta)
    d = points - origin.view(2, 1, 1)
    x = scale * (cos * d[0] - sin * d[1]) + center[0]
    y = scale * (sin * d[0] + cos * d[1]) + center[1]
    return torch.stack([x, y])


def affine_inverse(points: torch.Tensor, center: torch.Tensor, scale: float, theta: float, origin: torch.Tensor) -> torch.Tensor:
    """``affine_points`` dönüşümünün tersi."""
    cos, sin = math.cos(theta), math.sin(theta)
    d = (points - center.view(2, 1, 1)) / scale
    x = cos * d[0] + sin * d[1] + origin[0]
    y = -sin * d[0] + cos * d[1] + origin[1]
    return torch.stack([x, y])


# ---- gerçekçi sıkıştırma bozulması ---------------------------------------------------------------
# Gerçek videolarda gömülü altyazının kenarında YUV420 renk alt örneklemesi (renk kanaması) ve H.264/HEVC
# DCT blok halkaları (sivrisinek gürültüsü) vardır. Keskin metni temiz kareye basıp öyle eğitilen model,
# gerçek MP4'te harf kenarındaki bu izleri "yazının parçası değil" sanıp bırakır. Girdi bu yüzden aynı
# zincirden geçirilir; hedef (GT) sıkıştırılmamış temiz karedir.

def _ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


FFMPEG = _ffmpeg()


def yuv420_roundtrip(frames: torch.Tensor, matrix: str = "bt601") -> torch.Tensor:
    """[T, 3, H, W] 0..1 RGB → Y′CbCr (sınırlı aralık, 8 bit) → renk 2× alt örnekleme → RGB."""
    kr, kb = (0.299, 0.114) if matrix == "bt601" else (0.2126, 0.0722)
    r, g, b = frames.unbind(1)
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    q = lambda v, lo, span: torch.round(v * span + lo) / 255.0          # 8 bit sınırlı aralık nicemleme
    y = (q(y, 16, 219) * 255 - 16) / 219
    c = torch.stack([cb, cr], 1)
    t, _, h, w = frames.shape
    c = F.avg_pool2d(F.pad(c, (0, w % 2, 0, h % 2), mode="replicate"), 2)
    c = (torch.round(c * 224 + 128) - 128) / 224
    c = F.interpolate(c, scale_factor=2.0, mode="bilinear", align_corners=False)[..., :h, :w]
    cb, cr = c.unbind(1)
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / (1 - kr - kb)
    return torch.stack([r, g, b], 1).clamp(0, 1)


def codec_roundtrip(frames: torch.Tensor, codec: str, crf: int) -> Optional[torch.Tensor]:
    """Gerçek kodlayıcıdan geçirme (ffmpeg; libx264 / libx265, yuv420p). ffmpeg yoksa veya hata olursa None."""
    if FFMPEG is None:
        return None
    t, _, h, w = frames.shape
    ph, pw = h + h % 2, w + w % 2
    raw = (frames.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    if (ph, pw) != (h, w):
        raw = np.pad(raw, ((0, 0), (0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    lib, fmt, extra = ("libx265", "hevc", ["-x265-params", "log-level=none:pools=1"]) if codec == "hevc" else ("libx264", "h264", ["-threads", "1"])
    try:
        enc = subprocess.run(
            [FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{pw}x{ph}", "-r", "30", "-i", "-",
             "-c:v", lib, "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", *extra, "-f", fmt, "-"],
            input=raw.tobytes(), capture_output=True, timeout=60, check=True,
        ).stdout
        dec = subprocess.run(
            [FFMPEG, "-v", "error", "-f", fmt, "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            input=enc, capture_output=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    out = np.frombuffer(dec, np.uint8)
    if out.size != t * ph * pw * 3:
        return None
    out = torch.from_numpy(out.reshape(t, ph, pw, 3)[:, :h, :w].copy()).permute(0, 3, 1, 2).float() / 255.0
    return out


def jpeg_roundtrip(frames: torch.Tensor, quality: int) -> torch.Tensor:
    out = []
    for f in frames:
        buffer = io.BytesIO()
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
        out.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


DEGRADATIONS = ["yok", "yuv420", "yuv420+jpeg", "h264", "hevc"]


def degrade(frames: torch.Tensor, rng: random.Random) -> Tuple[torch.Tensor, str]:
    """Kaplama basılmış kareleri gerçekçi sıkıştırma zincirinden geçirir. Dönen: (girdi, kullanılan yol)."""
    pick = rng.random()
    if pick < 0.45:
        codec = "hevc" if rng.random() < 0.25 else "h264"
        out = codec_roundtrip(frames, codec, rng.randint(20, 38))
        if out is not None:
            return out, codec
        pick = 0.5                                           # ffmpeg yok: JPEG yoluna düş
    matrix = "bt709" if rng.random() < 0.5 else "bt601"
    if pick < 0.75:
        return jpeg_roundtrip(yuv420_roundtrip(frames, matrix), rng.randint(35, 95)), "yuv420+jpeg"
    if pick < 0.92:
        return yuv420_roundtrip(frames, matrix), "yuv420"
    return frames, "yok"


class ClipSynthesizer:
    """Temiz klip + sentetik kaplama → eğitim örneği."""

    def __init__(self, cfg: TrainConfig, seed: int, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError("split 'train' veya 'val' olmalı")
        self.cfg = cfg
        self.split = split
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.fonts = find_fonts(cfg.fonts)
        self.lines: Optional[List[str]] = None
        if cfg.text_file:
            self.lines = [ln.strip() for ln in Path(cfg.text_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.clips: List[List[Path]] = clip_folders(cfg, split)


    # ---- temiz klipler -------------------------------------------------------

    def _scene_text_canvas(self, canvas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tuvale sahneye ait yazı (tabela benzeri) gömer. Dönen: (tuval, maske [1, h, w])."""
        _, h, w = canvas.shape
        rng = self.rng
        image = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        mask = Image.new("L", (w, h), 0)
        for _ in range(rng.randint(1, 3)):
            font = load_font(self.fonts, rng.randint(max(10, h // 14), max(12, h // 5)), rng)
            text = random_text(rng, self.lines).replace("\n", " ")
            color = tuple(rng.randint(0, 255) for _ in range(3))
            xy = (rng.randint(0, max(1, w // 2)), rng.randint(0, max(1, h - h // 4)))
            if rng.random() < 0.5:      # tabela zemini
                box = ImageDraw.Draw(image).textbbox(xy, text, font=font)
                pad = rng.randint(2, 10)
                ImageDraw.Draw(image).rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=tuple(rng.randint(0, 255) for _ in range(3)))
            ImageDraw.Draw(image).text(xy, text, font=font, fill=color)
            ImageDraw.Draw(mask).text(xy, text, font=font, fill=255)
        out = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        return out, torch.from_numpy(np.asarray(mask).copy()).float().unsqueeze(0) / 255.0

    def procedural_clip(self) -> Batch:
        """Bilinen benzerlik hareketiyle (öteleme + ölçek + dönme) kayan dokulu sahne; gerçek akış bilinir."""
        cfg, rng, gen = self.cfg, self.rng, self.gen
        t, (h, w) = cfg.frames, cfg.crop
        margin = 64
        ch, cw = h + 2 * margin, w + 2 * margin
        canvas = smooth_noise(gen, 3, ch, cw)
        ys, xs = torch.meshgrid(torch.arange(ch, dtype=torch.float32), torch.arange(cw, dtype=torch.float32), indexing="ij")
        for _ in range(rng.randint(2, 7)):      # yumuşak kenarlı düz renkli şekiller
            cx, cy = rng.uniform(0, cw), rng.uniform(0, ch)
            rx, ry = rng.uniform(cw * 0.03, cw * 0.25), rng.uniform(ch * 0.03, ch * 0.25)
            if rng.random() < 0.5:
                dist = torch.maximum((xs - cx).abs() / rx, (ys - cy).abs() / ry)
            else:
                dist = torch.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
            cover = ((1.0 - dist) * rng.uniform(4, 40)).clamp(0, 1) * rng.uniform(0.5, 1.0)
            color = torch.tensor([rng.random(), rng.random(), rng.random()]).view(3, 1, 1)
            canvas = canvas * (1 - cover) + color * cover
        scene_mask = torch.zeros(1, ch, cw)
        if rng.random() < 0.6:
            canvas, scene_mask = self._scene_text_canvas(canvas)

        # Kamera hareketi: %40 neredeyse sabit, %40 orta, %20 hızlı.
        speed = rng.choice([0.3, 0.3, 3.0, 3.0, 12.0])
        velocity = torch.tensor([rng.uniform(-1, 1), rng.uniform(-1, 1)]) * speed
        accel = torch.tensor([rng.uniform(-1, 1), rng.uniform(-1, 1)]) * speed * 0.02
        zoom, spin = rng.uniform(-0.002, 0.002), rng.uniform(-0.0015, 0.0015)
        origin = torch.tensor([(w - 1) / 2, (h - 1) / 2])
        start = torch.tensor([(cw - 1) / 2, (ch - 1) / 2])
        pose = [(start + velocity * i + 0.5 * accel * i * i, math.exp(zoom * i), spin * i) for i in range(t)]

        grid = coords_grid(h, w, torch.device("cpu"))[0]
        stack = torch.cat([canvas, scene_mask], dim=0).unsqueeze(0)
        frames, flow_fw, flow_bw = [], [], []
        for i in range(t):
            q = affine_points(grid, *pose[i], origin)
            gx = 2 * q[0] / (cw - 1) - 1
            gy = 2 * q[1] / (ch - 1) - 1
            # Yansıtmalı dolgu: tuval sonsuz düzleme yansıtılarak uzatılır; akış yine kesin kalır.
            frames.append(F.grid_sample(stack, torch.stack([gx, gy], -1).unsqueeze(0), mode="bilinear", padding_mode="reflection", align_corners=True)[0])
            if i + 1 < t:
                flow_fw.append(affine_inverse(q, *pose[i + 1], origin) - grid)
                flow_bw.append(affine_inverse(affine_points(grid, *pose[i + 1], origin), *pose[i], origin) - grid)
        frames = torch.stack(frames)
        return {
            "clean": frames[:, :3].clamp(0, 1),
            "scene_text": frames[:, 3:4] > 0.5,
            "flow_fw": torch.stack(flow_fw),
            "flow_bw": torch.stack(flow_bw),
            "has_flow": torch.tensor(True),
        }

    def folder_clip(self) -> Batch:
        """Diskteki bir klipten rastgele WINDOW (15) kare ve rastgele kırpım (gerçek akış bilinmez)."""
        cfg, rng = self.cfg, self.rng
        t, (h, w) = cfg.frames, cfg.crop
        files = rng.choice(self.clips)
        first = rng.randint(0, len(files) - t)
        frames = []
        box = None
        for path in files[first: first + t]:
            image = Image.open(path).convert("RGB")
            if box is None:
                scale = max(h / image.height, w / image.width, rng.uniform(0.5, 1.0))
                size = (max(w, round(image.width * scale)), max(h, round(image.height * scale)))
                left, top = rng.randint(0, size[0] - w), rng.randint(0, size[1] - h)
                box = (size, (left, top, left + w, top + h))
            image = image.resize(box[0], Image.BICUBIC).crop(box[1])
            frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
        clean = torch.stack(frames)
        if rng.random() < 0.5:
            clean = clean.flip(-1)
        zero = torch.zeros(t - 1, 2, h, w)
        return {
            "clean": clean,
            "scene_text": torch.zeros(t, 1, h, w, dtype=torch.bool),
            "flow_fw": zero,
            "flow_bw": zero.clone(),
            "has_flow": torch.tensor(False),
        }

    # ---- kaplama --------------------------------------------------------------

    def _render_text_layers(self, h: int, w: int) -> Dict[str, torch.Tensor]:
        """Bir altyazı/filigran durumunu çizer: yazı gövdesi + kontur, gölge/parıltı/kutu katmanı."""
        rng = self.rng
        text = random_text(rng, self.lines)
        # %30 yarı saydam logo/filigran (köşe, %20–60 opaklık), %70 opak altyazı (kontur, gölge, kutu)
        watermark = rng.random() < 0.30
        size = rng.randint(max(8, h // 22), max(10, h // 7)) if not watermark else rng.randint(max(8, h // 26), max(10, h // 10))
        font = load_font(self.fonts, size, rng)
        stroke = rng.choice([0, 0, 1, 2, 3]) if size >= 14 else rng.choice([0, 1])
        probe = ImageDraw.Draw(Image.new("L", (8, 8)))
        left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke, align="center")
        tw, th = right - left, bottom - top
        place = rng.random()
        if watermark:
            x = rng.choice([w * 0.03, w - tw - w * 0.03])
            y = rng.choice([h * 0.04, h - th - h * 0.04])
        elif place < 0.7:
            x, y = (w - tw) / 2 + rng.uniform(-w * 0.05, w * 0.05), h - th - rng.uniform(h * 0.03, h * 0.15)
        elif place < 0.85:
            x, y = (w - tw) / 2, rng.uniform(h * 0.03, h * 0.15)
        else:
            x, y = rng.uniform(0, max(1, w - tw)), rng.uniform(0, max(1, h - th))
        xy = (x - left, y - top)

        def draw(stroke_width: int) -> Image.Image:
            image = Image.new("L", (w, h), 0)
            ImageDraw.Draw(image).multiline_text(xy, text, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255, align="center")
            return image

        to_tensor = lambda im: torch.from_numpy(np.asarray(im).copy()).float().unsqueeze(0) / 255.0
        body_img = draw(0)
        full_img = draw(stroke) if stroke else body_img
        body, full = to_tensor(body_img), to_tensor(full_img)

        bright = rng.random() < 0.75
        text_color = torch.tensor(rng.choice([(1, 1, 1), (1, 1, 1), (1, 0.92, 0.2), (0.9, 0.9, 0.9)]) if bright else [rng.random() for _ in range(3)], dtype=torch.float32)
        if rng.random() < 0.15:
            text_color = torch.rand(3, generator=self.gen)
        outline_color = torch.zeros(3) if bright or rng.random() < 0.7 else torch.rand(3, generator=self.gen)
        outline = (full - body).clamp(0, 1)
        color = (body * text_color.view(3, 1, 1) + outline * outline_color.view(3, 1, 1)) / full.clamp_min(1e-4)
        if rng.random() < 0.15:     # dikey renk geçişi
            ramp = torch.linspace(0.75, 1.0, h).view(1, h, 1)
            color = color * ramp

        # Alt katman: gölge / parıltı / yarı saydam kutu (hepsi "shadow" sınıfı).
        under = torch.zeros(1, h, w)
        under_color = torch.zeros(3)
        effect = rng.random()
        if effect < 0.35:
            dx, dy = rng.randint(1, max(2, size // 8)), rng.randint(1, max(2, size // 8))
            shifted = full_img.transform((w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy))
            under = to_tensor(shifted.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, size / 10)))) * rng.uniform(0.5, 1.0)
        elif effect < 0.5:
            under = (to_tensor(full_img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, size / 5)))) * rng.uniform(1.2, 2.5)).clamp(0, 1)
            under_color = torch.rand(3, generator=self.gen) if rng.random() < 0.5 else torch.zeros(3)
        elif effect < 0.65:
            pad = rng.randint(2, max(3, size // 3))
            box = Image.new("L", (w, h), 0)
            ImageDraw.Draw(box).rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=255)
            under = to_tensor(box) * rng.uniform(0.3, 0.8)
        opacity = rng.uniform(0.2, 0.6) if watermark else 1.0
        if watermark and rng.random() < 0.5:
            # Logo: yazının yanında/arkasında basit şekil (daire veya yuvarlatılmış kutu), aynı opaklıkla.
            shape = Image.new("L", (w, h), 0)
            pad = max(2, size // 4)
            box = (x - pad, y - pad, x + tw + pad, y + th + pad)
            if rng.random() < 0.5:
                ImageDraw.Draw(shape).rounded_rectangle(box, radius=pad, outline=255, width=max(1, size // 10))
            else:
                r = max(4, size // 2)
                cx, cy = x - r - pad, y + th / 2
                ImageDraw.Draw(shape).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            full = torch.maximum(full, to_tensor(shape))
            color = torch.where(full > body, color.new_ones(1) * color.mean(dim=(1, 2), keepdim=True), color)
        return {"alpha_o": full * opacity, "color_o": color, "alpha_s": under * opacity, "color_s": under_color}

    def add_overlay(self, sample: Batch) -> Batch:
        """Temiz klibe kaplama ekler ve tüm gerçek (GT) haritaları üretir."""
        cfg, rng = self.cfg, self.rng
        clean = sample["clean"]
        t, _, h, w = clean.shape

        alpha_o = torch.zeros(t, 1, h, w)
        color_o = torch.zeros(t, 3, h, w)
        alpha_s = torch.zeros(t, 1, h, w)
        color_s = torch.zeros(t, 3, h, w)
        if rng.random() < 0.92:          # %8 kaplamasız örnek: yanlış pozitifleri öğretir
            # Pencere içinde 1-3 ardışık altyazı; aralarında boşluk olabilir, yumuşak giriş/çıkış olabilir.
            cursor = rng.randint(0, 10) if rng.random() < 0.5 else 0
            while cursor < t:
                length = rng.randint(6, t) if rng.random() < 0.7 else t
                end = min(t, cursor + length)
                layers = self._render_text_layers(h, w)
                fade = rng.randint(0, 4) if rng.random() < 0.3 else 0
                for i in range(cursor, end):
                    gain = 1.0
                    if fade:
                        gain = min(1.0, (i - cursor + 1) / (fade + 1), (end - i) / (fade + 1))
                    alpha_o[i] = layers["alpha_o"] * gain
                    color_o[i] = layers["color_o"]
                    alpha_s[i] = layers["alpha_s"] * gain
                    color_s[i] = layers["color_s"].view(3, 1, 1)
                cursor = end + (rng.randint(0, 8) if rng.random() < 0.5 else 0)

        # İki katmanı tek eşdeğer katmana indir: α = α_o + (1 − α_o)·α_s
        under = (1.0 - alpha_o) * alpha_s
        alpha = (alpha_o + under).clamp(0, 1)
        fg = (alpha_o * color_o + under * color_s) / alpha.clamp_min(1e-4)
        space = "linear" if rng.random() < 0.3 else "gamma"
        comp = composite(clean, fg, alpha, space)

        # Gerçekçi bozulma: YUV420 renk alt örnekleme + gerçek H.264/HEVC veya JPEG (bkz. degrade).
        frames, path = degrade(comp, rng)

        overlay = alpha_o > 0.02
        sample.update(
            frames=frames,
            composite=comp,
            overlay=overlay,
            shadow=(under > 0.02) & ~overlay,
            alpha=alpha,
            fg_rgb=fg.clamp(0, 1),
            linear_blend=torch.tensor(space == "linear"),
            degradation=torch.tensor(DEGRADATIONS.index(path)),
            valid=torch.ones(t, dtype=torch.bool),
        )
        return sample

    def sample(self) -> Batch:
        use_folder = bool(self.clips) and self.rng.random() >= self.cfg.procedural_ratio
        return self.add_overlay(self.folder_clip() if use_folder else self.procedural_clip())


def held_out(folder_name: str, fraction: float) -> bool:
    """Klasör doğrulamaya mı ayrıldı? Ada bağlı, çalıştırmalar arasında değişmeyen karar."""
    return zlib.crc32(folder_name.encode("utf-8")) % 10_000 < int(round(fraction * 10_000))


def clip_folders(cfg: TrainConfig, split: str) -> List[List[Path]]:
    """``split`` için kullanılacak klipler (her klip = bir klasördeki sıralı kare dosyaları).

    Eğitim ve doğrulama kümeleri ayrıktır: aynı klasör asla ikisinde birden yer almaz.
    """
    def scan(root: str) -> Dict[str, List[Path]]:
        found: Dict[str, List[Path]] = {}
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            if len(files) >= cfg.frames:
                found[folder.name] = files
        return found

    if not cfg.data:
        if cfg.val_data and split == "val":
            return list(scan(cfg.val_data).values())
        return []
    train_all = scan(cfg.data)
    if not train_all:
        raise FileNotFoundError(f"{cfg.data}: en az {cfg.frames} kare içeren klip klasörü bulunamadı")
    if cfg.val_data:
        if Path(cfg.val_data).resolve() == Path(cfg.data).resolve():
            raise ValueError("--val-data, --data ile aynı klasör olamaz (doğrulama eğitim verisinden ayrı olmalı)")
        val = scan(cfg.val_data)
        shared = sorted(set(val) & set(train_all))
        if shared:
            raise ValueError(f"aynı adlı klipler hem eğitim hem doğrulama klasöründe: {shared[:5]}")
        return list((val if split == "val" else train_all).values())
    picked = {name: files for name, files in train_all.items() if held_out(name, cfg.val_fraction) == (split == "val")}
    if split == "train" and not picked:
        raise ValueError("tüm klipler doğrulamaya ayrıldı; --val-fraction değerini küçültün veya --val-data verin")
    return list(picked.values())


class ClipStream(torch.utils.data.IterableDataset):
    """Sonsuz örnek akışı; her işçi kendi tohumuyla üretir."""

    def __init__(self, cfg: TrainConfig, seed: int):
        super().__init__()
        self.cfg, self.seed = cfg, seed

    def __iter__(self) -> Iterator[Batch]:
        info = torch.utils.data.get_worker_info()
        synth = ClipSynthesizer(self.cfg, self.seed + 1000 * (info.id + 1 if info else 0))
        while True:
            yield synth.sample()


def make_loader(cfg: TrainConfig, seed: int) -> Iterator[Batch]:
    loader = torch.utils.data.DataLoader(
        ClipStream(cfg, seed), batch_size=cfg.batch, num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=cfg.workers > 0,
    )
    return iter(loader)


def to_device(batch: Batch, device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def truncate_window(batch: Batch, rng: random.Random, prob: float) -> Batch:
    """Sahne başı/sonu dolgusunu taklit eder: pencerenin başını veya sonunu geçersiz kılar."""
    if rng.random() >= prob:
        return batch
    t = batch["valid"].shape[1]
    cut = rng.randint(1, t - 10)
    valid = batch["valid"].clone()
    if rng.random() < 0.5:
        valid[:, :cut] = False
    else:
        valid[:, t - cut:] = False
    batch = dict(batch)
    batch["valid"] = valid
    vm = valid.view(*valid.shape, 1, 1, 1)
    for key in ("frames", "composite", "clean"):
        batch[key] = batch[key] * vm
    for key in ("overlay", "shadow", "scene_text"):
        batch[key] = batch[key] & vm
    batch["alpha"] = batch["alpha"] * vm
    return batch


# =============================================================================
# 3. Kayıplar
# =============================================================================

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``mask`` ile ağırlıklı ortalama; kanal boyutu yayınlanır."""
    mask = mask.to(x.dtype)
    return (x * mask).sum() / (mask.expand_as(x).sum() + 1e-6)


def pair_mask(valid: torch.Tensor) -> torch.Tensor:
    """Her iki karesi de geçerli komşu çiftler: [B, T-1, 1, 1, 1]."""
    return (valid[:, :-1] & valid[:, 1:]).view(valid.shape[0], -1, 1, 1, 1)


def warp_pairs(x: torch.Tensor, flow_fw: torch.Tensor) -> torch.Tensor:
    """x[:, 1:] karelerini bir önceki kareye hizalar: çıktı[:, t] ≈ x[:, t]."""
    b, t = x.shape[:2]
    return warp(x[:, 1:].flatten(0, 1).float(), flow_fw.flatten(0, 1).float()).view(b, t - 1, *x.shape[2:])


def temporal_validity(
    clean: torch.Tensor, flow_fw: torch.Tensor, valid: torch.Tensor, reliable: Optional[torch.Tensor] = None,
    occlusion: float = 0.05, cut: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zamansal kayıp/ölçüt için geçerlilik maskesi [B, T-1, 1, H, W] ve gerçek çarpıtma hatası.

    Dışlananlar (planın "geçerlilik maskeli temporal loss" kuralı):
      * örtülen (occluded) pikseller: temiz videoda bile çarpıtma hatası ``occlusion`` üstünde,
      * VC-Flow'un güvenilmez dediği pikseller (``reliable`` = ``out["reliable_fw"]``),
      * sahne geçişi (scene_cut) olan çiftler: ortalama çarpıtma hatası ``cut`` üstünde — çiftin tamamı,
      * dolgulu (geçersiz) karelere dokunan çiftler.
    """
    e_true = (clean[:, :-1] - warp_pairs(clean, flow_fw)).abs().mean(2, keepdim=True)
    scene_cut = e_true.mean((2, 3, 4), keepdim=True) > cut
    keep = (e_true < occlusion) & pair_mask(valid) & ~scene_cut
    if reliable is not None and reliable.shape[1] == keep.shape[1]:
        keep = keep & reliable.bool()
    return keep, e_true


class TemporalPatchGAN(nn.Module):
    """Zamansal PatchGAN ayrıştırıcı: [B, T, 3+1, H, W] (görüntü + onarım maskesi) → yama başına gerçeklik logiti.

    3B evrişimler (zamanda 3, uzayda 4×4 adım-2) titreşen/plastik dokuyu kareler arası tutarlılıkla birlikte
    yakalar. Spektral normalizasyon eğitimi kararlı tutar. Telefona gitmez.
    """

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers: List[nn.Module] = []
        cin = 4
        for i in range(depth):
            cout = width * 2 ** min(i, 3)
            layers += [sn(nn.Conv3d(cin, cout, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))), nn.LeakyReLU(0.2, inplace=True)]
            cin = cout
        layers.append(sn(nn.Conv3d(cin, 1, (3, 3, 3), padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frames * 2.0 - 1.0, mask.to(frames.dtype)], dim=2)          # [B, T, 4, H, W]
        return self.net(x.transpose(1, 2))                                        # [B, 1, T, h, w]


class Adversarial:
    """Onarıcı (üreteç) kaybı ve ayrıştırıcı adımı; hinge kaybı. ``active`` değilken hiçbir şey yapmaz."""

    def __init__(self, disc: TemporalPatchGAN, optimizer: torch.optim.Optimizer, weight: float):
        self.disc, self.optimizer, self.weight = disc, optimizer, weight
        self.active = False
        self._pair: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def generator_loss(self, fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.active or self.weight <= 0:
            return fake.new_zeros(())
        self._pair = (fake.detach(), real.detach(), mask.detach())
        self.disc.requires_grad_(False)
        loss = -self.disc(fake, mask).float().mean()
        self.disc.requires_grad_(True)
        return loss

    def discriminator_step(self) -> Dict[str, float]:
        if not self.active or self._pair is None:
            return {}
        fake, real, mask = self._pair
        self._pair = None
        d_real = self.disc(real, mask).float()
        d_fake = self.disc(fake, mask).float()
        loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"gan/d": loss.item(), "gan/real": d_real.mean().item(), "gan/fake": d_fake.mean().item()}


class VGGPerceptual(nn.Module):
    """İsteğe bağlı algısal kayıp (torchvision VGG16 özellikleri; ağırlık dosyası kullanıcıdan)."""

    def __init__(self, weights_path: str):
        super().__init__()
        from torchvision.models import vgg16

        net = vgg16(weights=None)
        if weights_path:
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.blocks = nn.ModuleList([net.features[:4], net.features[4:9], net.features[9:16]]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred.flatten(0, 1).float() - self.mean) / self.std
        y = (target.flatten(0, 1).float() - self.mean) / self.std
        loss = pred.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def detector_loss(det: Batch, batch: Batch, w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    vm = batch["valid"].view(*batch["valid"].shape, 1, 1, 1).float()
    overlay, shadow, scene = batch["overlay"].float(), batch["shadow"].float(), batch["scene_text"].float()
    alpha_gt, fg_gt = batch["alpha"].float(), batch["fg_rgb"].float()
    logit = lambda k: det[k].float()
    bce = lambda k, target, pos: masked_mean(F.binary_cross_entropy_with_logits(logit(k), target, pos_weight=torch.tensor(pos, device=target.device), reduction="none"), vm)

    prob = torch.sigmoid(logit("overlay_logit")) * vm
    dice = 1.0 - (2.0 * (prob * overlay).sum() + 1.0) / (prob.sum() + (overlay * vm).sum() + 1.0)
    l_overlay = bce("overlay_logit", overlay, 3.0) + dice
    l_shadow = bce("shadow_logit", shadow, 3.0)
    # λ₆: sahne yazısını tanı VE onu kaplama sanma.
    l_scene = bce("scene_text_logit", scene, 3.0) + masked_mean(prob, scene * vm)

    alpha, fg = det["alpha"].float(), det["fg_rgb"].float()
    region = dilate((alpha_gt > 0).float(), 4)
    l_alpha = masked_mean((alpha - alpha_gt).abs(), vm * (0.2 + 0.8 * region))
    l_fg = masked_mean((fg - fg_gt).abs(), vm * (alpha_gt > 0.05))
    # Yeniden birleştirme: tahmin edilen (α, F) temiz arka planla girdiyi geri vermeli.
    linear = batch["linear_blend"].view(-1, 1, 1, 1, 1)
    clean = batch["clean"].float()
    recomposed = torch.where(linear, composite(clean, fg, alpha, "linear"), composite(clean, fg, alpha, "gamma"))
    l_recomp = masked_mean((recomposed - batch["composite"].float()).abs(), vm * region)
    # Güven: ayrıştırmanın gerçekten doğru olduğu yerde yüksek olmalı.
    with torch.no_grad():
        correct = torch.exp(-(alpha - alpha_gt).abs() / 0.1) * (1.0 - (torch.sigmoid(logit("overlay_logit")) - overlay).abs())
    l_conf = masked_mean(F.binary_cross_entropy_with_logits(logit("confidence_logit"), correct, reduction="none"), vm)

    total = l_overlay + 0.5 * l_shadow + w.scene_text * l_scene + w.alpha * (l_alpha + l_fg + l_recomp) + 0.2 * l_conf
    with torch.no_grad():
        hard = (prob > 0.5).float()
        iou = ((hard * overlay).sum() / ((hard + overlay).clamp(max=1) * vm).sum().clamp_min(1.0)).item()
    return total, {
        "det/overlay": l_overlay.item(), "det/shadow": l_shadow.item(), "det/scene_text": l_scene.item(),
        "det/alpha": l_alpha.item(), "det/fg": l_fg.item(), "det/recomp": l_recomp.item(), "det/conf": l_conf.item(), "det/iou": iou,
    }


def flow_loss(flows: Batch, batch: Batch, gamma: float = 0.85) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Gerçek akış biliniyorsa dizi kaybı; bilinmiyorsa temiz kareler üzerinde fotometrik kayıp."""
    pm = pair_mask(batch["valid"]).float()
    clean = batch["clean"].float()
    b, t = clean.shape[:2]
    stats: Dict[str, float] = {}
    iters_fw = flows.get("iters_fw", [flows["flow_fw"]])
    iters_bw = flows.get("iters_bw", [flows["flow_bw"]])
    n = len(iters_fw)
    total = clean.new_zeros(())

    if bool(batch["has_flow"].all()):
        gt_fw, gt_bw = batch["flow_fw"].float(), batch["flow_bw"].float()
        for i, (fw, bw) in enumerate(zip(iters_fw, iters_bw)):
            weight = gamma ** (n - 1 - i)
            total = total + weight * (masked_mean((fw.float() - gt_fw).abs(), pm) + masked_mean((bw.float() - gt_bw).abs(), pm))
        with torch.no_grad():
            epe_fw = (flows["flow_fw"].float() - gt_fw).pow(2).sum(2, keepdim=True).sqrt()
            epe_bw = (flows["flow_bw"].float() - gt_bw).pow(2).sum(2, keepdim=True).sqrt()
            target_fw, target_bw = torch.exp(-epe_fw), torch.exp(-epe_bw)
            stats["flow/epe"] = masked_mean(epe_fw, pm).item()
    else:
        # Fotometrik: model kaplamalı kareyi görür ama kayıp TEMİZ karelerde ölçülür;
        # böylece maske altındaki arka plan akışını gerçek akış olmadan öğrenir.
        prev, nxt = clean[:, :-1].flatten(0, 1), clean[:, 1:].flatten(0, 1)
        for i, (fw, bw) in enumerate(zip(iters_fw, iters_bw)):
            weight = gamma ** (n - 1 - i)
            fw2, bw2 = fw.float().flatten(0, 1), bw.float().flatten(0, 1)
            e_fw = charbonnier(warp(nxt, fw2) - prev).mean(1, keepdim=True)
            e_bw = charbonnier(warp(prev, bw2) - nxt).mean(1, keepdim=True)
            smooth = sum((f[..., :, 1:] - f[..., :, :-1]).abs().mean() + (f[..., 1:, :] - f[..., :-1, :]).abs().mean() for f in (fw2, bw2))
            total = total + weight * (masked_mean(e_fw, pm.flatten(0, 1)) + masked_mean(e_bw, pm.flatten(0, 1)) + 0.05 * smooth)
        with torch.no_grad():
            fw2, bw2 = flows["flow_fw"].float().flatten(0, 1), flows["flow_bw"].float().flatten(0, 1)
            target_fw = torch.exp(-(warp(nxt, fw2) - prev).abs().mean(1, keepdim=True) / 0.05).view(b, t - 1, 1, *clean.shape[-2:])
            target_bw = torch.exp(-(warp(prev, bw2) - nxt).abs().mean(1, keepdim=True) / 0.05).view(b, t - 1, 1, *clean.shape[-2:])
            stats["flow/photo"] = masked_mean((warp(nxt, fw2) - prev).abs().mean(1, keepdim=True), pm.flatten(0, 1)).item()

    conf = masked_mean(F.binary_cross_entropy_with_logits(flows["conf_fw"].float(), target_fw, reduction="none"), pm) + masked_mean(
        F.binary_cross_entropy_with_logits(flows["conf_bw"].float(), target_bw, reduction="none"), pm
    )
    total = total + 0.1 * conf
    stats["flow/loss"] = total.item()
    stats["flow/conf"] = conf.item()
    return total, stats


def restoration_loss(
    out: Batch, batch: Batch, flow_fw: Optional[torch.Tensor], w: LossWeights, perceptual: Optional[VGGPerceptual],
    adv: Optional["Adversarial"] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mamba için λ₁..λ₅ (λ₆, λ₇ detector_loss içindedir)."""
    index = out["index"]
    clean = batch["clean"][:, index].float()
    raw = out["raw"].float()
    mask = out["mask"].float()
    valid = batch["valid"][:, index]
    vm = valid.view(*valid.shape, 1, 1, 1).float()
    log_b = out["log_b"].float()
    err = (raw - clean).abs()

    l_rec = masked_mean(charbonnier(raw - clean), mask)                                      # λ₁
    l_nll = masked_mean(err.detach().mean(2, keepdim=True) * torch.exp(-log_b) + log_b, mask)
    band = (dilate(mask, 8) - mask).clamp(0, 1) * vm
    l_keep = masked_mean(err, band)                                                          # λ₂

    l_temp = raw.new_zeros(())                                                               # λ₃
    if flow_fw is not None and raw.shape[1] > 1:
        fw = flow_fw.float()
        e_pred = (raw[:, :-1] - warp_pairs(raw, fw)).abs().mean(2, keepdim=True)
        region = torch.maximum(mask[:, :-1], warp_pairs(mask, fw))
        # Örtülen, VC-Flow'un güvenmediği ve sahne geçişine uğrayan pikseller kayıptan dışlanır.
        visible, e_true = temporal_validity(clean, fw, valid, out.get("reliable_fw"))
        # Yalnızca gerçek videodaki zamansal farkın ÜSTÜNDE kalan kısım cezalandırılır.
        l_temp = masked_mean(F.relu(e_pred - e_true), region * visible.float())

    l_perc = raw.new_zeros(())                                                               # λ₄
    if perceptual is not None and w.perceptual > 0:
        blend = mask * raw + (1 - mask) * clean
        l_perc = perceptual(blend, clean)

    l_edge = raw.new_zeros(())                                                               # λ₅
    a, bb, m = raw, clean, dilate(mask, 2)
    for _ in range(2):
        a, da = haar_dwt(a)
        bb, db = haar_dwt(bb)
        m = F.max_pool2d(m.flatten(0, 1), 2).view(*m.shape[:3], m.shape[3] // 2, m.shape[4] // 2)
        l_edge = l_edge + masked_mean((da - db).abs(), m)

    l_adv = raw.new_zeros(())                                                                # PatchGAN
    if adv is not None and adv.active:
        fake = mask * raw.clamp(0, 1) + (1 - mask) * clean
        l_adv = adv.generator_loss(fake, clean, mask)

    total = (
        w.mask_reconstruction * l_rec + w.nll * l_nll + w.preservation * l_keep
        + w.temporal_warp * l_temp + w.perceptual * l_perc + w.edge_frequency * l_edge + w.adversarial * l_adv
    )
    with torch.no_grad():
        mse = masked_mean((raw.clamp(0, 1) - clean) ** 2, mask).clamp_min(1e-10)
    return total, {
        "rec/mask": l_rec.item(), "rec/nll": l_nll.item(), "rec/keep": l_keep.item(), "rec/temporal": l_temp.item(),
        "rec/perceptual": l_perc.item(), "rec/edge": l_edge.item(), "rec/adv": l_adv.item(), "rec/psnr_mask": (10 * torch.log10(1 / mse)).item(),
    }


# =============================================================================
# 4. Aşama adımları
# =============================================================================

def ground_truth_override(batch: Batch, rng: random.Random, with_flow: bool) -> Batch:
    """Mamba aşaması: Detector/Flow yerine (hafif bozulmuş) gerçek haritalar kullanılır.

    Maske rastgele genişletilir ve alfa/renk gürültülenir; böylece Mamba, kusursuz
    olmayan bir Detector'la çalışmaya da hazırlanır.
    """
    hard = (batch["overlay"] | batch["shadow"]).float()
    hard = dilate(hard, rng.randint(1, 4)) > 0.5
    alpha = (batch["alpha"] + torch.randn_like(batch["alpha"]) * 0.03).clamp(0, 1) * hard
    fg = (batch["fg_rgb"] + torch.randn_like(batch["fg_rgb"]) * 0.03).clamp(0, 1)
    override = {"mask_soft": hard.float(), "mask_hard": hard, "alpha": alpha, "fg_rgb": fg}
    if with_flow:
        override["flow_fw"], override["flow_bw"] = batch["flow_fw"], batch["flow_bw"]
    return override


def stage_step(
    engine: VideoCleaner, batch: Batch, cfg: TrainConfig, rng: random.Random, perceptual: Optional[VGGPerceptual],
    adv: Optional[Adversarial] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Bir mikro-yığın için kayıp. Hangi modelin eğitildiği ``cfg.stage`` ile belirlenir."""
    frames, valid, w = batch["frames"], batch["valid"], cfg.weights
    if cfg.stage == "detector":
        return detector_loss(engine.detector(frames, valid), batch, w)

    if cfg.stage == "flow":
        soft = dilate((batch["overlay"] | batch["shadow"]).float(), rng.randint(0, 3))
        return flow_loss(engine.flow(frames, soft, valid), batch)

    mode = "A" if rng.random() < cfg.mode_a_prob else "B"
    if cfg.stage == "mamba":
        known_flow = bool(batch["has_flow"].all())
        override = ground_truth_override(batch, rng, with_flow=known_flow)
        if not known_flow:
            with torch.no_grad():       # gerçek akış yoksa dondurulmuş Flow modelinin tahmini önsel olur
                flows = engine.flow(frames, override["mask_soft"], valid)
            override["flow_fw"], override["flow_bw"] = flows["flow_fw"], flows["flow_bw"]
        out = engine(frames, valid, mode=mode, override=override, detach_priors=True)
        return restoration_loss(out, batch, override["flow_fw"] if mode == "B" else None, w, perceptual, adv)

    if cfg.stage == "joint":
        out = engine(frames, valid, mode=mode, detach_priors=False)
        l_det, s_det = detector_loss(out["det"], batch, w)
        l_flow, s_flow = flow_loss(out["flows"], batch)
        # Uçtan uca kayıp GERÇEK silinecek bölgede ölçülür: Detector kaçırırsa ceza görür.
        target_mask = (batch["overlay"] | batch["shadow"])[:, out["index"]]
        scored = dict(out)
        scored["mask"] = target_mask | out["mask"]
        l_rec, s_rec = restoration_loss(scored, batch, out["flows"]["flow_fw"].detach() if mode == "B" else None, w, perceptual, adv)
        return l_rec + w.detector * l_det + w.flow * l_flow, {**s_det, **s_flow, **s_rec}
    raise ValueError(f"bilinmeyen aşama: {cfg.stage}")


def trained_modules(engine: VideoCleaner, stage: str) -> List[nn.Module]:
    return {"detector": [engine.detector], "flow": [engine.flow], "mamba": [engine.mamba], "joint": [engine.detector, engine.flow, engine.mamba]}[stage]


def build_optimizer(engine: VideoCleaner, cfg: TrainConfig, lr: float, device: torch.device) -> torch.optim.Optimizer:
    for p in engine.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for module in trained_modules(engine, cfg.stage):
        for p in module.parameters():
            p.requires_grad_(True)
            (decay if p.dim() >= 2 else no_decay).append(p)       # norm, önyargı ve ölçek parametrelerine ağırlık azaltma yok
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.99), fused=device.type == "cuda")


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def update_ema(ema: VideoCleaner, engine: VideoCleaner, decay: float) -> None:
    for pe, p in zip(ema.parameters(), engine.parameters()):
        if p.requires_grad:
            pe.lerp_(p.detach(), 1.0 - decay)


# =============================================================================
# 5. Değerlendirme — üç ayrı kalite boyutu
# =============================================================================

@torch.no_grad()
def evaluate(
    engine: VideoCleaner, cfg: TrainConfig, device: torch.device, clips: int, perceptual: Optional[VGGPerceptual] = None,
    lpips_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Onarım, koruma ve zamansal sadakati AYRI AYRI ölçer (birbirinin yerine geçmezler).

    Yalnızca DOĞRULAMA klipleri kullanılır (``clip_folders(cfg, "val")``) ve
    yordamsal klipler eğitimde hiç kullanılmayan bir tohumdan üretilir; aynı
    yapılandırmada her çağrı aynı klipleri değerlendirir.

    restoration_psnr      : silinen bölgede temiz görüntüye göre PSNR
    input_psnr            : aynı bölgede GİRDİNİN (altyazılı kare) PSNR'ı — karşılaştırma tabanı
    restoration_perceptual: VGG algısal uzaklık (``--vgg-weights`` verilmişse)
    preservation_max      : maske dışında |çıktı − girdi| en büyük değeri (kapı nedeniyle tam 0 olmalı)
    restoration_ssim      : silinen bölgenin kutusunda SSIM (11×11 Gauss)
    restoration_lpips     : LPIPS (yalnızca ``lpips`` paketi ve ağırlıkları varsa)
    temporal_excess       : maske bölgesinde, gerçek videonun üstünde kalan çarpıtma hatası
    temporal_ewarp        : onarılan bölgede E_warp (MSE; örtülme, güvenilmez akış ve sahne geçişi dışı)
    temporal_ewarp_source : aynı piksellerde TEMİZ videonun E_warp'ı (karşılaştırma tabanı)
    boundary_jump         : 7→8 (Mod A hedefi → sonraki) çiftinin fazlalık hatası − diğer çiftlerin ortancası
    detector_iou, missed_overlay, scene_text_erased, flow_epe : ara modellerin doğruluğu
    """
    engine.eval()
    synth = ClipSynthesizer(cfg, cfg.seed + 7_777_777, split="val")
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for _ in range(clips):
        batch = to_device({k: v.unsqueeze(0) for k, v in synth.sample().items()}, device)
        out = engine(batch["frames"], batch["valid"], mode="B")
        clean, frames, mask = batch["clean"].float(), batch["frames"].float(), out["mask"]
        truth = batch["overlay"] | batch["shadow"]
        region = (mask | truth).float()
        restored = out["restored"].float()
        mse = masked_mean((restored - clean) ** 2, region).clamp_min(1e-10)
        mse_in = masked_mean((frames - clean) ** 2, region).clamp_min(1e-10)
        outside = ~mask.expand_as(frames)
        metrics = {
            "restoration_psnr": (10 * torch.log10(1 / mse)).item() if region.sum() > 0 else float("nan"),
            "input_psnr": (10 * torch.log10(1 / mse_in)).item() if region.sum() > 0 else float("nan"),
            "preservation_max": (restored - frames)[outside].abs().max().item() if outside.any() else 0.0,
            "detector_iou": ((mask & truth).sum() / (mask | truth).sum().clamp_min(1)).item(),
            "missed_overlay": ((truth & ~mask).sum() / truth.sum().clamp_min(1)).item(),
            "scene_text_erased": ((batch["scene_text"] & mask).sum() / batch["scene_text"].sum().clamp_min(1)).item(),
        }
        if bool(batch["has_flow"].all()):
            fw = batch["flow_fw"].float()
            # Akış GERÇEK (bilinen) olduğundan modelin güven haritası değil, örtülme/sahne geçişi dışlanır.
            visible, e_true = temporal_validity(clean, fw, batch["valid"])
            diff = restored[:, :-1] - warp_pairs(restored, fw)
            e_out = diff.abs().mean(2, keepdim=True)
            pair_region = torch.maximum(region[:, :-1], warp_pairs(region, fw)) * visible.float()
            metrics["temporal_excess"] = masked_mean(F.relu(e_out - e_true), pair_region).item()
            # E_warp: onarılan bölgede (örtülme/sahne geçişi dışı) kareler arası çarpıtma hatası, MSE.
            sq = diff.pow(2).mean(2, keepdim=True)
            metrics["temporal_ewarp"] = masked_mean(sq, pair_region).item() if pair_region.sum() > 0 else float("nan")
            metrics["temporal_ewarp_source"] = (
                masked_mean((clean[:, :-1] - warp_pairs(clean, fw)).pow(2).mean(2, keepdim=True), pair_region).item()
                if pair_region.sum() > 0 else float("nan")
            )
            # Mod A hedefi → sonraki kare (7→8): bu çiftin fazlalık hatası diğer
            # çiftlerin ortancasını aşmamalı ("sıçrama yok").
            per_pair = [
                masked_mean(F.relu(e_out - e_true)[:, i], pair_region[:, i]).item()
                for i in range(e_out.shape[1]) if pair_region[:, i].sum() > 0
            ]
            if pair_region[:, TARGET_A].sum() > 0 and len(per_pair) > 1:
                boundary = masked_mean(F.relu(e_out - e_true)[:, TARGET_A], pair_region[:, TARGET_A]).item()
                metrics["boundary_jump"] = max(0.0, boundary - float(np.median(per_pair)))
            metrics["flow_epe"] = (out["flows"]["flow_fw"].float() - fw).pow(2).sum(2).sqrt().mean().item()
        if perceptual is not None and region.sum() > 0:
            metrics["restoration_perceptual"] = perceptual(restored, clean).item()
        if lpips_fn is not None and region.sum() > 0:
            m = region.flatten(0, 1).expand(-1, 3, -1, -1).bool()
            a = torch.where(m, restored.flatten(0, 1), clean.flatten(0, 1))      # maske dışı = temiz → yalnız onarım ölçülür
            metrics["restoration_lpips"] = lpips_fn(a * 2 - 1, clean.flatten(0, 1) * 2 - 1).mean().item()
        ssim = masked_ssim(restored, clean, region)
        if ssim is not None:
            metrics["restoration_ssim"] = ssim
        for k, v in metrics.items():
            if not math.isnan(v):       # her ölçüt yalnızca tanımlı olduğu kliplerde ortalanır
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    result = {k: sums[k] / counts[k] for k in sums}
    result["clips"] = float(clips)
    result["folder_clips_in_val"] = float(len(synth.clips))
    return result


def masked_ssim(x: torch.Tensor, y: torch.Tensor, region: torch.Tensor) -> Optional[float]:
    """SSIM (Wang 2004; 11×11, σ=1,5, C1=0,01², C2=0,03²) haritasının ``region`` üzerindeki ortalaması."""
    if region.sum() == 0:
        return None
    x2, y2 = x.flatten(0, 1).float(), y.flatten(0, 1).float()
    if min(x2.shape[-2:]) < 11:
        return None
    g = torch.exp(-((torch.arange(11, device=x.device, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :]).expand(x2.shape[1], 1, 11, 11).contiguous()
    blur = lambda z: F.conv2d(z, k, padding=5, groups=x2.shape[1])
    mx, my = blur(x2), blur(y2)
    sxx, syy, sxy = blur(x2 * x2) - mx * mx, blur(y2 * y2) - my * my, blur(x2 * y2) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return masked_mean(ssim, region.flatten(0, 1).float()).item()


def load_lpips(device: torch.device) -> Optional[nn.Module]:
    """``lpips`` paketi (AlexNet) varsa yükler; yoksa None — VC-Tester LPIPS'i "ölçülmedi" yazar."""
    try:
        import lpips  # type: ignore
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as error:      # paket yok veya ağırlıklar indirilemedi
        print(f"[vc-tester] LPIPS yüklenemedi ({type(error).__name__}: {error}); ölçülmeyecek")
        return None


# ---- VC-Tester: üç eksenli sürüm kapıları ---------------------------------------------------------
#
# Her kapı ölçülen bir değerle karşılaştırılır. Ölçülemeyen kapı "ÖLÇÜLMEDİ" olur ve genel
# karar asla "GEÇTİ" olmaz (ölçülmeyen şey geçmiş sayılmaz).

TESTER_GATES = (
    # (eksen, ad, ölçüt anahtarı, karşılaştırma, eşik, birim/açıklama)
    ("preservation", "maske dışı bayt eşitliği", "preservation_max", "==", 0.0, "en büyük |çıktı−girdi|"),
    ("preservation", "sahne yazısı silme hatası", "scene_text_erased", "<", 0.001, "oran (<%0,1)"),
    ("restoration", "maske PSNR", "restoration_psnr", ">=", 28.0, "dB"),
    ("restoration", "LPIPS", "restoration_lpips", "<", 0.04, "AlexNet LPIPS"),
    ("restoration", "kalan yazı hatası (TER)", "restoration_ter", "<", 0.005, "OCR gerekir (<%0,5)"),
    ("temporal", "E_warp", "temporal_ewarp", "<", 0.0020, "MSE, onarılan bölge"),
    ("temporal", "7→8 hedef sınırı sıçraması", "boundary_jump", "<=", 1e-4, "fazlalık − ortanca"),
)

_OPS = {"==": lambda a, b: a == b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


def tester_audit(metrics: Dict[str, float], source: str = "") -> Dict[str, object]:
    """``evaluate`` ölçütlerinden VC-Tester raporu: her kapı GEÇTİ / KALDI / ÖLÇÜLMEDİ."""
    axes: Dict[str, Dict[str, object]] = {}
    verdicts = []
    for axis, name, key, op, limit, unit in TESTER_GATES:
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            status = "ÖLÇÜLMEDİ"
            note = "OCR modeli yok" if key == "restoration_ter" else ("lpips paketi/ağırlığı yok" if key == "restoration_lpips" else "bu veride tanımsız")
        else:
            status = "GEÇTİ" if _OPS[op](float(value), limit) else "KALDI"
            note = ""
        verdicts.append(status)
        axes.setdefault(axis, {"gates": []})["gates"].append(
            {"name": name, "metric": key, "value": value, "rule": f"{op} {limit}", "unit": unit, "status": status, **({"note": note} if note else {})}
        )
    for axis in axes.values():
        states = [g["status"] for g in axis["gates"]]
        axis["status"] = "KALDI" if "KALDI" in states else ("ÖLÇÜLMEDİ" if "ÖLÇÜLMEDİ" in states else "GEÇTİ")
    overall = "KALDI" if "KALDI" in verdicts else ("EKSİK" if "ÖLÇÜLMEDİ" in verdicts else "GEÇTİ")
    return {
        "tester": "VC-Tester", "source": source, "overall": overall, "axes": axes,
        "context": {k: metrics[k] for k in ("input_psnr", "restoration_ssim", "temporal_ewarp_source", "temporal_excess", "detector_iou", "missed_overlay", "flow_epe", "clips") if k in metrics},
    }


def format_audit(report: Dict[str, object]) -> str:
    lines = [f"VC-Tester — {report['source'] or 'kaynak belirtilmedi'} → GENEL: {report['overall']}"]
    for axis, data in report["axes"].items():
        lines.append(f"  [{data['status']}] {axis}")
        for g in data["gates"]:
            value = "—" if g["value"] is None else f"{g['value']:.6g}"
            lines.append(f"      {g['status']:<10} {g['name']}: {value} (kural {g['rule']}; {g['unit']}){'  — ' + g['note'] if 'note' in g else ''}")
    return "\n".join(lines)


def audit(cfg: TrainConfig, device: torch.device, use_lpips: bool) -> Dict[str, object]:
    """Denetim noktasının EMA ağırlıklarını doğrulama klipleriyle denetler (``--audit``)."""
    state = load_checkpoint(cfg.resume, cfg) if cfg.resume else None
    engine = make_engine(cfg)
    source = "rastgele başlangıç (EĞİTİLMEMİŞ — kapıların kalması beklenir)"
    if state:
        engine.load_state_dict(state.get("ema", state["engine"]))
        source = f"{cfg.resume} (aşama {state.get('stage')}, adım {state.get('step')})"
    engine = engine.to(device).eval()
    perceptual = VGGPerceptual(cfg.vgg_weights).to(device) if cfg.vgg_weights else None
    lpips_fn = load_lpips(device) if use_lpips else None
    metrics = evaluate(engine, cfg, device, clips=cfg.val_clips, perceptual=perceptual, lpips_fn=lpips_fn)
    return tester_audit(metrics, source)


# =============================================================================
# 6. Eğitim, benchmark, kalibrasyon
# =============================================================================

def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def peak_memory_mib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    try:
        import resource
    except ImportError:            # Windows: süreç belleği ölçülemez
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KiB


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path: Path, engine: VideoCleaner, ema: VideoCleaner, optimizer, step: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"engine": engine.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "stage": cfg.stage, "config": asdict(cfg)}, tmp)
    tmp.replace(path)


def load_checkpoint(path: str, cfg: TrainConfig) -> Dict:
    """Denetim noktasını okur ve model boyutunu (``tiny``) ona göre ayarlar."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_tiny = bool(state.get("config", {}).get("tiny", cfg.tiny))
    if saved_tiny != cfg.tiny:
        print(f"[devam] denetim noktası {'küçük' if saved_tiny else 'tam boy'} modellerle kaydedilmiş; yapılandırma buna uyduruldu")
        cfg.tiny = saved_tiny
    return state


def train(cfg: TrainConfig, device: torch.device, benchmark_steps: int = 0) -> Dict[str, float]:
    """Bir aşamayı eğitir. ``benchmark_steps`` > 0 ise yalnızca ölçüm yapıp rapor döndürür."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    default_steps, default_lr = STAGE_DEFAULTS[cfg.stage]
    total_steps = cfg.steps or default_steps
    base_lr = cfg.lr or default_lr

    resumed = load_checkpoint(cfg.resume, cfg) if cfg.resume else None
    engine = make_engine(cfg)
    if resumed:
        engine.load_state_dict(resumed["engine"])
    ema = copy.deepcopy(engine)
    if resumed:
        ema.load_state_dict(resumed.get("ema", resumed["engine"]))
        print(f"[devam] {cfg.resume} (aşama {resumed.get('stage')}, adım {resumed.get('step')})")
    engine, ema = engine.to(device), ema.to(device).requires_grad_(False)
    start_step = 0
    optimizer = build_optimizer(engine, cfg, base_lr, device)
    if resumed and resumed.get("stage") == cfg.stage and not benchmark_steps:
        optimizer.load_state_dict(resumed["optimizer"])
        start_step = int(resumed["step"])
    engine.train()
    for module in (engine.detector, engine.flow, engine.mamba):
        if module not in trained_modules(engine, cfg.stage):
            module.eval()
    perceptual = VGGPerceptual(cfg.vgg_weights).to(device) if cfg.vgg_weights else None
    runner = torch.compile(engine) if cfg.compile else engine
    adv = None
    if cfg.stage in ("mamba", "joint") and cfg.weights.adversarial > 0:
        disc = TemporalPatchGAN(width=16 if cfg.tiny else 64).to(device)
        adv = Adversarial(disc, torch.optim.Adam(disc.parameters(), lr=cfg.gan_lr, betas=(0.0, 0.9)), cfg.weights.adversarial)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    amp = cfg.amp and device.type == "cuda"
    params = [p for p in engine.parameters() if p.requires_grad]
    print(f"[{cfg.stage}] cihaz={device}, eğitilen parametre={sum(p.numel() for p in params):,}, kırpım={cfg.crop[0]}×{cfg.crop[1]}, "
          f"kare={cfg.frames}, etkin yığın={cfg.batch * cfg.accum}, BF16={'açık' if amp else 'kapalı'}, denetim noktası={'açık' if cfg.checkpointing else 'kapalı'}")

    loader = make_loader(cfg, cfg.seed)
    out_dir = Path(cfg.out) / cfg.stage
    log_file = None
    if not benchmark_steps:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")

    timings = {"data": [], "forward": [], "backward": [], "optim": [], "step": []}
    last_stats: Dict[str, float] = {}
    end_step = start_step + benchmark_steps if benchmark_steps else total_steps
    window_start = time.perf_counter()
    for step in range(start_step, end_step):
        lr = lr_at(step, total_steps, base_lr, cfg.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        t_data = t_fwd = t_bwd = 0.0
        t_step = time.perf_counter()
        if adv is not None:
            adv.active = step >= cfg.gan_start
        for _ in range(cfg.accum):
            t0 = time.perf_counter()
            batch = truncate_window(to_device(next(loader), device), rng, cfg.truncate_prob)
            sync(device)
            t1 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss, stats = stage_step(runner, batch, cfg, rng, perceptual, adv)
            sync(device)
            t2 = time.perf_counter()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"adım {step}: kayıp sonlu değil ({loss.item()}); {stats}")
            (loss / cfg.accum).backward()
            sync(device)
            t3 = time.perf_counter()
            t_data, t_fwd, t_bwd = t_data + (t1 - t0), t_fwd + (t2 - t1), t_bwd + (t3 - t2)
            last_stats = {"loss": loss.item(), **stats}
        t4 = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if adv is not None and adv.active:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                last_stats.update(adv.discriminator_step())
        update_ema(ema, engine, min(cfg.ema_decay, (1 + step) / (10 + step)))
        sync(device)
        t5 = time.perf_counter()
        for key, value in (("data", t_data), ("forward", t_fwd), ("backward", t_bwd), ("optim", t5 - t4), ("step", t5 - t_step)):
            timings[key].append(value)

        if benchmark_steps:
            if (step - start_step + 1) % max(1, benchmark_steps // 10) == 0:
                print(f"  ölçüm {step - start_step + 1}/{benchmark_steps}: adım {timings['step'][-1]:.3f}s, kayıp {last_stats['loss']:.4f}")
            continue
        if (step + 1) % cfg.log_every == 0 or step == start_step:
            elapsed = time.perf_counter() - window_start
            record = {"step": step + 1, "lr": lr, "grad_norm": float(grad_norm), "sec_per_step": elapsed / (cfg.log_every if step != start_step else 1), **last_stats}
            window_start = time.perf_counter()
            print(f"[{cfg.stage} {step + 1}/{total_steps}] " + " ".join(f"{k}={v:.4g}" for k, v in record.items() if k != "step"))
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
        if (step + 1) % cfg.save_every == 0 or step + 1 == total_steps:
            save_checkpoint(out_dir / "last.pt", engine, ema, optimizer, step + 1, cfg)
        if cfg.val_every and ((step + 1) % cfg.val_every == 0 or step + 1 == total_steps):
            metrics = evaluate(ema, cfg, device, cfg.val_clips, perceptual)
            engine.train()
            for module in (engine.detector, engine.flow, engine.mamba):
                if module not in trained_modules(engine, cfg.stage):
                    module.eval()
            print(f"[doğrulama {step + 1}] " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            log_file.write(json.dumps({"step": step + 1, "validation": metrics}) + "\n")
            log_file.flush()
    if log_file:
        log_file.close()

    skip = min(len(timings["step"]) // 5, 20)        # ısınma adımlarını at
    report = {f"{k}_s": float(np.mean(v[skip:])) for k, v in timings.items() if v}
    if timings["step"]:
        report["step_p95_s"] = float(np.percentile(timings["step"][skip:], 95))
        report["peak_memory_mib"] = peak_memory_mib(device)
        report["projected_hours"] = report["step_s"] * total_steps / 3600.0
        report["total_steps"] = total_steps
        report["final_loss"] = last_stats.get("loss", float("nan"))
    return report


def print_benchmark(cfg: TrainConfig, device: torch.device, report: Dict[str, float]) -> None:
    kind = "VRAM (ayrılan tepe)" if device.type == "cuda" else "RSS (süreç tepe)"
    print(f"\n=== Benchmark: {cfg.stage} @ {cfg.crop[0]}×{cfg.crop[1]}×{cfg.frames}, etkin yığın {cfg.batch * cfg.accum} ===")
    print(f"  adım süresi   : {report['step_s']:.3f} s ortalama, {report['step_p95_s']:.3f} s (p95)")
    print(f"    veri        : {report['data_s']:.3f} s   ileri : {report['forward_s']:.3f} s   geri : {report['backward_s']:.3f} s   optimizer : {report['optim_s']:.3f} s")
    print(f"  {kind:<14}: {report['peak_memory_mib']:.0f} MiB")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 20
        print(f"  VRAM doluluk  : %{100 * report['peak_memory_mib'] / total:.0f} ({total:.0f} MiB içinden)")
    print(f"  tahmini süre  : {int(report['total_steps']):,} adım × {report['step_s']:.3f} s = {report['projected_hours']:.1f} saat (bu aşama)")
    if device.type != "cuda":
        print("  UYARI: CUDA yok — bu sayılar yalnızca hattın çalıştığını gösterir, RTX 4090 süresi DEĞİLDİR.")
    record = {
        "stage": cfg.stage, "crop": list(cfg.crop), "frames": cfg.frames, "batch": cfg.batch, "accum": cfg.accum,
        "seed": cfg.seed, "tiny": cfg.tiny, "checkpointing": cfg.checkpointing, "amp": cfg.amp and device.type == "cuda",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(), **report,
    }
    path = Path(cfg.out) / f"benchmark_{cfg.stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  rapor         : {path}")


def calibrate(cfg: TrainConfig, device: torch.device) -> None:
    """Veri hattı, renk dönüşümleri ve bellek için hızlı kalibrasyon."""
    print("=== Kalibrasyon ===")
    x = torch.rand(1, 3, 64, 64)
    print(f"renk gidiş-dönüş hatası (sRGB↔doğrusal): {(linear_to_srgb(srgb_to_linear(x)) - x).abs().max():.2e}")
    synth = ClipSynthesizer(cfg, cfg.seed)
    print(f"yazı tipi: {len(synth.fonts)} dosya" + ("" if synth.fonts else " (Pillow'un gömülü yazı tipi kullanılacak; çeşitlilik için --fonts verin)"))
    val_clips = clip_folders(cfg, "val")
    print(f"klip kaynağı: eğitim {len(synth.clips)} klasör klibi, doğrulama {len(val_clips)} klasör klibi (ayrık) + yordamsal (oran {cfg.procedural_ratio if synth.clips else 1.0})")
    t0 = time.perf_counter()
    samples = [synth.sample() for _ in range(4)]
    per_clip = (time.perf_counter() - t0) / 4
    cover = np.mean([float((s["overlay"] | s["shadow"]).float().mean()) for s in samples])
    print(f"veri üretimi: {per_clip:.2f} s/klip ({cfg.frames} kare, {cfg.crop[0]}×{cfg.crop[1]}); tek işçiyle {1 / per_clip:.2f} klip/s; ortalama maske kaplaması %{100 * cover:.1f}")
    for s in samples[:1]:
        recon = composite(s["clean"], s["fg_rgb"], s["alpha"], "linear" if bool(s["linear_blend"]) else "gamma")
        print(f"birleştirme tutarlılığı (GT α, F, B → I): en büyük hata {(recon - s['composite']).abs().max():.2e}")
    engine = make_engine(cfg)
    print(format_parameter_report(engine.parameter_report()))
    for stage in ("detector", "flow", "mamba"):
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.stage, stage_cfg.accum, stage_cfg.workers = stage, 1, 0
        report = train(stage_cfg, device, benchmark_steps=3)
        print(f"  {stage:<9} adım {report['step_s']:.2f} s, bellek tepe {report['peak_memory_mib']:.0f} MiB, kayıp {report['final_loss']:.4f}")
    print("Kalibrasyon tamam. Gerçek süre için: --stage <aşama> --benchmark 1000")


# =============================================================================
# 7. Dışa aktarma (mobil dağıtım için FP16 ağırlıklar)
# =============================================================================

def export_safetensors(state: Dict[str, torch.Tensor], path: Path, metadata: Dict[str, str]) -> int:
    """Ağırlıkları FP16 olarak safetensors biçiminde yazar (harici paket gerekmez).

    Biçim: 8 bayt küçük-endian başlık uzunluğu, JSON başlık (ad → dtype/shape/ofset),
    ardından ham tensör baytları. Rust tarafında ayrıştırması kolaydır.
    """
    header: Dict[str, object] = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name in sorted(state):
        tensor = state[name]
        if not tensor.is_floating_point():
            continue
        data = tensor.detach().to(torch.float16).cpu().numpy().tobytes()
        header[name] = {"dtype": "F16", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)
    return 8 + len(encoded) + offset


def export(cfg: TrainConfig, out_dir: str) -> None:
    """EMA ağırlıklarını model başına ayrı FP16 dosyası olarak yazar ve FP16'nın etkisini ölçer."""
    state = load_checkpoint(cfg.resume, cfg) if cfg.resume else None
    engine = make_engine(cfg)
    source = "rastgele başlangıç (eğitilmemiş)"
    if state:
        engine.load_state_dict(state.get("ema", state["engine"]))
        source = f"{cfg.resume} (aşama {state.get('stage')}, adım {state.get('step')})"
    engine.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("detector", "flow", "mamba"):
        module = getattr(engine, name)
        size = export_safetensors(module.state_dict(), out / f"video_{name}.safetensors", {"model": name, "format": "video_cleaner/2", "source": source})
        worst = max((p.float() - p.to(torch.float16).float()).abs().max().item() for p in module.parameters())
        overflow = sum(int((p.abs() > 65504).sum()) for p in module.parameters())
        print(f"  video_{name}.safetensors: {size / 2 ** 20:.1f} MiB, {count_parameters(module):,} parametre, FP16 en büyük yuvarlama hatası {worst:.2e}, taşan değer {overflow}")
    print(f"Dışa aktarıldı → {out}/  (kaynak: {source})")
    print("Not: Android çıkarım çalışma zamanı (Vulkan) henüz yazılmadı; INT8 nicemleme o aşamada ölçülecek.")


# =============================================================================
# 8. Kurulum denetimi
# =============================================================================

def selftest(device: torch.device) -> None:
    """Hızlı uçtan uca denetim: sözleşmeler, değişmezler ve her aşamadan birkaç adım."""
    print(f"=== Video_Cleaner kurulum denetimi (torch {torch.__version__}, cihaz {device}) ===")

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'HATA'}] {name}{' — ' + detail if detail else ''}")
        if not ok:
            raise SystemExit(f"denetim başarısız: {name}")

    torch.manual_seed(0)
    full = VideoCleaner().to(device).eval()
    report = full.parameter_report()
    print(format_parameter_report(report))
    check("parametre bütçeleri aşılmıyor", all(report[k]["params"] <= PARAM_BUDGET[k] for k in PARAM_BUDGET))

    # Mamba-2 SSD matris biçimi = ardışık özyineleme (dolgu kareleri dahil), çift hassasiyette
    from Video_Mamba import SSDMixer
    ssd = SSDMixer(64, 8, 3, 8).double()
    xs = torch.randn(9, WINDOW, 64, dtype=torch.float64)
    vs = torch.rand(9, WINDOW) > 0.2
    gap = 0.0
    for rev in (False, True):                       # ileri ve (dizi çevrilmeden) geri tarama
        xp, dtp, bp, cp = ssd.project(xs, vs, rev)
        gap = max(gap, float((ssd._mix(xp, dtp, bp, cp, rev) - ssd.reference(xp, dtp, bp, cp, rev)).abs().max()))
    flip_gap = float((ssd(xs, vs, reverse=True) - ssd(xs.flip(1), vs.flip(1)).flip(1)).abs().max())
    check("SSD (matris) = ardışık tarama; ters yön = çevrilmiş ileri", gap < 1e-10 and flip_gap < 1e-6,
          f"maks fark {gap:.1e}, çevirme farkı {flip_gap:.1e}")

    # Hibrit çıkarım bölmesi (telefon): kare başına kodlayıcı + zamansal çekirdek + kare başına çözücü
    # bileşimi tam modelle aynı olmalı. Sıfırla başlayan çıkış başlıkları rastgele doldurulur ki çözücü
    # gerçekten sınansın.
    with torch.no_grad():
        tiny = tiny_engine_config()
        det, mam = VideoDetector(tiny.detector).eval(), VideoMamba(tiny.mamba).eval()
        for head in (det.head[2], mam.head[2]):
            head.weight.normal_(0, 0.05)
            head.bias.normal_(0, 0.05)
        th, tw = 64, 96
        fr = torch.rand(1, WINDOW, 3, th, tw)
        vd = torch.ones(1, WINDOW, dtype=torch.bool)
        vd[0, -4:] = False
        ref = det(fr, vd)
        ref = torch.cat([ref[n] for n, _ in DETECTOR_HEADS], 2)[0]
        a0, a1, a8 = det.encode_frame(fr[0])
        a8, a16 = det.temporal_core(a8, vd)
        gap_d = float((det.decode_frame(a16, a0, a1, a8) - ref).abs().max())
        msk = (torch.rand(1, WINDOW, 1, th, tw) > 0.8).float()
        pri = {k: torch.rand(1, WINDOW, c, th, tw) for k, c in (("bg_gamma", 3), ("bg_linear", 3), ("analytic_conf", 1), ("pulled", 3), ("pulled_conf", 1))}
        ctx = torch.randn(1, tiny.mamba.widths[3])
        full_out = mam(fr, msk, pri, vd, context_state=ctx)
        packed = torch.cat([fr, msk, pri["bg_gamma"], pri["bg_linear"], pri["analytic_conf"], pri["pulled"], pri["pulled_conf"]], 2)[0]
        m1, m2, m8 = mam.encode_frame(packed * vd[0].view(-1, 1, 1, 1).float())
        m3, m16, nxt = mam.temporal_core(m8, vd, ctx)
        rgb, conf = mam.decode_frame(packed, m16, m3, m2, m1)
        gap_m = max(float((rgb - full_out["rgb"][0]).abs().max()), float((conf - full_out["confidence"][0]).abs().max()),
                    float((nxt - full_out["context_state"]).abs().max()))
        flw = VideoFlow(tiny.flow).eval()
        for conv in flw.refine.out:
            conv.weight.normal_(0, 0.05)
        fo = flw(fr, msk, vd)
        fmap, cmap = flw.encode_frame(torch.cat([fr, msk], 2)[0])
        mlr = F.avg_pool2d(msk[0], 8)
        hist, cf = flw.pair(fmap[:-1], fmap[1:], cmap[:-1], mlr[:-1], mlr[1:], th, tw)
        ok_pairs = (vd[0, :-1] & vd[0, 1:]).view(-1, 1, 1, 1)
        gap_f = float(((hist[-1] - fo["flow_fw"][0]) * ok_pairs).abs().max())
    check("hibrit bölme (kare/çekirdek/kare) = tam model", gap_d < 1e-4 and gap_m < 1e-4 and gap_f < 1e-4,
          f"algılayıcı {gap_d:.1e}, Mamba {gap_m:.1e}, akış {gap_f:.1e}")

    cfg = TrainConfig(tiny=True, crop=(64, 96), workers=0, accum=1, checkpointing=False, val_every=0, warmup=2)
    sample = ClipSynthesizer(cfg, 7).sample()
    batch = to_device({k: v.unsqueeze(0) for k, v in sample.items()}, device)
    # Gerçekçi bozulma zinciri: YUV420 gidiş-dönüşü düzgün görüntüde kayıpsıza yakın (renk kayması yok);
    # ffmpeg varsa H.264 gerçekten kodlanıp çözülür.
    grad = torch.linspace(0, 1, 96).view(1, 1, 1, 96).expand(2, 3, 64, 96).clone()
    grad[:, 0] *= 0.5
    yuv_psnr = 10 * math.log10(1.0 / max(float(((yuv420_roundtrip(grad) - grad) ** 2).mean()), 1e-12))
    h264 = codec_roundtrip(grad, "h264", 30) if FFMPEG else None
    check("bozulma: YUV420 (+ H.264)", yuv_psnr > 45 and (FFMPEG is None or (h264 is not None and h264.shape == grad.shape)),
          f"YUV420 PSNR {yuv_psnr:.1f} dB, ffmpeg {'var' if FFMPEG else 'yok (JPEG yedeği)'}")
    check("veri sözleşmesi", batch["frames"].shape == (1, WINDOW, 3, 64, 96) and batch["flow_fw"].shape == (1, WINDOW - 1, 2, 64, 96))
    space = "linear" if bool(sample["linear_blend"]) else "gamma"
    recon = composite(sample["clean"], sample["fg_rgb"], sample["alpha"], space)
    check("GT kaplama tutarlı: αF + (1−α)B = I", float((recon - sample["composite"]).abs().max()) < 2e-3)
    warped = warp_pairs(batch["clean"], batch["flow_fw"])
    inside = (batch["clean"][:, :-1] - warped).abs().mean().item()
    naive = (batch["clean"][:, :-1] - batch["clean"][:, 1:]).abs().mean().item()
    check("GT akış kareleri hizalıyor", inside <= naive + 1e-6 and inside < 0.03, f"hizalı hata {inside:.4f}, hizasız {naive:.4f}")

    with torch.no_grad():
        mask = batch["overlay"] | batch["shadow"]
        if not bool(mask.any()):
            mask[..., 40:56, 16:80] = True
        override = {"mask_soft": mask.float(), "mask_hard": mask, "alpha": batch["alpha"], "fg_rgb": batch["fg_rgb"],
                    "flow_fw": batch["flow_fw"], "flow_bw": batch["flow_bw"]}
        out_b = full(batch["frames"], batch["valid"], mode="B", override=override)
        out_a = full(batch["frames"], batch["valid"], mode="A", override=override)
    outside = ~out_b["mask"].expand_as(batch["frames"])
    check("bit-exact kapı: maskesiz pikseller girdiyle birebir aynı", torch.equal(out_b["restored"][outside], batch["frames"][outside]))
    check("Mod A (7+1+7) = Mod B'nin 7. karesi", torch.allclose(out_a["restored"][:, 0], out_b["restored"][:, TARGET_A], atol=1e-4))
    check("Mod B 15 kare, Mod A 1 kare üretir", out_b["restored"].shape[1] == WINDOW and out_a["restored"].shape[1] == 1
          and WINDOW == 15 and STRIDE == 11 and TARGET_A == WINDOW // 2)

    # Dinamik ROI tüpü ve akıllı bypass (çıkarım yolu)
    with torch.no_grad():
        empty = torch.zeros_like(mask)
        bypass = full(batch["frames"], batch["valid"], mode="B", override={**override, "mask_soft": empty.float(), "mask_hard": empty})
        small = torch.zeros_like(mask)
        small[..., 8:24, 8:40] = True
        margin, full.config.roi_margin = full.config.roi_margin, 8     # 64×96 denetim karesinde 32 px pay tüm kareyi kaplar
        try:
            cropped = full(batch["frames"], batch["valid"], mode="B", override={**override, "mask_soft": small.float(), "mask_hard": small})
        finally:
            full.config.roi_margin = margin
    check("akıllı bypass: maske yoksa model atlanır, çıktı = girdi", bypass["roi"] == (0, 0, 0, 0) and torch.equal(bypass["restored"], batch["frames"]))
    roi = cropped["roi"]
    check("ROI tüpü: küçük maskede kırpım, dışarısı bit-exact",
          isinstance(roi, tuple) and roi != (0, 0, 0, 0) and (roi[1] - roi[0]) * (roi[3] - roi[2]) < 64 * 96
          and torch.equal(cropped["restored"][~cropped["mask"].expand_as(batch["frames"])], batch["frames"][~cropped["mask"].expand_as(batch["frames"])])
          # yumuşatma bandı maskeden en fazla feather_px taşar
          and bool((cropped["mask"] <= dilate(small, full.config.feather_px)).all()),
          f"ROI {roi}")

    # SEDiT tarzı bağlam aktarımı: Mod B penceresi bir özet üretir; FiLM sıfırla başladığı için
    # eğitilmemiş modelde bağlam çıktıyı DEĞİŞTİRMEZ (eğitim kararlılığı).
    state = out_b["context_state"]
    check("context_state üretilir", state is not None and state.dim() == 2 and state.shape[0] == 1 and bool(torch.isfinite(state).all()),
          f"boyut {tuple(state.shape) if state is not None else None}")
    with torch.no_grad():
        carried = full(batch["frames"], batch["valid"], mode="B", override=override, context_state=torch.randn_like(state))
    check("bağlam köprüsü sıfır başlatılmış (çıktı değişmez)", torch.allclose(carried["restored"], out_b["restored"], atol=1e-5))

    # Geçerlilik maskeli zamansal kayıp: sahne geçişi olan çift tamamen, örtülen pikseller dışlanır.
    spliced = batch["clean"].clone()
    spliced[:, 8:] = 1 - spliced[:, 8:].flip(-1)
    keep, _ = temporal_validity(spliced, batch["flow_fw"], batch["valid"])
    check("sahne geçişi çifti (7→8) zamansal kayıptan dışlanır", not bool(keep[:, 7].any()) and bool(keep[:, :7].any()))
    unreliable = torch.zeros_like(keep)
    keep2, _ = temporal_validity(batch["clean"], batch["flow_fw"], batch["valid"], unreliable)
    check("VC-Flow'un güvenmediği pikseller dışlanır", not bool(keep2.any()))

    # VC-Tester kapı mantığı (ölçülmeyen kapı asla geçmiş sayılmaz)
    good = {"preservation_max": 0.0, "scene_text_erased": 0.0, "restoration_psnr": 31.0, "temporal_ewarp": 0.001, "boundary_jump": 0.0}
    check("VC-Tester: LPIPS/TER ölçülmeden genel karar EKSİK", tester_audit(good)["overall"] == "EKSİK")
    check("VC-Tester: tüm kapılar ölçülüp sağlanınca GEÇTİ", tester_audit({**good, "restoration_lpips": 0.02, "restoration_ter": 0.001})["overall"] == "GEÇTİ")
    check("VC-Tester: koruma ihlali KALDI", tester_audit({**good, "preservation_max": 1 / 255})["axes"]["preservation"]["status"] == "KALDI")

    # Zamansal PatchGAN: hinge kaybı, üreteç ve ayrıştırıcı adımları sonlu; ayrıştırıcı gerçek ile sahteyi ayırmaya başlar
    disc = TemporalPatchGAN(width=8, depth=2).to(device)
    gan = Adversarial(disc, torch.optim.Adam(disc.parameters(), lr=1e-3, betas=(0.0, 0.9)), 0.01)
    gan.active = True
    real = torch.rand(1, 6, 3, 32, 32, device=device)
    fake = (real * 0.0 + 0.5).requires_grad_(True)
    hole = torch.ones(1, 6, 1, 32, 32, device=device)
    g_loss = gan.generator_loss(fake, real, hole)
    g_loss.backward()
    first = gan.discriminator_step()
    for _ in range(20):
        gan.generator_loss(fake.detach(), real, hole)
        last = gan.discriminator_step()
    check("PatchGAN: üreteç gradyanı ve ayrıştırıcı öğrenmesi", fake.grad is not None and math.isfinite(g_loss.item()) and last["gan/d"] < first["gan/d"],
          f"D kaybı {first['gan/d']:.3f} → {last['gan/d']:.3f}")

    for stage in ("detector", "flow", "mamba", "joint"):
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg.stage, stage_cfg.steps = stage, 8
        stage_cfg.gan_start = 2                     # PatchGAN yolu da sınansın (adım 2'den itibaren)
        report = train(stage_cfg, device, benchmark_steps=4)
        check(f"eğitim adımı: {stage}", math.isfinite(report["final_loss"]), f"kayıp {report['final_loss']:.4f}, adım {report['step_s']:.2f} s")
    metrics = evaluate(VideoCleaner(tiny_engine_config()).to(device), cfg, device, clips=2)
    check("değerlendirme: koruma hatası tam 0", metrics["preservation_max"] == 0.0, f"ölçütler: {sorted(metrics)}")
    report = tester_audit(metrics, "küçük, eğitilmemiş model")
    print(format_audit(report))
    check("VC-Tester raporu üretildi (eğitilmemiş modelde kapıların kalması beklenir)",
          set(report["axes"]) == {"preservation", "restoration", "temporal"} and report["axes"]["preservation"]["gates"][0]["status"] == "GEÇTİ")
    print("Kurulum denetimi BAŞARILI.")


# =============================================================================
# 9. Komut satırı
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video_Cleaner eğitim / ölçüm / dışa aktarma", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["calibrate", "detector", "flow", "mamba", "joint"], default="detector")
    p.add_argument("--steps", type=int, default=0, help="0 = aşamanın varsayılanı")
    p.add_argument("--benchmark", type=int, default=0, metavar="N", help="N adım ölç, süre/bellek raporu ver, eğitme")
    p.add_argument("--selftest", action="store_true", help="kurulum denetimini çalıştır")
    p.add_argument("--export", default="", metavar="DİZİN", help="FP16 ağırlıkları dışa aktar")
    p.add_argument("--audit", action="store_true", help="VC-Tester: --resume ağırlıklarını sürüm kapılarıyla denetle (JSON + tablo)")
    p.add_argument("--audit-json", default="", metavar="DOSYA", help="VC-Tester raporunu bu dosyaya da yaz")
    p.add_argument("--lpips", action="store_true", help="VC-Tester'da LPIPS ölç (lpips paketi gerekir)")
    p.add_argument("--crop", default="256x448", help="eğitim kırpımı YxG (16'nın katı); tam kare için 720x1280")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0, help="0 = aşamanın varsayılanı")
    p.add_argument("--data", default="", help="klip klasörlerinin kökü (yoksa yordamsal veri)")
    p.add_argument("--val-data", default="", help="ayrı doğrulama klipleri (yoksa --data'nın --val-fraction kadarı ayrılır)")
    p.add_argument("--val-fraction", type=float, default=0.1, help="--val-data yokken doğrulamaya ayrılan klip oranı")
    p.add_argument("--val-clips", type=int, default=8, help="her doğrulamada değerlendirilen klip sayısı")
    p.add_argument("--loss-weights", default="", metavar="AD=DEĞER,...", help="ablasyon için kayıp ağırlıkları, ör. temporal_warp=0,edge_frequency=0.5")
    p.add_argument("--fonts", default="", help="ek yazı tipi dizini (çok dilli altyazı için)")
    p.add_argument("--text-file", default="", help="satır başına bir altyazı metni içeren UTF-8 dosya")
    p.add_argument("--procedural-ratio", type=float, default=0.25)
    p.add_argument("--mode-a-prob", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default="")
    p.add_argument("--vgg-weights", default="", help="algısal kayıp için VGG16 ağırlık dosyası (isteğe bağlı)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-amp", action="store_true", help="BF16 autocast'i kapat")
    p.add_argument("--no-checkpointing", action="store_true", help="aktivasyon denetim noktasını kapat (daha hızlı, daha çok VRAM)")
    p.add_argument("--compile", action="store_true", help="torch.compile kullan")
    p.add_argument("--tiny", action="store_true", help="küçük modeller (yalnızca hat denemesi)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    if args.selftest:
        selftest(device)
        return
    try:
        h, w = (int(v) for v in args.crop.lower().split("x"))
    except ValueError:
        raise SystemExit("--crop YxG biçiminde olmalı, ör. 256x448")
    if h % 16 or w % 16:
        raise SystemExit("--crop değerleri 16'nın katı olmalı")
    weights = LossWeights()
    for item in filter(None, (part.strip() for part in args.loss_weights.split(","))):
        name, _, value = item.partition("=")
        if not hasattr(weights, name.strip()):
            raise SystemExit(f"--loss-weights: bilinmeyen ağırlık '{name}'. Geçerli adlar: {', '.join(asdict(weights))}")
        setattr(weights, name.strip(), float(value))
    cfg = TrainConfig(
        stage=args.stage if args.stage != "calibrate" else "detector", steps=args.steps, crop=(h, w), batch=args.batch, accum=args.accum,
        lr=args.lr, data=args.data, val_data=args.val_data, val_fraction=args.val_fraction, val_clips=args.val_clips, weights=weights,
        fonts=args.fonts, text_file=args.text_file, procedural_ratio=args.procedural_ratio,
        mode_a_prob=args.mode_a_prob, workers=args.workers, out=args.out, resume=args.resume, vgg_weights=args.vgg_weights,
        seed=args.seed, amp=not args.no_amp, checkpointing=not args.no_checkpointing, compile=args.compile, tiny=args.tiny,
        log_every=args.log_every, save_every=args.save_every, val_every=args.val_every,
    )
    if args.export:
        export(cfg, args.export)
        return
    if args.audit:
        report = audit(cfg, device, args.lpips)
        print(format_audit(report))
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.audit_json:
            Path(args.audit_json).write_text(text, encoding="utf-8")
        raise SystemExit(0 if report["overall"] == "GEÇTİ" else 1)
    try:        # veri yapılandırmasını işçi süreçler başlamadan, anlaşılır bir hatayla doğrula
        train_clips, val_clips = clip_folders(cfg, "train"), clip_folders(cfg, "val")
    except (ValueError, OSError) as error:
        raise SystemExit(f"Veri yapılandırması hatalı: {error}")
    if cfg.data:
        print(f"[veri] eğitim {len(train_clips)} klip, doğrulama {len(val_clips)} klip (ayrık kümeler)")
    if args.stage == "calibrate":
        calibrate(cfg, device)
    elif args.benchmark:
        print_benchmark(cfg, device, train(cfg, device, benchmark_steps=args.benchmark))
    else:
        report = train(cfg, device)
        print(f"Aşama tamamlandı: ortalama adım {report.get('step_s', float('nan')):.3f} s, bellek tepe {report.get('peak_memory_mib', float('nan')):.0f} MiB")


if __name__ == "__main__":
    main()
