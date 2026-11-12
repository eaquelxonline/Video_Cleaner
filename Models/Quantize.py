"""VC-Quantize — hibrit grafiklerin simetrik, kanal başına statik QDQ INT8 nicelemesi.

Neden statik QDQ: dinamik INT8 evrişimleri ``ConvInteger``'a çevirir ve ONNX Runtime'da optimize SIMD
çekirdeği yoktur (x86 ölçümü: 6,7× YAVAŞ). Statik QDQ'da ağırlıklar kanal başına simetrik int8, ara
değerler kalibrasyonla ölçeklenmiş simetrik int8'dir; ARM'da XNNPACK/KleidiAI SDOT / I8MM komutları ve
Qualcomm HTP (QNN) bu biçimi doğrudan yürütür.

Kalibrasyon verisi: AndroidTest/assets/calib/c01–c08 (depodaki doğrulama klipleri). Her klipten ardışık
kareler alınır; algılayıcı için tüm kare 288×512'ye, akış ve onarım için alt 128×512 altyazı şeridine
ölçeklenir. Kodlayıcı dışındaki grafiklerin girdileri, FP32 grafikler zincirlenerek ÜRETİLİR (her grafik
gerçek üretim dağılımını görür).

Not: eğitilmemiş ağırlıklarla kalibrasyon yalnızca hız ve operatör uyumluluğu ölçümü içindir. Eğitimden
sonra aynı betik eğitilmiş FP32 grafiklerle yeniden çalıştırılır; doğruluk raporu o zaman anlamlıdır.

Kullanım:
    python Models/Quantize.py --onnx DIR --calib AndroidTest/assets/calib [--frames 30] [--measure]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

T = 30
DET_HW = (288, 512)
ROI_HW = (128, 512)
GRAPHS = ["det_enc", "det_core", "det_dec", "flow_enc", "flow_pair", "mamba_enc", "mamba_core", "mamba_dec"]


def decode(path: Path, frames: int, size: tuple, band: bool) -> np.ndarray:
    """Klipten ilk ``frames`` kare, RGB 0..1, [N, 3, H, W]. band=True: genişlik 512'ye ölçeklenip alt şerit."""
    h, w = size
    vf = f"scale={w}:-2,crop={w}:{h}:0:ih-{h}" if band else f"scale={w}:{h}"
    try:
        import imageio_ffmpeg  # type: ignore

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg") or "ffmpeg"
    raw = subprocess.run([ff, "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8)
    n = arr.size // (h * w * 3)
    return arr[: n * h * w * 3].reshape(n, h, w, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255.0


class Feeds(dict):
    """Grafik başına kalibrasyon girdileri; yalnızca istenen grafikler ve en çok ``limit`` örnek saklanır
    (bellek: 30 karelik ara özellikler örnek başına onlarca MB)."""

    def __init__(self, wanted: List[str], limit: int):
        super().__init__({g: [] for g in GRAPHS})
        self.wanted, self.limit = set(wanted), limit

    def add(self, graph: str, item: Dict[str, np.ndarray]) -> None:
        if graph in self.wanted and len(self[graph]) < self.limit:
            self[graph].append(item)


def collect(onnx_dir: Path, clips: List[Path], frames: int, wanted: List[str], limit: int) -> Feeds:
    """FP32 grafikleri zincirleyerek her grafik için kalibrasyon girdileri üretir."""
    import onnxruntime as ort

    sess = {g: ort.InferenceSession(str(onnx_dir / f"vc_{g}_fp32.onnx"), providers=["CPUExecutionProvider"]) for g in GRAPHS}
    feeds = Feeds(wanted, limit)
    valid = np.ones((1, T), np.float32)
    ctx_dim = sess["mamba_core"].get_inputs()[2].shape[1]
    for clip in clips:
        det = decode(clip, frames, DET_HW, band=False)
        roi = decode(clip, frames, ROI_HW, band=True)
        if len(det) < T or len(roi) < T:
            continue
        det, roi = det[:T], roi[:T]
        # Algılayıcı
        enc = []
        for f in det:
            item = {"frame": f[None]}
            feeds.add("det_enc", item)
            enc.append(sess["det_enc"].run(None, item))
        core_in = {"x8": np.concatenate([e[2] for e in enc]), "valid": valid}
        feeds.add("det_core", core_in)
        s2, x16 = sess["det_core"].run(None, core_in)
        for i in range(0, T, 3):
            feeds.add("det_dec", {"x16": x16[i:i + 1], "s0": enc[i][0], "s1": enc[i][1], "s2": s2[i:i + 1]})
        # Akış (kalibrasyonda maske boş: gerçek altyazısız klipler)
        mask_full = np.zeros((1, 1, *ROI_HW), np.float32)
        fe = []
        for f in roi:
            item = {"frame_mask": np.concatenate([f[None], mask_full], 1)}
            feeds.add("flow_enc", item)
            fe.append(sess["flow_enc"].run(None, item))
        mlr = np.zeros((1, 1, ROI_HW[0] // 8, ROI_HW[1] // 8), np.float32)
        for i in range(0, T - 1, 2):
            feeds.add("flow_pair", {"feat_src": fe[i][0], "feat_dst": fe[i + 1][0], "ctx_src": fe[i][1], "mask_src": mlr, "mask_dst": mlr})
        # Onarım: 15 kanal = kare, maske, bg_gamma, bg_linear, analitik güven, çekilen, çekilen güveni
        packed = [np.concatenate([f, np.zeros((1, *ROI_HW), np.float32), f, f, np.full((1, *ROI_HW), 0.5, np.float32), f,
                                  np.full((1, *ROI_HW), 0.5, np.float32)])[None] for f in roi]
        me = []
        for p in packed:
            feeds.add("mamba_enc", {"packed": p})
            me.append(sess["mamba_enc"].run(None, {"packed": p}))
        mcore = {"x8": np.concatenate([e[2] for e in me]), "valid": valid, "context": np.zeros((1, ctx_dim), np.float32)}
        feeds.add("mamba_core", mcore)
        s3, m16, _ = sess["mamba_core"].run(None, mcore)
        for i in range(0, T, 3):
            feeds.add("mamba_dec", {"packed": packed[i], "x16": m16[i:i + 1], "s3": s3[i:i + 1], "s2": me[i][1], "s1": me[i][0]})
    return feeds


class Reader:
    """onnxruntime.quantization.CalibrationDataReader arayüzü."""

    def __init__(self, items: List[Dict[str, np.ndarray]]):
        self.items = iter(items)

    def get_next(self):
        return next(self.items, None)

    def rewind(self):
        pass


# Çekirdek grafikler (30 kare) kalibrasyonda tüm ara tensörlerini çıktı yapar: tam boyda 15 GB'ı aşıyor.
# Etkinleştirme aralıkları evrişim/MatMul'da uzamsal boyuttan büyük ölçüde bağımsız olduğundan, AYNI
# ağırlıklarla küçük bir kopya (T=10, 1/8 ölçekte 16×16) kalibre edilir; aralıklar önbelleğe yazılır ve tam
# boy grafik aynı önbellekle nicemlenir (tensör adları aynı modül yollarından gelir).
SMALL = {"det_core": (10, 16, 16), "mamba_core": (10, 16, 16)}

def activation_matmuls(path: Path) -> List[str]:
    """İki girdisi de etkinleştirme olan MatMul düğümleri (ağırlık içermeyen: SSD'nin 30×30 sönüm ⊙ C·Bᵀ
    çarpımları, pencere dikkatinin q·kᵀ ve p·v'si, akışın tüm-çiftler korelasyonu). Bunlar FP32 kalır: x86
    ölçümünde int8'de bağıl hata mamba_core %22, flow_pair %23 idi. Ağırlıklı Conv/Linear int8 olur."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    weights = {t.name for t in model.graph.initializer}
    consts = {o for n in model.graph.node if n.op_type == "Constant" for o in n.output}
    fixed = weights | consts
    return [n.name for n in model.graph.node if n.op_type == "MatMul" and not any(i in fixed for i in n.input)]


def small_core(onnx_dir: Path, name: str, feeds: List[Dict[str, np.ndarray]]):
    """Küçük çekirdek grafiği dışa aktarır; tam boy kalibrasyon girdilerini kırparak küçük girdiler üretir."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import Export

    t, h, w = SMALL[name]
    det, _, mamba = Export.models()
    owner = det if name == "det_core" else mamba
    module = Export.Fn(owner, owner.temporal_core)
    small_feeds = []
    for item in feeds:
        x8 = item["x8"]
        y0, x0 = (x8.shape[2] - h) // 2, (x8.shape[3] - w) // 2
        crop = {"x8": np.ascontiguousarray(x8[:t, :, y0:y0 + h, x0:x0 + w]), "valid": np.ascontiguousarray(item["valid"][:, :t])}
        if "context" in item:
            crop["context"] = item["context"]
        small_feeds.append(crop)
    names = list(small_feeds[0])
    path = onnx_dir / f"vc_{name}_calib_small.onnx"
    with torch.no_grad():
        torch.onnx.export(module, tuple(torch.from_numpy(small_feeds[0][n]) for n in names), str(path), input_names=names,
                          opset_version=18, dynamo=False, do_constant_folding=True)
    return path, small_feeds


def quantize(onnx_dir: Path, feeds: Dict[str, List[Dict[str, np.ndarray]]], only: List[str] = GRAPHS) -> list:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    report = []
    for g in only:
        src = onnx_dir / f"vc_{g}_fp32.onnx"
        pre = onnx_dir / f"vc_{g}_pre.onnx"
        out = onnx_dir / f"vc_{g}_int8.onnx"
        started = time.time()
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        cache = None
        calib_feeds = feeds[g]
        if g in SMALL:
            small, small_feeds = small_core(onnx_dir, g, feeds[g])
            small_pre = onnx_dir / f"vc_{g}_calib_small_pre.onnx"
            quant_pre_process(str(small), str(small_pre), skip_symbolic_shape=True)
            cache = onnx_dir / f"vc_{g}_ranges.json"
            cache.unlink(missing_ok=True)
            quantize_static(
                str(small_pre), str(onnx_dir / f"vc_{g}_calib_small_int8.onnx"), Reader(small_feeds),
                quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=activation_matmuls(small_pre),
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
                calibration_cache_path=str(cache),
            )
            for f in (small, small_pre, onnx_dir / f"vc_{g}_calib_small_int8.onnx"):
                f.unlink(missing_ok=True)
            calib_feeds = None
        quantize_static(
            str(pre), str(out), Reader(calib_feeds) if calib_feeds is not None else None,
            quant_format=QuantFormat.QDQ, per_channel=True,
            weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            nodes_to_exclude=activation_matmuls(pre),
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
            calibration_cache_path=str(cache) if cache else None,
        )
        pre.unlink(missing_ok=True)
        report.append({"name": g, "calib_items": len(feeds[g]), "fp32_matmuls": len(activation_matmuls(src)), "int8_mb": round(out.stat().st_size / 2**20, 1),
                       "fp32_mb": round(src.stat().st_size / 2**20, 1), "quant_s": round(time.time() - started, 1)})
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    return report


def measure(onnx_dir: Path, feeds: Dict[str, List[Dict[str, np.ndarray]]], threads: int = 4) -> list:
    """Bu makinede grafik başına süre (FP32 ↔ INT8) ve çıktı sapması (FP32'ye göre bağıl hata)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    rows = []
    for g in [g for g in GRAPHS if feeds[g]]:
        item = feeds[g][0]
        res = {}
        outs = {}
        for kind in ("fp32", "int8"):
            s = ort.InferenceSession(str(onnx_dir / f"vc_{g}_{kind}.onnx"), so, providers=["CPUExecutionProvider"])
            s.run(None, item)
            t0 = time.time()
            for _ in range(3):
                outs[kind] = s.run(None, item)
            res[f"{kind}_ms"] = round((time.time() - t0) / 3 * 1000, 1)
        pairs = list(zip(outs["fp32"], outs["int8"]))
        rel_max = max(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9)) for a, b in pairs)
        rel_rmse = max(float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((a ** 2).mean()) + 1e-9)) for a, b in pairs)
        rows.append({"name": g, **res, "speedup": round(res["fp32_ms"] / max(res["int8_ms"], 1e-3), 2),
                     "rel_rmse": round(rel_rmse, 4), "rel_max": round(rel_max, 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="vc_*_fp32.onnx dosyalarının klasörü (Export.py çıktısı)")
    ap.add_argument("--calib", default="AndroidTest/assets/calib")
    ap.add_argument("--clips", default="c01,c02,c03,c04,c05,c06,c07,c08")
    ap.add_argument("--frames", type=int, default=T)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--only", default="", help="yalnızca bu grafikler (virgülle)")
    ap.add_argument("--limit", type=int, default=64, help="grafik başına en çok kalibrasyon örneği")
    args = ap.parse_args()
    onnx_dir = Path(args.onnx)
    calib = Path(args.calib)
    clips = [p for name in args.clips.split(",") for p in sorted(calib.glob(f"{name}_*"))]
    only = args.only.split(",") if args.only else GRAPHS
    feeds = collect(onnx_dir, clips, max(args.frames, T), only, args.limit)
    print(json.dumps({g: len(v) for g, v in feeds.items()}, ensure_ascii=False))
    report = quantize(onnx_dir, feeds, only)
    if args.measure:
        report = {"quantize": report, "measure": measure(onnx_dir, feeds)}
    (onnx_dir / "quantize_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
