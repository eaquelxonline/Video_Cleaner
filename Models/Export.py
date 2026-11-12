"""VC-Export — üç modelin (VC-Detector, VC-Flow, VC-Mamba) mobil çalışma zamanı iskelet testi.

Amaç: tek bir GPU saati harcanmadan ÖNCE mimarinin telefonda çalışıp çalışmadığını görmek.
Ağırlıklar eğitilmemiştir (sabit tohumla rastgele); çıktı görüntüsü anlamsızdır. Ölçülen şeyler:
operatör uyumluluğu (ONNX Runtime CPU/XNNPACK/NNAPI), bellek ve pencere başına süre.

Her model sabit şekilli bir ONNX grafiğine dışa aktarılır ve dinamik INT8 (ağırlıklar int8) kopyası
üretilir. Şekiller üretim hattındaki gerçek iş yükünü temsil eder:

    detector : 30 kare × 3 × 288 × 512   (tüm kare, küçültülmüş — maske/alfa/renk)
    flow     : 30 kare × (3+1) × 128 × 512 (ROI şeridi; ileri + geri 29'ar çift)
    mamba    : 30 kare × 15 × 128 × 512    (ROI şeridi; Mod B: 30 kare girer, 30 çıkar)

Not: deterministik kurtarma (analitik alfa çözümü + tek seferlik piksel çekme, 5 boyutlu
``grid_sample``) sinir ağı değildir; telefonda Rust çekirdeğinde koşacak şekilde ayrı tutulur ve
burada dışa aktarılmaz.

Kullanım:
    python Models/Export.py --out AndroidTest/assets/onnx            # dışa aktar + INT8 + yerel ölçüm
    python Models/Export.py --out DIR --check                        # ayrıca torch ↔ ORT sayısal karşılaştırma
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Video_Detector import DetectorConfig, VideoDetector  # noqa: E402
from Video_Flow import FlowConfig, VideoFlow  # noqa: E402
from Video_Mamba import MambaConfig, VideoMamba  # noqa: E402

T = 30
SHAPES = {
    "detector": (T, 288, 512),
    "flow": (T, 128, 512),
    "mamba": (T, 128, 512),
}


class DetectorGraph(nn.Module):
    def __init__(self, model: VideoDetector):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        out = self.model(frames)
        return torch.cat([v for v in out.values()], dim=2)        # [1, T, 8, H, W]


class FlowGraph(nn.Module):
    def __init__(self, model: VideoFlow):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor, mask: torch.Tensor):
        out = self.model(frames, mask)
        return out["flow_fw"], out["flow_bw"], out["conf_fw"], out["conf_bw"]


class MambaGraph(nn.Module):
    def __init__(self, model: VideoMamba):
        super().__init__()
        self.model = model

    def forward(self, frames, mask, bg_gamma, bg_linear, analytic_conf, pulled, pulled_conf, context):
        priors = {"bg_gamma": bg_gamma, "bg_linear": bg_linear, "analytic_conf": analytic_conf, "pulled": pulled, "pulled_conf": pulled_conf}
        out = self.model(frames, mask, priors, context_state=context)
        return out["rgb"], out["confidence"], out["context_state"]


CHUNK = 0     # >0: modellerin çıkarım dilimlemesi (kare/çift ve SSM konum dilimi) grafiğe açılır → daha az bellek


def build(name: str):
    """Eğitilmemiş model (sabit tohum) + örnek girdiler."""
    torch.manual_seed(1234)
    t, h, w = SHAPES[name]
    g = torch.Generator().manual_seed(7)
    rand = lambda *s: torch.rand(*s, generator=g)
    if name == "detector":
        model = VideoDetector(DetectorConfig(infer_chunk=CHUNK)).eval()
        return DetectorGraph(model), (rand(1, t, 3, h, w),), ["frames"], ["heads"]
    if name == "flow":
        model = VideoFlow(FlowConfig(infer_chunk=CHUNK)).eval()
        mask = (rand(1, t, 1, h, w) > 0.8).float()
        return FlowGraph(model), (rand(1, t, 3, h, w), mask), ["frames", "mask"], ["flow_fw", "flow_bw", "conf_fw", "conf_bw"]
    model = VideoMamba(MambaConfig(infer_chunk=CHUNK, ssm_token_chunk=256 if CHUNK else 0)).eval()
    # Eğitilmemiş modelde başlıklar sıfırla başlar (çıktı = taban); süre ölçümü bundan etkilenmez.
    inputs = (
        rand(1, t, 3, h, w), (rand(1, t, 1, h, w) > 0.8).float(), rand(1, t, 3, h, w), rand(1, t, 3, h, w),
        rand(1, t, 1, h, w), rand(1, t, 3, h, w), rand(1, t, 1, h, w), rand(1, model.config.widths[3]),
    )
    names = ["frames", "mask", "bg_gamma", "bg_linear", "analytic_conf", "pulled", "pulled_conf", "context"]
    return MambaGraph(model), inputs, names, ["rgb", "confidence", "next_context"]


def export(name: str, out_dir: Path, opset: int) -> dict:
    graph, inputs, in_names, out_names = build(name)
    path = out_dir / f"vc_{name}_fp32.onnx"
    params = sum(p.numel() for p in graph.parameters())
    started = time.time()
    with torch.no_grad():
        torch.onnx.export(graph, inputs, str(path), input_names=in_names, output_names=out_names, opset_version=opset, dynamo=False,
                          do_constant_folding=True)
    info = {"name": name, "params": params, "shape": SHAPES[name], "export_s": round(time.time() - started, 1),
            "fp32_mb": round(path.stat().st_size / 2**20, 1)}
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    ops: dict = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    info["nodes"] = len(m.graph.node)
    info["ops"] = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
    # Dinamik INT8: ağırlıklar int8, etkinleştirmeler çalışma anında nicemlenir (MatMul + Conv).
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = out_dir / f"vc_{name}_int8.onnx"
    quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    info["int8_mb"] = round(q.stat().st_size / 2**20, 1)
    return info


def measure(name: str, out_dir: Path, check: bool) -> dict:
    """Bu makinede (x86 CPU) ORT ile süre, en yüksek bellek ve (isteğe bağlı) torch ile sayısal fark."""
    import numpy as np
    import onnxruntime as ort
    graph, inputs, in_names, out_names = build(name)
    feeds = {n: x.numpy() for n, x in zip(in_names, inputs)}
    result = {}
    for kind in ("fp32", "int8"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(out_dir / f"vc_{name}_{kind}.onnx"), so, providers=["CPUExecutionProvider"])
        sess.run(None, feeds)                                       # ısınma
        runs = []
        for _ in range(2):
            t0 = time.time()
            outs = sess.run(None, feeds)
            runs.append(time.time() - t0)
        result[f"{kind}_ms"] = round(min(runs) * 1000)
        if check:
            with torch.no_grad():
                ref = graph(*inputs)
            ref = ref if isinstance(ref, tuple) else (ref,)
            diff = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))
            result[f"{kind}_maxdiff"] = diff
    result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--only", default="detector,flow,mamba")
    ap.add_argument("--check", action="store_true", help="torch ↔ ORT sayısal karşılaştırma")
    ap.add_argument("--no-measure", action="store_true")
    ap.add_argument("--chunk", type=int, default=0, help="çıkarım dilimi (kare); 0 = tek parça")
    args = ap.parse_args()
    global CHUNK
    CHUNK = args.chunk
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for name in args.only.split(","):
        info = export(name, out_dir, args.opset)
        if not args.no_measure:
            info.update(measure(name, out_dir, args.check))
        report.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
    (out_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
