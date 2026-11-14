"""VC-Export — üç modelin hibrit çıkarım grafikleri (telefon) ve iskelet ölçümü.

Mimari ilke: ONNX yalnızca evrişim / matris çarpımı hızlandırıcısıdır. Kontrol akışı, 15 karelik
pencere, ara özelliklerin saklanması ve kareler arası yönetim grafiğin DIŞINDA (Rust halka tamponu /
Kotlin) yapılır. Her model üç parçaya bölünür; bileşimleri tam modelle birebir aynıdır (Trainer
selftest "hibrit bölme" denetimi):

    VC-Detector  det_enc  (kare başına, 1×3×288×512)      → 1/2, 1/4 atlama + 1/8 özellik
                 det_core (pencere, 30 × 1/8 özellik)     → zamansal karıştırma, 1/16 özellik
                 det_dec  (kare başına)                    → maske/alfa/renk/güven (8 kanal)
    VC-Flow      flow_enc (kare başına, 1×4×128×512 ROI)  → 1/8 eşleştirme + bağlam özelliği
                 flow_pair(çift başına)                    → tek MatMul tüm-çiftler korelasyonu +
                                                             2 adım evrişim iyileştirme → akış, güven
    VC-Mamba     mamba_enc (kare başına, 1×15×128×512 ROI) → 1/2, 1/4 atlama + 1/8 özellik
                 mamba_core(pencere, 30 × 1/8 özellik)     → Bi-SSM (Mamba-2 SSD, MatMul) + dikkat
                 mamba_dec (kare başına)                   → onarılmış RGB + güven

Ağırlıklar eğitilmemiştir (sabit tohum); çıktılar anlamsızdır. Ölçülen: operatör kümesi, düğüm sayısı,
dosya boyu, süre, bellek. Her grafik FP32 ve FP16 (ağırlık + ara değerler; giriş/çıkış FP32) üretilir.
Deterministik kurtarma (analitik alfa çözümü, piksel çekme / çarpıtma) sinir ağı değildir; Rust'ta koşar.

Kullanım:
    python Models/Export.py --out AndroidTest/assets/onnx             # dışa aktar + FP16
    python Models/Export.py --out DIR --measure                       # + bu makinede graf ve hat ölçümü
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
from Core import STRIDE, WINDOW, fold_layer_scale  # noqa: E402
from Video_Detector import DetectorConfig, VideoDetector  # noqa: E402
from Video_Flow import FlowConfig, VideoFlow  # noqa: E402
from Video_Mamba import MambaConfig, VideoMamba  # noqa: E402

T = WINDOW            # 15 (Core.WINDOW)
DET_HW = (288, 512)       # algılayıcı: tüm kare (küçültülmüş)
ROI_HW = (128, 512)       # akış ve onarım: altyazı şeridi ROI'si


class Fn(nn.Module):
    """Bir model yöntemini ONNX'e aktarılabilir modüle saran ince katman."""

    def __init__(self, owner: nn.Module, fn):
        super().__init__()
        self.owner = owner
        self.fn = fn
        # torch.onnx.export dışa aktarımdan sonra modülün eğitim kipini geri yükler; sarmalayıcı eğitim
        # kipinde kalırsa sahibini de eğitim kipine sokar ve BatchNorm yığın istatistiğine geçer.
        self.eval()

    def forward(self, *args):
        return self.fn(*args)


def models():
    torch.manual_seed(1234)
    det = VideoDetector(DetectorConfig(infer_chunk=0)).eval()
    flow = VideoFlow(FlowConfig(infer_chunk=0, query_chunk=0)).eval()
    mamba = VideoMamba(MambaConfig(infer_chunk=0, ssm_token_chunk=0)).eval()
    # Katman ölçeği son evrişime katlanır (birebir aynı sonuç; grafikte ayrı Mul kalmaz). BN zaten eval'de.
    return tuple(fold_layer_scale(m) for m in (det, flow, mamba))


def graphs():
    """(ad, modül, örnek girdiler, girdi adları, çıktı adları) — örnekler gerçek ara şekilleri taşır."""
    det, flow, mamba = models()
    g = torch.Generator().manual_seed(7)
    rand = lambda *s: torch.rand(*s, generator=g)
    dh, dw = DET_HW
    rh, rw = ROI_HW
    out = []
    with torch.no_grad():
        frame = rand(1, 3, dh, dw)
        d0, d1, d8 = det.encode_frame(frame)
        out.append(("det_enc", Fn(det, det.encode_frame), (frame,), ["frame"], ["s0", "s1", "x8"]))
        d8w = d8.repeat(T, 1, 1, 1)
        valid = torch.ones(1, T)
        d8o, d16 = det.temporal_core(d8w, valid)
        out.append(("det_core", Fn(det, det.temporal_core), (d8w, valid), ["x8", "valid"], ["s2", "x16"]))
        out.append(("det_dec", Fn(det, det.decode_frame), (d16[:1], d0, d1, d8o[:1]), ["x16", "s0", "s1", "s2"], ["heads"]))

        fm = rand(1, 4, rh, rw)
        f1, c1 = flow.encode_frame(fm)
        out.append(("flow_enc", Fn(flow, flow.encode_frame), (fm,), ["frame_mask"], ["feat", "ctx"]))
        m_lr = (rand(1, 1, rh // 8, rw // 8) > 0.8).float()

        def pair(fa, fb, ca, ma, mb):
            hist, conf = flow.pair(fa, fb, ca, ma, mb, rh, rw)
            return hist[-1], conf
        out.append(("flow_pair", Fn(flow, pair), (f1, f1.flip(-1), c1, m_lr, m_lr), ["feat_src", "feat_dst", "ctx_src", "mask_src", "mask_dst"],
                    ["flow", "conf"]))

        packed = rand(1, 15, rh, rw)
        m1, m2, m8 = mamba.encode_frame(packed)
        out.append(("mamba_enc", Fn(mamba, mamba.encode_frame), (packed,), ["packed"], ["s1", "s2", "x8"]))
        m8w = m8.repeat(T, 1, 1, 1)
        ctx = rand(1, mamba.config.widths[3])
        m3, m16, _ = mamba.temporal_core(m8w, valid, ctx)
        out.append(("mamba_core", Fn(mamba, mamba.temporal_core), (m8w, valid, ctx), ["x8", "valid", "context"], ["s3", "x16", "next_context"]))
        out.append(("mamba_dec", Fn(mamba, mamba.decode_frame), (packed, m16[:1], m3[:1], m2, m1), ["packed", "x16", "s3", "s2", "s1"],
                    ["rgb", "confidence"]))
    return out


def export_all(out_dir: Path, opset: int, check: bool) -> list:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers import float16

    report = []
    for name, module, inputs, in_names, out_names in graphs():
        path = out_dir / f"vc_{name}_fp32.onnx"
        started = time.time()
        with torch.no_grad():
            torch.onnx.export(module, inputs, str(path), input_names=in_names, output_names=out_names, opset_version=opset,
                              dynamo=False, do_constant_folding=True)
        model = onnx.load(str(path))
        ops: dict = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        info = {"name": name, "params": sum(p.numel() for p in module.parameters()), "export_s": round(time.time() - started, 1),
                "fp32_mb": round(path.stat().st_size / 2**20, 1), "nodes": len(model.graph.node),
                "inputs": {n: list(x.shape) for n, x in zip(in_names, inputs)},
                "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}
        # FP16: ağırlıklar ve ara değerler yarım hassasiyet; giriş/çıkış FP32 kalır (çağıran taraf değişmez).
        # Dışa aktarıcının yazdığı ara tür bilgisi (value_info) dönüştürücüyle çelişiyor; silinir, ORT yeniden çıkarır.
        # Normalleştirme ve softmax/üstel işlemleri FP32'de kalır (yarım hassasiyette taşma/kayıp riski).
        del model.graph.value_info[:]
        half = float16.convert_float_to_float16(
            model, keep_io_types=True,
            op_block_list=["InstanceNormalization", "LayerNormalization", "ReduceMean", "Softmax", "Exp", "Pow", "Sqrt", "CumSum"],
        )
        # Dönüştürücü aynı girdiye aynı Cast'i iki kez ekleyebiliyor (aynı çıktı adı): ikincisi atılır;
        # yinelenen düğüm adları benzersizleştirilir.
        defined: set = set()
        names: set = set()
        kept = []
        for k, node in enumerate(half.graph.node):
            if node.op_type == "Cast" and all(o in defined for o in node.output):
                continue
            defined.update(node.output)
            if not node.name or node.name in names:
                node.name = f"{node.name or node.op_type}_{k}"
            names.add(node.name)
            kept.append(node)
        del half.graph.node[:]
        half.graph.node.extend(kept)
        hpath = out_dir / f"vc_{name}_fp16.onnx"
        onnx.save(half, str(hpath))
        info["fp16_mb"] = round(hpath.stat().st_size / 2**20, 1)
        if check:
            with torch.no_grad():
                ref = module(*inputs)
            ref = ref if isinstance(ref, tuple) else (ref,)
            feeds = {n: x.numpy() for n, x in zip(in_names, inputs)}
            for kind, p in (("fp32", path), ("fp16", hpath)):
                outs = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, feeds)
                info[f"{kind}_maxdiff"] = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))
        report.append(info)
        print(json.dumps({k: v for k, v in info.items() if k != "ops"}, ensure_ascii=False), flush=True)
    return report


def measure_pipeline(out_dir: Path, kind: str, threads: int = 4) -> dict:
    """Tek pencere (15 kare) boyunca tüm hattı zincirler: kare başına kodlayıcılar, çekirdekler, çiftler,
    çözücüler. Ara özellikler süreç belleğinde (telefonda Rust halka tamponu) tutulur."""
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    names = ["det_enc", "det_core", "det_dec", "flow_enc", "flow_pair", "mamba_enc", "mamba_core", "mamba_dec"]
    t0 = time.time()
    sess = {n: ort.InferenceSession(str(out_dir / f"vc_{n}_{kind}.onnx"), so, providers=["CPUExecutionProvider"]) for n in names}
    load_s = time.time() - t0
    rng = np.random.default_rng(7)
    rnd = lambda *s: rng.random(s, dtype=np.float32)
    dh, dw = DET_HW
    rh, rw = ROI_HW
    times = {n: 0.0 for n in names}

    def run(n, feeds):
        t = time.time()
        out = sess[n].run(None, feeds)
        times[n] += time.time() - t
        return out

    valid = np.ones((1, T), np.float32)
    # Algılayıcı
    enc = [run("det_enc", {"frame": rnd(1, 3, dh, dw)}) for _ in range(T)]
    s2, x16 = run("det_core", {"x8": np.concatenate([e[2] for e in enc]), "valid": valid})
    for i in range(T):
        run("det_dec", {"x16": x16[i:i + 1], "s0": enc[i][0], "s1": enc[i][1], "s2": s2[i:i + 1]})
    del enc
    # Akış
    fe = [run("flow_enc", {"frame_mask": rnd(1, 4, rh, rw)}) for _ in range(T)]
    mlr = (rnd(1, 1, rh // 8, rw // 8) > 0.8).astype(np.float32)
    for i in range(T - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            run("flow_pair", {"feat_src": fe[a][0], "feat_dst": fe[b][0], "ctx_src": fe[a][1], "mask_src": mlr, "mask_dst": mlr})
    del fe
    # Onarım
    packed = [rnd(1, 15, rh, rw) for _ in range(T)]
    me = [run("mamba_enc", {"packed": p}) for p in packed]
    ctx_dim = sess["mamba_core"].get_inputs()[2].shape[1]
    s3, m16, _ = run("mamba_core", {"x8": np.concatenate([e[2] for e in me]), "valid": valid, "context": rnd(1, ctx_dim)})
    for i in range(T):
        run("mamba_dec", {"packed": packed[i], "x16": m16[i:i + 1], "s3": s3[i:i + 1], "s2": me[i][1], "s1": me[i][0]})
    # Mod B: pencere STRIDE (11) kare kayar → kare başına maliyet: kodlayıcı + 2 çift + çözücü × 15/11
    # (örtüşen 4 kare iki pencerede çözülür) + çekirdek/11. Kodlayıcı ve çift halkada yeniden kullanılır.
    per = {n: round(v * 1000 / (T if not n.endswith("core") else STRIDE), 1) for n, v in times.items()}
    per["flow_pair"] = round(times["flow_pair"] * 1000 / T, 1)
    per["det_dec"] = round(times["det_dec"] * 1000 / STRIDE, 1)
    per["mamba_dec"] = round(times["mamba_dec"] * 1000 / STRIDE, 1)
    total = round(sum(per.values()), 1)
    return {"kind": kind, "threads": threads, "load_s": round(load_s, 1), "frame_ms": per, "frame_ms_total": total,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--check", action="store_true", help="torch ↔ ORT sayısal karşılaştırma")
    ap.add_argument("--measure", action="store_true", help="bu makinede tüm hattı bir pencere boyunca ölç")
    ap.add_argument("--measure-only", default="", help="dışa aktarmadan yalnızca ölç (fp32 veya fp16)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.measure_only:
        print(json.dumps(measure_pipeline(out_dir, args.measure_only), ensure_ascii=False))
        return
    report = export_all(out_dir, args.opset, args.check)
    (out_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if args.measure:
        for kind in ("fp32", "fp16"):
            print(json.dumps(measure_pipeline(out_dir, kind), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
