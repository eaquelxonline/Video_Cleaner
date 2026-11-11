#!/usr/bin/env python3
"""Video Cleaner çıktı doğrulaması: cihazda üretilen her çıktıyı kendi kaynağıyla karşılaştırır.

Hem emülatör kalibrasyonunda (bitrate-calibration.yml) hem gerçek cihazda (firebase-device-test.yml)
aynı betik kullanılır. Kalite ölçümü kaynağa göredir: yalnızca Video Cleaner'ın yeniden kodlama kaybını
ölçer; kaynağın kendi sıkıştırma kusurları ölçüme girmez.

Girdi:
  --logcat   VCCAL satırlarını içeren logcat dökümü (EncoderCalibrationTest)
  --outputs  cihazdan çekilen çıktı klasörü (<klip>__<aday>.mp4)
  --sources  kaynak klipler (AndroidTest/assets/calib)
  --vmaf     libvmaf `vmaf` aracı (yoksa VMAF "-")
  --label    rapor başlığı (ör. "Pixel 8 · API 35 · emülatör değil")
Çıktı:
  --csv, --markdown

Denetimler (PASS/FAIL):
  kare: kaynak ve çıktının her karesinin PTS'i (ilk kareye göre) ≤ 1 ms farkla aynı; kare sayısı aynı
  boyut: çözünürlük (tek boyutta yalnızca son satır/sütun), piksel en-boy oranı (SAR), görüntü oranı, yön
  ses: kodek, örnekleme hızı, kanal; "kopyalandı" ise paketlerin MD5'i birebir; süre ve A/V ofseti
  renk: MP4 colr ve bit akışı VUI ayrı okunur; oynatıcı kuralıyla etkin renk kaynakla aynı mı
"""
import argparse
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

LINE = re.compile(r"VCCAL\s*:\s*(RESULT|SOURCE|CHECK|AUDIOPROBE)\s+(.*)$")
FIELD = re.compile(r'(\w+)=("([^"]*)"|\S+)')
VMAF_MODELS = [("vmaf", "vmaf_v0.6.1"), ("vmaf_neg", "vmaf_v0.6.1neg")]
MATRIX = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020", 0: "gbr(RGB)", 2: "belirtilmemiş"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def parse_log(path):
    out = {"RESULT": [], "SOURCE": {}, "CHECK": {}, "AUDIOPROBE": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            fields = {k: (q if v.startswith('"') else v) for k, v, q in FIELD.findall(m.group(2))}
            kind = m.group(1)
            if kind == "SOURCE":
                out["SOURCE"][fields["clip"]] = fields
            elif kind == "CHECK":
                out["CHECK"][(fields["clip"], fields["cand"])] = fields
            else:
                out[kind].append(fields)
    return out


def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]).stdout)


def packets(path, sel):
    j = json.loads(run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "packet=pts_time,dts_time,duration_time,size,flags", "-of", "json", path]).stdout or "{}")
    return j.get("packets", [])


def f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def stream_stats(path, sel):
    pk = packets(path, sel)
    if not pk:
        return None
    times = sorted(f(p.get("pts_time"), f(p.get("dts_time"))) for p in pk)
    last_dur = f(pk[-1].get("duration_time"))
    span = times[-1] - times[0] + last_dur
    total = sum(int(p.get("size", 0)) for p in pk)
    return {"times": times, "count": len(pk), "bps": int(total * 8 / span) if span > 0 else 0, "start": times[0], "span": span}


def audio_hashes(path):
    """Ses paketlerinin MD5 listesi (yeniden kodlamadan, paket verisi)."""
    out = run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy", "-f", "framemd5", "-"]).stdout
    # framemd5 sütunları: stream, dts, pts, duration, size, hash[, yan veri...] — hash 6. sütundur.
    return [line.split(",")[5].strip() for line in out.splitlines() if line and not line.startswith("#")]


def colr_box(path):
    """MP4 'colr' (nclx) kutusu: (primaries, transfer, matrix, full_range) veya None."""
    with open(path, "rb") as fh:
        data = fh.read()
    i = data.find(b"colrnclx")
    if i < 0:
        return None
    p, t, m = struct.unpack(">HHH", data[i + 8:i + 14])
    full = (data[i + 14] >> 7) & 1
    return {"primaries": p, "transfer": t, "matrix": m, "full": full}


def vui(path):
    """H.264/HEVC bit akışındaki VUI renk alanları (trace_headers ile)."""
    err = run(["ffmpeg", "-v", "trace", "-i", path, "-map", "0:v:0", "-frames:v", "1", "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"]).stderr
    keys = {"video_signal_type_present_flag": "signal", "video_full_range_flag": "full", "colour_description_present_flag": "desc",
            "colour_primaries": "primaries", "transfer_characteristics": "transfer", "matrix_coefficients": "matrix", "matrix_coeffs": "matrix"}
    found = {}
    for line in err.splitlines():
        m = re.search(r"\b(" + "|".join(keys) + r")\b.*=\s*(\d+)\s*$", line)
        if m and keys[m.group(1)] not in found:
            found[keys[m.group(1)]] = int(m.group(2))
    return found or None


def effective_color(colr, v, width, height):
    """Oynatıcı kuralı: bit akışı VUI > MP4 colr > boyuta göre varsayılan (mpv: w ≥ 1280 veya h > 576 → BT.709)."""
    default = 1 if (width >= 1280 or height > 576) else 6
    if v and v.get("desc"):
        matrix, src = v.get("matrix", 2), "VUI"
    elif colr:
        matrix, src = colr["matrix"], "colr"
    else:
        matrix, src = 2, "varsayılan"
    if matrix == 2:
        matrix = default
    if v and v.get("signal"):
        full = v.get("full", 0)
    elif colr:
        full = colr["full"]
    else:
        full = 0
    return {"matrix": matrix, "full": full, "from": src}


def render_psnr(src, out, src_col, out_col):
    """Kaynağı ve çıktıyı kendi etkin renk matrisleriyle RGB'ye çevirip PSNR (oynatımda görülen renk farkı)."""
    def chain(col):
        name = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020"}.get(col["matrix"])
        if name is None:
            return None
        return f"scale=in_color_matrix={name}:in_range={'pc' if col['full'] else 'tv'},format=gbrp"
    a, b = chain(out_col), chain(src_col)
    if a is None or b is None:
        return None
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", out, "-i", src, "-lavfi",
               f"[0:v]setpts=PTS-STARTPTS,{a}[o];[1:v]setpts=PTS-STARTPTS,{b}[s];[o][s]psnr", "-f", "null", "-"]).stderr
    m = re.search(r"PSNR .*average:([\d.]+|inf)", err)
    return float(m.group(1)) if m else None


def psnr_ssim(ref, dist):
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", dist, "-i", ref, "-lavfi",
               "[0:v]setpts=PTS-STARTPTS,split[d1][d2];[1:v]setpts=PTS-STARTPTS,split[r1][r2];[d1][r1]psnr;[d2][r2]ssim", "-f", "null", "-"]).stderr
    p = re.search(r"PSNR y:([\d.]+|inf)", err)
    s = re.search(r"SSIM .*All:([\d.]+)", err)
    return (float(p.group(1)) if p else float("nan")), (float(s.group(1)) if s else float("nan"))


def percentile(values, q):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def vmaf_scores(tool, ref, dist, threads):
    if not tool:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        r, d, js = os.path.join(tmp, "r.y4m"), os.path.join(tmp, "d.y4m"), os.path.join(tmp, "v.json")
        for src, dst in ((ref, r), (dist, d)):
            subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1", dst, "-y"], check=True)
        cmd = [tool, "-r", r, "-d", d, "--threads", str(threads), "--json", "-o", js, "-q"]
        for name, version in VMAF_MODELS:
            cmd += ["--model", f"version={version}:name={name}"]
        proc = run(cmd)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return {}
        with open(js) as fh:
            j = json.load(fh)
        out = {}
        for name, _ in VMAF_MODELS:
            frames = [fr["metrics"][name] for fr in j["frames"] if name in fr["metrics"]]
            if frames:
                out[name] = {"mean": sum(frames) / len(frames), "min": min(frames), "p5": percentile(frames, 0.05), "p10": percentile(frames, 0.10)}
        return out


def rotation(stream):
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(sd["rotation"]) % 360
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logcat", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--vmaf", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    vmaf_version = run([args.vmaf, "--version"]).stdout.strip() if args.vmaf else "-"
    log = parse_log(args.logcat)
    rows = []
    for r in log["RESULT"]:
        clip, cand = r["clip"], r["cand"]
        if r.get("status") != "OK":
            rows.append({"clip": clip, "cand": cand, "status": r.get("status"), "problems": r.get("error", "") or r.get("reason", "")})
            continue
        src = next((os.path.join(args.sources, x) for x in sorted(os.listdir(args.sources)) if x.rsplit(".", 1)[0] == clip), None)
        out = os.path.join(args.outputs, f"{clip}__{cand}.mp4")
        if not src or not os.path.exists(out):
            rows.append({"clip": clip, "cand": cand, "status": "MISSING"})
            continue
        sp, op = probe(src), probe(out)
        sv = next(s for s in sp["streams"] if s["codec_type"] == "video")
        ov = next(s for s in op["streams"] if s["codec_type"] == "video")
        sa = next((s for s in sp["streams"] if s["codec_type"] == "audio"), None)
        oa = next((s for s in op["streams"] if s["codec_type"] == "audio"), None)
        svs, ovs = stream_stats(src, "v:0"), stream_stats(out, "v:0")
        sas, oas = (stream_stats(src, "a:0") if sa else None), (stream_stats(out, "a:0") if oa else None)
        problems = []

        # Kare kare PTS
        st0, ot0 = svs["times"][0], ovs["times"][0]
        diffs = [abs((o - ot0) - (s - st0)) for s, o in zip(svs["times"], ovs["times"])]
        pts_max_us = int(max(diffs) * 1e6) if diffs else 0
        if svs["count"] != ovs["count"]:
            problems.append(f"kare {svs['count']}→{ovs['count']}")
        if pts_max_us > 1000:
            problems.append(f"PTS sapması {pts_max_us}µs")
        # Boyut, SAR, DAR, yön
        sw, sh, ow, oh = sv["width"], sv["height"], ov["width"], ov["height"]
        if (sw & ~1, sh & ~1) != (ow, oh):
            problems.append(f"çözünürlük {sw}x{sh}→{ow}x{oh}")
        ssar, osar = sv.get("sample_aspect_ratio", "1:1"), ov.get("sample_aspect_ratio", "1:1")
        ssar = "1:1" if ssar in ("0:1", "N/A") else ssar
        osar = "1:1" if osar in ("0:1", "N/A") else osar
        if ssar != osar:
            problems.append(f"SAR {ssar}→{osar}")
        if rotation(sv) != rotation(ov):
            problems.append(f"yön {rotation(sv)}→{rotation(ov)}")
        # Ses
        audio_mode = "-"
        packets_same = "-"
        av_src = av_out = dur_src = dur_out = None
        if sa:
            if not oa:
                problems.append("ses yok")
            else:
                copied = "kopyalandı" in r.get("audio", "")
                audio_mode = "kopya" if copied else "yeniden kodlandı"
                same = audio_hashes(src) == audio_hashes(out) if sa["codec_name"] == oa["codec_name"] else False
                packets_same = "birebir" if same else "farklı"
                if copied and not same:
                    problems.append("ses 'kopyalandı' ama paketler farklı")
                if int(sa.get("sample_rate", 0)) != int(oa.get("sample_rate", 0)):
                    problems.append(f"örnekleme {sa.get('sample_rate')}→{oa.get('sample_rate')}")
                if sa.get("channels") != oa.get("channels"):
                    problems.append(f"kanal {sa.get('channels')}→{oa.get('channels')}")
                av_src = sas["start"] - st0
                av_out = oas["start"] - ot0
                dur_src, dur_out = sas["span"], oas["span"]
                # Senkron: sesin görüntüye göre başlangıç ofseti kaynakla aynı olmalı (≤ 1 ms; MP4 zaman tabanı yuvarlaması).
                if abs(av_out - av_src) > (0.001 if copied else 0.025):
                    problems.append(f"A/V ofseti {av_src * 1000:.1f}→{av_out * 1000:.1f} ms")
                if abs(dur_out - dur_src) > 0.05:
                    problems.append(f"ses süresi {dur_src:.3f}→{dur_out:.3f} sn")
        # Renk
        s_colr, o_colr = colr_box(src) if src.endswith(".mp4") else None, colr_box(out)
        s_vui, o_vui = vui(src), vui(out)
        if not src.endswith(".mp4"):
            # VP9/WebM: renk bilgisi ffprobe'un okuduğu akış alanlarından
            mx = {"bt709": 1, "smpte170m": 6, "bt470bg": 5, "bt2020nc": 9}.get(sv.get("color_space", ""), 2)
            s_vui = {"desc": 1 if mx != 2 else 0, "matrix": mx, "signal": 1, "full": 1 if sv.get("color_range") == "pc" else 0}
        s_eff = effective_color(s_colr, s_vui, sw, sh)
        o_eff = effective_color(o_colr, o_vui, ow, oh)
        o_eff_colr = effective_color(o_colr, None, ow, oh)
        color_ok = (s_eff["matrix"] == o_eff["matrix"] or {s_eff["matrix"], o_eff["matrix"]} <= {5, 6}) and s_eff["full"] == o_eff["full"]
        conflict = bool(o_colr and o_vui and o_vui.get("desc") and o_vui.get("matrix") != o_colr["matrix"])
        if not color_ok:
            problems.append(f"renk matrisi {MATRIX.get(s_eff['matrix'], s_eff['matrix'])}→{MATRIX.get(o_eff['matrix'], o_eff['matrix'])} ({o_eff['from']})")
        rgb_eff = render_psnr(src, out, s_eff, o_eff)
        rgb_ideal = render_psnr(src, out, s_eff, s_eff)
        # Kalite
        psnr, ssim = psnr_ssim(src, out)
        vm = vmaf_scores(args.vmaf, src, out, args.threads)
        dev = log["CHECK"].get((clip, cand), {})
        row = {
            "clip": clip, "cand": cand, "status": "PASS" if not problems else "FAIL", "device_check": dev.get("status", "-"),
            "encoder": r.get("encoder"), "out_codec": ov["codec_name"], "mode": r.get("mode"),
            "requested_kbps": int(r.get("targetBps", 0)) // 1000, "out_vkbps": ovs["bps"] // 1000, "src_vkbps": svs["bps"] // 1000,
            "overshoot": round(ovs["bps"] / max(1, int(r.get("targetBps", 1))), 2),
            "src_bytes": int(sp["format"]["size"]), "out_bytes": int(op["format"]["size"]),
            "size_ratio": round(int(op["format"]["size"]) / int(sp["format"]["size"]), 3),
            "src_akbps": (sas["bps"] // 1000) if sas else 0, "out_akbps": (oas["bps"] // 1000) if oas else 0,
            "src_frames": svs["count"], "out_frames": ovs["count"], "pts_max_us": pts_max_us,
            "src_fps": sv.get("avg_frame_rate"), "out_fps": ov.get("avg_frame_rate"),
            "size": f"{sw}x{sh}→{ow}x{oh}", "sar": f"{ssar}→{osar}", "dar": f"{sv.get('display_aspect_ratio', '-')}→{ov.get('display_aspect_ratio', '-')}",
            "rot": f"{rotation(sv)}→{rotation(ov)}",
            "audio": f"{sa['codec_name']}/{sa.get('sample_rate')}/{sa.get('channels')}ch→{oa['codec_name']}/{oa.get('sample_rate')}/{oa.get('channels')}ch" if sa and oa else ("-" if not sa else "kayıp"),
            "audio_mode": audio_mode, "audio_packets": packets_same,
            "av_offset_ms": f"{av_src * 1000:.1f}→{av_out * 1000:.1f}" if av_src is not None else "-",
            "audio_dur_s": f"{dur_src:.3f}→{dur_out:.3f}" if dur_src is not None else "-",
            "colr_out": f"{o_colr['primaries']}/{o_colr['transfer']}/{o_colr['matrix']}/{'full' if o_colr['full'] else 'tv'}" if o_colr else "yok",
            "vui_out": (f"{o_vui.get('primaries', '-')}/{o_vui.get('transfer', '-')}/{o_vui.get('matrix', '-')}/{'full' if o_vui.get('full') else 'tv'}" if o_vui and o_vui.get("desc") else "renk yok") if o_vui else "-",
            "color_src": f"{MATRIX.get(s_eff['matrix'], s_eff['matrix'])}/{'full' if s_eff['full'] else 'tv'} ({s_eff['from']})",
            "color_out": f"{MATRIX.get(o_eff['matrix'], o_eff['matrix'])}/{'full' if o_eff['full'] else 'tv'} ({o_eff['from']})",
            "color_out_colr_only": f"{MATRIX.get(o_eff_colr['matrix'], o_eff_colr['matrix'])}",
            "colr_vui_conflict": "evet" if conflict else "hayır",
            "rgb_psnr_playback": round(rgb_eff, 2) if rgb_eff else "-", "rgb_psnr_ideal": round(rgb_ideal, 2) if rgb_ideal else "-",
            "psnr_y": round(psnr, 2), "ssim": round(ssim, 4),
            "encode_ms": int(r.get("ms", 0)), "problems": "; ".join(problems),
        }
        for name, _ in VMAF_MODELS:
            for k in ("mean", "min", "p5", "p10"):
                row[f"{name}_{k}"] = round(vm[name][k], 2) if name in vm else "-"
        rows.append(row)
        print(f"{clip:28s} {cand:16s} {row['status']:4s} {row['out_vkbps']:>6} kb/s (istenen {row['requested_kbps']})  VMAF {row.get('vmaf_mean')}  {row['problems']}", flush=True)

    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") in ("PASS", "FAIL")]
    with open(args.markdown, "w") as fh:
        fh.write(f"## Doğrulama: {args.label}\n\n")
        fh.write(f"VMAF: libvmaf `{vmaf_version}`, modeller {', '.join(v for _, v in VMAF_MODELS)}. Kalite kaynağa göredir; yalnızca yeniden kodlama kaybını ölçer.\n\n")
        enc = sorted({r['encoder'] for r in ok if r.get('encoder')})
        fh.write(f"Kodlayıcı(lar): {', '.join(f'`{e}`' for e in enc) or '-'}\n\n")
        fh.write(f"**Denetim:** {sum(r['status'] == 'PASS' for r in ok)}/{len(ok)} PASS\n\n")
        fh.write("| klip | profil | kodlayıcı | istenen | gerçek | aşım | boyut | VMAF ort/min/p5/p10 | VMAF-NEG ort | PSNR-Y | SSIM | kare | PTS maks µs | ses | renk (çıktı) | durum |\n")
        fh.write("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---|\n")
        for r in ok:
            fh.write(
                f"| {r['clip']} | {r['cand']} | {r['encoder']} | {r['requested_kbps']} | {r['out_vkbps']} | {r['overshoot']}× | {r['size_ratio']}× | "
                f"{r.get('vmaf_mean')}/{r.get('vmaf_min')}/{r.get('vmaf_p5')}/{r.get('vmaf_p10')} | {r.get('vmaf_neg_mean')} | {r['psnr_y']} | {r['ssim']} | "
                f"{r['src_frames']}→{r['out_frames']} | {r['pts_max_us']} | {r['audio_mode']}, {r['audio_packets']} | {r['color_out']} | {r['status']} {r['problems']} |\n"
            )
        bad = [r for r in rows if r.get("status") not in ("PASS", "FAIL")]
        if bad:
            fh.write("\n### Kodlanamayanlar\n\n" + "".join(f"- {r['clip']} / {r['cand']}: {r['status']} {r.get('problems', '')}\n" for r in bad))
        if log["AUDIOPROBE"]:
            fh.write("\n### Sesin MP4'e yeniden kodlanmadan taşınması\n\n")
            for p in log["AUDIOPROBE"]:
                fh.write(f"- API {p.get('api')} · {p.get('mime')} ({p.get('source')}): {p.get('result')}\n")


if __name__ == "__main__":
    main()
