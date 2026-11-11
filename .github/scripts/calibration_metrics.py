#!/usr/bin/env python3
"""Video Cleaner bit hızı kalibrasyonu: cihazda kodlanan çıktıları kaynaklarıyla karşılaştırır.

Girdi:
  --logcat   VCCAL satırlarını içeren logcat dökümü (EncoderCalibrationTest)
  --outputs  cihazdan çekilen çıktı klasörü (<klip>__<aday>.mp4)
  --sources  kalibrasyon klipleri (AndroidTest/assets/calib)
  --vmaf     libvmaf'ın `vmaf` aracı (yoksa VMAF atlanır ve raporda "-" yazar)
Çıktı:
  --csv      her kodlama için bir satır
  --markdown özet tablo (GitHub iş özeti)

Ölçülenler: dosya boyutu, video/ses bit hızı (paket boyutlarından), PSNR-Y, SSIM (All), VMAF (v0.6.1),
kare sayısı, süre, FPS, çözünürlük, yön ve renk etiketleri; kaynakla birebir karşılaştırma.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

LINE = re.compile(r"VCCAL\s*:\s*(RESULT|SOURCE)\s+(.*)$")
FIELD = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def parse_log(path):
    results, sources = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            fields = {k: (q if q is not None and v.startswith('"') else v) for k, v, q in FIELD.findall(m.group(2))}
            if m.group(1) == "SOURCE":
                sources[fields["clip"]] = fields
            else:
                results.append(fields)
    return results, sources


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-count_packets", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def stream_bitrate(path, kind):
    """Akışın gerçek bit hızı: paket boyutlarının toplamı / süre (kapsayıcı beyanına güvenmez)."""
    sel = "v:0" if kind == "video" else "a:0"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "packet=size,pts_time,duration_time", "-of", "json", path],
        capture_output=True, text=True,
    ).stdout
    packets = json.loads(out or "{}").get("packets", [])
    if not packets:
        return 0, 0.0
    total = sum(int(p.get("size", 0)) for p in packets)
    times = [float(p["pts_time"]) for p in packets if p.get("pts_time") not in (None, "N/A")]
    durs = [float(p["duration_time"]) for p in packets if p.get("duration_time") not in (None, "N/A")]
    span = (max(times) - min(times) + (durs[-1] if durs else 0.0)) if times else 0.0
    return (int(total * 8 / span) if span > 0 else 0), span


def video_info(path):
    j = ffprobe(path)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    return {
        "codec": v["codec_name"], "width": v["width"], "height": v["height"],
        "fps": round(num / den, 3) if den else 0, "frames": int(v.get("nb_read_packets", 0)),
        "duration": float(j["format"].get("duration", 0)), "rotation": rotation,
        "color": "/".join(str(v.get(k, "?")) for k in ("color_range", "color_primaries", "color_transfer", "color_space")),
        "audio": a["codec_name"] if a else "-", "audio_frames": int(a.get("nb_read_packets", 0)) if a else 0,
        "bytes": int(j["format"]["size"]),
    }


def psnr_ssim(ref, dist):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", dist, "-i", ref, "-lavfi", "[0:v]setpts=N[d];[1:v]setpts=N[r];[d][r]psnr;[0:v]setpts=N[d2];[1:v]setpts=N[r2];[d2][r2]ssim", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    psnr = re.search(r"PSNR y:([\d.]+|inf)", proc.stderr)
    ssim = re.search(r"SSIM .*All:([\d.]+)", proc.stderr)
    return (float(psnr.group(1)) if psnr else float("nan")), (float(ssim.group(1)) if ssim else float("nan"))


def vmaf_score(tool, ref, dist, threads):
    if not tool:
        return float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        r, d, js = os.path.join(tmp, "r.y4m"), os.path.join(tmp, "d.y4m"), os.path.join(tmp, "v.json")
        for src, dst in ((ref, r), (dist, d)):
            subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "-strict", "-1", dst, "-y"], check=True)
        proc = subprocess.run([tool, "-r", r, "-d", d, "--threads", str(threads), "--model", "version=vmaf_v0.6.1", "--json", "-o", js, "-q"], capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return float("nan")
        with open(js) as f:
            return float(json.load(f)["pooled_metrics"]["vmaf"]["mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logcat", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--vmaf", default="")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    results, sources = parse_log(args.logcat)
    rows, src_cache = [], {}
    for r in results:
        if r.get("status") != "OK":
            rows.append({"clip": r["clip"], "cand": r["cand"], "status": r.get("status"), "error": r.get("error", "")})
            continue
        clip, cand = r["clip"], r["cand"]
        src = os.path.join(args.sources, clip + ".mp4")
        out = os.path.join(args.outputs, f"{clip}__{cand}.mp4")
        if not os.path.exists(out):
            rows.append({"clip": clip, "cand": cand, "status": "MISSING"})
            continue
        if clip not in src_cache:
            info = video_info(src)
            info["vbps"], _ = stream_bitrate(src, "video")
            info["abps"], _ = stream_bitrate(src, "audio")
            src_cache[clip] = info
        s = src_cache[clip]
        o = video_info(out)
        vbps, _ = stream_bitrate(out, "video")
        abps, _ = stream_bitrate(out, "audio")
        psnr, ssim = psnr_ssim(src, out)
        vmaf = vmaf_score(args.vmaf, src, out, args.threads)
        rows.append({
            "clip": clip, "cand": cand, "status": "OK", "encoder": r.get("encoder"), "mode": r.get("mode"), "cq": r.get("cq"),
            "target_kbps": int(r.get("targetBps", 0)) // 1000, "gop_s": r.get("gopSec"), "encode_ms": int(r.get("ms", 0)),
            "src_bytes": s["bytes"], "out_bytes": o["bytes"], "size_ratio": round(o["bytes"] / s["bytes"], 3),
            "src_vkbps": s["vbps"] // 1000, "out_vkbps": vbps // 1000, "vbitrate_ratio": round(vbps / s["vbps"], 3) if s["vbps"] else 0,
            "src_akbps": s["abps"] // 1000, "out_akbps": abps // 1000,
            "psnr_y": round(psnr, 2), "ssim": round(ssim, 4), "vmaf": round(vmaf, 2),
            "src_frames": s["frames"], "out_frames": o["frames"], "src_fps": s["fps"], "out_fps": o["fps"],
            "src_dur": round(s["duration"], 3), "out_dur": round(o["duration"], 3),
            "src_size": f'{s["width"]}x{s["height"]}', "out_size": f'{o["width"]}x{o["height"]}',
            "src_rot": s["rotation"], "out_rot": o["rotation"], "src_color": s["color"], "out_color": o["color"],
            "src_audio_frames": s["audio_frames"], "out_audio_frames": o["audio_frames"], "out_codec": o["codec"],
        })
        print(f"{clip:28s} {cand:18s} {rows[-1]['out_vkbps']:>6} kb/s  PSNR {psnr:6.2f}  SSIM {ssim:.4f}  VMAF {vmaf:6.2f}", flush=True)

    keys = sorted({k for row in rows for k in row}, key=lambda k: list(rows[0].keys()).index(k) if rows and k in rows[0] else 999)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "OK"]
    with open(args.markdown, "w") as f:
        f.write("## Bit hızı kalibrasyonu\n\n")
        f.write("| klip | aday | kodlayıcı | mod | hedef kb/s | çıktı video kb/s | kaynak video kb/s | boyut oranı | PSNR-Y | SSIM | VMAF | kare | FPS | ms |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|\n")
        for r in ok:
            f.write(
                f"| {r['clip']} | {r['cand']} | {r['encoder']} | {r['mode']}{'/'+r['cq'] if r['cq'] not in (None, '-1') else ''} | {r['target_kbps']} | "
                f"{r['out_vkbps']} | {r['src_vkbps']} | {r['size_ratio']} | {r['psnr_y']} | {r['ssim']} | {r['vmaf']} | "
                f"{r['src_frames']}→{r['out_frames']} | {r['src_fps']}→{r['out_fps']} | {r['encode_ms']} |\n"
            )
        bad = [r for r in rows if r.get("status") != "OK"]
        if bad:
            f.write("\n### Çalışmayan adaylar\n\n")
            for r in bad:
                f.write(f"- {r['clip']} / {r['cand']}: {r.get('status')} {r.get('error', '')}\n")
        mismatch = [r for r in ok if r["src_frames"] != r["out_frames"] or r["src_size"] != r["out_size"] or abs(r["src_fps"] - r["out_fps"]) > 0.01 or r["src_rot"] != r["out_rot"]]
        f.write(f"\n**Korunan özellikler:** {len(ok) - len(mismatch)}/{len(ok)} çıktıda kare sayısı, çözünürlük, FPS ve yön kaynakla aynı.\n")
        for r in mismatch:
            f.write(f"- UYUMSUZ {r['clip']} / {r['cand']}: kare {r['src_frames']}→{r['out_frames']}, boyut {r['src_size']}→{r['out_size']}, FPS {r['src_fps']}→{r['out_fps']}, yön {r['src_rot']}→{r['out_rot']}\n")


if __name__ == "__main__":
    main()
