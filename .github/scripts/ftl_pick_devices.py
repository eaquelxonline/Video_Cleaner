#!/usr/bin/env python3
"""Firebase Test Lab fiziksel cihaz seçimi.

`gcloud firebase test android models list --format=json` çıktısından, uygulamanın desteklediği (API ≥ 30)
fiziksel telefonlar arasından farklı üretici ve API seviyelerini kapsayan en fazla N cihaz seçer:
  1) en yüksek API'li Google cihazı, 2) en yüksek API'li Samsung cihazı, 3) API 30'a en yakın cihaz,
  sonra kalan üreticilerden en yüksek API'liler. Çıktı: her satırda bir `--device model=…,version=…`.
"""
import json
import sys


def main():
    models = json.load(open(sys.argv[1]))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    phones = []
    for m in models:
        if m.get("form") != "PHYSICAL" or m.get("formFactor", "PHONE") != "PHONE":
            continue
        if any("deprecated" in t for t in m.get("tags", [])):
            continue
        versions = sorted(int(v) for v in m.get("supportedVersionIds", []) if str(v).isdigit() and int(v) >= 30)
        if versions:
            phones.append({"id": m["id"], "maker": m.get("manufacturer", "?"), "name": m.get("name", m["id"]), "versions": versions})
    picked, used = [], set()

    def take(device, version):
        key = (device["id"], version)
        if key in used or len(picked) >= limit:
            return
        used.add(key)
        picked.append((device, version))

    by_max = sorted(phones, key=lambda d: d["versions"][-1], reverse=True)
    for maker in ("Google", "Samsung"):
        d = next((x for x in by_max if x["maker"].lower() == maker.lower()), None)
        if d:
            take(d, d["versions"][-1])
    low = sorted(phones, key=lambda d: d["versions"][0])
    if low:
        take(low[0], low[0]["versions"][0])
    makers = {d["maker"] for d, _ in picked}
    for d in by_max:
        if d["maker"] not in makers:
            take(d, d["versions"][-1])
            makers.add(d["maker"])
    for d, v in picked:
        print(f"--device=model={d['id']},version={v},locale=tr,orientation=portrait")
        print(f"# {d['maker']} {d['name']} · API {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
