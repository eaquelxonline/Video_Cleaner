#!/usr/bin/env python3
"""Firebase Test Lab fiziksel cihaz seçimi.

Kullanım: ftl_pick_devices.py MODELS_JSON N DEVICES_OUT

`gcloud firebase test android models list --format=json` çıktısından, uygulamanın desteklediği (API ≥ 30)
fiziksel telefonlar arasından farklı üretici ve API seviyelerini kapsayan en fazla N cihaz seçer:
  1) en yüksek API'li Google cihazı, 2) en yüksek API'li Samsung cihazı, 3) API 30'a en yakın cihaz,
  sonra kalan üreticilerden en yüksek API'liler.
DEVICES_OUT dosyasına her satırda bir `--device=model=…,version=…` yazılır; okunur özet ve
tanılama stdout'a gider (iş akışı günlüğünde görünür). Model listesindeki alanlar eksik olabilir;
eksik alanlı kayıtlar atlanır, betik çökmez. Uygun cihaz yoksa çıkış kodu 2'dir.
"""
import json
import sys


def as_versions(raw):
    out = []
    for v in raw or []:
        s = str(v).strip()
        if s.isdigit() and int(s) >= 30:
            out.append(int(s))
    return sorted(set(out))


def main():
    models_path, limit, out_path = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3, sys.argv[3] if len(sys.argv) > 3 else "devices.txt"
    with open(models_path, encoding="utf-8") as fh:
        models = json.load(fh)
    if isinstance(models, dict):        # bazı gcloud sürümleri {"models": [...]} döndürebilir
        models = models.get("models", [])
    print(f"model kaydı: {len(models)}")

    phones = []
    forms = {}
    for m in models:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        form = str(m.get("form", "")).upper()
        factor = str(m.get("formFactor", "PHONE")).upper()
        forms[form] = forms.get(form, 0) + 1
        if form != "PHYSICAL" or factor not in ("PHONE", ""):
            continue
        tags = [str(t).lower() for t in (m.get("tags") or [])]
        if any("deprecated" in t for t in tags):
            continue
        versions = as_versions(m.get("supportedVersionIds"))
        if versions:
            phones.append({"id": m["id"], "maker": m.get("manufacturer") or "?", "name": m.get("name") or m["id"], "versions": versions})
    print(f"biçim dağılımı: {forms}; API ≥ 30 fiziksel telefon: {len(phones)}")

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
    for d in by_max:                    # hâlâ yer varsa kalan en yeni cihazlar
        take(d, d["versions"][-1])

    with open(out_path, "w", encoding="utf-8") as fh:
        for d, v in picked:
            fh.write(f"--device=model={d['id']},version={v},locale=tr,orientation=portrait\n")
    for d, v in picked:
        print(f"# {d['maker']} {d['name']} ({d['id']}) · API {v}")
    if not picked:
        print("Uygun fiziksel cihaz bulunamadı (API ≥ 30, PHYSICAL, PHONE).")
        sys.exit(2)


if __name__ == "__main__":
    main()
