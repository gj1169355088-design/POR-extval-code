#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build analysis-ready derivation (Shanghai) and prospective external (Zunyi/Jining)
cohorts following the manuscript's inclusion criteria:
  - female 18-50 y, FIRST stimulated fresh IVF/ICSI cycle with intention to retrieve
  - complete baseline age, BMI, basal FSH (+AMH once merged)
  - outcome: Bologna POR = oocytes retrieved <= 3 (incl. cancelled-after->=7d-Gn, flagged)
Cleaning rules (documented for Methods):
  - BMI plausible range 14-50 (questionnaire unit errors flagged -> dropped)
  - basal FSH 0.1-40 IU/L (Jining outlier problem in original manuscript)
  - AMH >0, <25 ng/mL
External basal FSH uses Gn-start-day FSH (cycle day 2-4 equivalent) -- must be
confirmed by the clinical team before manuscript use.
Outputs: shanghai_analysis.csv, external_analysis.csv, cohort_summary.txt
"""
import csv
from datetime import datetime, timedelta
from collections import Counter, defaultdict

BASE = "."

def num(s):
    try:
        v = float(s)
        return v
    except Exception:
        return None

def excel_date(serial):
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(serial))
    except Exception:
        return None

def build_shanghai():
    rows = list(csv.DictReader(open(f"{BASE}/shanghai_cycles.csv", encoding="utf-8-sig")))
    kept, drop_reasons = [], Counter()
    for r in rows:
        age = num(r["age"]); bmi = num(r["bmi"]); fsh = num(r["base_FSH"]); oc = num(r["oocytes_total"])
        cyc = r["cycle_type"]; no = num(r["cycle_no_in_hospital"])
        d = excel_date(r["stat_date"])
        proto = (r["planned_protocol"] or "") + "|" + (r["final_protocol"] or "")
        if cyc != "新鲜周期":
            drop_reasons["not fresh cycle"] += 1; continue
        if no != 1:
            drop_reasons["not first cycle"] += 1; continue
        if "人工授精" in proto or "IUI" in proto.upper():
            drop_reasons["IUI"] += 1; continue
        if "IVF" not in proto.upper() and "ICSI" not in proto.upper():
            drop_reasons["not IVF/ICSI"] += 1; continue
        if d is None or d < datetime(2021, 1, 1):
            drop_reasons["before 2021 (manuscript period)"] += 1; continue
        if age is None or not (18 <= age <= 50):
            drop_reasons["age outside 18-50"] += 1; continue
        if bmi is None or not (14 <= bmi <= 50):
            drop_reasons["BMI missing/implausible"] += 1; continue
        if fsh is None or not (0.1 <= fsh <= 40):
            drop_reasons["FSH missing/outlier"] += 1; continue
        if oc is None:
            drop_reasons["oocyte count missing"] += 1; continue
        kept.append(dict(patient_id=r["patient_id"], date=d.strftime("%Y-%m-%d"),
                         age=age, bmi=bmi, base_fsh=fsh, oocytes=int(oc),
                         por=int(oc <= 3), amh=""))
    return kept, drop_reasons

def build_external():
    rows = list(csv.DictReader(open(f"{BASE}/zjy_external_cohort.csv", encoding="utf-8-sig")))
    kept, drop_reasons = [], Counter()
    for r in rows:
        pid = r["patient_id"]
        if pid[:3] not in ("072", "710", "288", "712"):
            drop_reasons["not ART population (071 disease cohort etc.)"] += 1; continue
        if r["gender"] != "2":
            drop_reasons["male"] += 1; continue
        if r["cycle_type"] != "2":   # first-cycle fresh
            drop_reasons["no first fresh cycle"] += 1; continue
        age = num(r["age"]); h = num(r["height_cm"]); wt = num(r["weight_kg"])
        amh = num(r["base_AMH"]); fsh = num(r["gn_day_FSH"]); oc = num(r["oocytes_retrieved"])
        bmi = wt / (h / 100) ** 2 if h and wt and h > 50 and wt > 25 else None
        if age is None or not (18 <= age <= 50):
            drop_reasons["age missing/outside 18-50"] += 1; continue
        if bmi is None or not (14 <= bmi <= 50):
            drop_reasons["BMI missing/implausible"] += 1; continue
        if amh is None or not (0 < amh < 25):
            drop_reasons["AMH missing/implausible"] += 1; continue
        if fsh is None or not (0.1 <= fsh <= 40):
            drop_reasons["FSH missing/outlier"] += 1; continue
        if oc is None:
            drop_reasons["oocyte count missing"] += 1; continue
        center = "Zunyi" if pid[:3] == "072" else "Jining"
        kept.append(dict(patient_id=pid, center=center, age=age, bmi=round(bmi, 2),
                         base_fsh=fsh, oocytes=int(oc), por=int(oc <= 3), amh=amh,
                         site=r["site_name"]))
    return kept, drop_reasons

def main():
    sh, shd = build_shanghai()
    ext, extd = build_external()
    cols = ["patient_id", "date", "age", "bmi", "base_fsh", "oocytes", "por", "amh"]
    with open(f"{BASE}/shanghai_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sh)
    cols2 = ["patient_id", "center", "site", "age", "bmi", "base_fsh", "amh", "oocytes", "por"]
    with open(f"{BASE}/external_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols2); w.writeheader(); w.writerows(ext)

    lines = ["# Cohort build summary", "", "## Shanghai derivation (first fresh IVF/ICSI, 2021+)",
             f"- kept: {len(sh)} cycles, POR: {sum(r['por'] for r in sh)} "
             f"({sum(r['por'] for r in sh)/max(len(sh),1):.1%})",
             f"- AMH: EMPTY (not present in exports; must merge lab DB)",
             "- exclusions:"]
    lines += [f"  - {k}: {v}" for k, v in shd.most_common()]
    by = defaultdict(lambda: [0, 0])
    for r in ext:
        by[r["center"]][0] += 1
        by[r["center"]][1] += r["por"]
    lines += ["", "## Prospective external (Zunyi-Jining registry, ART women, first fresh cycle)",
              f"- kept: {len(ext)} cycles, POR: {sum(r['por'] for r in ext)} "
              f"({sum(r['por'] for r in ext)/max(len(ext),1):.1%})"]
    for c, (n, p) in by.items():
        lines.append(f"  - {c}: n={n}, POR={p} ({p/n:.1%})")
    lines += ["- exclusions:"]
    lines += [f"  - {k}: {v}" for k, v in extd.most_common()]
    open(f"{BASE}/cohort_summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
