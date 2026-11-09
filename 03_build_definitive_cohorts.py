#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the definitive Shanghai derivation cohort from shanghai_raw.csv
(RAW data 0.xlsx, 103,899 cycles, includes AMH) + refresh external cohort.

Manuscript criteria: first stimulated fresh IVF/ICSI cycle, age 18-50,
complete age/BMI/AMH/basal FSH, Jan 2021 - Oct 2026 (standardized AMH era).
Outcome: Bologna POR = oocytes <=3 (retrieval records only in this frame;
cancellation handling is a sensitivity flag -> column 'cancelled_no_retrieval').

Outputs: shanghai_analysis.csv, cohort_summary.txt (appended funnel)
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

def plausible(v, lo, hi):
    return v is not None and lo <= v <= hi

def build_shanghai():
    rows = list(csv.DictReader(open(f"{BASE}/shanghai_raw.csv", encoding="utf-8-sig")))
    funnel = Counter()
    kept = []
    for r in rows:
        age = num(r["age"]); amh = num(r["amh"]); fsh = num(r["base_fsh"])
        bmi = num(r["bmi"]); oc = num(r["oocytes_total"]); no = num(r["cycle_no"])
        d = excel_date(r["stat_date"])
        proto = f"{r['planned_protocol']}|{r['final_protocol']}"
        if r["cycle_type"] != "新鲜周期":
            funnel["1 not fresh cycle"] += 1; continue
        if no != 1:
            funnel["2 not first cycle in hospital"] += 1; continue
        if "人工授精" in proto or "IUI" in proto.upper():
            funnel["3 IUI"] += 1; continue
        if "IVF" not in proto.upper() and "ICSI" not in proto.upper():
            funnel["4 not IVF/ICSI"] += 1; continue
        if d is None or not (datetime(2021, 1, 1) <= d <= datetime(2026, 10, 31)):
            funnel["5 outside 2021.01-2026.10"] += 1; continue
        if not plausible(age, 18, 50):
            funnel["6 age outside 18-50"] += 1; continue
        if not plausible(bmi, 14, 50):
            funnel["7 BMI missing/implausible"] += 1; continue
        if not plausible(amh, 0.01, 25):
            funnel["8 AMH missing/implausible"] += 1; continue
        if not plausible(fsh, 0.1, 40):
            funnel["9 basal FSH missing/implausible"] += 1; continue
        cancelled = (r["cancellation"] or "").strip() not in ("", "无", "0")
        if oc is None and not cancelled:
            funnel["10 no oocyte count and not cancelled"] += 1; continue
        kept.append(dict(patient_id=r["patient_id"], date=d.strftime("%Y-%m-%d"),
                         age=age, bmi=bmi, amh=amh, base_fsh=fsh,
                         oocytes=int(oc) if oc is not None else "",
                         cancelled=1 if (oc is None and cancelled) else 0,
                         por=int(oc is not None and oc <= 3) if oc is not None else
                             int((r["cancellation"] or "").find("卵") >= 0 and "未" in (r["cancellation"] or ""))))
    return kept, funnel

def refresh_external():
    """Rebuild external from the earlier extraction (unchanged, kept for the pipeline)."""
    rows = list(csv.DictReader(open(f"{BASE}/zjy_external_cohort.csv", encoding="utf-8-sig")))
    kept = []
    for r in rows:
        if r["patient_id"][:3] not in ("072", "710", "288", "712") or r["gender"] != "2" \
           or r["cycle_type"] != "2":
            continue
        age = num(r["age"]); h = num(r["height_cm"]); wt = num(r["weight_kg"])
        amh = num(r["base_AMH"]); fsh = num(r["gn_day_FSH"]); oc = num(r["oocytes_retrieved"])
        bmi = wt / (h / 100) ** 2 if h and wt and h > 50 and wt > 25 else None
        if None in (age, amh, fsh, oc) or bmi is None:
            continue
        if not (18 <= age <= 50 and 14 <= bmi <= 50 and 0 < amh < 25 and 0.1 <= fsh <= 40):
            continue
        kept.append(dict(patient_id=r["patient_id"],
                         center="Zunyi" if r["patient_id"][:3] == "072" else "Jining",
                         age=age, bmi=round(bmi, 2), amh=amh, base_fsh=fsh,
                         oocytes=int(oc), por=int(oc <= 3)))
    return kept

def main():
    sh, funnel = build_shanghai()
    cols = ["patient_id", "date", "age", "bmi", "amh", "base_fsh", "oocytes", "cancelled", "por"]
    with open(f"{BASE}/shanghai_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sh:
            w.writerow({k: r[k] for k in cols})
    ext = refresh_external()
    cols2 = ["patient_id", "center", "age", "bmi", "amh", "base_fsh", "oocytes", "por"]
    with open(f"{BASE}/external_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols2); w.writeheader()
        for r in ext:
            w.writerow({k: r[k] for k in cols2})

    n_ret = sum(1 for r in sh if r["cancelled"] == 0)
    lines = ["# Definitive cohort build (RAW data 0.xlsx, 103,899 cycles)", "",
             "## Shanghai derivation funnel (cumulative drops in filter order)"]
    lines += [f"  {k}: {v}" for k, v in sorted(funnel.items())]
    lines += [f"  KEPT: {len(sh)} (retrieval outcomes: {n_ret}, cancelled-no-retrieval: {len(sh)-n_ret})",
              f"  POR events (<=3 oocytes): {sum(r['por'] for r in sh)} "
              f"({sum(r['por'] for r in sh)/max(n_ret,1):.1%} of retrievals)",
              "", "## External (Zunyi-Jining registry, refreshed)"]
    by = defaultdict(lambda: [0, 0])
    for r in ext:
        by[r["center"]][0] += 1
        by[r["center"]][1] += r["por"]
    for c, (n, p) in sorted(by.items()):
        lines.append(f"  {c}: n={n}, POR={p} ({p/n:.1%})")
    lines.append(f"  TOTAL: n={len(ext)}, POR={sum(r['por'] for r in ext)}")
    txt = "\n".join(lines)
    open(f"{BASE}/cohort_summary_definitive.txt", "w", encoding="utf-8").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
