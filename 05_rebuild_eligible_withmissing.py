#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct the eligible-with-missing Shanghai cohort (manuscript funnel step 28,420)
from shanghai_raw.csv, then validate against the existing complete-case files:
  - complete-case subset must equal shanghai_analysis_allcycles.csv (20,248)
  - first-cycle-per-patient must equal shanghai_analysis.csv (11,749)
Outputs: shanghai_eligible_withmissing.csv (28,420 rows, missing retained)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE = "."

raw = pd.read_csv(f"{BASE}/shanghai_raw.csv", dtype=str)
print("raw:", len(raw))

def xd(s):
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(s))
    except Exception:
        return None

def num(s):
    try:
        return float(s)
    except Exception:
        return None

# funnel step 1: fresh
fresh = raw[raw["cycle_type"] == "新鲜周期"].copy()
print("fresh:", len(fresh))

# step 2: window
fresh["dt"] = fresh["stat_date"].map(xd)
win = fresh[(fresh["dt"] >= datetime(2021, 1, 1)) & (fresh["dt"] <= datetime(2026, 10, 31))].copy()
print("in window:", len(win))

# step 3: IVF or ICSI (no IUI)
proto = (win["planned_protocol"].fillna("") + "|" + win["final_protocol"].fillna("")).str.upper()
art = win[proto.str.contains("IVF|ICSI") & ~proto.str.contains("人工授精|IUI")].copy()
print("IVF/ICSI:", len(art))

# step 4: stimulated -- exclude natural-cycle and HRT preparations (per stim_regimen class)
NAT = ("自然周期", "改良自然周期方案", "自然周期方案", "HRT周期")
stim = art[~art["stim_regimen"].isin(NAT)].copy()
print("stimulated (eligible, missing retained):", len(stim))

# outcome: oocytes known, or cancelled-no-retrieval flagged (definitive logic)
oc = stim["oocytes_total"].map(num)
canc = stim["cancellation"].fillna("").str.strip()
cancelled = (~canc.isin(["", "无", "0"])).astype(int)
por = np.where(oc.notna(), (oc <= 3).astype(float),
               ((canc.str.contains("卵")) & (canc.str.contains("未"))).astype(float))
has_outcome = oc.notna() | (cancelled == 1)

elig = pd.DataFrame({
    "patient_id": stim["patient_id"], "date": stim["dt"].dt.strftime("%Y-%m-%d"),
    "age": stim["age"].map(num), "bmi": stim["bmi"].map(num),
    "amh": stim["amh"].map(num), "base_fsh": stim["base_fsh"].map(num),
    "oocytes": oc, "cancelled": cancelled, "por": por,
    "stim_regimen": stim["stim_regimen"],
})
elig = elig[has_outcome.values].reset_index(drop=True)
print("eligible with outcome (manuscript 28,420 target):", len(elig))

# ---- validation vs complete-case files ----
def plausible(v, lo, hi):
    return v.notna() & (v >= lo) & (v <= hi)

cc_mask = (plausible(elig["age"], 18, 50) & plausible(elig["bmi"], 14, 50)
           & plausible(elig["amh"], 0.01, 25) & plausible(elig["base_fsh"], 0.1, 40))
cc = elig[cc_mask].reset_index(drop=True)
print("complete-case within eligible:", len(cc), "(target 20,248)")

allc = pd.read_csv(f"{BASE}/shanghai_analysis_allcycles.csv", encoding="utf-8-sig")
key_cc = set(zip(cc["patient_id"].astype(str), cc["date"]))
key_ref = set(zip(allc["patient_id"].astype(str), allc["date"].str[:10]))
print("allcycles n:", len(allc), "| key match:", len(key_cc & key_ref),
      "| only_recon:", len(key_cc - key_ref), "| only_ref:", len(key_ref - key_cc))

# numeric equality spot check on matched keys
m = allc.copy()
m["pid_s"] = m["patient_id"].astype(str); m["date_s"] = m["date"].str[:10]
cc2 = cc.copy(); cc2["pid_s"] = cc2["patient_id"].astype(str)
mg = m.merge(cc2, left_on=["pid_s", "date_s"], right_on=["pid_s", "date"], suffixes=("_ref", "_rec"))
print("merged rows:", len(mg))
for col in ["age", "bmi", "amh", "base_fsh", "oocytes", "por"]:
    a, b = mg[f"{col}_ref"], mg[f"{col}_rec"]
    eq = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
    print(f"  {col}: {eq.mean():.4f} equal")

# ---- first cycle per patient ----
first = elig.sort_values(["patient_id", "date"], kind="stable").drop_duplicates("patient_id", keep="first")
print("first eligible cycle per patient:", len(first))
prim = pd.read_csv(f"{BASE}/shanghai_analysis.csv", encoding="utf-8-sig")
kp = set(zip(first["patient_id"].astype(str), first["date"]))
kr = set(zip(prim["patient_id"].astype(str), prim["date"].str[:10]))
print("primary n:", len(prim), "| key match:", len(kp & kr), "| only_recon:", len(kp - kr), "| only_ref:", len(kr - kp))

# missingness in eligible first-cycle set
f_miss = first[["age", "bmi", "amh", "base_fsh"]].isna().mean()
print("\nmissingness (first eligible per patient):", f_miss.round(3).to_dict())
print("POR rate (first eligible):", first["por"].mean().round(4))

first.to_csv(f"{BASE}/shanghai_eligible_first_withmissing.csv", index=False, encoding="utf-8-sig")
elig.to_csv(f"{BASE}/shanghai_eligible_withmissing.csv", index=False, encoding="utf-8-sig")
print("\nsaved: shanghai_eligible_withmissing.csv / shanghai_eligible_first_withmissing.csv")
