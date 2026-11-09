#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream-extract Shanghai derivation cycles from 2019-2025.xlsx and 2025-2026-10.16.xlsx
(single header row, 216 cols). Columns selected BY HEADER NAME (robust to reordering).

NOTE: neither export contains an AMH column (verified by full header scan).
      AMH must be merged from the lab database by 病历号/统计日期 before the
      definitive 4-feature model can be trained. This script therefore emits
      the complete analysis frame with an empty amh column.

Output: shanghai_cycles.csv  (one row per cycle record)
"""
import zipfile, re, csv, time

FILES = ["<raw EHR export 2019-2025>",
         "<raw EHR export 2025-2026>"]
OUT = "./shanghai_cycles.csv"

WANT = ["病历号", "统计日期", "默认基本信息|女方年龄", "周期信息|本院第几周期",
        "周期类型", "周期信息|拟行助孕方案", "周期信息|最终助孕方案",
        "女方病历|体重指数", "女方病历|基础FSH", "取卵记录|卵子总数",
        "周期信息|用药方案", "周期信息|取消类型"]
COL_ALIASES = {  # 2025-2026 file uses 默认基本信息| prefix on 病历号
    "默认基本信息|病历号": "病历号",
}

def col_to_idx(c):
    n = 0
    for ch in c:
        n = n * 26 + ord(ch) - 64
    return n

def extract(path):
    z = zipfile.ZipFile(path)
    ss = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
    strings = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", s))
               for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    sheet = z.open("xl/worksheets/sheet1.xml")
    buf = b""
    header = None
    hmap = {}          # col idx -> canonical name
    n = 0
    carry = b""
    row_re = re.compile(rb"<row[^>]*>.*?</row>", re.S)
    cell_re = re.compile(
        r'<c r="([A-Z]+)(\d+)"([^>]*)>(?:<is><t[^>]*>([^<]*)</t></is>|<v>([^<]*)</v>)')
    def flush(chunk):
        nonlocal header, hmap, n
        for rm in row_re.finditer(chunk):
            x = rm.group(0).decode("utf-8", "ignore")
            rowno = int(re.search(r'<row[^>]* r="(\d+)"', x).group(1))
            vals = {}
            for m in cell_re.finditer(x):
                col, attrs, a, b = m.group(1), m.group(3), m.group(4), m.group(5)
                tm = re.search(r't="(\w+)"', attrs)
                v = strings[int(b)] if (tm and tm.group(1) == "s" and b and int(b) < len(strings)) else (a if a else b)
                vals[col_to_idx(col)] = v
            if rowno == 1:
                header = vals
                name2idx = {}
                for ci, nm in vals.items():
                    canon = COL_ALIASES.get(nm, nm)
                    if canon in WANT and canon not in name2idx:
                        name2idx[canon] = ci
                hmap = name2idx
                missing = [w for w in WANT if w not in hmap]
                print(f"  header mapped {len(hmap)}/{len(WANT)} cols; missing: {missing}")
            elif rowno >= 2 and hmap:
                yield_row = [vals.get(hmap[w], "") for w in WANT]
                yield_row  # noqa
                rows_out.append([str(v) for v in yield_row])
                n += 1
    rows_out = []
    t0 = time.time()
    while True:
        chunk = sheet.read(1 << 22)
        if not chunk:
            break
        buf = carry + chunk
        cut = buf.rfind(b"</row>")
        if cut == -1:
            carry = buf
            continue
        flush(buf[:cut + 6])
        carry = buf[cut + 6:]
    if carry:
        flush(carry)
    print(f"  rows extracted: {n} [{time.time()-t0:.1f}s]")
    return rows_out

def main():
    all_rows = []
    for p in FILES:
        print(f"== {p.split('/')[-1]}")
        all_rows.extend(extract(p))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "stat_date", "age", "cycle_no_in_hospital", "cycle_type",
                    "planned_protocol", "final_protocol", "bmi", "base_FSH",
                    "oocytes_total", "stim_regimen", "cancellation_type"])
        w.writerows(all_rows)
    print("TOTAL rows:", len(all_rows), "->", OUT)

if __name__ == "__main__":
    main()
