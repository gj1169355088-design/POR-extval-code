#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the definitive Shanghai derivation cycles from 'RAW data 0.xlsx'
(103,899 rows, 184 cols, includes AMH at col 9). Column indices fixed from header.

Output: shanghai_raw.csv (one row per cycle record)
"""
import zipfile, re, csv, time

SRC = "<raw EHR export>"
OUT = "./shanghai_raw.csv"

# 1-based Excel column indices (verified against header)
KEEP = {
    1: "patient_id",       # 病历号
    2: "stat_date",        # 统计日期 (Excel serial)
    4: "age",              # 女方年龄
    9: "amh",              # 'AMH '
    16: "cycle_no",        # 本院第几周期
    19: "cycle_type",      # 周期类型
    20: "planned_protocol",# 拟行助孕方案
    21: "final_protocol",  # 最终助孕方案
    18: "stim_regimen",    # 用药方案 (antagonist/long/mild/PPOS classification)
    22: "cancellation",    # 取消类型
    25: "bmi",             # 体重指数
    32: "base_fsh",        # 基础FSH
    82: "oocytes_total",   # 卵子总数
    86: "mii",             # 成熟卵子
    103: "good_embryos",   # 优良胚胎
    112: "blastocyst_cultured",  # 囊胚|囊胚培养数
    113: "blastocyst_formed",    # 囊胚|囊胚形成数
    183: "p_id",           # P_ID
}
MAXCOL = max(KEEP)

def col_to_idx(c):
    n = 0
    for ch in c:
        n = n * 26 + ord(ch) - 64
    return n

def main():
    t0 = time.time()
    z = zipfile.ZipFile(SRC)
    ss = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
    strings = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", s))
               for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    sheet = z.open("xl/worksheets/sheet1.xml")
    row_re = re.compile(rb"<row[^>]*>.*?</row>", re.S)
    cell_re = re.compile(
        r'<c r="([A-Z]+)(\d+)"([^>]*)>(?:<is><t[^>]*>([^<]*)</t></is>|<v>([^<]*)</v>)')
    n = 0
    out_rows = []
    buf_tail = b""

    def flush(chunk):
        nonlocal n
        for rm in row_re.finditer(chunk):
            x = rm.group(0).decode("utf-8", "ignore")
            rowno = int(re.search(r'<row[^>]* r="(\d+)"', x).group(1))
            if rowno < 2:
                continue
            vals = {}
            for m in cell_re.finditer(x):
                col, attrs, a, b = m.group(1), m.group(3), m.group(4), m.group(5)
                ci = col_to_idx(col)
                if ci > MAXCOL:
                    continue
                tm = re.search(r't="(\w+)"', attrs)
                v = strings[int(b)] if (tm and tm.group(1) == "s" and b and int(b) < len(strings)) else (a if a else b)
                vals[ci] = v
            out_rows.append([vals.get(k, "") for k in KEEP])
            n += 1

    while True:
        chunk = sheet.read(1 << 22)
        if not chunk:
            break
        buf = buf_tail + chunk
        cut = buf.rfind(b"</row>")
        if cut == -1:
            buf_tail = buf
            continue
        flush(buf[:cut + 6])
        buf_tail = buf[cut + 6:]
    if buf_tail:
        flush(buf_tail)

    cols = ["patient_id","stat_date","age","amh","cycle_no","cycle_type",
            "planned_protocol","final_protocol","stim_regimen","cancellation","bmi","base_fsh",
            "oocytes_total","mii","good_embryos","blastocyst_cultured","blastocyst_formed","p_id"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(out_rows)
    print(f"rows: {n} [{time.time()-t0:.1f}s] -> {OUT}")

if __name__ == "__main__":
    main()
