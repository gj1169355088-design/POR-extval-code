#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the analysis-ready prospective external-validation cohort from the
Zunyi-Jining Reproductive Health Registry xlsx (multi-level header, wide format).

Source: 遵义-济宁生殖健康队列数据.xlsx (Sheet1)
  - row 1: section (男性临床数据/女性临床数据/第一周期-新鲜周期/...)
  - row 2: sub-category (门诊信息/性激素六项/促排监测-.../胚胎培养/移植实验总结)
  - row 3: variable label (Chinese)
  - row 4: machine variable name (English)
  - data rows: one woman per row

Key columns (1-based index):
   1   编号            patient id (072*=Zunyi, 710/288/712*=Jining)
   10  site_name
   248 age            249 Height   250 Weight
   858 性激素六项 FSH  (baseline)
   866 性激素六项 AMH  (baseline)
   868 窦卵泡计数 左侧 LO, 869 右侧 RO (AFC)
   924 周期类型(第一周期-新鲜周期)   925 助孕方案   930 用药方案
   974 FSH (Gn启动日)  978 AMH (Gn启动日)
   1330 获卵数 (移植实验总结)  1347 成熟卵子数  1349 优良胚胎数  1355 冷冻胚胎总数
Outputs:
   zjy_external_cohort.csv   (one row per woman, first fresh cycle)
   extraction_log.txt
"""
import zipfile, re, sys, time, csv

XLSX = "<registry export>"
OUT_CSV = "./zjy_external_cohort.csv"
OUT_LOG = "./extraction_log.txt"

# columns of interest (1-based Excel column index) -> short name
KEEP = {
    1: "patient_id",
    10: "site_name",
    11: "gender",
    248: "age",
    249: "height_cm",
    250: "weight_kg",
    858: "base_FSH",          # 女性临床数据 性激素六项 FSH
    866: "base_AMH",          # 女性临床数据 AMH
    868: "AFC_left",
    924: "cycle_type",        # 第一周期-新鲜周期 周期类型 2=新鲜
    925: "art_protocol",      # 助孕方案
    930: "stim_protocol",     # 用药方案 1=降解方案 2=拮抗 3=长方案 4=其它
    933: "gn_start_date",
    974: "gn_day_FSH",
    978: "gn_day_AMH",
    1330: "oocytes_retrieved",  # 获卵数
    1347: "mii_oocytes",        # 成熟卵子数
    1349: "good_embryos",       # 优良胚胎数
    1355: "frozen_embryos",     # 冷冻胚胎总数
}
MAXCOL = max(KEEP)

def col_to_idx(c):
    n = 0
    for ch in c:
        n = n * 26 + ord(ch) - 64
    return n

def main():
    t0 = time.time()
    log = []
    z = zipfile.ZipFile(XLSX)
    # shared strings
    ss = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
    strings = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", s))
               for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    log.append(f"[{time.time()-t0:6.1f}s] shared strings: {len(strings)}")

    sheet = z.open("xl/worksheets/sheet1.xml")
    buf = b""
    row_re = re.compile(rb"<row[^>]*>.*?</row>", re.S)
    cell_re = re.compile(
        r'<c r="([A-Z]+)(\d+)"([^>]*)>(?:<is><t[^>]*>([^<]*)</t></is>|<v>([^<]*)</v>)')
    header = None
    n_data = 0
    written = 0
    fout = open(OUT_CSV, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(fout)
    w.writerow(["patient_id","site_name","gender","age","height_cm","weight_kg","BMI",
                "base_FSH","base_AMH","AFC_left",
                "cycle_type","art_protocol","stim_protocol","gn_start_date",
                "gn_day_FSH","gn_day_AMH",
                "oocytes_retrieved","mii_oocytes","good_embryos","frozen_embryos"])

    def flush_rows(chunk):
        nonlocal header, n_data, written
        for rm in row_re.finditer(chunk):
            x = rm.group(0).decode("utf-8", "ignore")
            rowno = int(re.search(r'<row[^>]* r="(\d+)"', x).group(1))
            vals = {}
            for m in cell_re.finditer(x):
                col, attrs, a, b = m.group(1), m.group(3), m.group(4), m.group(5)
                ci = col_to_idx(col)
                if ci > MAXCOL:
                    continue
                tm = re.search(r't="(\w+)"', attrs)
                v = strings[int(b)] if (tm and tm.group(1) == "s" and b and int(b) < len(strings)) else (a if a else b)
                vals[ci] = v
            if rowno == 1:
                header = vals
                log.append(f"[{time.time()-t0:6.1f}s] header row1 sections: "
                           + str({k: v for k, v in sorted(vals.items()) if k <= MAXCOL}))
            elif rowno == 3:
                # sanity: confirm variable labels at expected columns
                chk = {KEEP[k]: vals.get(k, "MISSING") for k in KEEP}
                log.append(f"[{time.time()-t0:6.1f}s] row3 label check: {chk}")
            elif rowno >= 4:
                n_data += 1
                rec = [vals.get(k, "") for k in KEEP]
                try:
                    h = float(rec[5]); wgt = float(rec[6])
                    bmi = round(wgt / ((h / 100) ** 2), 2) if h > 50 and wgt > 25 else ""
                except Exception:
                    bmi = ""
                w.writerow(rec[:7] + [bmi] + rec[7:])
                written += 1
    carry = b""
    while True:
        chunk = sheet.read(1 << 22)
        if not chunk:
            break
        buf = carry + chunk
        # only process complete rows; keep tail as carry
        cut = buf.rfind(b"</row>")
        if cut == -1:
            carry = buf
            continue
        flush_rows(buf[:cut + 6])
        carry = buf[cut + 6:]
    if carry:
        flush_rows(carry)
    fout.close()
    log.append(f"[{time.time()-t0:6.1f}s] data rows read: {n_data}, written: {written}")
    open(OUT_LOG, "w", encoding="utf-8").write("\n".join(log))
    print("\n".join(log))

if __name__ == "__main__":
    main()
