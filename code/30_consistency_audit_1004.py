# ============================================================
# 30_consistency_audit_1004.py —— 跨脚本一致性核对（1013；在 21、22 之前运行）
#   同一个量如果出现在多个脚本的输出里，必须是同一个数。本脚本逐项核对并给出 ✔ / ✘：
#   A 格网与基准唯一：06 各情景基准相同；09 各情景基准相同（CO2_CF_BASELINE="withco2" 时两套 CO2 也相同，
#     "own" 时每套 CO2 各自一个基准）；14 各情景基准产值相同
#     （24 号逐成员基准产值的情景间差只作提示：成员缺值格网在两情景可能略有不同，属补充材料的成员离散度）
#   B 气候：05 分解的基准、总变化 = 06 气候汇总；05 天数 = 06 GF_days；07 灌浆天数（MA−HE）≈ 06 窗口日数（差 ≤ 0.5 d）
#   C 经济：14 全国 = 16 各省合计 = 16 数值框 = 25（Q_HIGH 主值）= 28 核算表；HRY 变化 = 16 数值框
#   D 分区：11 写出的阈值 = config；23 主方案集合基准优质占比 = 11 分区栅格
#   E 代码：除 config 外不得再写死锚点、价格、汇率；除 01/03b/config 外不得直接读原始支撑区
#   F 旧输出：08 旧外推表、旧 SHAP 缓存等已失效文件
# 输出：output/consistency_audit_{TAG}/consistency_audit.csv（21 收录进 Excel）
# ============================================================
import re
from pathlib import Path
import numpy as np
import pandas as pd

import config_1004 as C
import stats_common_1004 as S

OUT = C.OUTPUT / f"consistency_audit_{C.TAG}"; OUT.mkdir(parents=True, exist_ok=True)
F_05 = C.OUTPUT / f"decomposition_373_{C.TAG}" / "summary.csv"
F_06 = C.FIGURES / f"climate_change_{C.TAG}" / "climate_change_summary.csv"
F_07 = C.FIGURES / f"phenology_{C.TAG}" / "phenology_summary.csv"
F_09 = C.FIGURES / f"rice_quality_{C.TAG}_{C.PRED_MODE}" / "quality_change_summary.csv"
LOSS = C.out_dir("loss"); SENS = C.out_dir("sensitivity")
ZDIR = C.out_dir("quality_zones") / C.PRED_MODE
rows = []


def rd(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def check(group, item, a, b, tol, src_a, src_b, note=""):
    a = float(a) if a is not None else np.nan; b = float(b) if b is not None else np.nan
    d = abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
    ok = bool(np.isfinite(d) and d <= tol)
    rows.append(dict(group=group, item=item, value_a=a, source_a=src_a, value_b=b, source_b=src_b,
                     abs_diff=d, tolerance=tol, ok="✔" if ok else "✘", note=note))


def missing(group, item, f):
    rows.append(dict(group=group, item=item, ok="–", note=f"缺 {f}（对应步骤未运行）"))


# ---------------- A：每个季型只有一个基准 ----------------
d06, d05, d07, d09 = rd(F_06), rd(F_05), rd(F_07), rd(F_09)
if d06 is not None:
    for (v, ri), g in d06.groupby(["var", "ri"]):
        check("A 基准唯一", f"06 {v} {ri} 基准（{C.SSPS[0]} vs {C.SSPS[-1]}）",
              g[g.ssp == C.SSPS[0]].baseline.iloc[0], g[g.ssp == C.SSPS[-1]].baseline.iloc[0], 1e-6, "06", "06")
else:
    missing("A 基准唯一", "06", F_06)
if d09 is not None:
    # CO2_CF_BASELINE="withco2"：两套 CO2 共用一个基准；"own"：每套 CO2 各自一个基准（情景间仍必须相同）
    keys = ["trait", "ri"] if C.CO2_CF_BASELINE == "withco2" else ["trait", "ri", "co2"]
    for k, g in d09.groupby(keys):
        lab = " ".join(map(str, k)) + ("（所有情景 × 两套 CO2）" if C.CO2_CF_BASELINE == "withco2" else "（所有情景）")
        check("A 基准唯一", f"09 {lab} 基准", g.base_mean.min(), g.base_mean.max(), 1e-6,
              "09 最小", "09 最大", f"CO2_CF_BASELINE={C.CO2_CF_BASELINE}")
else:
    missing("A 基准唯一", "09", F_09)
d14 = rd(LOSS / "production_loss_summary.csv")
if d14 is not None:
    for ri in C.RI_TYPES + ["ri1+ri2"]:
        g = d14[d14.ri == ri]
        check("A 基准唯一", f"14 {ri} 基准产值（亿元）", g.V_his_yi.min(), g.V_his_yi.max(), 1e-3, "14 最小", "14 最大")
else:
    missing("A 基准唯一", "14", "production_loss_summary.csv")
d24 = rd(SENS / "econ_members_long.csv")
if d24 is not None:
    sp = d24.groupby(["ri", "model", "gcm"]).V_hist.agg(lambda x: x.max() - x.min())
    rows.append(dict(group="A 基准唯一", item="24 逐成员基准产值的情景间最大差（亿元，提示）", value_a=float(sp.max()),
                     source_a="24", ok="ℹ", note="成员缺值格网在两情景略有不同所致；只影响补充材料的成员范围"))

# ---------------- B：气候、物候 ----------------
MAP56 = {"days": "GF_days"}
if d05 is not None and d06 is not None:
    for _, r in d05.iterrows():
        v6 = MAP56.get(r["var"], r["var"])
        x = d06[(d06["var"] == v6) & (d06.ri == r.ri) & (d06.ssp == r.ssp)]
        if x.empty: continue
        x = x.iloc[0]
        check("B 气候", f"{r['var']} {r.ri} {r.ssp} 基准", r.baseline, x.baseline, 1e-3, "05", "06")
        check("B 气候", f"{r['var']} {r.ri} {r.ssp} 总变化", r.total, x.change, 1e-3, "05", "06")
        if r["var"] == "Tavg":
            check("B 气候", f"Tavg {r.ri} {r.ssp} 基准窗口项 A′−A（应≈0）", r.base_window, 0.0, 0.05, "05", "0")
            check("B 气候", f"Tavg {r.ri} {r.ssp} 气候+窗口+基准窗口 = 总变化",
                  r.climate + r["shift"] + r.base_window, r.total, 1e-3, "05", "05", "CSV 保留 4 位小数")
elif d05 is None:
    missing("B 气候", "05", F_05)
if d07 is not None and d06 is not None:
    e7 = d07[d07.member == "ensemble_mean"]
    for _, r in e7.iterrows():
        x = d06[(d06["var"] == "GF_days") & (d06.ri == r.ri) & (d06.ssp == r.ssp)]
        if x.empty: continue
        x = x.iloc[0]
        check("B 物候", f"灌浆天数 {r.ri} {r.ssp} 基准（d）", r.GF_base, x.baseline, 0.5, "07 物候",
              "06 窗口日数", "同一定义；差值来自跨年窗口截断或格网差异")
        check("B 物候", f"灌浆天数 {r.ri} {r.ssp} 变化（d）", r.GF_change, x.change, 0.5, "07 物候", "06 窗口日数")
elif d07 is None:
    missing("B 物候", "07", F_07)

# ---------------- C：经济 ----------------
if d14 is not None:
    nat14 = {s: float(d14[(d14.ssp == s) & (d14.ri == "ri1+ri2")].loss_yi.iloc[0]) for s in C.SSPS}
    pl = rd(LOSS / "province_loss_long.csv")
    if pl is not None:
        for s in C.SSPS:
            x = pl[(pl.ssp == s) & (pl.ri == "all") & (pl.province != "National")].loss_yi.sum()
            check("C 经济", f"{s} 全国损失（亿元）：14 vs 16 各省合计", nat14[s], x, max(0.05, 1e-3 * abs(nat14[s])), "14", "16 分省")
    bv = rd(LOSS / "bivariate_summary.csv")
    if bv is not None:
        for _, r in bv.iterrows():
            y = d14[(d14.ssp == r.ssp) & (d14.ri == r.ri)]
            if len(y):
                check("C 经济", f"{r.ssp} {r.ri} 损失（亿元）：14 vs 16 数值框", y.loss_yi.iloc[0],
                      r.economic_loss_1e8CNY, 0.01, "14", "16 bivariate_summary")
            # HRY 变化：在 14 的核算格网上重算（= 21 号 07_head_rice_yield）
            H = C.out_dir("head_rice_yield", thresh=False)
            yh, yf = S.read(H / f"head_rice_yield_{r.ri}_historical.tif"), S.read(H / f"head_rice_yield_{r.ri}_{r.ssp}.tif")
            lo, aw = S.read(LOSS / f"loss_{r.ri}_{r.ssp}.tif"), S.read(C.area_file(r.ri))
            if all(z is not None for z in (yh, yf, lo, aw)):
                k = np.isfinite(lo) & np.isfinite(yh) & np.isfinite(yf) & np.isfinite(aw)
                pct = 100 * (np.sum(aw[k] * yf[k]) / np.sum(aw[k] * yh[k]) - 1)
                check("C 经济", f"{r.ssp} {r.ri} Δ整精米产量（%）：21 vs 16 数值框", pct,
                      r.head_rice_production_change_pct, 0.01, "21 m07 口径", "16 bivariate_summary")
    q25 = rd(SENS / "econ_qhigh_long.csv")
    if q25 is not None:
        for s in C.SSPS:
            x = q25[(q25.Q_HIGH.round(4) == round(C.Q_HIGH, 4)) & (q25.ssp == s) & (q25.ri == "ri1+ri2")]
            if len(x):
                check("C 经济", f"{s} 全国损失（亿元）：14 vs 25（Q_HIGH={C.Q_HIGH}）", nat14[s], x.loss_yi.iloc[0],
                      max(0.05, 1e-3 * abs(nat14[s])), "14", "25")
    n28 = rd(LOSS / "econ_accounting_numeric.csv")
    if n28 is not None:
        for s in C.SSPS:
            x = n28[(n28.ssp == s) & (n28.ri == "ri1+ri2") & (n28.quantity == "net")]
            if len(x):
                check("C 经济", f"{s} 全国净变化（十亿美元）：14 vs 28", -nat14[s] / 10 / C.CNY_PER_USD,
                      x.ensemble.iloc[0], 0.002, "14", "28")

# ---------------- D：分区 ----------------
import json
th = C.thresholds_file()
if th.exists():
    t = json.loads(th.read_text())["thresholds"]
    for ri, v in t.items():
        if C.Q_THRESH_MODE == "fixed":
            check("D 分区", f"{ri} Q_HIGH：11 vs config", v["Q_HIGH"], C.Q_HIGH, 1e-9, "11", "config")
            check("D 分区", f"{ri} Q_LOW：11 vs config", v["Q_LOW"], C.Q_LOW, 1e-9, "11", "config")
q23 = rd(SENS / "q_sensitivity_long.csv")
if q23 is not None:
    for ri in C.RI_TYPES:
        z = S.read(ZDIR / f"zone_{ri}_baseline.tif"); w = S.area(ri)
        if z is None: continue
        k = np.isfinite(z) & (z > 0) & np.isfinite(w)
        sh = 100 * w[k & (z == 3)].sum() / w[k].sum()
        x = q23[(q23.scheme == "main (1:1:1)") & (q23.threshold == "fixed") & (q23.ri == ri) & (q23.period == "baseline")]
        if len(x):
            check("D 分区", f"{ri} 基准优质占比（%）：11 分区栅格 vs 23", sh, x.premium_ens.iloc[0], 0.01, "11", "23")

# ---------------- E：代码里不得有第二套常数 / 第二套格网 ----------------
HERE = Path(__file__).resolve().parent
PAT = {r"\b5\.23\b|\b5\.96\b": "价格", r"23\.38|73\.77|44\.59|12\.18": "锚点",
       r"CNY_PER_USD\s*=\s*7": "汇率", r"_baseline_support_": "原始支撑区路径"}
ALLOW = {"config_1004.py", "30_consistency_audit_1004.py", "01_make_baseline_support_1004.py",
         "03b_make_analysis_domain_1004.py"}
for f in sorted(HERE.glob("*.py")):
    if f.name in ALLOW: continue
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0]
        for pat, lab in PAT.items():
            if re.search(pat, code):
                rows.append(dict(group="E 代码", item=f"{f.name}:{i} 写死了{lab}", ok="✘", note=line.strip()[:120]))
if not any(r["group"] == "E 代码" for r in rows):
    rows.append(dict(group="E 代码", item="锚点/价格/汇率/原始支撑区只在 config 定义", ok="✔"))

# ---------------- F：已失效的旧输出（22 打包前删除）----------------
stale = [C.FIGURES / f"feature_distribution_{C.TAG}" / "extrapolation_check.csv"]
for p in stale:
    rows.append(dict(group="F 旧输出", item=f"{p.parent.name}/{p.name}", ok="✘" if p.exists() else "✔",
                     note="旧口径文件仍在，22 会把它打包；请删除" if p.exists() else "不存在"))

R = pd.DataFrame(rows)
R.to_csv(OUT / "consistency_audit.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
cols = [c for c in ["ok", "group", "item", "value_a", "value_b", "abs_diff", "source_a", "source_b"] if c in R]
for _, r in R.iterrows():
    v = "" if pd.isna(r.get("value_a")) else (f"  {r.value_a:.4g}" if pd.isna(r.get("value_b")) else
                                               f"  {r.value_a:.4g} vs {r.value_b:.4g}（{r.source_a} / {r.source_b}）")
    print(f"{r.ok} [{r.group}] {r['item']}{v}")
n_bad = int((R.ok == "✘").sum()); n_ok = int((R.ok == "✔").sum())
print(f"\n一致性核对：✔ {n_ok} 项  ✘ {n_bad} 项  → {OUT / 'consistency_audit.csv'}")
if n_bad:
    print("有 ✘ 的项目请先处理再引用对应数字（✘ 不会中断后续步骤）。")
