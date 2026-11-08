# ============================================================
# 25_econ_qhigh_sensitivity_1004.py —— 经济损失对优质阈值 Q_HIGH 的敏感性（补充材料）
# [1004] 运行顺序 25（依赖 10、11；与 14 口径完全一致）
#   价格只取决于是否优质（Q ≥ Q_HIGH → 5.96，否则 5.23 元/kg），与 Q_LOW 无关，
#   所以只需改变 Q_HIGH 重新赋价，不用重跑分区。Q 栅格直接读 11 的输出（Q 与阈值无关）。
#   集合口径：Q 由集合平均性状计算（同 Fig. 5、14）；另报 25 成员净损失中位数 [范围]（同 24）。
# 输入：
#   out_dir("quality_zones")/{PRED_MODE}/Q_{ri}_baseline.tif、Q_{ri}_{ssp}_future.tif
#   out_dir("head_rice_yield", thresh=False)/head_rice_yield_{ri}_{historical|ssp}.tif
#   area_weights、共同支撑区；逐成员性状与 HRY（成员范围，经 q_common_1004）
# 输出（out_dir("sensitivity")）：
#   econ_qhigh_long.csv     每个 Q_HIGH × 情景 × 稻作类型（含 ri1+ri2）：价值、净损失、数量/等级效应、优质面积
#   econ_qhigh_table.csv    补充表格式（全国，十亿美元 yr⁻¹）
#   econ_qhigh_text.txt     可直接填入正文的句子（X、Y 与范围）
# ============================================================
import numpy as np
import pandas as pd
import rasterio
import config_1004 as C

Q_HIGHS     = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72]   # [1031] 与 Table S7 同范围；主方案 = config.Q_HIGH（0.66）
Q_MAIN      = C.Q_HIGH                                     # 0.66
P_ORD, P_HQ = C.P_ORD, C.P_HQ                        # 元/kg   [1013] 价格、汇率只在 config 定义
CNY_PER_USD = C.CNY_PER_USD
UNIT, YI    = 1e5, 1e8                               # t/ha·km²·元/kg → 元；元 → 亿元
DO_MEMBERS  = True                                   # 同时计算 25 成员范围（约几分钟）

ZDIR = C.out_dir("quality_zones") / C.PRED_MODE
HRY  = C.out_dir("head_rice_yield", thresh=False)
OUT  = C.out_dir("sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
usd  = lambda yi: yi / 10.0 / CNY_PER_USD            # 亿元 → 十亿美元


def rd(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a[a == ds.nodata] = np.nan
    return a


def terms(aw, yh, yf, qh, qf, valid, qhi):
    ph = np.where(qh >= qhi, P_HQ, P_ORD); pf = np.where(qf >= qhi, P_HQ, P_ORD)
    s = lambda x: float(np.nansum(np.where(valid, x, 0.0))) / YI
    Vh, Vf = s(aw * yh * ph * UNIT), s(aw * yf * pf * UNIT)
    return dict(V_his_yi=Vh, V_fut_yi=Vf, loss_yi=Vh - Vf,
                output_effect_yi=s(aw * UNIT * ph * (yh - yf)),
                grade_effect_yi=s(aw * UNIT * yf * (ph - pf)),
                prem_area_his_km2=float(aw[valid & (qh >= qhi)].sum()),
                prem_area_fut_km2=float(aw[valid & (qf >= qhi)].sum()),
                area_km2=float(aw[valid].sum()))


# ---------------- 集合平均（正文口径）----------------
rows = []
for ri in C.RI_TYPES:
    aw = rd(C.area_file(ri)); sup = rd(C.support_file(ri))
    qh = rd(ZDIR / f"Q_{ri}_baseline.tif"); yh = rd(HRY / f"head_rice_yield_{ri}_historical.tif")
    for ssp in C.SSPS:
        qf = rd(ZDIR / f"Q_{ri}_{ssp}_future.tif"); yf = rd(HRY / f"head_rice_yield_{ri}_{ssp}.tif")
        valid = (np.isfinite(yh) & np.isfinite(yf) & np.isfinite(aw) & (aw > 0)
                 & np.isfinite(qh) & np.isfinite(qf) & np.isfinite(sup))
        for qhi in Q_HIGHS:
            rows.append(dict(Q_HIGH=qhi, ssp=ssp, ri=ri, **terms(aw, yh, yf, qh, qf, valid, qhi)))
df = pd.DataFrame(rows)
num = [c for c in df.columns if c.endswith(("_yi", "_km2"))]
nat = df.groupby(["Q_HIGH", "ssp"], as_index=False)[num].sum(); nat["ri"] = "ri1+ri2"
df = pd.concat([df, nat], ignore_index=True)
df["loss_pct"] = 100 * df.loss_yi / df.V_his_yi
df["prem_share_his_pct"] = 100 * df.prem_area_his_km2 / df.area_km2
df["prem_share_fut_pct"] = 100 * df.prem_area_fut_km2 / df.area_km2
for c in ["V_his_yi", "V_fut_yi", "loss_yi", "output_effect_yi", "grade_effect_yi"]:
    df[c.replace("_yi", "_bnUSD")] = usd(df[c])

# ---------------- 25 成员范围（同 24 口径）----------------
if DO_MEMBERS:
    import q_common_1004 as Q
    mem = []
    for ri in C.RI_TYPES:
        a = Q.area_and_mask(ri)
        qh_g = {g: Q.q_index(*Q.traits_hist(ri, g)) for g in Q.GCMS}
        for ssp in C.SSPS:
            for m in Q.GGCMS:
                for g in Q.GCMS:
                    yh = Q.read(Q.find("hry_hist", ri=ri, model=m, gcm=g))
                    yf = Q.read(Q.find("hry_fut", ri=ri, ssp=ssp, model=m, gcm=g))
                    qf = Q.q_index(*Q.traits_fut(ri, ssp, m, g))
                    v = np.isfinite(a) & np.isfinite(yh) & np.isfinite(yf) & np.isfinite(qh_g[g]) & np.isfinite(qf)
                    aa = np.where(v, a, 0.0)
                    for qhi in Q_HIGHS:
                        t = terms(aa, yh, yf, qh_g[g], qf, v, qhi)
                        mem.append(dict(Q_HIGH=qhi, ssp=ssp, ri=ri, model=m, gcm=g,
                                        loss_yi=t["loss_yi"], V_his_yi=t["V_his_yi"]))
        print("members done:", ri)
    M = pd.DataFrame(mem)
    Mn = M.groupby(["Q_HIGH", "ssp", "model", "gcm"], as_index=False)[["loss_yi", "V_his_yi"]].sum(); Mn["ri"] = "ri1+ri2"
    M = pd.concat([M, Mn], ignore_index=True)
    agg = (M.assign(loss_bnUSD=usd(M.loss_yi))
             .groupby(["Q_HIGH", "ssp", "ri"])["loss_bnUSD"]
             .agg(member_median="median", member_min="min", member_max="max").reset_index())
    df = df.merge(agg, on=["Q_HIGH", "ssp", "ri"], how="left")

df = df.round(4)
df.to_csv(OUT / "econ_qhigh_long.csv", index=False)

# ---------------- 一致性检查：Q_HIGH = 0.66 应与 14 的结果相同 ----------------
ref = C.out_dir("loss") / "production_loss_summary.csv"
if ref.exists():
    r = pd.read_csv(ref)
    for ssp in C.SSPS:
        a = float(r[(r.ssp == ssp) & (r.ri == "ri1+ri2")].loss_yi.iloc[0])
        b = float(df[(df.Q_HIGH == Q_MAIN) & (df.ssp == ssp) & (df.ri == "ri1+ri2")].loss_yi.iloc[0])
        flag = "✔" if abs(a - b) <= max(0.5, 0.005 * abs(a)) else "✘ 与 14 不一致，请检查"
        print(f"  一致性 {ssp}: 14 号 {a:.2f} 亿元  vs 本脚本 {b:.2f} 亿元  {flag}")

# ---------------- 补充表（全国）----------------
N = df[df.ri == "ri1+ri2"]
tab = []
for qhi in Q_HIGHS:
    row = {"Q_HIGH": qhi}
    for ssp in C.SSPS:
        x = N[(N.Q_HIGH == qhi) & (N.ssp == ssp)].iloc[0]
        lab = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}[ssp]
        row[f"{lab} net loss, ensemble (bn US$ yr-1)"] = f"{x.loss_bnUSD:.2f}"
        if DO_MEMBERS:   # 成员各自分级；分级非线性，集合值不一定落在成员范围内
            row[f"{lab} net loss, members median [range]"] = f"{x.member_median:.2f} [{x.member_min:.2f}–{x.member_max:.2f}]"
        row[f"{lab} loss (%)"] = f"{x.loss_pct:.1f}"
        row[f"{lab} output effect"] = f"{x.output_effect_bnUSD:.2f}"
        row[f"{lab} grade effect"] = f"{x.grade_effect_bnUSD:.2f}"
        row[f"{lab} premium share base→future (%)"] = f"{x.prem_share_his_pct:.0f}→{x.prem_share_fut_pct:.0f}"
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(OUT / "econ_qhigh_table.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab.to_string(index=False))

# ---------------- 正文句子 ----------------
def val(ssp, qhi): return float(N[(N.Q_HIGH == qhi) & (N.ssp == ssp)].loss_bnUSD.iloc[0])
X, Y = val("ssp126", Q_MAIN), val("ssp585", Q_MAIN)
rx = [val("ssp126", q) for q in Q_HIGHS]; ry = [val("ssp585", q) for q in Q_HIGHS]
txt = (f"Under Q_HIGH = {Q_MAIN:.2f}, the projected mean annual loss in head-rice value was US${X:.2f} billion "
       f"(SSP1-2.6) and US${Y:.2f} billion (SSP5-8.5); across Q_HIGH = {Q_HIGHS[0]:.2f}–{Q_HIGHS[-1]:.2f}, "
       f"the corresponding ranges were US${min(rx):.2f}–{max(rx):.2f} billion and US${min(ry):.2f}–{max(ry):.2f} billion.")
hi585 = [f"{q:.2f}" for q in Q_HIGHS if val("ssp585", q) > val("ssp126", q)]
txt += ("\n\nNote: at Q_HIGH = 0.62 the moderate class is empty (Q_LOW = 0.62); value depends only on Q_HIGH."
        "\nQ_HIGH values at which the SSP5-8.5 loss exceeds the SSP1-2.6 loss: " + (", ".join(hi585) or "none"))
(OUT / "econ_qhigh_text.txt").write_text(txt, encoding="utf-8")
print("\n" + txt)
print(f"\n[saved] {OUT}/econ_qhigh_long.csv / econ_qhigh_table.csv / econ_qhigh_text.txt")
