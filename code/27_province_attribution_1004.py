# ============================================================
# 27_province_attribution_1004.py —— 分省归因：CO2 与各气象因子对品质、Q、优质占比、等级效应的贡献
# [1004] 运行顺序 27（依赖 04 两套预测 withco2 / fixedco2、10、11、14；16 的 province_frac_1km.npz；
#                    18/19 的 SHAP 缓存 shap_arrays/sv_{性状}_{ri}_{hist|ssp126|ssp585}.npy）
# 格网：该季、该情景经济核算域（loss 有值），每格按 frac（1 km 水稻像元落在该省的比例）分给各省；
#   [1013] 性状、Q、ΔSHAP 按 config.SUMMARY_WEIGHT 汇总；优质占比、等级效应 US$/ha 为面积量（frac × area_weight）
#
# A. CO2 与气候（反事实；与主结果同口径：逐年预测、完整集成模型）
#      总变化   = 未来(随情景 CO2) − 基准            ← withco2
#      气候部分 = 未来(CO2 固定为基准均值) − 基准     ← fixedco2
#      CO2 部分 = 总变化 − 气候部分 = 未来(随情景 CO2) − 未来(CO2 固定)
#      [1013] 基准见 config.CO2_CF_BASELINE（定稿 "withco2"：唯一基准，与 09 相同；等级效应同理）
#    对象：HRR、CD、PC、AC；由集合性状算出的 Q（同 11 号）、优质占比（Q ≥ Q_HIGH）；
#          等级效应 US$/ha（HRY 用主结果，只替换未来价格 → 单独看品质降档里 CO2 的份额）
# B. 各特征的贡献（ΔSHAP = 未来 SHAP − 基准 SHAP；19 号缓存；树模型、时段均值特征）
#      HRR、CD、PC 逐特征（CO2、Tavg、Tdif、CTSR、TSR、Rain、RHavg）的省内面积加权 ΔSHAP；
#      ΔQ 的特征贡献（一阶近似）：ΔQ_j ≈ [ΔSHAP_HRR,j/(hi−lo) − ΔSHAP_CD,j/(hi−lo) − ΔSHAP_PC,j/(hi−lo)] / 3
#      （hi/lo = 11 号锚点；性状落在锚点区间外的格点，该性状的贡献记 0）
#      各特征 ΔSHAP 之和 ≈ 树模型、时段均值口径下的预测变化，与主结果（逐年、全集成）并列输出供对照
# 输出（out_dir("loss")）：
#   province_attr_co2_long.csv     A：省 × 季别 × 情景 × 变量：total、climate、co2
#   province_attr_shap_long.csv    B：省 × 季别 × 情景 × 性状(HRR/CD/PC/Q) × 特征：ΔSHAP 或 ΔQ 贡献
#   province_attribution_wide.csv  ri = all，每省一行：ΔQ 总/CO2/气候、优质占比变化、等级效应 CO2 份额、
#                                  ΔQ 的前两位驱动特征、单位面积损失（来自 16）
# ============================================================
import numpy as np
import pandas as pd
import rasterio
import config_1004 as C
import q_common_1004 as Q

OUT    = C.out_dir("loss")
PW     = C.pred_dir(C.PRED_MODE, "withco2") / "ensemble"
PF     = C.pred_dir(C.PRED_MODE, "fixedco2") / "ensemble"
HRYD   = C.out_dir("head_rice_yield", thresh=False)
ZDIR   = C.out_dir("quality_zones") / C.PRED_MODE
FEATD  = C.out_dir("shap_features", thresh=False)
SVD    = C.fig_dir("shap", thresh=False) / "shap_arrays"
FEATS  = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]       # 与 19 号 FEATURES_BASE 顺序一致
TRAITS = ["HRR", "CD", "PC", "Acontent"]
QT     = ["HRR", "CD", "PC"]
P_ORD, P_HQ, CNY_PER_USD = C.P_ORD, C.P_HQ, C.CNY_PER_USD     # [1013] 只在 config 定义
SIGN   = {"HRR": 1.0, "CD": -1.0, "PC": -1.0}                         # Q 中 HRR 越高越好，CD、PC 越低越好


def rd(p):
    if not p.exists(): return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None and not np.isnan(ds.nodata): a[a == ds.nodata] = np.nan
        return a


npz = np.load(OUT / "province_frac_1km.npz", allow_pickle=False)
U = [str(x) for x in npz["names"]]; K = len(U)
GRP = [str(x) for x in npz["groups"]] if "groups" in npz.files else U       # < 0.5 Mha 及非主产区合并为一行（16 号写入）
G = list(dict.fromkeys(GRP)); GIDX = {g: [k for k in range(K) if GRP[k] == g] for g in G}
NAMES = G + ["National"]
FRAC = {ri: npz[f"frac_{ri}"].astype("float64") for ri in C.RI_TYPES}
with rasterio.open(OUT / f"loss_{C.RI_TYPES[0]}_{C.SSPS[0]}.tif") as ds:
    GT, GH, GW = ds.transform, ds.height, ds.width

acc = {}                                                   # key → [Σw·x, Σw]（K+1 行）


def add(key, x, w, F):
    m = np.isfinite(x) & (w > 0)
    ww, xx = np.where(m, w, 0.0), np.where(m, x, 0.0)
    num = np.append((F * (ww * xx)).sum((1, 2)), (ww * xx).sum())
    den = np.append((F * ww).sum((1, 2)), ww.sum())
    for k in (key, key[:1] + ("all",) + key[2:]):
        a = acc.get(k); acc[k] = np.stack([num, den], 1) if a is None else a + np.stack([num, den], 1)


def grid_from_csv(period, ri):
    df = pd.read_csv(FEATD / f"features_{period}_{ri}.csv")
    r = np.rint((GT.f - df.lat.to_numpy()) / abs(GT.e) - 0.5).astype(int)
    c = np.rint((df.lon.to_numpy() - GT.c) / GT.a - 0.5).astype(int)
    ok = (r >= 0) & (r < GH) & (c >= 0) & (c < GW)
    return r, c, ok, len(df)


def shap_grid(trait, ri, key, period):
    p = SVD / f"sv_{trait}_{ri}_{key}.npy"
    if not p.exists(): return None
    sv = np.load(p); r, c, ok, n = grid_from_csv(period, ri)
    if sv.shape[0] != n:
        print(f"  ⚠️ {p.name}: SHAP 行数 {sv.shape[0]} ≠ 特征表 {n}，跳过"); return None
    G = np.full((len(FEATS), GH, GW), np.nan)
    G[:, r[ok], c[ok]] = sv[ok].T
    return G


Q.check_anchors()
qchk = []
for ri in C.RI_TYPES:
    aw = rd(C.area_file(ri)); F = FRAC[ri]
    tw = {t: rd(PW / f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif") for t in TRAITS}
    tf = {t: rd(PF / f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif") for t in TRAITS}
    if C.CO2_CF_BASELINE == "withco2":
        tf = dict(tw)                         # [1013] 固定 CO2 的未来也与同一个基准比较
    yh = rd(HRYD / f"head_rice_yield_{ri}_historical.tif")
    sv_h = {t: shap_grid(t, ri, "hist", "historical") for t in QT}
    for ssp in C.SSPS:
        L = rd(OUT / f"loss_{ri}_{ssp}.tif")
        w = np.where(np.isfinite(L) & np.isfinite(aw), aw, 0.0)          # 经济核算域（面积量用）
        wc = np.where(w > 0, 1.0, 0.0) if C.SUMMARY_WEIGHT == "cell" else w   # [1013] 性状、Q、SHAP：汇总权重
        fw = {t: rd(PW / f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif") for t in TRAITS}
        ff = {t: rd(PF / f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif") for t in TRAITS}
        # ---------- A：性状 ----------
        for t in TRAITS:
            if any(x is None for x in (tw[t], tf[t], fw[t], ff[t])):
                print(f"  [缺] {t} {ri} {ssp}（withco2 或 fixedco2 集合预测）"); continue
            tot, cli = fw[t] - tw[t], ff[t] - tf[t]
            m = np.isfinite(tot) & np.isfinite(cli)
            add(("A", ri, ssp, t, "total"),   np.where(m, tot, np.nan), wc, F)
            add(("A", ri, ssp, t, "climate"), np.where(m, cli, np.nan), wc, F)
            add(("A", ri, ssp, t, "co2"),     np.where(m, tot - cli, np.nan), wc, F)
        if any(x is None for t in QT for x in (tw[t], tf[t], fw[t], ff[t])): continue
        # ---------- A：Q、优质占比、等级效应 ----------
        qwb, qwf = Q.q_index(*[tw[t] for t in QT]), Q.q_index(*[fw[t] for t in QT])
        qfb, qff = Q.q_index(*[tf[t] for t in QT]), Q.q_index(*[ff[t] for t in QT])
        q11 = rd(ZDIR / f"Q_{ri}_{ssp}_future.tif")
        if q11 is not None:
            mm = np.isfinite(q11) & np.isfinite(qwf) & (w > 0)
            qchk.append(float(np.max(np.abs(q11[mm] - qwf[mm]))) if mm.any() else 0.0)
        m = np.isfinite(qwb) & np.isfinite(qwf) & np.isfinite(qfb) & np.isfinite(qff)
        for comp, val in (("total", qwf - qwb), ("climate", qff - qfb), ("co2", (qwf - qwb) - (qff - qfb))):
            add(("A", ri, ssp, "Q", comp), np.where(m, val, np.nan), wc, F)
        H = C.Q_HIGH
        pw_b, pw_f = (qwb >= H).astype(float), (qwf >= H).astype(float)
        pf_b, pf_f = (qfb >= H).astype(float), (qff >= H).astype(float)
        for comp, val in (("total", pw_f - pw_b), ("climate", pf_f - pf_b),
                          ("co2", (pw_f - pw_b) - (pf_f - pf_b))):
            add(("A", ri, ssp, "premium_share", comp), np.where(m, 100 * val, np.nan), w, F)    # 百分点
        yf = rd(HRYD / f"head_rice_yield_{ri}_{ssp}.tif")
        if yf is not None:
            price = lambda q: np.where(q >= H, P_HQ, P_ORD)
            g_w = yf * 1000 * (price(qwb) - price(qwf)) / CNY_PER_USD        # US$/ha：t/ha × kg/t × 元/kg
            g_f = yf * 1000 * (price(qfb) - price(qff)) / CNY_PER_USD        # 固定 CO2 未来 vs 基准 qfb（"withco2" 时 qfb 即唯一基准）
            mg = m & np.isfinite(yf)
            add(("A", ri, ssp, "grade_USD_ha", "total"),   np.where(mg, g_w, np.nan), w, F)
            add(("A", ri, ssp, "grade_USD_ha", "climate"), np.where(mg, g_f, np.nan), w, F)
            add(("A", ri, ssp, "grade_USD_ha", "co2"),     np.where(mg, g_w - g_f, np.nan), w, F)
        # ---------- B：ΔSHAP ----------
        sv_f = {t: shap_grid(t, ri, ssp, ssp) for t in QT}
        if any(sv_h[t] is None or sv_f[t] is None for t in QT):
            print(f"  [缺] SHAP 缓存 {ri} {ssp}（先跑 19 号）"); continue
        dQ = np.zeros((len(FEATS), GH, GW)); okQ = np.ones((GH, GW), bool)
        for t in QT:
            d = sv_f[t] - sv_h[t]                                              # 特征 × GH × GW
            lo, hi = Q.ANCHORS[t]
            inr = (tw[t] > lo) & (tw[t] < hi) & (fw[t] > lo) & (fw[t] < hi)   # 锚点区间内才对 Q 有一阶贡献
            for j, f in enumerate(FEATS):
                add(("B", ri, ssp, t, f), d[j], wc, F)
            add(("B", ri, ssp, t, "sum_SHAP"), d.sum(0), wc, F)
            add(("B", ri, ssp, t, "main_delta"), np.where(np.isfinite(d[0]), fw[t] - tw[t], np.nan), wc, F)
            dQ += np.where(inr, SIGN[t] * np.nan_to_num(d) / (hi - lo), 0.0) / 3.0
            okQ &= np.isfinite(d[0])
        for j, f in enumerate(FEATS):
            add(("B", ri, ssp, "Q", f), np.where(okQ, dQ[j], np.nan), wc, F)
        add(("B", ri, ssp, "Q", "sum_SHAP"), np.where(okQ, dQ.sum(0), np.nan), wc, F)
        add(("B", ri, ssp, "Q", "main_delta"), np.where(okQ, qwf - qwb, np.nan), wc, F)
    print("done:", ri)
if qchk: print(f"  [核对] 由集合性状重算的 Q 与 11 号 Q 栅格最大差 {max(qchk):.2e}（应接近 0）")

# ---------------- 长表 ----------------
rows = {"A": [], "B": []}
for (part, ri, ssp, var, comp), a in acc.items():
    ag = [a[GIDX[g]].sum(0) for g in G] + [a[K]]                             # 按分组合并
    for n, (num, den) in zip(NAMES, ag):
        if den <= 0: continue
        rows[part].append(dict(province=n, ri=ri, ssp=ssp, variable=var, component=comp,
                               value=num / den, weight_sum=den))
A = pd.DataFrame(rows["A"]); B = pd.DataFrame(rows["B"])
A.round(5).sort_values(["province", "ri", "ssp", "variable", "component"]).to_csv(
    OUT / "province_attr_co2_long.csv", index=False, encoding="utf-8-sig")
B.round(6).sort_values(["province", "ri", "ssp", "variable", "component"]).to_csv(
    OUT / "province_attr_shap_long.csv", index=False, encoding="utf-8-sig")

# ---------------- 宽表（ri = all）----------------
PL = pd.read_csv(OUT / "province_loss.csv")
keep = ["province", "in_ranking"] + [c for c in PL.columns if c.endswith(("_area_kha", "_USD_ha"))]
W = PL[[c for c in keep if c in PL.columns]].copy()
Aa = A[A.ri == "all"].pivot_table(index="province", columns=["ssp", "variable", "component"], values="value")
Ba = B[B.ri == "all"]
for ssp in C.SSPS:
    for var, comps, nd in (("Q", ("total", "co2", "climate"), 4), ("premium_share", ("total", "co2"), 1),
                           ("grade_USD_ha", ("total", "co2"), 1), ("HRR", ("total", "co2"), 2),
                           ("CD", ("total", "co2"), 2), ("PC", ("total", "co2"), 3)):
        for comp in comps:
            col = (ssp, var, comp)
            if col in Aa.columns:
                lab = {"premium_share": "dPrem_pp", "grade_USD_ha": "grade_USD_ha"}.get(var, f"d{var}")
                W[f"{ssp}_{lab}_{comp}"] = W.province.map(Aa[col]).round(nd)
    q = Ba[(Ba.ssp == ssp) & (Ba.variable == "Q") & Ba.component.isin(FEATS)]
    if not q.empty:
        def top2(g):
            g = g.reindex(g.value.abs().sort_values(ascending=False).index)
            return "; ".join(f"{r.component}({r.value:+.4f})" for r in g.head(2).itertuples())
        W[f"{ssp}_dQ_top2_features"] = W.province.map(q.groupby("province").apply(top2))
        co2s = q.assign(a=q.value.abs()).groupby("province").apply(
            lambda g: 100 * g.loc[g.component == "CO2", "a"].sum() / g.a.sum() if g.a.sum() > 0 else np.nan)
        W[f"{ssp}_dQ_CO2_share_of_abs_pct"] = W.province.map(co2s).round(1)
W.to_csv(OUT / "province_attribution_wide.csv", index=False, encoding="utf-8-sig")

# ---------------- 屏幕摘要 ----------------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for ssp in C.SSPS:
    cols = ["province", f"{ssp}_USD_ha", f"{ssp}_dQ_total", f"{ssp}_dQ_co2", f"{ssp}_dQ_climate",
            f"{ssp}_dPrem_pp_total", f"{ssp}_dPrem_pp_co2", f"{ssp}_grade_USD_ha_total", f"{ssp}_grade_USD_ha_co2",
            f"{ssp}_dQ_top2_features"]
    print(f"\n==== {ssp}（ri1+ri2；dQ 为 Q 指数变化，dPrem 为优质占比变化百分点）====")
    sub = W[(W.in_ranking == True) | (W.province == "National") | W.province.str.startswith("Other")]
    print(sub[[c for c in cols if c in W.columns]].to_string(index=False))
nat = B[(B.province == "National") & (B.ri == "all") & (B.variable == "Q")]
if not nat.empty:
    print("\n全国 ΔQ 的特征贡献（ri1+ri2；sum_SHAP 与 main_delta 对照一阶近似的覆盖程度）：")
    print(nat.pivot_table(index="component", columns="ssp", values="value").round(4).to_string())
print(f"\n[saved] {OUT}/province_attr_co2_long.csv / province_attr_shap_long.csv / province_attribution_wide.csv")
