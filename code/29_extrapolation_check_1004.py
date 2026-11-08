# ============================================================
# 29_extrapolation_check_1004.py —— 补充表 S10：预测输入超出训练范围的比例（逐年、逐成员口径）
# [1004] 运行顺序 29（依赖 02/03 的逐年气候 nc、01 的共同支撑区；不依赖预测结果）
#
# 模型实际吃的是逐年 × 逐成员的灌浆期气候 + 当年 CO2（主结果为 annual 预测）。本脚本按模型真实输入计算
#   （[1013] 08 号的分布图也已改为同一批输入，08 不再输出外推表）：
#   · 基准 1981–2010：5 个 GCM × 30 年，成员自己的逐年物候窗口；CO2 = 当年历史值
#   · 未来 2041–2070：25 个成员（GGCM × GCM）× 30 年，成员自己的未来物候窗口；CO2 = 当年情景值
#   · 固定窗口（回应 #373：SSP1-2.6 灌浆期增温偏高）：未来气候套用基准期平均窗口（5 GCM × 30 年），
#     即 05 号分解里的 C；与上一列对比 = 物候窗口移动的影响
# 只统计统一格网域 D（03b）内的格点；每个"格点 × 年 × 成员"算一个值。
# [1013] 统一口径（config.DIAG_WEIGHT = "cell"）：每个"格点 × 年 × 成员"等权计数；
#        S10b 均值为全部输入值的精确均值（分位数由抽样估计）；#373 的增温分解只由 05 号给出，这里不再另算
#
# 指标
#   单变量（NT1）：低于训练最小值 / 高于训练最大值的比例（%）
#   任一预测因子越界：七个因子中至少一个超出训练范围的比例
#   组合新颖性（NT2，Mesgaran et al. 2014, Divers. Distrib. 20:1147）：
#       在单变量范围内的点，计算到训练集中心的马氏距离 D，NT2 = D / max(D_训练)；NT2 > 1 = 组合新颖
#   预测因子的均值 [5%, 95%]（S10b），可直接看固定窗口下 Tavg 少升多少
# 输出（fig_dir("feature_distribution", thresh=False) → 汇总到 08_feature_distribution）：
#   extrapolation_annual_long.csv        长表：因子 × 列 × 低于/高于、样本数
#   table_S10_extrapolation.csv          论文表格式："低于 / 高于"（%）+ 任一越界 + 组合新颖
#   table_S10b_predictor_means.csv       各列预测因子的均值 [5%, 95%]
#   table_S10_caption.txt                表注（训练样本数、口径、NT2 说明、与 08 号的差别）
# ============================================================
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import config_1004 as C

OUT = C.FIGURES / f"feature_distribution_{C.TAG}"; OUT.mkdir(parents=True, exist_ok=True)
S10_WEIGHT = "area" if C.DIAG_WEIGHT == "area" else "count"      # 随 config：定稿 count（格点×年×成员等权）
FEATS = ["Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg", "CO2"]
LAB = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "CTSR": "cumul_solar_MJ_m2",
       "TSR": "mean_daily_solar_MJ_m2", "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct"}
UNIT = {"Tavg": "°C", "Tdif": "°C", "CTSR": "MJ m⁻²", "TSR": "MJ m⁻² d⁻¹", "Rain": "mm",
        "RHavg": "%", "CO2": "ppm"}
H, F = C.CLIM_HIS, C.CLIM_FUT
HL, FL = C.HIS_LABEL, C.FUT_LABEL

# ---------------- 训练数据 ----------------
T = pd.read_excel(C.FACE_XLSX)
T = T[FEATS].apply(pd.to_numeric, errors="coerce").dropna()
LO, HI = T.min(), T.max()
MU = T.mean().to_numpy()
SINV = np.linalg.pinv(np.cov(T.to_numpy(), rowvar=False))
def maha(X):
    d = X - MU
    return np.sqrt(np.einsum("ij,jk,ik->i", d, SINV, d))
DMAX = maha(T.to_numpy()).max()
print(f"训练记录 {len(T)} 条；训练集马氏距离最大值 {DMAX:.2f}")
print("训练范围：" + "  ".join(f"{f} {LO[f]:.2f}–{HI[f]:.2f}" for f in FEATS))

# ---------------- 各列的输入文件 ----------------
def cols():
    yield ("baseline", f"Baseline {HL}", "historical", C.HIS_YEARS,
           lambda ri: [H / "annual" / f"{g}_historical_{ri}_{C.HIS_P}_annual.nc" for g in C.GCMS])
    for s, sl in (("ssp126", "SSP1-2.6"), ("ssp585", "SSP5-8.5")):
        yield (s, f"{sl} {FL}", s, C.FUT_YEARS,
               lambda ri, s=s: [F / "annual" / s / m / g / f"{m}_{g}_{s}_{ri}_{C.FUT_P}_annual.nc"
                                for m in C.CROP_MODELS for g in C.GCMS])
    for s, sl in (("ssp126", "SSP1-2.6"), ("ssp585", "SSP5-8.5")):
        yield (f"{s}_fixedwindow", f"{sl} {FL}, fixed baseline window", s, C.FUT_YEARS,
               lambda ri, s=s: [F / "fixed_window" / "annual" / f"{g}_{s}_{ri}_{C.FUT_P}_fixedwindow_annual.nc"
                                for g in C.GCMS])


def rd(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None and not np.isnan(ds.nodata): a[a == ds.nodata] = np.nan
    return a


rows, means = [], []
for key, label, co2scen, years, files in cols():
    co2 = np.array(C.co2_series(co2scen, years))
    acc = {f: np.zeros(4) for f in FEATS}                  # 低于数、高于数、低于权重、高于权重
    n_tot = w_tot = 0.0; any_n = any_w = 0.0; nov_n = nov_w = 0.0; inr_n = 0.0; cl_n = cl_w = 0.0
    vals = {f: [] for f in FEATS}; wts = []; nfile = 0
    swx = {f: 0.0 for f in FEATS}; sw = 0.0                 # 精确均值（权重随口径）
    for ri in C.RI_TYPES:
        aw = rd(C.area_file(ri)); sup = np.isfinite(rd(C.support_file(ri)))
        dom = sup & np.isfinite(aw) & (aw > 0)
        for fp in files(ri):
            if not fp.exists():
                print(f"  [缺] {fp}"); continue
            with xr.open_dataset(fp) as ds:
                yy = [y for y in years if y in ds.year.values]
                A = {f: ds[LAB[f]].sel(year=yy).values[:, dom] for f in LAB}      # 年 × 格点
            c = np.array(C.co2_series(co2scen, yy))
            A["CO2"] = np.repeat(c[:, None], dom.sum(), axis=1)
            wv = aw[dom] if S10_WEIGHT == "area" else np.ones(int(dom.sum()))
            w = np.repeat(wv[None, :], len(yy), axis=0)
            X = np.stack([A[f].ravel() for f in FEATS], 1); w = w.ravel()
            ok = np.isfinite(X).all(1); X, w = X[ok], w[ok]
            nfile += 1; n_tot += len(X); w_tot += w.sum()
            sw += w.sum()
            for j, f in enumerate(FEATS): swx[f] += float(np.sum(w * X[:, j]))
            below = X < LO[FEATS].to_numpy(); above = X > HI[FEATS].to_numpy()
            for j, f in enumerate(FEATS):
                acc[f] += [below[:, j].sum(), above[:, j].sum(), w[below[:, j]].sum(), w[above[:, j]].sum()]
            out = (below | above).any(1)
            any_n += out.sum(); any_w += w[out].sum()
            outc = (below | above)[:, :6].any(1)                       # 只看气候因子（不含 CO2）
            cl_n += outc.sum(); cl_w += w[outc].sum()
            inr = ~out; inr_n += inr.sum()
            if inr.any():
                nov = maha(X[inr]) / DMAX > 1.0
                nov_n += nov.sum(); nov_w += w[inr][nov].sum()
            # 均值与分位数（抽样，避免内存过大）
            idx = np.random.default_rng(0).choice(len(X), size=min(len(X), 20000), replace=False)
            for j, f in enumerate(FEATS): vals[f].append(X[idx, j])
            wts.append(w[idx])
    if n_tot == 0:
        print(f"[skip] {label}: 无数据"); continue
    for f in FEATS:
        b, a, bw, aw_ = acc[f]
        rows.append(dict(column=key, label=label, predictor=f, unit=UNIT[f],
                         train_min=LO[f], train_max=HI[f],
                         pct_below=100 * b / n_tot, pct_above=100 * a / n_tot,
                         pct_below_areaw=100 * bw / w_tot, pct_above_areaw=100 * aw_ / w_tot,
                         n_values=int(n_tot), n_files=nfile))
        v = np.concatenate(vals[f]); ww = np.concatenate(wts)
        o = np.argsort(v); cw = np.cumsum(ww[o]) / ww.sum()
        q = lambda p: float(v[o][np.searchsorted(cw, p)])
        means.append(dict(column=key, label=label, predictor=f, unit=UNIT[f],
                          mean=swx[f] / sw, p05=q(0.05), p95=q(0.95)))
    rows.append(dict(column=key, label=label, predictor="Any predictor", unit="",
                     pct_below=np.nan, pct_above=100 * any_n / n_tot, pct_above_areaw=100 * any_w / w_tot,
                     n_values=int(n_tot), n_files=nfile))
    rows.append(dict(column=key, label=label, predictor="Any climate predictor (excl. CO2)", unit="",
                     pct_below=np.nan, pct_above=100 * cl_n / n_tot, pct_above_areaw=100 * cl_w / w_tot,
                     n_values=int(n_tot), n_files=nfile))
    rows.append(dict(column=key, label=label, predictor="Novel combination (NT2 > 1)", unit="",
                     pct_below=np.nan, pct_above=100 * nov_n / n_tot, pct_above_areaw=100 * nov_w / w_tot,
                     n_values=int(n_tot), n_files=nfile))
    print(f"[{label}] 文件 {nfile}，值 {int(n_tot):,}；任一越界 {100*any_n/n_tot:.2f}%；组合新颖 {100*nov_n/n_tot:.2f}%")

L = pd.DataFrame(rows); M = pd.DataFrame(means)
if S10_WEIGHT == "count":                                     # 只保留一套比例（计数），避免长表出现两套口径
    L = L.drop(columns=[c for c in L.columns if c.endswith("_areaw")])
L.round(3).to_csv(OUT / "extrapolation_annual_long.csv", index=False, encoding="utf-8-sig")

# ---------------- 论文表 S10 ----------------
order = [k for k, *_ in cols() if k in set(L.column)]
lab = dict(zip(L.column, L.label))
f2 = lambda x: "0" if x == 0 else (f"{x:.2f}" if x >= 0.01 else "<0.01")
tab = []
for p in FEATS + ["Any predictor", "Any climate predictor (excl. CO2)", "Novel combination (NT2 > 1)"]:
    r = {"Predictor": f"{p} ({UNIT[p]})" if p in UNIT else p,
         "Training range": f"{LO[p]:.2f}–{HI[p]:.2f}" if p in FEATS else ""}
    for k in order:
        x = L[(L.column == k) & (L.predictor == p)]
        if x.empty: continue
        x = x.iloc[0]
        if S10_WEIGHT == "area":
            r[lab[k]] = f"{f2(x.pct_below_areaw)} / {f2(x.pct_above_areaw)}" if p in FEATS else f2(x.pct_above_areaw)
        else:
            r[lab[k]] = f"{f2(x.pct_below)} / {f2(x.pct_above)}" if p in FEATS else f2(x.pct_above)
    tab.append(r)
S10 = pd.DataFrame(tab)
S10.to_csv(OUT / "table_S10_extrapolation.csv", index=False, encoding="utf-8-sig")

# ---------------- S10b：预测因子均值 ----------------
tb = []
for p in FEATS:
    r = {"Predictor": f"{p} ({UNIT[p]})"}
    for k in order:
        x = M[(M.column == k) & (M.predictor == p)]
        if x.empty: continue
        x = x.iloc[0]; nd = 0 if p in ("CTSR", "Rain", "CO2") else 2
        r[lab[k]] = f"{x['mean']:.{nd}f} [{x.p05:.{nd}f}, {x.p95:.{nd}f}]"
    tb.append(r)
S10b = pd.DataFrame(tb)
S10b.to_csv(OUT / "table_S10b_predictor_means.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n==== Table S10：低于 / 高于 训练范围（%，格点 × 年 × 成员）====")
print(S10.to_string(index=False))
print("\n==== Table S10b：均值 [5%, 95%] ====")
print(S10b.to_string(index=False))

_wtxt = ("Percentages are weighted by rice area (single/early and late rice pooled), i.e. they give the share of "
         "planted area × years × members" if S10_WEIGHT == "area" else "Percentages are unweighted counts of grid cell × year × member values")
cap = (f"Supplementary Table S10 | Projection inputs outside the training range. Percentage of predictor values "
       f"below / above the minimum and maximum of the training data (n = {len(T)} records). Values are evaluated "
       f"for every grid cell × year × ensemble member that enters the annual predictions, within the common "
       f"model-support domain: baseline {HL} (5 GCMs × 30 years), future {FL} (25 GGCM × GCM members × 30 years, "
       f"member-specific grain-filling windows) and, for comparison, future climate in the fixed baseline "
       f"grain-filling window (5 GCMs × 30 years). CO₂ enters as the annual atmospheric concentration of each "
       f"year (period means {C.co2_mean('historical', C.HIS_YEARS):.2f}, {C.co2_mean('ssp126', C.FUT_YEARS):.2f} "
       f"and {C.co2_mean('ssp585', C.FUT_YEARS):.2f} ppm). 'Any predictor' gives the share of values with at "
       f"least one predictor outside its training range ('Any climate predictor' excludes CO₂). "
       f"'Novel combination' gives the share of values that lie "
       f"within all single-predictor ranges but whose Mahalanobis distance to the training centroid exceeds the "
       f"largest distance found among the training records (NT2 > 1; Mesgaran et al. 2014). {_wtxt}.")
(OUT / "table_S10_caption.txt").write_text(cap, encoding="utf-8")
print(f"\n[saved] {OUT}/table_S10_extrapolation.csv / table_S10b_predictor_means.csv / "
      f"extrapolation_annual_long.csv / table_S10_caption.txt")
