# -*- coding: utf-8 -*-
# ============================================================
# 21_export_paper_data_1004.py  (v2 → 1004)
# [1004] 运行顺序 21（最后一步）——全部路径、时段、阈值来自 config_1004
#   ★ 仅共同支撑区内、面积>0 的格网；CO2 由 ISIMIP3b 文件计算；气候含 TSR
#   ★ 新增 sheet：11_decomp_373（#373 分解）、12_uncertainty（ANOVA）、13_shap、14_bivariate、15_thresholds
#   输出：out_dir("excel")/paper_data_{TAG}.xlsx
# 把全流程各图/各步的可计算数据汇总成一个 Excel，供论文撰写取数。
#
# ★ [1013] 全流程唯一口径（取代 v2 的"逐格等权 mean"说法，那与 Methods 和 06/09/14 不一致）：
#   · 格网 = 统一格网域 D（03b）；全国汇总 = D 内面积加权平均（config.SUMMARY_WEIGHT）；中心值 = 集合平均场
#   · SHAP、方差分解、外推比例为模型诊断量，按格网等权（config.DIAG_WEIGHT）
#   · 相对变化 = (未来均值 − 基准均值) / 基准均值；基准与各情景同一组格网
#   · 面积量（经济、整精米总产、优质与各等级面积占比、迁移面积）按 1 km 聚合面积求和
#   · 成员范围 = 25 成员各自同法计算 → 中位数 [最小, 最大]
#   本脚本对品质、气候、物候不再自行重算，而是直接收录 09/06/07 的统计表（同一段代码算出的唯一数字）；
#   HRY 在经济核算格网上计算（与 16 号 Fig. 6 数值框同一数字）
#
# 用法（在服务器上）:
#   python export_paper_data.py
#   python export_paper_data.py --raw                 # 额外导出逐像元长表
#   python export_paper_data.py --base /path/ISIMIP3b \
#          --bench-tables /path/3_1_blending_RESULTS_V3AC/Tables \
#          --china /home/debian/pythonfile/china --out paper_data.xlsx
#
# 输出工作簿 paper_data.xlsx，sheet 见文件末尾 SHEETS 说明。
# ============================================================
import argparse, sys, json, csv as _csv
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError:
    sys.exit("需要 rasterio：pip install rasterio")

# ---------------- 默认路径（来自 config_1004，可命令行覆盖）----------------
import config_1004 as C
BASE_D   = str(C.BASE)
CHINA_D  = str(C.CHINA_SHP.parent)
BENCH_D  = str(C.MODEL_DIR.parent / "Tables")      # 模型训练输出的 Tables/（★ 与 results/Models 同级，确认）

TARGETS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC", "PCA"]
QUAL7   = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]     # 展示 7 指标
RI      = ["ri1", "ri2"]
RILAB   = {"ri1": "Single/early rice", "ri2": "Late rice"}   # [1105] 与正文统一为 Single/early
SSP     = ["ssp126", "ssp585"]
GRADES  = {3: "Premium", 2: "Moderate", 1: "Common"}
CLIM    = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC",
           "CTSR": "cumul_solar_MJ_m2", "TSR": "mean_daily_solar_MJ_m2",
           "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct",
           "GF_days": "gf_days"}                # CO2 是标量，另处理
CO2_VALS = {"baseline": C.co2_mean("historical", C.HIS_YEARS),
            "ssp126":   C.co2_mean("ssp126", C.FUT_YEARS),
            "ssp585":   C.co2_mean("ssp585", C.FUT_YEARS)}
EPS = 1e-9

# ==================== 栅格工具 ====================
def rd(path, clean_nonpos=False):
    p = Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    if clean_nonpos:
        a[a <= 0] = np.nan
    return a

def lonlat(path):
    with rasterio.open(path) as ds:
        H, W, T = ds.height, ds.width, ds.transform
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    lon = T.c + (cols + 0.5) * T.a
    lat = T.f + (rows + 0.5) * T.e
    return lon, lat

def wq(x, w, q):
    if x.size == 0: return np.nan
    o = np.argsort(x); x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cw, x))

def pct(fut, base):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (fut-base)/np.where(np.abs(base)<EPS, np.nan, base)*100.0

# ==================== 各模块 ====================
def m01_benchmark(BENCH):
    rows = []
    mats = {}
    for name, f in [("R2","r2_matrix.csv"),("RMSE","rmse_matrix.csv"),("MAE","mae_matrix.csv")]:
        p = Path(BENCH)/f
        mats[name] = pd.read_csv(p, index_col=0) if p.exists() else None
    if mats["R2"] is not None:
        for model in mats["R2"].index:
            for t in [c for c in mats["R2"].columns if c in TARGETS]:
                rows.append(dict(model=model, trait=t,
                    R2=mats["R2"].loc[model,t],
                    RMSE=(mats["RMSE"].loc[model,t] if mats["RMSE"] is not None else np.nan),
                    MAE=(mats["MAE"].loc[model,t] if mats["MAE"] is not None else np.nan)))
    bench = pd.DataFrame(rows)
    # 集成散点统计（每指标 train/test R2/RMSE/MAE）
    sca_rows = []
    pp = Path(BENCH)/"Blending_predictions.csv"
    if pp.exists():
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        pred = pd.read_csv(pp)
        for t in pred["Target"].unique():
            for st in ["Train","Test"]:
                s = pred[(pred.Target==t)&(pred.Set==st)]
                if len(s)==0: continue
                y,p = s["Observed"].values, s["Predicted"].values
                sca_rows.append(dict(trait=t, set=st, n=len(s),
                    R2=round(r2_score(y,p),4),
                    RMSE=round(float(np.sqrt(mean_squared_error(y,p))),4),
                    MAE=round(float(mean_absolute_error(y,p)),4)))
    return bench, pd.DataFrame(sca_rows)

def _aw(BASE, ri):
    """面积权重，限定在统一格网域 D 内（[1013]；与分区、经济核算口径一致）"""
    a = rd(C.area_file(ri)); s = rd(C.support_file(ri))
    if a is None: return None
    return np.where(np.isfinite(s) & np.isfinite(a) & (a > 0), a, np.nan)

def m02_quality(BASE, want_raw):
    """[1013] 收录 09 号 quality_change_summary.csv（面积加权、集合平均、成员中位数 [范围]、CO2 效应）"""
    qfig = C.FIGURES / f"rice_quality_{C.TAG}_{C.PRED_MODE}"
    p = qfig / "quality_change_summary.csv"
    qual = pd.read_csv(p) if p.exists() else pd.DataFrame()
    co2 = pd.DataFrame()
    if len(qual):
        w = qual[qual.co2 == "with_CO2"].set_index(["trait", "ri", "ssp"])
        f = qual[qual.co2 == "fixed_CO2"].set_index(["trait", "ri", "ssp"])
        co2 = pd.DataFrame({"baseline": w.base_mean,
                            "change_with_co2_pct": w.rel_change_pct,
                            "change_fixed_co2_pct": f.rel_change_pct.reindex(w.index),
                            "co2_effect_pp": w.rel_change_pct - f.rel_change_pct.reindex(w.index)}).reset_index()
        co2["baseline_reference"] = C.CO2_CF_BASELINE
    rawdf = None
    if want_raw:                                    # 逐像元长表（仅含 CO2；统一格网域）
        hd = C.pred_dir(C.PRED_MODE, "withco2") / "ensemble"
        raw = []
        for ri in RI:
            aw = _aw(BASE, ri)
            for t in TARGETS:
                base = rd(hd/f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif")
                if base is None or aw is None: continue
                lon, lat = lonlat(hd/f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif")
                for ssp in SSP:
                    fut = rd(hd/f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif")
                    if fut is None: continue
                    c = pct(fut, base); vc = np.isfinite(c) & np.isfinite(aw) & (aw > 0)
                    for lo, la, b_, f2, cc, ww in zip(lon[vc], lat[vc], base[vc], fut[vc], c[vc], aw[vc]):
                        raw.append((t, ri, ssp, round(float(lo), 3), round(float(la), 3),
                                    float(b_), float(f2), float(cc), float(ww)))
        rawdf = pd.DataFrame(raw, columns=["trait","ri","ssp","lon","lat","baseline","future","change_pct","area_weight"])
    return qual, co2, rawdf

def m04_05_zones(BASE):
    Z = C.out_dir("quality_zones") / C.PRED_MODE
    area_rows, mig_rows = [], []
    for ri in RI:
        aw = _aw(BASE, ri)
        zb = rd(Z/f"zone_{ri}_baseline.tif")
        if zb is None: continue
        w = aw if aw is not None else np.ones_like(zb)
        tot_layers = [("baseline", zb)] + [(s, rd(Z/f"zone_{ri}_{s}_future.tif")) for s in SSP]
        for scen, zf in tot_layers:
            if zf is None: continue
            base_ok = np.isfinite(zf)&np.isfinite(w)&(w>0)
            tot = float(np.sum(w[base_ok]))
            for code,g in GRADES.items():
                sel = base_ok&(zf==code)
                a = float(np.sum(w[sel]))
                area_rows.append(dict(ri=ri, scenario=scen, grade=g,
                                      area_wsum=a, area_pct=round(100*a/tot,3) if tot else np.nan,
                                      n_px=int(sel.sum())))
        for ssp in SSP:
            zf = rd(Z/f"zone_{ri}_{ssp}_future.tif")
            if zf is None: continue
            ok = np.isfinite(zb)&np.isfinite(zf)&np.isfinite(w)&(w>0)
            for cb,gb in GRADES.items():
                for cf,gf in GRADES.items():
                    sel = ok&(zb==cb)&(zf==cf)
                    mig_rows.append(dict(ri=ri, scenario=ssp, from_grade=gb, to_grade=gf,
                                         area_wsum=float(np.sum(w[sel])), n_px=int(sel.sum())))
    return pd.DataFrame(area_rows), pd.DataFrame(mig_rows)

def m06_migration_index(BASE):
    S = C.out_dir("migration_stats") / C.PRED_MODE
    rows = []
    for ri in RI:
        for ssp in SSP:
            jp = S/f"sankey_{ri}_{ssp}.json"
            if not jp.exists(): continue
            d = json.loads(jp.read_text())
            mg = d.get("migration",{})
            rt = mg.get("robust_top33_meanlat",{})
            pc = mg.get("premium_centroid",{})
            rows.append(dict(ri=ri, ssp=ssp,
                total_area_km2=d.get("total_area_km2"),
                robust_lat_base=rt.get("baseline"), robust_lat_future=rt.get("future"),
                robust_lat_delta_deg=rt.get("delta_deg"), robust_direction=rt.get("direction"),
                premium_centroid_shift_km=pc.get("shift_km"), premium_compass=pc.get("compass"),
                premium_centroid_reliable=pc.get("reliable")))
    return pd.DataFrame(rows)

def m07_hry(BASE):
    """[1013] 整精米产量（面积量）：经济核算格网（14 号 loss 有值处）上的总产与单产（总产 / 面积）
       相对变化 = 总产变化 %（与 16 号 Fig. 6 数值框"Δ Head rice yield"同一数字）"""
    H = C.out_dir("head_rice_yield", thresh=False); L = C.out_dir("loss")
    rows = []
    for ri in RI:
        aw = _aw(BASE, ri)
        yh = rd(H/f"head_rice_yield_{ri}_historical.tif")
        if yh is None or aw is None: continue
        for ssp in SSP:
            yf = rd(H/f"head_rice_yield_{ri}_{ssp}.tif"); lo = rd(L/f"loss_{ri}_{ssp}.tif")
            if yf is None or lo is None: continue
            k = np.isfinite(lo) & np.isfinite(aw) & np.isfinite(yh) & np.isfinite(yf)
            a = aw[k]; A = a.sum()
            bm, fm = float(np.sum(a*yh[k])/A), float(np.sum(a*yf[k])/A)
            rows.append(dict(ri=ri, scenario=ssp, n_cells=int(k.sum()), area_Mha=A/1e4,
                             baseline_t_ha=bm, future_t_ha=fm, change_pct=100*(fm-bm)/bm,
                             baseline_output_Mt=float(np.sum(a*100*yh[k]))/1e6,
                             future_output_Mt=float(np.sum(a*100*yf[k]))/1e6))
    return pd.DataFrame(rows)

def m08_loss(BASE):
    L = C.out_dir("loss")
    summ = pd.read_csv(L/"production_loss_summary.csv") if (L/"production_loss_summary.csv").exists() else pd.DataFrame()
    prov = pd.read_csv(L/"province_loss.csv") if (L/"province_loss.csv").exists() else pd.DataFrame()
    return summ, prov

def m09_climate(BASE):
    """[1013] 收录 06 号 climate_change_summary.csv（集合平均，面积加权；各情景同一组格网）+ CO2 标量"""
    p = C.FIGURES / f"climate_change_{C.TAG}" / "climate_change_summary.csv"
    d = pd.read_csv(p) if p.exists() else pd.DataFrame()
    co2 = pd.DataFrame([dict(var="CO2", ri="both", ssp=s_, baseline=CO2_VALS["baseline"], future=CO2_VALS[s_],
                             change=CO2_VALS[s_] - CO2_VALS["baseline"],
                             change_pct=100 * (CO2_VALS[s_] / CO2_VALS["baseline"] - 1)) for s_ in SSP])
    return pd.concat([co2, d], ignore_index=True)


def m10_phenology(BASE):
    """[1013] 收录 07 号 phenology_summary.csv：集合平均行 + 25 成员中位数 [最小, 最大]
       灌浆天数 = 窗口日数 floor(MA) − ceil(HE) + 1（与 05/06 的 gf_days 同一定义）"""
    p = C.FIGURES / f"phenology_{C.TAG}" / "phenology_summary.csv"
    if not p.exists(): return pd.DataFrame()
    st = pd.read_csv(p)
    e = st[st.member == "ensemble_mean"].drop(columns="member").copy()
    m = st[st.member != "ensemble_mean"]
    for c in ["HE_change", "MA_change", "GF_change", "GF_change_pct"]:
        if c in m:
            g = m.groupby(["ri", "ssp"])[c]
            e = e.merge(g.median().rename(f"{c}_member_median").reset_index(), on=["ri", "ssp"], how="left")
            e = e.merge(g.min().rename(f"{c}_member_min").reset_index(), on=["ri", "ssp"], how="left")
            e = e.merge(g.max().rename(f"{c}_member_max").reset_index(), on=["ri", "ssp"], how="left")
    return e


def extra_sheets():
    """1004 新增：直接汇入各步已生成的统计表（缺文件自动跳过）"""
    def rcsv(p):
        p = Path(p)
        return pd.read_csv(p) if p.exists() else None
    th = C.thresholds_file()
    th_df = None
    if th.exists():
        d = json.loads(th.read_text())
        th_df = pd.DataFrame([dict(ri=ri, Q_LOW=v["Q_LOW"], Q_HIGH=v["Q_HIGH"], mode=d.get("mode"))
                              for ri, v in d["thresholds"].items()])
    zdir = C.out_dir("quality_zones") / C.PRED_MODE
    zshare = [rcsv(zdir / f"member_zone_share_{ri}.csv") for ri in C.RI_TYPES]
    zshare = pd.concat([z.assign(ri=ri) for z, ri in zip(zshare, C.RI_TYPES) if z is not None]) \
        if any(z is not None for z in zshare) else None
    qfig  = C.FIGURES / f"rice_quality_{C.TAG}_{C.PRED_MODE}"
    zdiag = C.out_dir("quality_zones") / "diagnosis"
    mdir  = C.out_dir("migration_stats") / C.PRED_MODE

    def cat_csv(d, pat, ssp=False):
        """按 ri（及 ssp）拼接同类小表，加 ri / ssp 列"""
        parts = []
        for ri in C.RI_TYPES:
            for sp in (C.SSPS if ssp else [None]):
                p = Path(d) / pat.format(ri=ri, ssp=sp)
                if p.exists():
                    t = pd.read_csv(p); t.insert(0, "ri", ri)
                    if sp: t.insert(1, "ssp", sp)
                    parts.append(t)
        return pd.concat(parts, ignore_index=True) if parts else None

    def texts():
        """各脚本写出的正文/图注句子，逐行放进一个 sheet"""
        src = [(qfig / "quality_change_text.txt", "09 quality change"),
               (C.out_dir("sensitivity") / "econ_qhigh_text.txt", "25 econ Q_HIGH sensitivity"),
               (C.out_dir("loss") / "econ_accounting_text.txt", "28 econ accounting (#910)"),
               (C.FIGURES / f"feature_distribution_{C.TAG}" / "table_S10_caption.txt", "29 Table S10 caption")]
        rows = [dict(source=lab, line=i + 1, text=ln) for p, lab in src if Path(p).exists()
                for i, ln in enumerate(Path(p).read_text(encoding="utf-8").splitlines()) if ln.strip()]
        return pd.DataFrame(rows) if rows else None

    return [
        (rcsv(C.OUTPUT / f"decomposition_373_{C.TAG}" / "summary.csv"), "11_decomp_373"),
        (rcsv(C.OUTPUT / f"decomposition_373_{C.TAG}" / "members.csv"), "11_decomp_373_members"),
        (rcsv(C.fig_dir("uncertainty", thresh=False) / "uncertainty_4factor_wide.csv"), "12_uncertainty"),
        (rcsv(C.fig_dir("shap", thresh=False) / "shap_mean_abs.csv"), "13_shap"),
        (rcsv(C.out_dir("loss") / "bivariate_summary.csv"), "14_bivariate"),
        (th_df, "15_thresholds"),
        (zshare, "16_zone_member_share"),
        # [1013] 17–19 已并入 02/09/10（同一张表，不再重复收录）；20（08 号旧外推表）已由 30_S10 取代
        (rcsv(C.domain_file("ri1").parent / "domain_diagnostics.csv"), "00_domain_diagnostics"),
        (rcsv(C.OUTPUT / f"consistency_audit_{C.TAG}" / "consistency_audit.csv"), "00_consistency_audit"),
        (rcsv(C.out_dir("sensitivity") / "q_sensitivity_checks.csv"), "21_q_sensitivity"),
        (rcsv(C.out_dir("sensitivity") / "econ_accounting_table.csv"), "22_econ_members"),
        (rcsv(C.out_dir("sensitivity") / "econ_qhigh_long.csv"), "23_econ_qhigh"),
        (rcsv(C.out_dir("sensitivity") / "province_loss_members.csv"), "24b_prov_loss_members"),   # [1106] Fig. 4 箱线图
        (rcsv(C.out_dir("sensitivity") / "valuation_tiers_summary.csv"), "43_valuation_tiers"),      # [1106] 补表 13
        (rcsv(C.out_dir("sensitivity") / "valuation_tiers_members.csv"), "43_valuation_members"),
        (rcsv(C.out_dir("loss") / "province_loss_long.csv"), "24_prov_loss_long"),
        (rcsv(C.out_dir("loss") / "province_drivers_wide.csv"), "25_prov_drivers"),
        (rcsv(C.out_dir("loss") / "province_driver_corr.csv"), "26_prov_driver_corr"),
        (rcsv(C.out_dir("loss") / "province_attribution_wide.csv"), "27_prov_attribution"),
        (rcsv(C.out_dir("loss") / "econ_accounting_ssp126.csv"), "28_accounting_ssp126"),
        (rcsv(C.out_dir("loss") / "econ_accounting_ssp585.csv"), "28_accounting_ssp585"),
        (rcsv(C.out_dir("loss") / "econ_accounting_numeric.csv"), "28_accounting_numeric"),
        (rcsv(C.out_dir("loss") / "bivariate_breaks.csv"), "29_fig6_map_breaks"),
        (rcsv(C.FIGURES / f"feature_distribution_{C.TAG}" / "table_S10_extrapolation.csv"), "30_S10_extrapolation"),
        (rcsv(C.FIGURES / f"feature_distribution_{C.TAG}" / "table_S10b_predictor_means.csv"), "30_S10b_predictor_means"),
        (rcsv(C.FIGURES / f"feature_distribution_{C.TAG}" / "extrapolation_annual_long.csv"), "30_S10_long"),
        (rcsv(C.out_dir("sensitivity") / "annual_classification.csv"), "41_annual_classification"),          # [1031] 31 号
        (rcsv(C.out_dir("sensitivity") / "annual_classification_members.csv"), "41_annual_class_members"),
        (rcsv(C.out_dir("sensitivity") / "quality_error_mc.csv"), "42_quality_error_mc"),                   # [1031] 32 号
        (rcsv(C.out_dir("sensitivity") / "table_S6_thresholds_from_limits.csv"), "43_table_S6_limits"),   # [1031] 35 号
        # ---- 以下为此前未进 Excel 的明细/诊断表（补齐，正文与审稿回复取数都在本文件里）----
        (rcsv(qfig / "quality_change_members.csv"), "31_quality_change_members"),
        (rcsv(zdiag / "threshold_sensitivity.csv"), "32_zone_threshold_sens"),
        (rcsv(zdiag / "Q_distribution.csv"), "32_zone_Q_distribution"),
        (rcsv(zdiag / "baseline_percentile_zones.csv"), "32_zone_baseline_pct"),
        (rcsv(zdiag / "regional.csv"), "32_zone_regional"),
        (cat_csv(mdir, "transition_{ri}_{ssp}.csv", ssp=True), "33_migration_transition"),
        (cat_csv(mdir, "migration_{ri}_{ssp}.csv", ssp=True), "33_migration_shift"),
        (cat_csv(mdir, "class_centroid_{ri}.csv"), "33_class_centroid"),
        (rcsv(C.fig_dir("uncertainty", thresh=False) / "uncertainty_4factor_long.csv"), "34_uncertainty_long"),
        (rcsv(C.out_dir("sensitivity") / "q_sensitivity_long.csv"), "35_q_sensitivity_long"),
        (rcsv(C.out_dir("sensitivity") / "econ_members_long.csv"), "36_econ_members_long"),
        (rcsv(C.out_dir("sensitivity") / "econ_qhigh_table.csv"), "37_econ_qhigh_table"),
        (rcsv(C.out_dir("loss") / "province_drivers_long.csv"), "38_prov_drivers_long"),
        (rcsv(C.out_dir("loss") / "province_attr_co2_long.csv"), "39_prov_attr_co2_long"),
        (rcsv(C.out_dir("loss") / "province_attr_shap_long.csv"), "39_prov_attr_shap_long"),
        (texts(), "40_text_snippets"),
    ]


def readme():
    return pd.DataFrame([
        ("★ 统一口径（1013 定稿）", "格网 = 统一格网域 D（03b：基准支撑区 ∩ 分季面积>0）；中心值 = 集合平均场；全国汇总（气候、物候、品质、Q）= D 内面积加权；"
                                  "相对变化 = (未来均值 − 基准均值)/基准均值；基准与各情景同一组格网；成员范围 = 25 成员同法计算的中位数 [最小, 最大]；"
                                  "SHAP、方差分解、外推比例按格网等权；面积量（经济、整精米总产、等级面积占比、迁移）按 1 km 聚合面积求和"),
        ("灌浆天数", "窗口 [HE, MA] 内日数 = floor(MA) − ceil(HE) + 1（05/06/07/26 同一定义，与 CTSR、Rain 累计窗口一致）"),
        ("CO2 反事实", "固定 CO2 情景的变化相对于同样在固定 CO2 下预测的基准；CO2 效应 = 随情景 CO2 的变化 − 固定 CO2 的变化（百分点）"),
        ("经济核算格网", "D ∩ 基准与两个情景的 HRY、分级都有值（14/16/24/25/26/27/28 同一组格网；基准产值唯一）"),
        ("00_domain_diagnostics", "03b：D 的格网数/面积；未来成员齐全面积占比；全部格网 vs 成员齐全格网的基准均温"),
        ("00_consistency_audit", "30 号：跨脚本同一量的核对结果（✔ 一致 / ✘ 不一致）"),
        ("01_model_benchmark", "§3.1 每模型每指标 R2/RMSE/MAE(来自 Tables/*_matrix.csv) + 集成散点 train/test 统计"),
        ("02_quality_change", "§2.2 品质：09 号 quality_change_summary（含/固定 CO2；基准/未来均值、相对变化、格网分位数、"
                              "下降面积占比、成员中位数 [范围]、同号比例、co2_effect_pp）"),
        ("03_co2_effect", "§2.2/4.4 由 02 派生：随情景 CO2 与固定 CO2 的相对变化及其差（百分点）"),
        ("04_zone_area", "§2.4 各情景各等级面积占比（集合平均性状 → Q → 分级；面积量）"),
        ("05_zone_migration", "§2.4 基线→未来 3×3 迁移面积矩阵"),
        ("06_migration_index", "§2.4 稳健前 1/3 纬度位移 + 优质质心位移（13 号 sankey json）"),
        ("07_head_rice_yield", "§2.5 整精米产量（面积量）：经济核算格网上的单产（总产/面积，t/ha）、总产（Mt）与变化 %（= Fig. 6 数值框）"),
        ("08_econ_loss / _prov", "§2.5 全国损失（数量/等级分解、价比敏感性）+ 分省损失（1 km 像元拆分；总量、US$/ha）"),
        ("09_climate_change", "§2.2/附图 S1–S6 气候：06 号 climate_change_summary（含 GF_days）+ CO2 时段均值"),
        ("10_phenology", "附图 物候：07 号 phenology_summary 的集合行 + 成员中位数/最小/最大"),
        ("raw_quality(可选)", "--raw 时导出米质逐像元长表（仅含 CO2；统一格网域）"),
        ("11_decomp_373(_members)", "#373 灌浆期气候分解（05 号；基准、总变化与 09_climate_change 相同）"),
        ("12_uncertainty / 34_uncertainty_long", "方差分解（GGCM/GCM/SSP/RS；对象为变化量；逐像元等权汇总）（20 号）"),
        ("13_shap", "SHAP mean|SHAP| 与 mean SHAP，逐格等权（19 号；输入为集合时段均值特征；AC 按籼稻占比 f 加权）"),
        ("14_bivariate / 29_fig6_map_breaks", "Fig.6 数值框三个全国量；地图图例数值断点（16 号）"),
        ("15_thresholds / 16_zone_member_share", "分区阈值；25 成员各等级面积占比（分母 = 该成员有值面积，另列覆盖率）（11 号）"),
        ("21_q_sensitivity / 35_q_sensitivity_long", "PC 权重 × 阈值模式敏感性（23 号，#774/#809）"),
        ("22_econ_members / 36_econ_members_long", "25 成员经济核算：汇总（中位数 [范围]）/ 逐成员明细（24 号，#910）"),
        ("23_econ_qhigh / 37_econ_qhigh_table", "Q_HIGH 0.62–0.72 经济敏感性（25 号）"),
        ("24_prov_loss_long / 08_econ_loss_prov", "分省损失（1 km 像元拆分；<0.5 Mha 合并为一行）（16 号）"),
        ("25_prov_drivers / 38_prov_drivers_long / 26_prov_driver_corr", "分省气候、性状、Q、HRY 变化与单位面积损失（26 号）"),
        ("27_prov_attribution / 39_prov_attr_*", "分省归因：CO2 vs 气候（固定 CO2 反事实）；各气象因子 ΔSHAP（27 号）"),
        ("28_accounting_ssp126/585/numeric", "#910 经济核算表：每个情景一张，集合估计 [25 成员最小, 最大]（28 号）"),
        ("30_S10_*", "表 S10：逐年×逐成员输入越界比例（格点×年×成员计数）、组合新颖性、固定窗口对照；S10b 因子均值（29 号）"),
        ("31_quality_change_members", "25 成员品质变化（09 号）"),
        ("32_zone_*", "分区诊断：Q 分布、阈值敏感性、基准分位分区、分区域（12 号）"),
        ("33_migration_* / 33_class_centroid", "优质区迁移：转移矩阵、纬度/质心位移、各等级质心（13 号）"),
        ("40_text_snippets", "各脚本生成的正文句子与图注（09/25/28/29 号）"),
        ("41_annual_classification(_members)", "汇总顺序敏感性：先平均再分级 vs 逐成员 vs 逐年逐成员先分级再平均；期望单价核算（31 号）"),
        ("42_quality_error_mc", "品质模型误差蒙特卡洛传播（iid / 逐格持续偏差 / 共同偏差）：优质占比、面积损失、等级效应的 5–95 百分位（32 号）"),
        ("43_table_S6_limits", "新 Table S6：国标三级限值换算的 Q 阈值（籼/粳/平均；PC 上限 9.0 与 11.0）及实测/CV 预测 Q 上的 AUC、敏感度、特异度（35 号）"),
    ], columns=["项","说明"])

# ==================== 主程序 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_D)
    ap.add_argument("--china", default=CHINA_D)
    ap.add_argument("--bench-tables", default=BENCH_D)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--raw-max", type=int, default=800000)
    a = ap.parse_args()
    BASE = a.base
    # ← 新增：默认输出到 BASE/output/excel/paper_data.xlsx；--out 给相对名则也落到该目录
    EXCEL_DIR = C.out_dir("excel")
    EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    if a.out is None:
        a.out = str(EXCEL_DIR / f"paper_data_{C.TAG}.xlsx")
    elif not Path(a.out).is_absolute():
        a.out = str(EXCEL_DIR / a.out)

    print("== 01 benchmark =="); bench, scatter = m01_benchmark(a.bench_tables)
    print("== 02 quality  =="); qual, co2eff, qraw = m02_quality(BASE, a.raw)
    print("== 04/05 zones =="); zarea, zmig = m04_05_zones(BASE)
    print("== 06 migration=="); mig = m06_migration_index(BASE)
    print("== 07 HRY      =="); hry = m07_hry(BASE)
    print("== 08 loss     =="); loss, prov = m08_loss(BASE)
    print("== 09 climate  =="); clim = m09_climate(BASE)
    print("== 10 phenology=="); phen = m10_phenology(BASE)

    # 用 openpyxl 直接写（避免个别 pandas 版本 ExcelWriter 的兼容问题）
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils import get_column_letter
    HDR_F = Font(bold=True, color="FFFFFF"); HDR_B = PatternFill("solid", fgColor="2C6FB0")
    def add_sheet(wb, name, df):
        if df is None or df.empty: return
        ws = wb.create_sheet(name[:31])
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(["" if (isinstance(v,float) and pd.isna(v)) else v for v in r])
        for c in range(1, df.shape[1]+1):
            cell = ws.cell(row=1, column=c); cell.font = HDR_F; cell.fill = HDR_B
            cell.alignment = Alignment(horizontal="left")
            _l = df.iloc[:,c-1].astype(str).str.len().max() if len(df) else 10
            w = max(10, min(40, int(_l if pd.notna(_l) else 10) + 2))
            ws.column_dimensions[get_column_letter(c)].width = w
        ws.freeze_panes = "A2"
    wb = Workbook(); wb.remove(wb.active)
    add_sheet(wb, "说明", readme())
    for df, nm in [(bench,"01_model_benchmark"),(scatter,"01_ensemble_scatter"),
                   (qual,"02_quality_change"),(co2eff,"03_co2_effect"),
                   (zarea,"04_zone_area"),(zmig,"05_zone_migration"),
                   (mig,"06_migration_index"),(hry,"07_head_rice_yield"),
                   (loss,"08_econ_loss"),(prov,"08_econ_loss_prov"),
                   (clim,"09_climate_change"),(phen,"10_phenology")] + extra_sheets():
        add_sheet(wb, nm, df)
    if a.raw and qraw is not None:
        if len(qraw) <= a.raw_max: add_sheet(wb, "raw_quality", qraw)
        else:
            qraw.to_csv(Path(a.out).with_name("raw_quality.csv"), index=False)
            print(f"[raw] {len(qraw)} 行超上限，转存 raw_quality.csv")
    if not wb.sheetnames:
        wb.create_sheet("说明")   # 保底，避免空工作簿报错
    wb.save(a.out)
    print(f"[saved] {a.out}")

if __name__ == "__main__":
    main()
