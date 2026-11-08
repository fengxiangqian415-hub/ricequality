# ============================================================
# 26_province_drivers_1004.py —— 分省驱动因子：灌浆期气候、品质性状、Q/优质占比、整精米产量的变化，
#                                 与单位面积损失（含数量/等级效应）并表，用于"气温敏感区"讨论
# [1004] 运行顺序 26（依赖 16：province_frac_1km.npz、province_loss.csv；以及 02/03、04、10、11 的输出）
#   格网：只取该季、该情景经济核算域内的格（loss 栅格有值）；每格按 frac（1 km 水稻像元落在该省的比例）分给各省
#   [1013] 统一口径：气候、性状、Q 为省内汇总平均（config.SUMMARY_WEIGHT：area → frac × area_weight，cell → frac）；
#          优质占比、整精米产量为面积量（权重 = frac × area_weight，与 16 号单位面积损失同一口径）；
#          变化% = (未来均值 − 基准均值)/基准均值
# 输出（out_dir("loss")）：
#   province_drivers_long.csv   省 × 季别(ri1/ri2/all) × 情景 × 变量：baseline、future、delta、delta_pct
#   province_drivers_wide.csv   ri = all，每省一行：主要变量变化 + 面积、US$/ha、数量/等级效应 US$/ha
#   province_driver_corr.csv    参与排名的省份之间（16 号 in_ranking），单位面积损失（及数量/等级效应）
#                               与各变量变化的 Spearman 秩相关（省份数少，仅作描述）
# ============================================================
import numpy as np
import pandas as pd
import rasterio
import config_1004 as C

MIN_AREA_KHA = 500           # 与 16 号一致（实际以 province_loss.csv 的 in_ranking 为准）
OUT  = C.out_dir("loss")
ZDIR = C.out_dir("quality_zones") / C.PRED_MODE
HRY  = C.out_dir("head_rice_yield", thresh=False)
PRED = C.pred_dir(C.PRED_MODE, "withco2") / "ensemble"          # 含情景 CO2 的预测（与分区、经济一致）
CH, CF = C.CLIM_HIS / "ensemble", C.CLIM_FUT / "ensemble"

# 变量：名称 → (基准文件, 未来文件模板, 单位, 变化表示)
CLIM = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "GF_days": "gf_days",
        "TSR": "mean_daily_solar_MJ_m2", "CTSR": "cumul_solar_MJ_m2",
        "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct"}
CLIM_UNIT = {"Tavg": "degC", "Tdif": "degC", "GF_days": "d", "TSR": "MJ m-2 d-1",
             "CTSR": "MJ m-2", "Rain": "mm", "RHavg": "%"}
TRAITS = ["HRR", "CD", "PC", "Acontent", "BRR", "MRR", "CGR"]

VARS = {}
for v, lab in CLIM.items():
    VARS[v] = (lambda ri, lab=lab: CH / f"ensemble_historical_{ri}_{lab}_{C.HIS_P}_mean.tif",
               lambda ri, ssp, lab=lab: CF / f"ensemble_{ssp}_{ri}_{lab}_{C.FUT_P}_mean.tif", CLIM_UNIT[v])
for t in TRAITS:
    VARS[t] = (lambda ri, t=t: PRED / f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif",
               lambda ri, ssp, t=t: PRED / f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif", "%")
VARS["Q"]   = (lambda ri: ZDIR / f"Q_{ri}_baseline.tif", lambda ri, ssp: ZDIR / f"Q_{ri}_{ssp}_future.tif", "index")
VARS["premium_share"] = (lambda ri: ZDIR / f"zone_{ri}_baseline.tif",
                         lambda ri, ssp: ZDIR / f"zone_{ri}_{ssp}_future.tif", "fraction")
VARS["HRY"] = (lambda ri: HRY / f"head_rice_yield_{ri}_historical.tif",
               lambda ri, ssp: HRY / f"head_rice_yield_{ri}_{ssp}.tif", "t/ha")
ABS_DELTA = {"Tavg", "Tdif", "GF_days", "RHavg", "Q", "premium_share"}   # 宽表报绝对变化；其余（辐射、降水、性状、HRY）报 %
AREA_VARS = {"premium_share", "HRY"}                                      # 面积量：按面积；其余按汇总权重


def rd(p):
    if not p.exists(): return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None and not np.isnan(ds.nodata): a[a == ds.nodata] = np.nan
    return a


npz = np.load(OUT / "province_frac_1km.npz", allow_pickle=False)
U = [str(x) for x in npz["names"]]; K = len(U)
GRP = [str(x) for x in npz["groups"]] if "groups" in npz.files else U       # < 0.5 Mha 及非主产区合并为一行（16 号写入）
G = list(dict.fromkeys(GRP))                                                 # 展示分组（保持顺序、去重）
GIDX = {g: [k for k in range(K) if GRP[k] == g] for g in G}
FRAC = {ri: npz[f"frac_{ri}"].astype("float64") for ri in C.RI_TYPES}

# 累加器：(var, ssp, ri) → [Σw·base, Σw·fut, Σw]（K+1 行，最后一行 = National）
acc = {}
for ri in C.RI_TYPES:
    aw = rd(C.area_file(ri)); F = FRAC[ri]
    for ssp in C.SSPS:
        L = rd(OUT / f"loss_{ri}_{ssp}.tif")
        if L is None or aw is None: print(f"[skip] {ri} {ssp}: 缺 loss 或面积"); continue
        dom = np.isfinite(L) & np.isfinite(aw) & (aw > 0)
        for v, (fb, ff, unit) in VARS.items():
            b, f = rd(fb(ri)), rd(ff(ri, ssp))
            if b is None or f is None:
                if ssp == C.SSPS[0]: print(f"  [缺] {v} {ri}: {fb(ri).name if b is None else ff(ri, ssp).name}")
                continue
            m = dom & np.isfinite(b) & np.isfinite(f)
            if v == "premium_share":                                   # 分区 3 = 优质 → 面积占比
                m &= (b > 0) & (f > 0); b, f = (b == 3).astype(float), (f == 3).astype(float)
            wa = aw if (v in AREA_VARS or C.SUMMARY_WEIGHT == "area") else np.ones_like(aw)
            w = np.where(m, wa, 0.0); b0, f0 = np.where(m, b, 0.0), np.where(m, f, 0.0)
            wk = F * w                                                  # K × GH × GW
            row = np.stack([(wk * b0).sum((1, 2)), (wk * f0).sum((1, 2)), wk.sum((1, 2))], 1)
            row = np.vstack([row, [(w * b0).sum(), (w * f0).sum(), w.sum()]])
            for key in ((v, ssp, ri), (v, ssp, "all")):
                acc[key] = acc.get(key, 0) + row
    print("done:", ri)

rows = []
for (v, ssp, ri), a in acc.items():
    ag = [a[GIDX[g]].sum(0) for g in G] + [a[K]]                             # 按分组合并（分子、分母分别加总）
    for name, (sb, sf, sw) in zip(G + ["National"], ag):
        if sw <= 0: continue
        b, f = sb / sw, sf / sw
        rows.append(dict(province=name, ri=ri, ssp=ssp, variable=v, unit=VARS[v][2],
                         baseline=b, future=f, delta=f - b,
                         delta_pct=100 * (f - b) / b if abs(b) > 1e-12 else np.nan,
                         weight_sum=sw, weight_unit="km2" if (v in AREA_VARS or C.SUMMARY_WEIGHT == "area") else "cells"))
D = pd.DataFrame(rows).round(5)
D.sort_values(["province", "ri", "ssp", "variable"]).to_csv(OUT / "province_drivers_long.csv",
                                                           index=False, encoding="utf-8-sig")

# ---------------- 宽表（ri = all）----------------
PL = pd.read_csv(OUT / "province_loss.csv")
A = D[D.ri == "all"]
W = PL[["province"] + [c for c in PL.columns if c.endswith(("_area_kha", "_USD_ha")) or c == "in_ranking"]].copy()
for ssp in C.SSPS:
    for v in ["Tavg", "Tdif", "GF_days", "TSR", "Rain", "HRR", "CD", "PC", "Acontent", "Q", "premium_share", "HRY"]:
        s = A[(A.ssp == ssp) & (A.variable == v)].set_index("province")
        if s.empty: continue
        if v == "premium_share":
            W[f"{ssp}_premium_base_pct"] = W.province.map(100 * s.baseline).round(1)
            W[f"{ssp}_premium_fut_pct"]  = W.province.map(100 * s.future).round(1)
        elif v in ABS_DELTA:
            W[f"{ssp}_d{v}"] = W.province.map(s.delta).round(3)
        else:
            W[f"{ssp}_d{v}_pct"] = W.province.map(s.delta_pct).round(2)
W.to_csv(OUT / "province_drivers_wide.csv", index=False, encoding="utf-8-sig")

# ---------------- 省际 Spearman 相关（参与排名的省份）----------------
R = W[W.in_ranking == True]
corr = []
for ssp in C.SSPS:
    ys = [f"{ssp}_USD_ha", f"{ssp}_qty_USD_ha", f"{ssp}_grade_USD_ha"]
    xs = [c for c in W.columns if c.startswith(f"{ssp}_d") or c.startswith(f"{ssp}_premium")]
    for y in ys:
        for x in xs:
            sub = R[[y, x]].dropna()
            if len(sub) < 5: continue
            corr.append(dict(ssp=ssp, loss_metric=y.replace(f"{ssp}_", ""), driver=x.replace(f"{ssp}_", ""),
                             n_prov=len(sub), spearman_rho=round(sub[y].rank().corr(sub[x].rank()), 3)))
CR = pd.DataFrame(corr)
CR.to_csv(OUT / "province_driver_corr.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
show = ["province"] + [f"{s}_{c}" for s in C.SSPS for c in ("USD_ha", "qty_USD_ha", "grade_USD_ha", "dTavg", "dHRR_pct", "dCD_pct", "dPC_pct", "dHRY_pct")]
print(W[(W.in_ranking == True) | (W.province == "National")][[c for c in show if c in W.columns]].to_string(index=False))
if not CR.empty:
    print("\n单位面积损失与驱动变量的省际 Spearman 相关（|rho| ≥ 0.5）：")
    print(CR[CR.spearman_rho.abs() >= 0.5].to_string(index=False))
print(f"\n[saved] {OUT}/province_drivers_long.csv / province_drivers_wide.csv / province_driver_corr.csv")
