# ============================================================
# 18_export_shap_features_1004.py —— 为 SHAP 导出格网特征矩阵（配置来自 config_1004）
#   每个格网一行：CO2（时段均值）+ 灌浆期气候（集合平均的多年均值）+ f_indica（籼稻占比，仅供 Acontent 按 f 加权）
#   历史：02 的 5 GCM 集合平均；未来：03 的 25 成员集合平均
#   仅统一格网域 D 内格网（03b：支撑区 ∩ 面积>0）；不做籼/粳二分（与 04 预测一致：Acontent = f·A(籼) + (1−f)·A(粳)）
#   [1013] 另写出分季面积 w（km²），仅当 config.DIAG_WEIGHT = "area" 时 19 号才用它加权（定稿为逐格等权）
#   注：SHAP 输入为时段均值特征（集合多年均值 + 时段均值 CO2），而主预测为逐年预测再平均；正文 4.9 需注明
# 输出：out_dir("shap_features", thresh=False)/features_{historical|ssp126|ssp585}_{ri}.csv
# ============================================================
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from config_1004 import (RI_TYPES, SSPS, HIS_YEARS, FUT_YEARS, HIS_P, FUT_P, CLIM_HIS, CLIM_FUT, F_TIF,
                         out_dir, support_file, area_file, co2_mean)
from pheno_common_1004 import N_LAT, N_LON, LAT_VALS, LON_VALS, DST_TRANSFORM, DST_CRS, read_tif

OUT = out_dir("shap_features", thresh=False); OUT.mkdir(parents=True, exist_ok=True)
CLIM = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "CTSR": "cumul_solar_MJ_m2",
        "TSR": "mean_daily_solar_MJ_m2", "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct"}

F = np.full((N_LAT, N_LON), np.nan, np.float32)
with rasterio.open(F_TIF) as src:
    reproject(rasterio.band(src, 1), F, src_transform=src.transform, src_crs=src.crs,
              dst_transform=DST_TRANSFORM, dst_crs=DST_CRS, resampling=Resampling.nearest)
FIND = np.where(np.isfinite(F), F, 0.0).astype(np.float32)     # 与 04 预测相同：缺失按 f=0（粳）
LA, LO = np.meshgrid(LAT_VALS, LON_VALS, indexing="ij")

CO2 = {"historical": co2_mean("historical", HIS_YEARS),
       **{s: co2_mean(s, FUT_YEARS) for s in SSPS}}
print("CO2 时段均值:", {k: round(v, 1) for k, v in CO2.items()})

for ri in RI_TYPES:
    sup = np.isfinite(read_tif(support_file(ri)))
    AW = read_tif(area_file(ri))
    sup &= np.isfinite(AW) & (AW > 0)
    for period in ["historical"] + SSPS:
        if period == "historical":
            arr = {k: read_tif(CLIM_HIS / "ensemble" / f"ensemble_historical_{ri}_{lab}_{HIS_P}_mean.tif")
                   for k, lab in CLIM.items()}
        else:
            arr = {k: read_tif(CLIM_FUT / "ensemble" / f"ensemble_{period}_{ri}_{lab}_{FUT_P}_mean.tif")
                   for k, lab in CLIM.items()}
        ok = sup.copy()
        for a in arr.values():
            ok &= np.isfinite(a)
        df = pd.DataFrame({"lat": LA[ok], "lon": LO[ok], "CO2": CO2[period],
                           **{k: a[ok] for k, a in arr.items()}, "f_indica": FIND[ok], "w": AW[ok]})
        fp = OUT / f"features_{period}_{ri}.csv"
        df.to_csv(fp, index=False, float_format="%.4f")
        print(f"  [{ri}/{period}] {len(df)} 格网  籼稻占比均值 {df.f_indica.mean():.2f}  → {fp.name}")
print(f"输出: {OUT}")
