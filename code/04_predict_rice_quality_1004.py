import numpy as np
import pandas as pd
import xarray as xr
import joblib
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config_1004 import (RI_TYPES, GCMS, CROP_MODELS, SSPS, HIS, FUT, CLIM_HIS, CLIM_FUT, PRED_ROOT,
                         MODEL_DIR as _MODEL_DIR, F_TIF as _F_TIF, CO2_FILES as _CO2_FILES)
from pheno_common_1004 import N_LAT, N_LON, DST_TRANSFORM, DST_CRS, save_tif, read_tif

MODEL_DIR = _MODEL_DIR
F_TIF     = _F_TIF
CO2_FILES = _CO2_FILES
PERIODS   = {"historical": HIS, "ssp126": FUT, "ssp585": FUT}
CLIM_DIR  = {"historical": CLIM_HIS, "future": CLIM_FUT}
PRED_OUT  = PRED_ROOT
CO2_VARIANTS = ["withco2", "fixedco2"]
MODES     = ["annual", "meanclim"]
FORCE     = False

TARGETS       = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC", "PCA"]
BASE_FEATURES = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]
TYPE_AWARE    = {"Acontent"}
RASTER_LABEL  = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC",
                 "CTSR": "cumul_solar_MJ_m2", "Rain": "cumul_precip_mm",
                 "RHavg": "mean_humidity_pct",
                 "TSR": "mean_daily_solar_MJ_m2"}

CO2 = {}
for scen, fp in CO2_FILES.items():
    if not fp.exists():
        raise FileNotFoundError(f"CO2 文件不存在: {fp}（请把 CO2_FILES 改成服务器上的实际文件名）")
    df = pd.read_csv(fp, sep=r"\s+", header=None, names=["year", "co2_ppm"])
    CO2[scen] = dict(zip(df["year"].astype(int), df["co2_ppm"].astype(float)))
for scen, (y0, y1) in PERIODS.items():
    miss = [y for y in range(y0, y1 + 1) if y not in CO2[scen]]
    if miss:
        raise ValueError(f"{scen} CO2 缺年份: {miss[:5]}...")
    print(f"CO2 {scen} {y0}-{y1} 均值: {np.mean([CO2[scen][y] for y in range(y0, y1 + 1)]):.2f} ppm")

MODELS = {}
for t in TARGETS:
    fp = MODEL_DIR / f"{t}_Blending.joblib"
    if fp.exists():
        MODELS[t] = joblib.load(fp)
    else:
        print(f"⚠️ 模型不存在: {fp.name}")
print(f"已加载模型: {list(MODELS)}")


def model_features(model, target):
    """优先读取模型保存的特征名（含顺序）；否则用 BASE_FEATURES（+TYPE）"""
    for obj in (model, getattr(model, "meta_model", None), getattr(model, "final_estimator", None)):
        if obj is None:
            continue
        for attr in ("feature_names_in_", "feature_names", "features", "feature_cols"):
            v = getattr(obj, attr, None)
            if v is not None and len(v) > 0 and all(isinstance(x, str) for x in v):
                return list(v), f"{type(obj).__name__}.{attr}"
    base = list(BASE_FEATURES) + (["TYPE"] if target in TYPE_AWARE else [])
    return base, "BASE_FEATURES（模型内未存特征名）"


FEATS = {}
for t, m in MODELS.items():
    FEATS[t], src = model_features(m, t)
    print(f"  {t:9s} 特征 {FEATS[t]}  ← {src}")
    unknown = [f for f in FEATS[t] if f not in set(RASTER_LABEL) | {"CO2", "TYPE"}]
    if unknown:
        raise ValueError(f"{t} 含未知特征 {unknown}，请在 RASTER_LABEL 中补充对应栅格")

F_GRID = np.full((N_LAT, N_LON), np.nan, np.float32)
with rasterio.open(F_TIF) as src:
    reproject(source=rasterio.band(src, 1), destination=F_GRID,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=DST_TRANSFORM, dst_crs=DST_CRS, resampling=Resampling.nearest)
print(f"f 栅格已对齐  有值像元={int(np.isfinite(F_GRID).sum())}")


def predict_quality(clim, co2_val):
    """clim: {feature: 2D}；返回 {target: 2D}（与 v3 逻辑相同）"""
    valid = np.ones((N_LAT, N_LON), bool)
    for f in RASTER_LABEL:
        valid &= np.isfinite(clim[f])
    out = {t: np.full((N_LAT, N_LON), np.nan, np.float32) for t in MODELS}
    if not valid.any():
        return out
    cols = {"CO2": np.full(valid.sum(), co2_val, np.float32)}
    cols.update({f: clim[f][valid].astype(np.float32) for f in RASTER_LABEL})
    fv = np.where(np.isfinite(F_GRID[valid]), F_GRID[valid], 0.0).astype(np.float32)
    for t, model in MODELS.items():
        feats = FEATS[t]
        if "TYPE" in feats:
            X1 = pd.DataFrame({f: (np.full(valid.sum(), 1, np.float32) if f == "TYPE" else cols[f]) for f in feats})
            X2 = X1.assign(TYPE=np.float32(2))[feats]
            pred = fv * model.predict(X1).astype(np.float32) + (1.0 - fv) * model.predict(X2).astype(np.float32)
        else:
            pred = model.predict(pd.DataFrame({f: cols[f] for f in feats})).astype(np.float32)
        out[t][valid] = pred
    return out


def members(scen):
    """返回 [(成员名, 气候文件前缀路径函数)]"""
    if scen == "historical":
        return [(gcm, gcm) for gcm in GCMS]
    return [(f"{m}_{gcm}", (m, gcm)) for m in CROP_MODELS for gcm in GCMS]


def clim_paths(scen, key, ri, period):
    if scen == "historical":
        gcm = key
        d = CLIM_DIR["historical"]
        ann = d / "annual" / f"{gcm}_historical_{ri}_{period}_annual.nc"
        mean = lambda lab: d / gcm / f"{gcm}_historical_{ri}_{lab}_{period}_mean.tif"
    else:
        m, gcm = key
        d = CLIM_DIR["future"]
        ann = d / "annual" / scen / m / gcm / f"{m}_{gcm}_{scen}_{ri}_{period}_annual.nc"
        mean = lambda lab: d / scen / m / gcm / f"{m}_{gcm}_{scen}_{ri}_{lab}_{period}_mean.tif"
    return ann, mean


Y0B, Y1B = PERIODS["historical"]
CO2_BASE = float(np.mean([CO2["historical"][y] for y in range(Y0B, Y1B + 1)]))
print(f"固定 CO2 情景取基准均值: {CO2_BASE:.2f} ppm")

for co2var in CO2_VARIANTS:
    ROOT = PRED_OUT if co2var == "withco2" else PRED_OUT / "fixedco2"
    for scen, (y0, y1) in PERIODS.items():
        period = f"{y0}_{y1}"
        years = list(range(y0, y1 + 1))
        co2_of = (lambda yr, sc=scen: CO2[sc][yr]) if co2var == "withco2" else (lambda yr: CO2_BASE)
        co2_mean = float(np.mean([co2_of(y) for y in years]))
        for ri in RI_TYPES:
            print(f"\n{'=' * 60}\n[{co2var} | {scen} / {ri}] {period}  CO2 均值 {co2_mean:.1f}")
            ens = {mode: {t: [] for t in MODELS} for mode in MODES}
            for name, key in members(scen):
                ann_fp, mean_fp = clim_paths(scen, key, ri, period)
                for mode in MODES:
                    od = ROOT / mode / scen
                    outs = {t: od / f"{name}_{scen}_{ri}_{t}_{period}.tif" for t in MODELS}
                    if not FORCE and all(p.exists() for p in outs.values()):
                        for t, p in outs.items():
                            ens[mode][t].append(read_tif(p))
                        continue

                    if mode == "meanclim":
                        clim = {f: read_tif(mean_fp(lab)) for f, lab in RASTER_LABEL.items()}
                        res = predict_quality(clim, co2_mean)
                    else:
                        with xr.open_dataset(ann_fp) as ds:
                            C = {f: ds[lab].sel(year=years).values for f, lab in RASTER_LABEL.items()}
                        stack = {t: np.full((len(years), N_LAT, N_LON), np.nan, np.float32) for t in MODELS}
                        for i, yr in enumerate(years):
                            p = predict_quality({f: C[f][i] for f in RASTER_LABEL}, co2_of(yr))
                            for t in MODELS:
                                stack[t][i] = p[t]
                        ann_out = od / "annual_nc" / f"{name}_{scen}_{ri}_{period}_annual_pred.nc"
                        ann_out.parent.mkdir(parents=True, exist_ok=True)
                        xr.Dataset({t: (("year", "lat", "lon"), v) for t, v in stack.items()},
                                   coords={"year": years}).to_netcdf(
                            ann_out, encoding={t: {"zlib": True, "complevel": 4} for t in MODELS})
                        with np.errstate(all="ignore"):
                            res = {t: np.nanmean(v, axis=0).astype(np.float32) for t, v in stack.items()}
                    for t, arr in res.items():
                        save_tif(arr, outs[t])
                        ens[mode][t].append(arr)
                print(f"  [{name}] done  HRR(annual)均值 "
                      f"{np.nanmean(ens['annual']['HRR'][-1]) if 'HRR' in MODELS and 'annual' in MODES else float('nan'):.3f}")

            for mode in MODES:
                for t, lst in ens[mode].items():
                    if lst:
                        save_tif(np.nanmean(np.stack(lst), axis=0),
                                 ROOT / mode / "ensemble" / f"ensemble_{scen}_{ri}_{t}_{period}.tif")
                print(f"  [ensemble/{mode}] {len(ens[mode][next(iter(MODELS))])} 成员")

print(f"\n完成。输出: {PRED_OUT}/{{annual|meanclim}}/ 与 {PRED_OUT}/fixedco2/{{annual|meanclim}}/")
