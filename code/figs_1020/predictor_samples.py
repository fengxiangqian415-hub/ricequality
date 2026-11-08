# ============================================================
# predictor_samples.py —— Fig. 1d 峰峦图用的预测输入抽样（与补充图 S18/S19 的 08 号脚本同一口径）
#   · 预测输入 = 模型实际吃进去的“逐年 × 逐成员”灌浆期气候（annual），只取统一格网域 D，两季合并
#   · 每个“格网 × 年 × 成员”按 config.DIAG_WEIGHT 加权（与 S10 相同），抽 N 个值做核密度
#   · CO₂ 在空间上均一，用各时段逐年浓度（co2_series）
#   · 结果缓存到 main_figs_1020/_cache/，第二次起直接读
# ============================================================
from pathlib import Path

import numpy as np

RASTER_LABEL = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "CTSR": "cumul_solar_MJ_m2",
                "TSR": "mean_daily_solar_MJ_m2", "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct"}
PERIODS = ["baseline", "ssp126", "ssp585"]


def _rd(path):
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a[a == ds.nodata] = np.nan
    return a


def load_samples(cache_dir, n=40000, seed=0):
    """返回 {period: {feature: 抽样数组}}，period ∈ baseline / ssp126 / ssp585；CO2 为逐年浓度"""
    import config_1004 as C
    cache = Path(cache_dir) / f"predictor_samples_{C.TAG}_{C.HIS_P}_{C.FUT_P}_{C.DIAG_WEIGHT}_{n}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"[Fig. 1d] 读缓存 {cache}")
        return {p: {f: z[f"{p}__{f}"] for f in list(RASTER_LABEL) + ["CO2"] if f"{p}__{f}" in z} for p in PERIODS}

    import xarray as xr
    dom = {}
    for ri in ("ri1", "ri2"):
        a_ = _rd(C.area_file(ri))
        d_ = np.isfinite(_rd(C.support_file(ri))) & np.isfinite(a_) & (a_ > 0)
        dom[ri] = (d_, np.where(d_, 1.0 if C.DIAG_WEIGHT == "cell" else a_, 0.0))

    def files(p, ri):
        if p == "baseline":
            return [C.CLIM_HIS / "annual" / f"{g}_historical_{ri}_{C.HIS_P}_annual.nc" for g in C.GCMS]
        return [C.CLIM_FUT / "annual" / p / m / g / f"{m}_{g}_{p}_{ri}_{C.FUT_P}_annual.nc"
                for m in C.CROP_MODELS for g in C.GCMS]

    rng = np.random.default_rng(seed)
    out, flat = {}, {}
    for p in PERIODS:
        vals = {f: [] for f in RASTER_LABEL}; wts = []
        for ri in ("ri1", "ri2"):
            d_, w2 = dom[ri]
            for fp in files(p, ri):
                if not fp.exists():
                    continue
                with xr.open_dataset(fp) as ds:
                    A = {f: ds[lab].values[:, d_] for f, lab in RASTER_LABEL.items()}
                X = np.stack([A[f].ravel() for f in RASTER_LABEL], 1)
                ww = np.repeat(w2[d_][None, :], next(iter(A.values())).shape[0], axis=0).ravel()
                ok = np.isfinite(X).all(1)
                for j, f in enumerate(RASTER_LABEL):
                    vals[f].append(X[ok, j])
                wts.append(ww[ok])
        out[p] = {}
        if wts:
            w = np.concatenate(wts)
            idx = rng.choice(len(w), size=min(n, len(w)), replace=True, p=w / w.sum())
            for f in RASTER_LABEL:
                out[p][f] = np.concatenate(vals[f])[idx]
        scen = "historical" if p == "baseline" else p
        out[p]["CO2"] = np.asarray(C.co2_series(scen, C.HIS_YEARS if p == "baseline" else C.FUT_YEARS), float)
        print(f"[Fig. 1d] {p}: 抽样 {len(out[p].get('Tavg', []))} 个输入值")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{f"{p}__{f}": v for p, d in out.items() for f, v in d.items()})
    return out
