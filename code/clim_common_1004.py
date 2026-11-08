import numpy as np
import pandas as pd
import xarray as xr

from pheno_common_1004 import BASE, N_LAT, N_LON, LAT_VALS, LON_VALS, OUT_HIS, read_tif
from config_1004 import HIS_YEARS, hist_pheno_tag

CN_HIST = BASE / "climate_daily_china"
CN_FUT  = BASE / "climate_daily_china_future"

GCM_UPPER = {"gfdl-esm4": "GFDL-ESM4", "ipsl-cm6a-lr": "IPSL-CM6A-LR",
             "mpi-esm1-2-hr": "MPI-ESM1-2-HR", "mri-esm2-0": "MRI-ESM2-0",
             "ukesm1-0-ll": "UKESM1-0-LL"}
GCM_MEM = {"gfdl-esm4": "gfdl-esm4_r1i1p1f1", "ipsl-cm6a-lr": "ipsl-cm6a-lr_r1i1p1f1",
           "mpi-esm1-2-hr": "mpi-esm1-2-hr_r1i1p1f1", "mri-esm2-0": "mri-esm2-0_r1i1p1f1",
           "ukesm1-0-ll": "ukesm1-0-ll_r1i1p1f2"}

VAR_AGG = {"tas": "mean", "tasmax": "mean", "tasmin": "mean",
           "pr": "sum", "rsds": "sum", "hurs": "mean"}
VAR_LABELS = {"tas": "mean_temp_degC", "tasmax": "mean_tmax_degC", "tasmin": "mean_tmin_degC",
              "tmpdiff": "mean_tmpdiff_degC", "pr": "cumul_precip_mm",
              "rsds": "cumul_solar_MJ_m2", "hurs": "mean_humidity_pct"}
EXTRA_LABELS = {"days": "gf_days", "rsds_daily": "mean_daily_solar_MJ_m2"}


def clean_window(he, ma, mask):
    """套 mask；剔除 HE/MA 缺失或 HE>=MA"""
    bad = ~mask | ~np.isfinite(he) | ~np.isfinite(ma) | (he >= ma)
    he, ma = he.copy(), ma.copy()
    he[bad] = np.nan
    ma[bad] = np.nan
    return he, ma


def baseline_window(ri, mask, years=None):
    """唯一历史物候（2000 年前 ens25 + 2000 年起 observed）基准期多年平均窗口 → 组 A′/C 的固定窗口"""
    years = HIS_YEARS if years is None else years
    H, M = [], []
    for yr in years:
        tag = hist_pheno_tag(yr)
        fh, fm = OUT_HIS / f"{ri}_HE_{yr}_{tag}.tif", OUT_HIS / f"{ri}_MA_{yr}_{tag}.tif"
        if fh.exists() and fm.exists():
            he, ma = clean_window(read_tif(fh), read_tif(fm), mask)
            H.append(he)
            M.append(ma)
    with np.errstate(all="ignore"):
        he_b, ma_b = np.nanmean(np.stack(H), axis=0), np.nanmean(np.stack(M), axis=0)
    return clean_window(he_b, ma_b, mask)


def load_daily(gcm, scen, var, y0, y1):
    """读取某 GCM×情景×变量 在 [y0, y1] 的中国区日值，返回 (values[T,lat,lon] float32, DatetimeIndex)"""
    up, mem = GCM_UPPER[gcm], GCM_MEM[gcm]
    d = CN_HIST / up if scen == "historical" else CN_FUT / scen / up
    files = []
    for fp in sorted(d.glob(f"{mem}_w5e5_{scen}_{var}_china_daily_*.nc")):
        s, e = map(int, fp.stem.split("_daily_")[-1].split("_"))
        if e >= y0 and s <= y1:
            files.append(fp)
    if not files:
        raise FileNotFoundError(f"无日值缓存: {d}/{mem}_w5e5_{scen}_{var}_china_daily_*.nc ({y0}-{y1})")
    arrs, times = [], []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            da = ds[var].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
            if da.sizes["time"] == 0:
                continue
            if da.shape[1:] != (N_LAT, N_LON):
                raise ValueError(f"格网不符 {da.shape[1:]}: {fp.name}")
            arrs.append(da.values.astype(np.float32))
            times.append(pd.DatetimeIndex(da.indexes["time"].to_datetimeindex()
                                          if hasattr(da.indexes["time"], "to_datetimeindex")
                                          else da.indexes["time"]))
    t = times[0].append(times[1:]) if len(times) > 1 else times[0]
    order = np.argsort(t.values)
    return np.concatenate(arrs, axis=0)[order], t[order]


def window_stats(block, doy, he, ma, agg):
    """block[T,lat,lon] 某一年日值；he/ma 为 DOY 栅格。返回 (统计值, 窗口天数)"""
    in_win = ((doy[:, None, None] >= he[None]) & (doy[:, None, None] <= ma[None])
              & np.isfinite(block))
    cnt = in_win.sum(axis=0)
    s = np.where(in_win, block, 0.0).sum(axis=0, dtype=np.float64)
    out = s / np.where(cnt > 0, cnt, 1) if agg == "mean" else s
    out = np.where(cnt > 0, out, np.nan).astype(np.float32)
    return out, np.where(cnt > 0, cnt, np.nan).astype(np.float32)


def extract_years(daily, times, years, pheno_of_year):
    """
    daily: {var: (values, times)}；pheno_of_year(yr) -> (he, ma) 或 None
    返回 {label: [year, lat, lon] 数组}（缺年为 NaN），标签含 tmpdiff、gf_days、日均辐射
    """
    ny = len(years)
    out = {lab: np.full((ny, N_LAT, N_LON), np.nan, np.float32)
           for lab in list(VAR_LABELS.values()) + list(EXTRA_LABELS.values())}
    year_idx = {yr: np.where(times.year == yr)[0] for yr in years}
    for i, yr in enumerate(years):
        ph = pheno_of_year(yr)
        if ph is None or len(year_idx[yr]) == 0:
            continue
        he, ma = ph
        idx = year_idx[yr]
        doy = times[idx].dayofyear.values
        for var, agg in VAR_AGG.items():
            v, cnt = window_stats(daily[var][idx], doy, he, ma, agg)
            out[VAR_LABELS[var]][i] = v
            if var == "tas":
                out[EXTRA_LABELS["days"]][i] = cnt
        out[VAR_LABELS["tmpdiff"]][i] = out[VAR_LABELS["tasmax"]][i] - out[VAR_LABELS["tasmin"]][i]
        out[EXTRA_LABELS["rsds_daily"]][i] = out[VAR_LABELS["rsds"]][i] / out[EXTRA_LABELS["days"]][i]
    return out


def save_annual_nc(res, years, path, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({k: (("year", "lat", "lon"), v) for k, v in res.items()},
                    coords={"year": np.asarray(years, int), "lat": LAT_VALS, "lon": LON_VALS},
                    attrs=attrs)
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 4, "dtype": "float32"} for k in res})
    tmp.replace(path)


def period_mean(arr):
    """多年均值 + 有效年数"""
    n = np.isfinite(arr).sum(axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        m = np.nanmean(arr, axis=0).astype(np.float32)
    return m, np.where(n > 0, n, np.nan)
