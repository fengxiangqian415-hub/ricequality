from pathlib import Path
import warnings

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

from config_1004 import BASE, CROP_MODELS, GCMS, SSPS, RI_TYPES, MASK_DIR

PHENO_HIS = BASE / "phenology" / "historical"
PHENO_FUT = BASE / "phenology" / "future"
OBS_DIR   = Path("/home/debian/IF10")
OUT_DIR   = BASE / "phenology_corrected"
OUT_HIS   = OUT_DIR / "historical_annual"
OUT_MEM   = OUT_DIR / "historical_member"
OUT_FUT   = OUT_DIR / "future"
OUT_BIAS  = OUT_DIR / "bias"

STAGES      = ["HE", "MA"]
OBS_PAT = {
    "ri1": {"HE": "CHN_Rice(SR&ER)_HE_{year}.tif", "MA": "CHN_Rice(SR&ER)_MA_{year}.tif"},
    "ri2": {"HE": "CHN_Rice(LR)_HE_{year}.tif",    "MA": "CHN_Rice(LR)_MA_{year}.tif"},
}
BIAS_YEARS = list(range(2000, 2015))

LON_VALS = np.arange(73.25, 136.0, 0.5)
LAT_VALS = np.arange(53.75, 17.75, -0.5)
N_LON, N_LAT, RES = len(LON_VALS), len(LAT_VALS), 0.5
DST_TRANSFORM = from_bounds(LON_VALS[0] - RES / 2, LAT_VALS[-1] - RES / 2,
                            LON_VALS[-1] + RES / 2, LAT_VALS[0] + RES / 2, N_LON, N_LAT)
DST_CRS = CRS.from_epsg(4326)
META_TIF = {"driver": "GTiff", "dtype": "float32", "nodata": np.nan,
            "width": N_LON, "height": N_LAT, "count": 1,
            "crs": DST_CRS, "transform": DST_TRANSFORM, "compress": "lzw"}
EXTENT = [LON_VALS[0] - RES / 2, LON_VALS[-1] + RES / 2,
          LAT_VALS[-1] - RES / 2, LAT_VALS[0] + RES / 2]


def save_tif(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **META_TIF) as dst:
        dst.write(np.asarray(arr, dtype=np.float32), 1)


def read_tif(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == np.float32(ds.nodata)] = np.nan
    return arr


def load_mask(ri):
    return ~np.isnan(read_tif(MASK_DIR / f"{ri}_mask_0.5deg.tif"))


def obs_to_0p5(fpath):
    with rasterio.open(fpath) as ds:
        src = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            src[src == np.float32(ds.nodata)] = np.nan
        src[src <= 0] = np.nan
        dst = np.full((N_LAT, N_LON), np.nan, np.float32)
        reproject(source=src, destination=dst,
                  src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=DST_TRANSFORM, dst_crs=DST_CRS,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    dst[dst <= 0] = np.nan
    return dst


def load_obs(ri, years):
    """返回 {yr: {"HE": arr, "MA": arr}}；优先读已有 *_observed.tif，没有则重采样并保存"""
    out = {}
    for yr in years:
        d = {}
        for st in STAGES:
            cached = OUT_HIS / f"{ri}_{st}_{yr}_observed.tif"
            if cached.exists():
                d[st] = read_tif(cached)
            else:
                src = OBS_DIR / OBS_PAT[ri][st].format(year=yr)
                if not src.exists():
                    break
                d[st] = obs_to_0p5(src)
                save_tif(d[st], cached)
        if len(d) == 2:
            out[yr] = d
    return out


def align_to_grid(da):
    li = np.array([np.argmin(np.abs(da.lon.values - v)) for v in LON_VALS])
    la = np.array([np.argmin(np.abs(da.lat.values - v)) for v in LAT_VALS])
    return da.isel(lon=li, lat=la)


def read_pheno_nc(nc_path, var_name):
    ds = xr.open_dataset(nc_path, decode_times=False)
    units = ds[var_name].attrs.get("units", "")
    fv = ds[var_name].attrs.get("_FillValue", ds[var_name].encoding.get("_FillValue", -9999.0))
    years = (ds["time"].values + 1601).astype(int)
    data = align_to_grid(ds[var_name]).values.astype(np.float32).copy()
    if fv is not None:
        data[data == np.float32(fv)] = np.nan
    if "day of year" in units and "plant" in var_name.lower():
        data[data > 350] = np.nan
    ds.close()
    return years, data, units


def sim_annual(pheno_dir, ri, years=None):
    """
    读取一个成员的模拟物候，返回 ({yr: (he_doy, ma_doy)}, n_cross)
    n_cross：因 DOY>365（跨年）或原始 MA<=HE（成熟日回绕到次年小 DOY 等无效模拟）被置为 NaN 的格网-年数
    """
    pheno_dir = Path(pheno_dir)
    pf = sorted(pheno_dir.glob(f"*plantday-{ri}-firr*"))
    af = sorted(pheno_dir.glob(f"*anthday-{ri}-firr*"))
    mf = sorted(pheno_dir.glob(f"*matyday-{ri}-firr*"))
    if not (pf and af and mf):
        return {}, 0
    yp, P, _  = read_pheno_nc(pf[0], f"plantday-{ri}-firr")
    ya, A, ua = read_pheno_nc(af[0], f"anthday-{ri}-firr")
    ym, M, um = read_pheno_nc(mf[0], f"matyday-{ri}-firr")
    if "days from planting" in ua:
        A[A <= 0] = np.nan
    if "days from planting" in um:
        M[M <= 0] = np.nan
    out, n_cross = {}, 0
    for yr in (years if years is not None else sorted(set(yp) & set(ya) & set(ym))):
        i, j, k = np.where(yp == yr)[0], np.where(ya == yr)[0], np.where(ym == yr)[0]
        if not (len(i) and len(j) and len(k)):
            continue
        he = P[i[0]] + A[j[0]] if "days from planting" in ua else A[j[0]].copy()
        ma = P[i[0]] + M[k[0]] if "days from planting" in um else M[k[0]].copy()
        he = np.where(he > 365, np.nan, he).astype(np.float32)
        ma = np.where(ma > 365, np.nan, ma).astype(np.float32)
        wrap = np.isfinite(he) & np.isfinite(ma) & (ma <= he)
        n_cross += int(np.sum(~np.isfinite(he) & np.isfinite(A[j[0]]))) + int(wrap.sum())
        he[wrap] = np.nan
        ma[wrap] = np.nan
        out[yr] = (he, ma)
    return out, n_cross


def nanmean_stack(lst):
    return np.nanmean(np.stack(lst), axis=0)


def nanmedian_stack(lst):
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(lst), axis=0)


def member_bias(obs, sim, years):
    """
    偏差 = 观测多年中位数 − 模拟多年中位数（逐格网，2000-2014）
    用中位数而非均值：少数离群年份（如成熟期回绕、作物失败）会把均值偏差放大到上百天
    """
    yrs = [y for y in years if y in obs and y in sim]
    if not yrs:
        return None, None, yrs
    bh = nanmedian_stack([obs[y]["HE"] for y in yrs]) - nanmedian_stack([sim[y][0] for y in yrs])
    bm = nanmedian_stack([obs[y]["MA"] for y in yrs]) - nanmedian_stack([sim[y][1] for y in yrs])
    return bh, bm, yrs


def apply_correction(he_raw, ma_raw, bias_he, bias_ma):
    """
    施加偏差；校正后超出 1-365 的日期不再截断，而是与 HE>=MA 一样置 NaN。
    （截断会把成熟期压到 12/31，造成延伸到冬季的虚假灌浆窗口，或 HE=MA 后被剔除）
    返回 (he, ma, 被剔除格网数)
    """
    hc = (he_raw + bias_he).astype(np.float32)
    mc = (ma_raw + bias_ma).astype(np.float32)
    both = np.isfinite(hc) & np.isfinite(mc)
    bad = both & ((hc < 1) | (hc > 365) | (mc < 1) | (mc > 365) | (hc >= mc))
    hc[bad] = np.nan
    mc[bad] = np.nan
    return hc, mc, int(bad.sum())
