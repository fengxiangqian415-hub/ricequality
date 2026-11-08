import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

FS_BASE   = 15
FS_TICK   = 13
FS_TITLE  = 15
FS_ROWLAB = 15
FS_PANEL  = 16
FS_CBLAB  = 14
FS_CBTICK = 12
FS_HLAB   = 14

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": FS_BASE, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

from config_1004 import (BASE, TAG, YIELD_DIR, MASK_DIR, PRED_MODE, HIS_P, FUT_P, HIS_LABEL, FUT_LABEL,
                         HIS_YEARS, FUT_YEARS, CROP_MODELS, GCMS, SSPS, RI_TYPES,
                         CHINA_SHP, POLY_SHP, pred_dir, out_dir, fig_dir, support_file)
HRR_ROOT = pred_dir()
OUT_TIF  = out_dir("head_rice_yield", thresh=False)
OUT_FIG  = fig_dir("head_rice_yield", thresh=False)
OUT_TIF.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


LON_VALS = np.arange(73.25, 136.0, 0.5)
LAT_VALS = np.arange(53.75, 17.75, -0.5)
N_LON, N_LAT = len(LON_VALS), len(LAT_VALS)
EXTENT   = [LON_VALS[0]-0.25, LON_VALS[-1]+0.25,
            LAT_VALS[-1]-0.25, LAT_VALS[0]+0.25]
INSET_EXTENT = [105, 2, 125, 26]

LON_TICKS = [80, 100, 120]
LAT_TICKS = [20, 30, 40, 50]

china   = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
polygon = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
nanhai  = polygon[polygon.geometry.apply(lambda g: g.bounds[1] < 25)]
jd_line = polygon[polygon["adchar"] == "JD"]
lon_min, lat_min, lon_max, lat_max = china.total_bounds

HIST_COLORS = {"baseline": "#2166AC", "ssp126": "#4DAF4A", "ssp585": "#D73027"}
ROW_LABELS  = {
    "baseline": f"Baseline\n({HIS_LABEL})",
    "ssp126":   f"SSP1-2.6\n({FUT_LABEL})",
    "ssp585":   f"SSP5-8.5\n({FUT_LABEL})",
}
RI_LABELS = {"ri1": "Single/early rice", "ri2": "Late rice"}

def add_nanhai_inset(ax, arr, norm, cmap):
    ax_pos = ax.get_position()
    iw = ax_pos.width  * 0.24
    ih = ax_pos.height * 0.30
    ix = ax_pos.x1 - iw * 0.9
    iy = ax_pos.y0 + ax_pos.height * 0.01
    ax_in = ax.figure.add_axes([ix, iy, iw, ih])
    if arr is not None:
        ax_in.imshow(arr, extent=EXTENT, origin="upper",
                     cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    nanhai.boundary.plot(ax=ax_in, linewidth=0.3, color="black", zorder=4)
    jd_line.boundary.plot(ax=ax_in, linewidth=0.5, color="black", linestyle="-", zorder=5)
    ax_in.set_xlim(INSET_EXTENT[0], INSET_EXTENT[2])
    ax_in.set_ylim(INSET_EXTENT[1], INSET_EXTENT[3])
    ax_in.set_aspect("equal")
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.5)

def draw_map(ax, arr, norm, cmap, ylabel="", show_lon=False, show_lat=False):
    if arr is not None:
        ax.imshow(arr, extent=EXTENT, origin="upper",
                  cmap=cmap, norm=norm, aspect="auto")
    china.boundary.plot(ax=ax, linewidth=0.35, color="black", zorder=5)
    ax.set_xlim(lon_min-0.5, lon_max+0.5)
    ax.set_ylim(lat_min-0.5, lat_max+0.5)
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    ax.set_xticklabels([f"{v}°E" for v in LON_TICKS] if show_lon else [])
    ax.set_yticklabels([f"{v}°N" for v in LAT_TICKS] if show_lat else [])
    ax.tick_params(labelsize=FS_TICK, length=2.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_ROWLAB, labelpad=6, fontweight="bold")
    add_nanhai_inset(ax, arr, norm, cmap)

def draw_hist(ax, arr, scen, xlabel, bins, xlim=None, show_ylabel=True):
    color = HIST_COLORS[scen]
    if arr is not None:
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            counts, edges = np.histogram(valid, bins=bins)
            freq = counts / counts.sum() * 100
            ax.bar(edges[:-1], freq, width=np.diff(edges),
                   align="edge", color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
    if xlim: ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, fontsize=FS_HLAB, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Frequency (%)", fontsize=FS_HLAB, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2.5)

def load_mask(ri):
    fp = MASK_DIR / f"{ri}_mask_0.5deg.tif"
    if not fp.exists(): return None
    with rasterio.open(fp) as ds:
        m = ds.read(1).astype(np.float32)
    return ~np.isnan(m)

masks = {ri: load_mask(ri) for ri in RI_TYPES}

def align_to_grid(da):
    src_lon = da.lon.values
    src_lat = da.lat.values
    lon_idx = np.array([np.argmin(np.abs(src_lon-v)) for v in LON_VALS])
    lat_idx = np.array([np.argmin(np.abs(src_lat-v)) for v in LAT_VALS])
    return da.values[:, lat_idx, :][:, :, lon_idx]

def read_yield_nc(nc_path, var_name, years, ri):
    if not Path(nc_path).exists(): return None
    ds  = xr.open_dataset(nc_path, decode_times=False)
    da  = ds[var_name].astype(np.float32)
    fv  = ds[var_name].attrs.get("_FillValue", 1e20)
    yrs = (ds["time"].values + 1601).astype(int)
    data = align_to_grid(da).copy()
    data[np.abs(data) > 1e10] = np.nan
    data[data == fv] = np.nan
    data[data < 0]  = np.nan
    idx = [i for i, y in enumerate(yrs) if y in set(years)]
    if not idx: ds.close(); return None
    mean_arr = np.nanmean(data[idx], axis=0)
    if masks[ri] is not None:
        mean_arr[~masks[ri]] = np.nan
    ds.close()
    return mean_arr

def find_yield_file(model, gcm, scenario, ri, ssp=None):
    if scenario == "historical":
        d = YIELD_DIR / "historical" / model / gcm
        files = list(d.glob(f"*historical*yield-{ri}-firr*.nc"))
    else:
        d = YIELD_DIR / "future" / model / gcm / ssp
        files = list(d.glob(f"*{ssp}*yield-{ri}-firr*.nc"))
    return files[0] if files else None

def read_tif(path):
    if not Path(path).exists(): return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == np.float32(ds.nodata)] = np.nan
    return arr

def save_tif(arr, path):
    transform = from_bounds(
        LON_VALS[0]-0.25, LAT_VALS[-1]-0.25,
        LON_VALS[-1]+0.25, LAT_VALS[0]+0.25,
        N_LON, N_LAT
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w",
                       driver="GTiff", height=N_LAT, width=N_LON,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=transform, compress="lzw",
                       nodata=np.nan) as dst:
        dst.write(arr.astype(np.float32), 1)

def pct_change(fut, base):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            ~np.isnan(base) & ~np.isnan(fut) & (base > 0),
            (fut - base) / base * 100, np.nan
        ).astype(np.float32)

print("计算整精米产量（产量 × HRR/100）...")

all_data = {}

for ri in RI_TYPES:
    print(f"\n=== {ri} ===")
    all_data[ri] = {}

    print(f"  [历史] 逐成员计算整精米产量...")
    hry_his_list = []
    for model in CROP_MODELS:
        for gcm in GCMS:
            fp = find_yield_file(model, gcm, "historical", ri)
            if not fp:
                continue
            arr = read_yield_nc(fp, f"yield-{ri}-firr", HIS_YEARS, ri)
            if arr is None:
                continue
            hrr_fp = HRR_ROOT / "historical" / f"{gcm}_historical_{ri}_HRR_{HIS_P}.tif"
            hrr = read_tif(hrr_fp)
            if hrr is None:
                print(f"    ⚠️  HRR缺失: {hrr_fp.name}")
                continue
            hry = arr * hrr / 100.0
            hry[np.isnan(arr) | np.isnan(hrr)] = np.nan
            save_tif(hry, OUT_TIF / "historical" / model / gcm /
                     f"{model}_{gcm}_historical_{ri}_HRY_{HIS_P}.tif")
            hry_his_list.append(hry)

    if hry_his_list:
        hry_his = np.nanmean(np.stack(hry_his_list), axis=0)
        save_tif(hry_his, OUT_TIF / f"head_rice_yield_{ri}_historical.tif")
        print(f"  [历史] 整精米产量: mean={np.nanmean(hry_his):.3f} t/ha  n={len(hry_his_list)}")
    else:
        hry_his = None
        print(f"  [历史] ⚠️  数据缺失")

    all_data[ri]["baseline"] = hry_his

    for ssp in SSPS:
        print(f"  [{ssp}] 计算整精米产量...")

        hry_list = []
        for model in CROP_MODELS:
            for gcm in GCMS:
                fp = find_yield_file(model, gcm, "future", ri, ssp=ssp)
                if not fp:
                    continue
                arr = read_yield_nc(fp, f"yield-{ri}-firr", FUT_YEARS, ri)
                if arr is None:
                    continue

                hrr_fp = HRR_ROOT / ssp / f"{model}_{gcm}_{ssp}_{ri}_HRR_{FUT_P}.tif"
                hrr = read_tif(hrr_fp)
                if hrr is None:
                    print(f"    ⚠️  HRR缺失: {hrr_fp.name}")
                    continue

                hry = arr * hrr / 100.0
                hry[np.isnan(arr) | np.isnan(hrr)] = np.nan

                single_dir = OUT_TIF / ssp / model / gcm
                single_dir.mkdir(parents=True, exist_ok=True)
                save_tif(hry, single_dir /
                         f"{model}_{gcm}_{ssp}_{ri}_HRY_{FUT_P}.tif")
                hry_list.append(hry)
                print(f"    [{model}/{gcm}] HRY mean={np.nanmean(hry):.3f} t/ha")

        if hry_list:
            hry_ens = np.nanmean(np.stack(hry_list), axis=0)
            ens_dir = OUT_TIF / ssp / "ensemble"
            ens_dir.mkdir(parents=True, exist_ok=True)
            save_tif(hry_ens, ens_dir / f"ensemble_{ssp}_{ri}_HRY_{FUT_P}.tif")
            save_tif(hry_ens, OUT_TIF / f"head_rice_yield_{ri}_{ssp}.tif")
            print(f"  [{ssp}] ensemble HRY: mean={np.nanmean(hry_ens):.3f} t/ha  n={len(hry_list)}")
            all_data[ri][ssp] = hry_ens
        else:
            all_data[ri][ssp] = None

for _ri in RI_TYPES:
    with rasterio.open(support_file(_ri)) as _ds:
        _dom = np.isfinite(_ds.read(1).astype(np.float32))
    for _k, _v in all_data[_ri].items():
        if _v is not None:
            all_data[_ri][_k] = np.where(_dom, _v, np.nan).astype(np.float32)

print("\n开始出图...")

panel_labels = list("abcdefghijkl")

SPACER_W = 0.08
MAP_HIST_WSPACE = 0.20
fig = plt.figure(figsize=(260/25.4, 200/25.4))
gs  = gridspec.GridSpec(
    3, 5, figure=fig,
    left=0.075, right=0.985,
    top=0.94, bottom=0.09,
    hspace=0.62, wspace=MAP_HIST_WSPACE,
    width_ratios=[1.4, 1.4, SPACER_W, 1.0, 1.0],
)

scenarios = ["baseline", "ssp126", "ssp585"]

all_base = []
for ri in RI_TYPES:
    arr = all_data[ri].get("baseline")
    if arr is not None:
        all_base.append(arr[~np.isnan(arr)])
if all_base:
    combined = np.concatenate(all_base)
    base_vmin = float(np.percentile(combined, 2))
    base_vmax = float(np.percentile(combined, 98))
else:
    base_vmin, base_vmax = 0, 10
norm_base = Normalize(vmin=base_vmin, vmax=base_vmax)

all_chg = []
for ri in RI_TYPES:
    base = all_data[ri].get("baseline")
    for ssp in SSPS:
        fut = all_data[ri].get(ssp)
        if base is not None and fut is not None:
            chg = pct_change(fut, base)
            v = chg[~np.isnan(chg)]
            if len(v): all_chg.append(v)
if all_chg:
    combined_chg = np.concatenate(all_chg)
    cp2, cp98 = np.percentile(combined_chg, [2, 98])
    abs_max = max(abs(cp2), abs(cp98))
    sym = max(int(np.ceil(abs_max / 5) * 5), 5)
    chg_vmin, chg_vmax = -sym, sym
else:
    chg_vmin, chg_vmax = -50, 50
norm_chg = TwoSlopeNorm(vmin=chg_vmin, vcenter=0, vmax=chg_vmax)

label_idx = 0

for row_i, scen in enumerate(scenarios):
    is_base = (scen == "baseline")
    is_bottom = (row_i == 2)
    norm  = norm_base if is_base else norm_chg
    cmap  = "YlOrRd"  if is_base else "RdBu_r"
    row_hist = []

    for col_i, ri in enumerate(RI_TYPES):
        base_arr = all_data[ri].get("baseline")
        fut_arr  = all_data[ri].get(scen) if not is_base else None
        arr = base_arr if is_base else pct_change(fut_arr, base_arr)

        ax_m = fig.add_subplot(gs[row_i, col_i])
        draw_map(ax_m, arr, norm, cmap,
                 ylabel=ROW_LABELS[scen] if col_i == 0 else "",
                 show_lon=(row_i == 2), show_lat=(col_i == 0))
        if row_i == 0:
            ax_m.set_title(RI_LABELS[ri], fontsize=FS_TITLE, pad=12)
        xoff = -0.16 if col_i == 0 else -0.06
        ax_m.text(xoff, 1.16, f"({panel_labels[label_idx]})",
                  transform=ax_m.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")
        label_idx += 1

        ax_h = fig.add_subplot(gs[row_i, col_i + 3])
        show_yl = (col_i == 0)
        if is_base:
            draw_hist(ax_h, arr, scen, r"HRY (t ha$^{-1}$)",
                      bins=np.arange(0, 12, 0.5), xlim=(0, 11),
                      show_ylabel=show_yl)
        else:
            step = max(1, (chg_vmax - chg_vmin) // 40)
            draw_hist(ax_h, arr, scen, "Change (%)",
                      bins=np.arange(chg_vmin, chg_vmax+step, step),
                      xlim=(chg_vmin, chg_vmax), show_ylabel=show_yl)
            ax_h.axvline(0, color="black", linewidth=0.5, linestyle="--")
        if col_i == 1:
            ax_h.tick_params(labelleft=False)
        row_hist.append(ax_h)
        if row_i == 0:
            ax_h.set_title(RI_LABELS[ri], fontsize=FS_TITLE, pad=12)
        xoff = -0.24 if col_i == 0 else -0.08
        ax_h.text(xoff, 1.16, f"({panel_labels[label_idx]})",
                  transform=ax_h.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")
        label_idx += 1

    if row_hist:
        ymax = max(ax.get_ylim()[1] for ax in row_hist)
        for ax in row_hist:
            ax.set_ylim(0, ymax)

    cbar_gap = 0.06 if is_bottom else 0.028
    pos_l = gs[row_i, 0].get_position(fig)
    pos_r = gs[row_i, 1].get_position(fig)
    cbar_left   = pos_l.x0 + pos_l.width * 0.15
    cbar_width  = (pos_r.x1 - pos_l.x0) * 0.70
    cbar_bottom = pos_l.y0 - cbar_gap
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, 0.016])
    if is_base:
        sm = ScalarMappable(cmap="YlOrRd", norm=norm_base)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(r"Head-rice yield (t ha$^{-1}$)", fontsize=FS_CBLAB, labelpad=2)
    else:
        sm = ScalarMappable(cmap="RdBu_r", norm=norm_chg)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Change (%)", fontsize=FS_CBLAB, labelpad=2)
    cb.ax.tick_params(labelsize=FS_CBTICK, length=2.5)

fname = OUT_FIG / "head_rice_yield"
fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{fname}.pdf", bbox_inches="tight")
fig.savefig(f"{fname}.svg", bbox_inches="tight")
plt.close(fig)
print(f"\n[saved] {fname}.png")
print(f"TIF输出: {OUT_TIF}")
