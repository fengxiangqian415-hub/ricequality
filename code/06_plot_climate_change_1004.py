import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import rasterio
import geopandas as gpd
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

from config_1004 import (CLIM_HIS, CLIM_FUT, HIS_P, FUT_P, HIS_LABEL, FUT_LABEL, HIS_YEARS, FUT_YEARS,
                         FIGURES, TAG, CHINA_SHP, POLY_SHP, support_file, area_file, co2_mean)
HIS_DIR  = CLIM_HIS / "ensemble"
FUT_DIR  = CLIM_FUT / "ensemble"
OUT_PATH = FIGURES / f"climate_change_{TAG}"
OUT_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]

FEAT_LABELS = {
    "CO2":   "CO₂ (ppm)",
    "Tavg":  "T$_{avg}$ (°C)",
    "Tdif":  "T$_{dif}$ (°C)",
    "CTSR":  "CTSR (MJ m$^{-2}$)",
    "TSR":   "TSR (MJ m$^{-2}$ d$^{-1}$)",
    "GF_days": "Grain-filling duration (d)",
    "Rain":  "Rain (mm)",
    "RHavg": "RH$_{avg}$ (%)",
}
RASTER_LABEL = {
    "Tavg":  "mean_temp_degC",
    "Tdif":  "mean_tmpdiff_degC",
    "CTSR":  "cumul_solar_MJ_m2",
    "TSR":   "mean_daily_solar_MJ_m2",
    "GF_days": "gf_days",
    "Rain":  "cumul_precip_mm",
    "RHavg": "mean_humidity_pct",
}
CHG_STEP = {"CO2": 50, "Tavg": 1, "Tdif": 0.5, "CTSR": 20, "TSR": 1, "GF_days": 2, "Rain": 10, "RHavg": 1}
CO2_VALS = {"baseline": co2_mean("historical", HIS_YEARS),
            "ssp126":   co2_mean("ssp126", FUT_YEARS),
            "ssp585":   co2_mean("ssp585", FUT_YEARS)}
print("CO2 时段均值:", {k: round(v, 1) for k, v in CO2_VALS.items()})

RI_TYPES  = ["ri1", "ri2"]
RI_LABELS = {"ri1": "Single/early rice", "ri2": "Late rice"}
SSPS      = ["ssp126", "ssp585"]

BASE_CMAP   = "YlOrRd"
CHANGE_CMAP = "RdBu_r"
HIST_COLORS = {"baseline": "#2166AC", "ssp126": "#4DAF4A", "ssp585": "#D73027"}
ROW_LABELS  = {
    "baseline": f"Baseline\n({HIS_LABEL})",
    "ssp126":   f"SSP1-2.6\n({FUT_LABEL})",
    "ssp585":   f"SSP5-8.5\n({FUT_LABEL})",
}

LON_VALS = np.arange(73.25, 136.0, 0.5)
LAT_VALS = np.arange(53.75, 17.75, -0.5)
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

def read_tif(path):
    if not Path(path).exists():
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == np.float32(ds.nodata)] = np.nan
    return arr

_SUP = {}
def supmask(arr, ri):
    """★ 共同支撑区外置 NaN"""
    if arr is None:
        return None
    if ri not in _SUP:
        s = read_tif(support_file(ri))
        _SUP[ri] = np.isfinite(s) if s is not None else np.ones_like(arr, bool)
    return np.where(_SUP[ri], arr, np.nan).astype(np.float32)

def abs_change(fut, base):
    return np.where(~np.isnan(base) & ~np.isnan(fut),
                    fut - base, np.nan).astype(np.float32)

def symmetric_range(cp2, cp98, step=5):
    abs_max = max(abs(cp2), abs(cp98))
    rounded = max(float(np.ceil(abs_max / step) * step), step)
    return -rounded, rounded

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

def draw_hist_abs(ax, arr, scen, xlabel, co2_val=None, show_ylabel=True):
    color = HIST_COLORS[scen]
    if co2_val is not None:
        ax.axvline(co2_val, color=color, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=FS_HLAB, labelpad=2)
    elif arr is not None:
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            counts, edges = np.histogram(valid, bins=30)
            freq = counts / counts.sum() * 100
            ax.bar(edges[:-1], freq, width=np.diff(edges),
                   align="edge", color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel, fontsize=FS_HLAB, labelpad=2)
        if show_ylabel:
            ax.set_ylabel("Frequency (%)", fontsize=FS_HLAB, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2.5)

def draw_hist_chg(ax, arr, scen, vmin, vmax, co2_val=None, show_ylabel=True, chg_label="Change"):
    color = HIST_COLORS[scen]
    if co2_val is not None:
        ax.axvline(co2_val, color=color, linewidth=1.5)
        ax.set_xlabel(chg_label, fontsize=FS_HLAB, labelpad=2)
    elif arr is not None:
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            bins = np.linspace(vmin, vmax, 41)
            counts, edges = np.histogram(valid, bins=bins)
            freq = counts / counts.sum() * 100
            ax.bar(edges[:-1], freq, width=np.diff(edges),
                   align="edge", color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
        ax.set_xlabel(chg_label, fontsize=FS_HLAB, labelpad=2)
        if show_ylabel:
            ax.set_ylabel("Frequency (%)", fontsize=FS_HLAB, labelpad=2)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlim(vmin, vmax)
    ax.tick_params(labelsize=FS_TICK, length=2.5)

PANEL_LABELS = list("abcdefghijkl")
print(f"开始绘制 {len(FEATURES)} 张气候指标变化图...")

for feat in FEATURES:
    print(f"  [{FEATURES.index(feat)+1}/{len(FEATURES)}] {feat}")
    is_co2 = (feat == "CO2")

    data = {}
    for ri in RI_TYPES:
        if is_co2:
            ref = supmask(read_tif(HIS_DIR / f"ensemble_historical_{ri}_mean_temp_degC_{HIS_P}_mean.tif"), ri)
            mask = ~np.isnan(ref) if ref is not None else None
            for scen in ["baseline", "ssp126", "ssp585"]:
                if mask is not None:
                    arr = np.full_like(ref, CO2_VALS[scen])
                    arr[np.isnan(ref)] = np.nan
                else:
                    arr = None
                data[f"{scen}_{ri}"] = arr
                if scen != "baseline":
                    base_arr = data[f"baseline_{ri}"]
                    data[f"{scen}_{ri}_chg"] = abs_change(arr, base_arr)
        else:
            label = RASTER_LABEL[feat]
            base = supmask(read_tif(HIS_DIR / f"ensemble_historical_{ri}_{label}_{HIS_P}_mean.tif"), ri)
            data[f"baseline_{ri}"] = base
            for ssp in SSPS:
                fut = supmask(read_tif(FUT_DIR / f"ensemble_{ssp}_{ri}_{label}_{FUT_P}_mean.tif"), ri)
                data[f"{ssp}_{ri}"] = fut
                data[f"{ssp}_{ri}_chg"] = abs_change(fut, base)

    all_base = []
    for ri in RI_TYPES:
        arr = data.get(f"baseline_{ri}")
        if arr is not None:
            all_base.append(arr[~np.isnan(arr)])
    if all_base:
        combined = np.concatenate(all_base)
        p2, p98 = np.percentile(combined, [2, 98])
        BASE_VMIN, BASE_VMAX = float(p2), float(p98)
    else:
        BASE_VMIN, BASE_VMAX = 0, 1
    norm_base = Normalize(vmin=BASE_VMIN, vmax=BASE_VMAX)

    all_chg = []
    for ssp in SSPS:
        for ri in RI_TYPES:
            arr = data.get(f"{ssp}_{ri}_chg")
            if arr is not None:
                v = arr[~np.isnan(arr)]
                if len(v): all_chg.append(v)
    if all_chg:
        combined_chg = np.concatenate(all_chg)
        cp2, cp98 = np.percentile(combined_chg, [2, 98])
        chg_vmin, chg_vmax = symmetric_range(cp2, cp98, step=CHG_STEP.get(feat, 5))
    else:
        chg_vmin, chg_vmax = -10, 10
    norm_chg = TwoSlopeNorm(vmin=chg_vmin, vcenter=0, vmax=chg_vmax)

    _unit = FEAT_LABELS[feat][FEAT_LABELS[feat].rfind("("):] if "(" in FEAT_LABELS[feat] else ""
    CHG_LABEL = f"Change {_unit}".strip()

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

    scenarios  = ["baseline", "ssp126", "ssp585"]
    label_idx  = 0
    hist_axes  = []

    for row_i, scen in enumerate(scenarios):
        is_base = (scen == "baseline")
        is_bottom = (row_i == 2)
        norm = norm_base if is_base else norm_chg
        cmap = BASE_CMAP  if is_base else CHANGE_CMAP
        xlabel = FEAT_LABELS[feat]

        for col_i, ri in enumerate(RI_TYPES):
            ax_m = fig.add_subplot(gs[row_i, col_i])
            arr  = data[f"{scen}_{ri}"] if is_base else data[f"{scen}_{ri}_chg"]
            draw_map(ax_m, arr, norm, cmap,
                     ylabel=ROW_LABELS[scen] if col_i == 0 else "",
                     show_lon=(row_i == 2), show_lat=(col_i == 0))
            if row_i == 0:
                ax_m.set_title(RI_LABELS[ri], fontsize=FS_TITLE, pad=12)
            xoff = -0.16 if col_i == 0 else -0.06
            ax_m.text(xoff, 1.16, f"({PANEL_LABELS[label_idx]})",
                      transform=ax_m.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")
            label_idx += 1

            ax_h = fig.add_subplot(gs[row_i, col_i + 3])
            show_yl = (col_i == 0)
            if is_base:
                co2_v = CO2_VALS[scen] if is_co2 else None
                draw_hist_abs(ax_h, data[f"{scen}_{ri}"], scen, xlabel,
                              co2_val=co2_v, show_ylabel=show_yl)
            else:
                co2_v = CO2_VALS[scen] - CO2_VALS["baseline"] if is_co2 else None
                draw_hist_chg(ax_h, data[f"{scen}_{ri}_chg"], scen,
                              chg_vmin, chg_vmax, co2_val=co2_v, show_ylabel=show_yl,
                              chg_label=CHG_LABEL)
            if col_i == 1:
                ax_h.tick_params(labelleft=False)
            hist_axes.append(ax_h)
            if row_i == 0:
                ax_h.set_title(RI_LABELS[ri], fontsize=FS_TITLE, pad=12)
            xoff = -0.24 if col_i == 0 else -0.08
            ax_h.text(xoff, 1.16, f"({PANEL_LABELS[label_idx]})",
                      transform=ax_h.transAxes, fontsize=FS_PANEL, fontweight="bold", va="top")
            label_idx += 1

        cbar_gap = 0.06 if is_bottom else 0.028
        pos_l = gs[row_i, 0].get_position(fig)
        pos_r = gs[row_i, 1].get_position(fig)
        cbar_left   = pos_l.x0 + pos_l.width * 0.15
        cbar_width  = (pos_r.x1 - pos_l.x0) * 0.70
        cbar_bottom = pos_l.y0 - cbar_gap
        cbar_height = 0.016
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        if is_base:
            sm = ScalarMappable(cmap=BASE_CMAP, norm=norm_base)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(FEAT_LABELS[feat], fontsize=FS_CBLAB, labelpad=2)
        else:
            sm = ScalarMappable(cmap=CHANGE_CMAP, norm=norm_chg)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(CHG_LABEL, fontsize=FS_CBLAB, labelpad=2)
        cb.ax.tick_params(labelsize=FS_CBTICK, length=2.5)

    if hist_axes:
        ymax = max(ax.get_ylim()[1] for ax in hist_axes)
        for ax in hist_axes:
            ax.set_ylim(0, ymax)

    fname = OUT_PATH / feat
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {feat}.png")

import pandas as pd
import stats_common_1004 as S
rows = []
for ri in RI_TYPES:
    w = S.weights(ri)
    a_km2 = S.area(ri)
    for feat, lab in RASTER_LABEL.items():
        b = S.read(HIS_DIR / f"ensemble_historical_{ri}_{lab}_{HIS_P}_mean.tif")
        fs = {ssp: S.read(FUT_DIR / f"ensemble_{ssp}_{ri}_{lab}_{FUT_P}_mean.tif") for ssp in SSPS}
        if b is None or any(f is None for f in fs.values()):
            continue
        k = S.cells(ri, b, *fs.values())
        bm = S.wmean(b, w, k)
        for ssp, f in fs.items():
            d = f - b
            fm = S.wmean(f, w, k)
            rows.append(dict(var=feat, ri=ri, ssp=ssp, n_cells=int(k.sum()),
                             area_Mha=float(np.nansum(a_km2[k])) / 1e4,
                             baseline=bm, future=fm, change=fm - bm, change_pct=S.rel(fm, bm),
                             cell_change_p10=S.wquant(d, w, k, 0.10), cell_change_median=S.wquant(d, w, k, 0.50),
                             cell_change_p90=S.wquant(d, w, k, 0.90),
                             increase_share_pct=100 * float(np.nansum(w[k & (d > 0)]) / np.nansum(w[k]))))
summ = pd.DataFrame(rows)
summ.to_csv(OUT_PATH / "climate_change_summary.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 220)
print("\n全国汇总（集合平均；统一格网域、各情景同一组格网；权重 SUMMARY_WEIGHT）:")
print(summ[["var", "ri", "ssp", "baseline", "future", "change", "change_pct",
            "cell_change_p10", "cell_change_p90", "increase_share_pct"]].round(2).to_string(index=False))
print(f"CO2: 基准 {CO2_VALS['baseline']:.1f} → SSP1-2.6 {CO2_VALS['ssp126']:.1f} / SSP5-8.5 {CO2_VALS['ssp585']:.1f} ppm")

print(f"\n全部完成！共输出 {len(FEATURES)} 张图")
print(f"输出目录: {OUT_PATH}")
