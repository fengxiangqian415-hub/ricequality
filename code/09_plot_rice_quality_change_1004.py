
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle
import rasterio
import geopandas as gpd
from pathlib import Path
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_palette as PAL

FS_BASE   = 15
FS_TICK   = 13
FS_TITLE  = 15
FS_ROWLAB = 15
FS_PANEL  = 16
FS_CBLAB  = 14
FS_CBTICK = 12
FS_HLAB   = 14
FS_SCALE  = 10
FS_LEG    = 13

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": FS_BASE,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

from config_1004 import (OUTPUT as _OUT, FIGURES, TAG, PRED_MODE, HIS_P, FUT_P, HIS_LABEL, FUT_LABEL,
                         CHINA_SHP, POLY_SHP, CROP_MODELS, GCMS, pred_dir, support_file, area_file,
                         CO2_CF_BASELINE, SUMMARY_WEIGHT)
BASE        = _OUT.parent
PRED_V5     = pred_dir(PRED_MODE, "withco2").parent
HIS_DIR     = pred_dir(PRED_MODE, "withco2") / "ensemble"
FUT_DIR     = HIS_DIR
HIS_DIR_RCO = pred_dir(PRED_MODE, "fixedco2") / "ensemble"
FUT_DIR_RCO = HIS_DIR_RCO

ZONE_SHP  = "/home/debian/pythonfile/china/rice_zones.shp"
SHOW_ZONES = True
ZONE_ONLY_FOR = {"Acontent"}
OUT_PATH  = FIGURES / f"rice_quality_{TAG}_{PRED_MODE}"
OUT_PATH.mkdir(parents=True, exist_ok=True)

TARGETS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC", "PCA"]
TARGET_LABELS = {
    "BRR":      "Brown rice rate (%)",
    "MRR":      "Milled rice rate (%)",
    "HRR":      "Head rice rate (%)",
    "CGR":      "Chalky grain rate (%)",
    "CD":       "Chalkiness degree (%)",
    "Acontent": "Amylose content (%)",
    "PC":       "Protein content (%)",
    "PCA":      "Protein content (albumin, %)",
}
RI_TYPES  = ["ri1", "ri2"]
RI_LABELS = {"ri1": "Single/early rice", "ri2": "Late rice"}
SSPS      = ["ssp126", "ssp585"]
CHANGE_CMAP = PAL.CMAP_DIV

CHANGE_RANGE_OVERRIDE = {
    "BRR": (-10, 10),
}

LON_TICKS = [80, 100, 120]
LAT_TICKS = [20, 30, 40, 50]

COL_WITH = PAL.CO2["with"]
COL_NOCO = PAL.CO2["without"]
ALPHA_WITH = 0.55
ALPHA_NOCO = 0.45

ROW_LABELS = {
    "baseline": f"Baseline\n({HIS_LABEL})",
    "ssp126":   f"SSP1-2.6\n({FUT_LABEL})",
    "ssp585":   f"SSP5-8.5\n({FUT_LABEL})",
}

LON_VALS = np.arange(73.25, 136.0, 0.5)
LAT_VALS = np.arange(53.75, 17.75, -0.5)
EXTENT   = [LON_VALS[0]-0.25, LON_VALS[-1]+0.25,
            LAT_VALS[-1]-0.25, LAT_VALS[0]+0.25]
INSET_EXTENT = [105, 2, 125, 26]
SCALE_KM = 1000
LAT_REF  = 35

def symmetric_range(cp2, cp98, step=5):
    abs_max = max(abs(cp2), abs(cp98))
    rounded = int(np.ceil(abs_max / step) * step)
    return -max(rounded, step), max(rounded, step)

china   = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
polygon = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
nanhai  = polygon[polygon.geometry.apply(lambda g: g.bounds[1] < 25)]
jd_line = polygon[polygon["adchar"] == "JD"]
lon_min, lat_min, lon_max, lat_max = china.total_bounds

zones = None
if SHOW_ZONES:
    try:
        zones = gpd.read_file(ZONE_SHP).to_crs("EPSG:4326")
    except Exception as e:
        print(f"⚠️  区划 shp 读取失败，跳过叠加: {e}")
        zones = None

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

BASE_MIN_FOR_PCT = {
    "CGR": 5.0,
    "CD":  2.0,
}

def pct_change(fut, base, base_min=None):
    """相对变化(%)；base 低于 base_min 的格子置 NaN。"""
    if fut is None or base is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (~np.isnan(base)) & (~np.isnan(fut)) & (base != 0)
        if base_min is not None:
            ok &= (np.abs(base) >= base_min)
        return np.where(ok, (fut - base) / np.abs(base) * 100,
                        np.nan).astype(np.float32)

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
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.5)
    return ax_in

def draw_map(ax, arr, norm, cmap, ylabel="", draw_zone=False,
             show_lon=False, show_lat=False):
    if arr is not None:
        ax.imshow(arr, extent=EXTENT, origin="upper",
                  cmap=cmap, norm=norm, aspect="auto")
    china.boundary.plot(ax=ax, linewidth=0.35, color="black", zorder=5)
    if draw_zone and zones is not None:
        zones.boundary.plot(ax=ax, linewidth=0.3, color="0.35",
                            linestyle=(0, (3, 2)), zorder=6)
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

def add_scale_bar(ax, x_left, cy, length_km=SCALE_KM, lat_ref=LAT_REF, n_seg=4, h=0.0040):
    """在 ax 底部画黑白分段比例尺(figure 坐标)。"""
    fig = ax.figure
    deg = length_km / (111.320 * np.cos(np.deg2rad(lat_ref)))
    x_lo, x_hi = ax.get_xlim(); pos = ax.get_position()
    w = deg / (x_hi - x_lo) * pos.width; seg = w / n_seg
    for i in range(n_seg):
        fig.patches.append(Rectangle((x_left + i*seg, cy), seg, h,
                           facecolor="black" if i % 2 == 0 else "white",
                           edgecolor="black", linewidth=0.4,
                           transform=fig.transFigure, zorder=10, figure=fig))
    fig.text(x_left + w + 0.006, cy + h/2, f"{length_km:,} km",
             ha="left", va="center", fontsize=FS_SCALE, zorder=10)


def _freq(arr, bins):
    if arr is None:
        return None
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return None
    counts, _ = np.histogram(valid, bins=bins)
    if counts.sum() == 0:
        return None
    return counts / counts.sum() * 100

def draw_hist(ax, arr_with, arr_rco, xlabel,
              vmin=None, vmax=None, is_change=False, show_ylabel=True):
    if is_change:
        step = max(1, (vmax - vmin) // 40)
        bins = np.arange(vmin, vmax + step, step)
    else:
        if arr_with is not None and np.any(~np.isnan(arr_with)):
            bins = np.histogram_bin_edges(arr_with[~np.isnan(arr_with)], bins=30)
        elif arr_rco is not None and np.any(~np.isnan(arr_rco)):
            bins = np.histogram_bin_edges(arr_rco[~np.isnan(arr_rco)], bins=30)
        else:
            bins = np.linspace(0, 1, 31)

    f_with = _freq(arr_with, bins)
    if f_with is not None:
        ax.bar(bins[:-1], f_with, width=np.diff(bins), align="edge",
               color=COL_WITH, alpha=ALPHA_WITH,
               edgecolor="white", linewidth=0.3, zorder=2, label="With CO₂")

    f_rco = _freq(arr_rco, bins)
    if f_rco is not None:
        ax.bar(bins[:-1], f_rco, width=np.diff(bins), align="edge",
               color=COL_NOCO, alpha=ALPHA_NOCO,
               edgecolor="white", linewidth=0.3, zorder=3, label="Fixed CO₂")

    if is_change:
        ax.axvline(0, color="0.35", linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
        ax.set_xlim(vmin, vmax)

    ax.set_xlabel(xlabel, fontsize=FS_HLAB, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Frequency (%)", fontsize=FS_HLAB, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2.5)

_AW = {}
def area_km2(ri):
    """D 内分季面积（km²）——只用于报告格网所代表的面积"""
    if ri not in _AW:
        a = read_tif(area_file(ri))
        _AW[ri] = supmask(np.where(np.isfinite(a) & (a > 0), a, np.nan), ri)
    return _AW[ri]

def area_w(ri):
    """汇总权重：cell → D 内每格 1；area → D 内面积"""
    a = area_km2(ri)
    return np.where(np.isfinite(a), 1.0, np.nan).astype(np.float32) if SUMMARY_WEIGHT == "cell" else a

def wstats(base, fut, ri, bmin=None, kfix=None):
    """同一组格网上的汇总统计；kfix = 各情景共用的格网（[1013]，保证基准值唯一）"""
    w = area_w(ri)
    ok = np.isfinite(base) & np.isfinite(fut) & np.isfinite(w)
    if kfix is not None:
        ok &= kfix
    if not ok.any():
        return None
    ww, b, f = w[ok], base[ok], fut[ok]
    bm, fm = np.sum(ww * b) / ww.sum(), np.sum(ww * f) / ww.sum()
    chg = pct_change(fut, base, bmin)[ok]
    okc = np.isfinite(chg)
    def wq(x, wt, q):
        o = np.argsort(x); c = np.cumsum(wt[o]) / wt[o].sum()
        return float(x[o][np.searchsorted(c, q)])
    return dict(n_cells=int(ok.sum()), area_Mha=round(float(np.nansum(area_km2(ri)[ok])) / 1e4, 3),
                base_mean=bm, fut_mean=fm, abs_change=fm - bm,
                rel_change_pct=100 * (fm - bm) / abs(bm),
                cell_chg_p10=wq(chg[okc], ww[okc], 0.10) if okc.any() else np.nan,
                cell_chg_median=wq(chg[okc], ww[okc], 0.50) if okc.any() else np.nan,
                cell_chg_p90=wq(chg[okc], ww[okc], 0.90) if okc.any() else np.nan,
                decrease_share_pct=100 * ww[f < b].sum() / ww.sum())

SUMMARY_ROWS = []
KFIX = {}

print(f"开始绘制 {len(TARGETS)} 张米质变化图 (v0817 布局优化版)...")

for t_idx, target in enumerate(TARGETS):
    print(f"  [{t_idx+1}/{len(TARGETS)}] {target}")

    data     = {}
    data_rco = {}
    bmin = BASE_MIN_FOR_PCT.get(target)

    for ri in RI_TYPES:
        base   = supmask(read_tif(HIS_DIR     / f"ensemble_historical_{ri}_{target}_{HIS_P}.tif"), ri)
        base_r = supmask(read_tif(HIS_DIR_RCO / f"ensemble_historical_{ri}_{target}_{HIS_P}.tif"), ri)
        if CO2_CF_BASELINE == "withco2" and base is not None:
            base_r = base
        data    [f"baseline_{ri}"] = base
        data_rco[f"baseline_{ri}"] = base_r

        if bmin is not None and base is not None:
            fin   = ~np.isnan(base)
            n_fin = int(fin.sum())
            n_msk = int((fin & (np.abs(base) < bmin)).sum())
            if n_fin:
                print(f"    [{ri}] baseline < {bmin:g} 掩膜: "
                      f"{n_msk}/{n_fin} 像元 ({100*n_msk/n_fin:.1f}%)")

        for ssp in SSPS:
            fut   = supmask(read_tif(FUT_DIR     / f"ensemble_{ssp}_{ri}_{target}_{FUT_P}.tif"), ri)
            fut_r = supmask(read_tif(FUT_DIR_RCO / f"ensemble_{ssp}_{ri}_{target}_{FUT_P}.tif"), ri)
            data    [f"{ssp}_{ri}"]     = fut
            data    [f"{ssp}_{ri}_chg"] = pct_change(fut,   base,   bmin)
            data_rco[f"{ssp}_{ri}"]     = fut_r
            data_rco[f"{ssp}_{ri}_chg"] = pct_change(fut_r, base_r, bmin)

    if all(data[f"baseline_{ri}"] is None for ri in RI_TYPES):
        print(f"    ⚠️ 无 {target} 预测结果（模型缺失？），跳过")
        continue

    for ri in RI_TYPES:
        _arrs = [data.get(f"baseline_{ri}"), data_rco.get(f"baseline_{ri}")] + \
                [dd.get(f"{s}_{ri}") for s in SSPS for dd in (data, data_rco)]
        kfix = np.ones(area_w(ri).shape, bool)
        for _a in _arrs:
            if _a is not None:
                kfix &= np.isfinite(_a)
        KFIX[(target, ri)] = kfix
        for ssp in SSPS:
            for var, dd in (("with_CO2", data), ("fixed_CO2", data_rco)):
                b, f = dd.get(f"baseline_{ri}"), dd.get(f"{ssp}_{ri}")
                if b is None or f is None:
                    continue
                st = wstats(b, f, ri, bmin, kfix)
                if st:
                    SUMMARY_ROWS.append(dict(trait=target, ri=ri, ssp=ssp, co2=var, **st))

    all_base_vals = []
    for ri in RI_TYPES:
        a = data[f"baseline_{ri}"]
        if a is not None:
            all_base_vals.append(a[~np.isnan(a)])
    if all_base_vals:
        combined = np.concatenate(all_base_vals)
        BASE_VMIN, BASE_VMAX = float(np.percentile(combined, 2)), float(np.percentile(combined, 98))
    else:
        BASE_VMIN, BASE_VMAX = 0, 1
    norm_base = Normalize(vmin=BASE_VMIN, vmax=BASE_VMAX)

    all_chg_vals = []
    for ssp in SSPS:
        for ri in RI_TYPES:
            a = data.get(f"{ssp}_{ri}_chg")
            if a is not None:
                v = a[~np.isnan(a)]
                if len(v):
                    all_chg_vals.append(v)
    if target in CHANGE_RANGE_OVERRIDE:
        chg_vmin, chg_vmax = CHANGE_RANGE_OVERRIDE[target]
    elif all_chg_vals:
        combined_chg = np.concatenate(all_chg_vals)
        cp2, cp98 = np.percentile(combined_chg, [2, 98])
        chg_vmin, chg_vmax = symmetric_range(cp2, cp98, step=5)
    else:
        chg_vmin, chg_vmax = -10, 10

    norm_chg  = TwoSlopeNorm(vmin=chg_vmin, vcenter=0, vmax=chg_vmax)
    base_cmap = PAL.CMAP_SEQ

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
    scenarios    = ["baseline", "ssp126", "ssp585"]
    panel_labels = list("abcdefghijkl")

    hist_axes = []

    for row_i, scen in enumerate(scenarios):
        is_base = (scen == "baseline")
        norm  = norm_base if is_base else norm_chg
        cmap  = base_cmap if is_base else CHANGE_CMAP
        base_label_idx = row_i * 4
        is_bottom = (row_i == 2)

        for col_i, ri in enumerate(RI_TYPES):
            label = panel_labels[base_label_idx + col_i]
            ax_m = fig.add_subplot(gs[row_i, col_i])
            if row_i == 0 and col_i == 0: ax_topleft = ax_m
            arr_map = (data[f"baseline_{ri}"] if is_base
                       else data[f"{scen}_{ri}_chg"])
            _zone_on = (SHOW_ZONES and zones is not None and
                        (ZONE_ONLY_FOR is None or target in ZONE_ONLY_FOR))
            draw_map(ax_m, arr_map, norm, cmap,
                     ylabel=ROW_LABELS[scen] if col_i == 0 else "",
                     draw_zone=_zone_on,
                     show_lon=is_bottom,
                     show_lat=(col_i == 0))
            if row_i == 0:
                ax_m.set_title(RI_LABELS[ri], fontsize=FS_TITLE - 1, pad=4)
            xoff = -0.16 if col_i == 0 else -0.06
            ax_m.text(xoff, 1.16, f"({label})",
                      transform=ax_m.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="top")

        for col_i, ri in enumerate(RI_TYPES):
            label = panel_labels[base_label_idx + 2 + col_i]
            ax_h = fig.add_subplot(gs[row_i, col_i + 3])
            show_yl = (col_i == 0)

            if is_base:
                draw_hist(
                    ax_h,
                    arr_with = data[f"baseline_{ri}"],
                    arr_rco  = None,
                    xlabel   = TARGET_LABELS[target],
                    is_change= False,
                    show_ylabel=show_yl,
                )
            else:
                draw_hist(
                    ax_h,
                    arr_with  = data    [f"{scen}_{ri}_chg"],
                    arr_rco   = data_rco[f"{scen}_{ri}_chg"],
                    xlabel    = "Change (%)",
                    vmin=chg_vmin, vmax=chg_vmax,
                    is_change = True,
                    show_ylabel=show_yl,
                )
            if col_i == 1:
                ax_h.tick_params(labelleft=False)
            if row_i == 0:
                ax_h.set_title(RI_LABELS[ri], fontsize=FS_TITLE - 1, pad=4)
            xoff = -0.24 if col_i == 0 else -0.08
            ax_h.text(xoff, 1.16, f"({label})",
                      transform=ax_h.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="top")
            hist_axes.append(ax_h)

        cbar_gap = 0.06 if is_bottom else 0.028
        pos_l = gs[row_i, 0].get_position(fig)
        pos_r = gs[row_i, 1].get_position(fig)
        cax = fig.add_axes([
            pos_l.x0 + pos_l.width * 0.15,
            pos_l.y0 - cbar_gap,
            (pos_r.x1 - pos_l.x0) * 0.70,
            0.016,
        ])
        if is_base:
            sm = ScalarMappable(cmap=base_cmap, norm=norm_base)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(TARGET_LABELS[target], fontsize=FS_CBLAB, labelpad=2)
        else:
            sm = ScalarMappable(cmap=CHANGE_CMAP, norm=norm_chg)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label("Change (%)", fontsize=FS_CBLAB, labelpad=2)
        cb.ax.tick_params(labelsize=FS_CBTICK, length=2.5)

        if scen == "ssp585":
            ph_l = gs[row_i, 3].get_position(fig)
            ph_r = gs[row_i, 4].get_position(fig)
            fig.legend(
                handles=[
                    Patch(facecolor=COL_WITH, alpha=ALPHA_WITH,
                          edgecolor="white", linewidth=0.3, label="With CO₂"),
                    Patch(facecolor=COL_NOCO, alpha=ALPHA_NOCO,
                          edgecolor="white", linewidth=0.3, label="Fixed CO₂ (baseline)"),
                ],
                ncol=2, fontsize=FS_LEG, frameon=False,
                loc="center",
                bbox_to_anchor=((ph_l.x0 + ph_r.x1) / 2, ph_l.y0 - 0.085),
                bbox_transform=fig.transFigure,
                handlelength=1.5, handleheight=0.9, handletextpad=0.5,
                columnspacing=2.5,
            )

    if hist_axes:
        ymax = max(ax.get_ylim()[1] for ax in hist_axes)
        for ax in hist_axes:
            ax.set_ylim(0, ymax)

    _p0 = ax_topleft.get_position()
    add_scale_bar(ax_topleft, x_left=_p0.x0 + 0.005, cy=_p0.y0 + 0.005)

    fname = OUT_PATH / target
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {target}.png")

import pandas as pd
mem_rows = []
ROOT_W = PRED_V5 / PRED_MODE
for target in TARGETS:
    for var, root in (("with_CO2", ROOT_W), ("fixed_CO2", PRED_V5 / "fixedco2" / PRED_MODE)):
        broot = ROOT_W if CO2_CF_BASELINE == "withco2" else root
        for ri in RI_TYPES:
            if (target, ri) not in KFIX:
                continue
            for g in GCMS:
                b = supmask(read_tif(broot / "historical" / f"{g}_historical_{ri}_{target}_{HIS_P}.tif"), ri)
                if b is None:
                    continue
                for m in CROP_MODELS:
                    fs = {ssp: supmask(read_tif(root / ssp / f"{m}_{g}_{ssp}_{ri}_{target}_{FUT_P}.tif"), ri)
                          for ssp in SSPS}
                    if any(f is None for f in fs.values()):
                        continue
                    km = KFIX[(target, ri)] & np.isfinite(b)
                    for f in fs.values():
                        km &= np.isfinite(f)
                    for ssp, f in fs.items():
                        st = wstats(b, f, ri, BASE_MIN_FOR_PCT.get(target), km)
                        if st:
                            mem_rows.append(dict(trait=target, ri=ri, ssp=ssp, co2=var, model=m, gcm=g,
                                                 rel_change_pct=st["rel_change_pct"],
                                                 abs_change=st["abs_change"],
                                                 decrease_share_pct=st["decrease_share_pct"]))
mem = pd.DataFrame(mem_rows)
summ = pd.DataFrame(SUMMARY_ROWS)
if len(mem):
    mem.to_csv(OUT_PATH / "quality_change_members.csv", index=False, float_format="%.4f")
    agg = (mem.groupby(["trait", "ri", "ssp", "co2"])["rel_change_pct"]
              .agg(member_median="median", member_min="min", member_max="max",
                   n_members="count",
                   frac_same_sign=lambda x: max((x > 0).mean(), (x < 0).mean()))
              .reset_index())
    summ = summ.merge(agg, on=["trait", "ri", "ssp", "co2"], how="left")
if len(summ):
    w = summ[summ.co2 == "with_CO2"].set_index(["trait", "ri", "ssp"])["rel_change_pct"]
    fx = summ[summ.co2 == "fixed_CO2"].set_index(["trait", "ri", "ssp"])["rel_change_pct"]
    summ["co2_effect_pp"] = [w.get(k, np.nan) - fx.get(k, np.nan) if c == "with_CO2" else np.nan
                             for k, c in zip(zip(summ.trait, summ.ri, summ.ssp), summ.co2)]
    summ.to_csv(OUT_PATH / "quality_change_summary.csv", index=False, float_format="%.4f")

    RI_TXT = {"ri1": "single/early rice", "ri2": "late rice"}
    SSP_TXT = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
    lines = [f"Grain-quality change, {FUT_P.replace('_', '–')} vs {HIS_P.replace('_', '–')} "
             f"({'area-weighted' if SUMMARY_WEIGHT == 'area' else 'grid-cell mean'} over the common model-support domain; "
             f"ensemble mean; member median [range]; {PRED_MODE})", ""]
    for (t, ri), sub in summ.groupby(["trait", "ri"], sort=False):
        bw = sub[sub.co2 == "with_CO2"].base_mean
        lines.append(f"{TARGET_LABELS.get(t, t)} — {RI_TXT[ri]}: baseline {(bw if len(bw) else sub.base_mean).iloc[0]:.2f}"
                     + ("" if CO2_CF_BASELINE == "withco2" else " (fixed-CO₂ rows use their own fixed-CO₂ baseline)"))
        for _, r in sub.iterrows():
            mm = (f"; members {r.member_median:+.1f}% [{r.member_min:+.1f}, {r.member_max:+.1f}], "
                  f"{100 * r.frac_same_sign:.0f}% same sign") if "member_median" in r and pd.notna(r.member_median) else ""
            co2 = f"; CO₂ effect {r.co2_effect_pp:+.1f} pp" if pd.notna(r.get("co2_effect_pp", np.nan)) else ""
            lines.append(f"  {SSP_TXT[r.ssp]} {r.co2:9s}: {r.rel_change_pct:+.1f}% "
                         f"({r.base_mean:.2f}→{r.fut_mean:.2f}); cells p10–p90 {r.cell_chg_p10:+.1f} to {r.cell_chg_p90:+.1f}%; "
                         f"decrease over {r.decrease_share_pct:.0f}% of {'the rice area' if SUMMARY_WEIGHT == 'area' else 'grid cells'}{mm}{co2}")
        lines.append("")
    (OUT_PATH / "quality_change_text.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] quality_change_summary.csv / quality_change_members.csv / quality_change_text.txt")

print(f"\n全部完成！共输出 {len(TARGETS)} 张图")
print(f"输出目录: {OUT_PATH}")
