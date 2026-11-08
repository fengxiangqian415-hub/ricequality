# ============================================================
# 米质变化图 v6 (0817 布局优化版)
# ★ v5 数据版（布局与配色不变）：
#   · 读取 04_predict_rice_quality_1004.py 输出；时段、路径来自 config_1004
#   · 主图 = 随情景 CO₂（v5_1004/{PRED_MODE}）；对照 = 固定 CO₂（v5_1004/fixedco2/{PRED_MODE}）
#   · 地图与直方图只显示共同支撑区（≥25/30 年，make_baseline_support.py）
#   · 无模型/无数据的指标（如 PCA）自动跳过
#   · ★ 同时输出正文描述用的统计表（统一格网域 D，权重 config.SUMMARY_WEIGHT，定稿为面积加权）：
#       quality_change_summary.csv   集合平均：基准/未来均值、相对变化、格网变化分位数、下降面积占比
#       quality_change_members.csv   逐成员全国相对变化 → 中位数 [范围] 与符号一致性
#       quality_change_text.txt      可直接引用的要点
# 本次改动（在 v0817 基础上）：
#   1. 字体整体加大到 ~15
#   2. 经纬度刻度精简：LON_TICKS=[80,100,120] / LAT_TICKS=[20,30,40,50]
#      · 去掉 "Lon (°E)" / "Lat (°N)" 轴标题
#      · 刻度标签带方向：80°E / 20°N ...
#      · 只显示最外圈：经度仅底行地图、纬度仅左列地图
#   3. 色条竖排，置于地图(左两列)与直方图(右两列)之间
#   4. 纵向排布收紧（hspace 减小、去掉地图下方横色条）
#   5. 右侧 6 张直方图纵轴统一到同一范围，仅左列显示刻度/标题
# 其余逻辑（数据路径、区划叠加、baseline 下限掩膜、配色）保持不变
# [1013] 统一口径：
#   · 统计格网 = 统一格网域 D 内、基准与两个情景（含固定 CO2）都有值的同一组格网 → 每个性状×季型一个基准值
#   · 固定 CO2 反事实的基准见 config.CO2_CF_BASELINE（定稿 "withco2"：唯一基准 = 历史逐年实际 CO2 预测的均值）：
#       CO2 效应 = 100 ×（F_随CO2 − F_固定CO2）/ B，与 Methods“两次预测之差”一致
#   · 成员：同一组格网中该成员（两个情景）都有值处
# ============================================================

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

# ── 全文统一配色: 同目录放 paper_palette.py ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_palette as PAL

# ── 全局字号 ~15 ────────────────────────────────────────
FS_BASE   = 15    # 基础
FS_TICK   = 13    # 刻度
FS_TITLE  = 15    # 列标题 (Single/Early rice ...)
FS_ROWLAB = 15    # 行标签 (Baseline ...)
FS_PANEL  = 16    # 面板编号 (a)(b)...
FS_CBLAB  = 14    # 色条标题
FS_CBTICK = 12    # 色条刻度
FS_HLAB   = 14    # 直方图 x/y 轴标题
FS_SCALE  = 10    # 比例尺标注
FS_LEG    = 13    # 图例

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

# ======================================================
# 路径配置
# ======================================================
from config_1004 import (OUTPUT as _OUT, FIGURES, TAG, PRED_MODE, HIS_P, FUT_P, HIS_LABEL, FUT_LABEL,
                         CHINA_SHP, POLY_SHP, CROP_MODELS, GCMS, pred_dir, support_file, area_file,
                         CO2_CF_BASELINE, SUMMARY_WEIGHT)
BASE        = _OUT.parent
PRED_V5     = pred_dir(PRED_MODE, "withco2").parent                    # v5_1004 根目录
HIS_DIR     = pred_dir(PRED_MODE, "withco2") / "ensemble"              # 随情景 CO₂
FUT_DIR     = HIS_DIR
HIS_DIR_RCO = pred_dir(PRED_MODE, "fixedco2") / "ensemble"             # 固定 CO₂
FUT_DIR_RCO = HIS_DIR_RCO

ZONE_SHP  = "/home/debian/pythonfile/china/rice_zones.shp"   # ★ 稻作区划（籼粳权重来源）
SHOW_ZONES = True          # ★ 是否在地图上叠加 6 大稻作区边界（仅装饰，可关）
ZONE_ONLY_FOR = {"Acontent"}   # 只在直链淀粉图上叠区划；设为 None 则所有指标都叠
OUT_PATH  = FIGURES / f"rice_quality_{TAG}_{PRED_MODE}"
OUT_PATH.mkdir(parents=True, exist_ok=True)

# ======================================================
# 配置
# ======================================================
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
RI_LABELS = {"ri1": "Single/early rice", "ri2": "Late rice"}   # [1105] 与正文统一为 Single/early
SSPS      = ["ssp126", "ssp585"]
CHANGE_CMAP = PAL.CMAP_DIV        # "RdBu_r" 发散(色盲安全)

# ── 手动锁定 change 范围（%）：仅对列出的指标生效 ────────
CHANGE_RANGE_OVERRIDE = {
    "BRR": (-10, 10),
}

# ── 经纬度刻度（精简）────────────────────────────────────
LON_TICKS = [80, 100, 120]
LAT_TICKS = [20, 30, 40, 50]

# ── 直方图：全局统一配色（两色实心柱重叠）────────────
COL_WITH = PAL.CO2["with"]        # With CO₂    → 蓝
COL_NOCO = PAL.CO2["without"]     # Without CO₂ → 橙
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
SCALE_KM = 1000                 # 比例尺长度(km)
LAT_REF  = 35                   # 比例尺参考纬度

# ======================================================
# 辅助函数
# ======================================================
def symmetric_range(cp2, cp98, step=5):
    abs_max = max(abs(cp2), abs(cp98))
    rounded = int(np.ceil(abs_max / step) * step)
    return -max(rounded, step), max(rounded, step)

china   = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
polygon = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
nanhai  = polygon[polygon.geometry.apply(lambda g: g.bounds[1] < 25)]
jd_line = polygon[polygon["adchar"] == "JD"]
lon_min, lat_min, lon_max, lat_max = china.total_bounds

# ★ 稻作区划（可选叠加）
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

# ✅ 相对变化的 baseline 下限（同原版）
BASE_MIN_FOR_PCT = {
    "CGR": 5.0,
    "CD":  2.0,
}

def pct_change(fut, base, base_min=None):
    """相对变化(%)；base 低于 base_min 的格子置 NaN。"""
    if fut is None or base is None:          # ★ 缺失指标（如 PCA）
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

    # ── 精简刻度 + 只显示最外圈 ──────────────────────────
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    ax.set_xticklabels([f"{v}°E" for v in LON_TICKS] if show_lon else [])
    ax.set_yticklabels([f"{v}°N" for v in LAT_TICKS] if show_lat else [])
    ax.tick_params(labelsize=FS_TICK, length=2.5)

    # ── 去掉 Lon/Lat 轴标题；仅左列保留行标签(粗体) ──────
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_ROWLAB, labelpad=6, fontweight="bold")

    add_nanhai_inset(ax, arr, norm, cmap)

# ======================================================
# 直方图：两色实心柱重叠（透明度混色）
# ======================================================
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
    # ── 确定 bins（刻度保持原版）────────────────────
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

# ======================================================
# 主循环
# ======================================================
# ======================================================
# ★ 全国汇总（[1013] 统一格网域 D；汇总权重 config.SUMMARY_WEIGHT，定稿为面积加权）
# ======================================================
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
                decrease_share_pct=100 * ww[f < b].sum() / ww.sum())          # 下降格网占比（%）

SUMMARY_ROWS = []
KFIX = {}          # (性状, ri) → 各情景共用格网（成员统计也在其上）

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
            base_r = base            # [1013] 固定 CO2 的未来与同一个基准（随情景 CO2）比较
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

    # ★ 集合平均统计（With CO₂ / Fixed CO₂）
    for ri in RI_TYPES:
        # [1013] 各情景、两套 CO2 共用一组格网
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

    # 地图色标（仅含 CO₂）
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

    # change 色标 & 直方图区间：仅由 With CO₂ 决定；BRR 手动锁定
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
    base_cmap = PAL.CMAP_SEQ       # "YlOrRd" 顺序(色盲安全)

    # ======================================================
    # 布局：3行 × 4列（色条恢复为横向，置于每行左侧两地图正下方）
    #   col0,1 = 地图(ri1/ri2)   col2,3 = 直方图(ri1/ri2)
    # ======================================================
    # ======================================================
    # 布局：3行 × 5列
    #   col0,1 = 地图   col2 = 空占位(隔开地图块与直方图块)   col3,4 = 直方图
    #   · wspace 只影响"块内"两图间距(地图↔地图 / 直方图↔直方图)，调小即可收紧
    #   · 地图块↔直方图块的间距 = 占位列宽(SPACER_W)，独立可调，
    #     保证给直方图纵轴留出空间，不受 wspace 缩小影响
    # ======================================================
    SPACER_W = 0.08        # ← 地图块↔直方图块 间距（越大越远）
    MAP_HIST_WSPACE = 0.20 # ← 块内两图间距（越小越紧）
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

    hist_axes = []   # 收集所有直方图 axes，循环后统一纵轴

    for row_i, scen in enumerate(scenarios):
        is_base = (scen == "baseline")
        norm  = norm_base if is_base else norm_chg
        cmap  = base_cmap if is_base else CHANGE_CMAP
        base_label_idx = row_i * 4
        is_bottom = (row_i == 2)

        # ── Step 1：地图（col 0,1）──────────────────────
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
                     show_lon=is_bottom,        # 经度仅底行
                     show_lat=(col_i == 0))     # 纬度仅左列
            if row_i == 0:
                ax_m.set_title(RI_LABELS[ri], fontsize=FS_TITLE - 1, pad=4)   # [1105] 列标题不加粗、略小，避免与 (c) 面板号重合
            xoff = -0.16 if col_i == 0 else -0.06
            ax_m.text(xoff, 1.16, f"({label})",
                      transform=ax_m.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="top")

        # ── Step 2：直方图（col 3,4）────────────────────
        for col_i, ri in enumerate(RI_TYPES):
            label = panel_labels[base_label_idx + 2 + col_i]
            ax_h = fig.add_subplot(gs[row_i, col_i + 3])
            show_yl = (col_i == 0)   # 纵轴标题仅左列

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
                ax_h.tick_params(labelleft=False)   # 右列隐藏纵轴刻度标签
            if row_i == 0:
                ax_h.set_title(RI_LABELS[ri], fontsize=FS_TITLE - 1, pad=4)   # [1105] 同上
            xoff = -0.24 if col_i == 0 else -0.08
            ax_h.text(xoff, 1.16, f"({label})",
                      transform=ax_h.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="top")
            hist_axes.append(ax_h)

        # ── 横色条：置于每行左侧两个地图正下方（原版布局）──────
        #    上面两行(baseline/ssp126)色条再上移一点；底行(ssp585)不动
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

        # ── 图例：底行直方图下方，横排 ───────────────────
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

    # ── 统一右侧直方图纵轴（0 ~ 全局最大频率）──────────────
    if hist_axes:
        ymax = max(ax.get_ylim()[1] for ax in hist_axes)
        for ax in hist_axes:
            ax.set_ylim(0, ymax)

    # 左上角地图加比例尺
    _p0 = ax_topleft.get_position()
    add_scale_bar(ax_topleft, x_left=_p0.x0 + 0.005, cy=_p0.y0 + 0.005)

    fname = OUT_PATH / target
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {target}.png")

# ======================================================
# ★ 逐成员统计（全国相对变化，同一口径；成员基准 = 对应 GCM 的基准）
# ======================================================
import pandas as pd
mem_rows = []
ROOT_W = PRED_V5 / PRED_MODE
for target in TARGETS:
    for var, root in (("with_CO2", ROOT_W), ("fixed_CO2", PRED_V5 / "fixedco2" / PRED_MODE)):
        broot = ROOT_W if CO2_CF_BASELINE == "withco2" else root      # [1013] 基准随 CO2_CF_BASELINE
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
    # CO2 贡献 = With − Fixed（相对变化，百分点）
    w = summ[summ.co2 == "with_CO2"].set_index(["trait", "ri", "ssp"])["rel_change_pct"]
    fx = summ[summ.co2 == "fixed_CO2"].set_index(["trait", "ri", "ssp"])["rel_change_pct"]
    summ["co2_effect_pp"] = [w.get(k, np.nan) - fx.get(k, np.nan) if c == "with_CO2" else np.nan
                             for k, c in zip(zip(summ.trait, summ.ri, summ.ssp), summ.co2)]
    summ.to_csv(OUT_PATH / "quality_change_summary.csv", index=False, float_format="%.4f")

    # 文字要点
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
