# ============================================================
# 物候验证图 v4
# 每张图：3行(baseline/ssp126/ssp585) × 4列(ri1地图|ri2地图|ri1直方图|ri2直方图)
# 共25张（5模型×5GCM），所有地图共用一个colorbar
# ★ v5（与气候/米质图统一）：
#   · 基准期 = 唯一历史物候（2000 年前 *_ens25 + 2000 年起 *_observed）；时段见 config_1004
#   · 未来期（step3 校正结果）；时段见 config_1004
#   · 只显示共同支撑区；逐年剔除 HE>=MA（与气候提取一致）
#   · 新增第 26 张：25 成员集合平均（ensemble）
#   · 同时输出全国汇总 phenology_summary.csv（抽穗/成熟 DOY、灌浆天数及其变化）
# [1013] 统一口径：统一格网域 D（03b）；汇总权重 config.SUMMARY_WEIGHT；集合行的格网 = 基准与两个情景
#        都有值的同一组格网（每个季型一个基准值）；成员行 = 同一组格网中该成员有值处。灌浆天数定义不变
# ============================================================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import rasterio
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── 全局字号 ~15（与 rice quality 图统一）────────────
FS_BASE   = 15    # 基础
FS_TICK   = 13    # 刻度
FS_TITLE  = 15    # 列标题 (Single/Early rice ...)
FS_ROWLAB = 15    # 行标签 (Baseline ...)
FS_PANEL  = 16    # 面板编号 (a)(b)...
FS_CBLAB  = 14    # 色条标题
FS_CBTICK = 12    # 色条刻度
FS_HLAB   = 14    # 直方图 / 地图 x/y 轴标题

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": FS_BASE,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

# ======================================================
# 路径配置
# ======================================================
from config_1004 import (HIS_ANN as PHENO_ANN, PHENO_FUT, CHINA_SHP, POLY_SHP, FIGURES, TAG,
                         HIS_YEARS, FUT_YEARS, HIS_LABEL, FUT_LABEL, CROP_MODELS, GCMS, SSPS, RI_TYPES,
                         support_file, area_file, hist_pheno_tag)
import stats_common_1004 as S
OUT_PATH  = FIGURES / f"phenology_{TAG}"
OUT_PATH.mkdir(parents=True, exist_ok=True)

# 成员、情景、时段来自 config_1004

LON_VALS = np.arange(73.25, 136.0, 0.5)
LAT_VALS = np.arange(53.75, 17.75, -0.5)
EXTENT   = [LON_VALS[0]-0.25, LON_VALS[-1]+0.25,
            LAT_VALS[-1]-0.25, LAT_VALS[0]+0.25]

# ★ 所有地图共用同一colorbar（ri1/ri2合并范围）
DOY_CMAP = "RdYlBu_r"
DOY_VMIN = 80
DOY_VMAX = 320

HIST_COLORS = {
    "baseline": "#2166AC",
    "ssp126":   "#4DAF4A",
    "ssp585":   "#D73027",
}
HIST_LABELS = {
    "baseline": f"Baseline ({HIS_LABEL})",
    "ssp126":   f"SSP1-2.6 ({FUT_LABEL})",
    "ssp585":   f"SSP5-8.5 ({FUT_LABEL})",
}
ROW_LABELS = {
    "baseline": f"Baseline\n({HIS_LABEL})",
    "ssp126":   f"SSP1-2.6\n({FUT_LABEL})",
    "ssp585":   f"SSP5-8.5\n({FUT_LABEL})",
}
RI_LABELS = {"ri1": "Single/early rice", "ri2": "Late rice"}   # [1105] 与正文统一为 Single/early
BINS = np.arange(0, 121, 5)

# ======================================================
# 工具函数
# ======================================================
china    = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
polygon  = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
nanhai   = polygon[polygon.geometry.apply(lambda g: g.bounds[1] < 25)]
jd_line  = polygon[polygon["adchar"] == "JD"]
lon_min, lat_min, lon_max, lat_max = china.total_bounds
norm_global = Normalize(vmin=DOY_VMIN, vmax=DOY_VMAX)

# 南海inset地理范围
INSET_EXTENT = [105, 2, 125, 26]   # [lon_min, lat_min, lon_max, lat_max]

# ── 经纬度刻度（精简；经度仅底行、纬度仅左列显示）──
LON_TICKS = [80, 100, 120]
LAT_TICKS = [20, 30, 40, 50]

def add_nanhai_inset(ax, arr, norm, cmap):
    """
    方案A：inset与主图共享同一imshow渲染，仅改变显示窗口范围
    → 像素块随显示区域自动缩放，视觉上是主图的缩小版
    """
    ax_pos = ax.get_position()
    iw = ax_pos.width  * 0.24
    ih = ax_pos.height * 0.30
    ix = ax_pos.x1 - iw * 0.9
    iy = ax_pos.y0 + ax_pos.height * 0.01
    ax_in = ax.figure.add_axes([ix, iy, iw, ih])

    # ★ 核心：与主图完全相同的imshow参数
    if arr is not None:
        ax_in.imshow(arr, extent=EXTENT, origin="upper",
                     cmap=cmap, norm=norm,
                     aspect="auto", interpolation="nearest")

    nanhai.boundary.plot(ax=ax_in, linewidth=0.3, color="black", zorder=4)
    jd_line.boundary.plot(ax=ax_in, linewidth=0.5, color="black",
                          linestyle="-", zorder=5)

    # ★ 核心：只改变显示窗口
    ax_in.set_xlim(INSET_EXTENT[0], INSET_EXTENT[2])
    ax_in.set_ylim(INSET_EXTENT[1], INSET_EXTENT[3])
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    return ax_in

def read_tif(path):
    if not Path(path).exists():
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == np.float32(ds.nodata)] = np.nan
        arr[arr <= 0] = np.nan
    return arr

_SUP = {}
def support(ri):
    if ri not in _SUP:
        s_ = read_tif(support_file(ri))
        _SUP[ri] = np.isfinite(s_) if s_ is not None else np.ones((len(LAT_VALS), len(LON_VALS)), bool)
    return _SUP[ri]

def read_pair(f_he, f_ma, ri):
    """★ 读取一年 HE/MA：支撑区外、缺失、HE>=MA 均置 NaN（与气候提取一致）"""
    he, ma = read_tif(f_he), read_tif(f_ma)
    if he is None or ma is None:
        return None, None
    bad = ~support(ri) | ~np.isfinite(he) | ~np.isfinite(ma) | (he >= ma)
    he[bad] = np.nan; ma[bad] = np.nan
    return he, ma

def nanmean_list(lst):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(lst), axis=0) if lst else None

def filling_dur(he, ma):
    if he is None or ma is None:
        return np.array([])
    d = ma - he
    return d[(~np.isnan(d)) & (d > 0) & (d <= 120)]

def draw_map(ax, he_arr, ylabel="", show_lon=False, show_lat=False):
    """绘制开花期DOY地图，使用全局norm"""
    if he_arr is not None:
        ax.imshow(he_arr, extent=EXTENT, origin="upper",
                  cmap=DOY_CMAP, norm=norm_global, aspect="auto")
    china.boundary.plot(ax=ax, linewidth=0.35, color="black", zorder=5)
    ax.set_xlim(lon_min-0.5, lon_max+0.5)
    ax.set_ylim(lat_min-0.5, lat_max+0.5)
    # ── 精简刻度 + 只显示最外圈（经度仅底行、纬度仅左列）──
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    ax.set_xticklabels([f"{v}°E" for v in LON_TICKS] if show_lon else [])
    ax.set_yticklabels([f"{v}°N" for v in LAT_TICKS] if show_lat else [])
    ax.tick_params(labelsize=FS_TICK, length=2.5)
    # ── 去掉 Lon/Lat 轴标题；仅左列保留行标签(粗体)──
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_ROWLAB, labelpad=6, fontweight="bold")
    add_nanhai_inset(ax, he_arr, norm_global, DOY_CMAP)

def draw_hist(ax, dur, scen, show_ylabel=True, show_xlabel=True):
    """绘制灌浆期频率直方图"""
    color = HIST_COLORS[scen]
    if len(dur) > 0:
        counts, edges = np.histogram(dur, bins=BINS)
        freq = counts / counts.sum() * 100
        ax.bar(edges[:-1], freq, width=np.diff(edges),
               align="edge", color=color, alpha=0.85,
               edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, 120)
    if show_xlabel:
        ax.set_xlabel("Grain filling (d)", fontsize=FS_HLAB, labelpad=2)
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Frequency (%)", fontsize=FS_HLAB, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2.5)

# ======================================================
# 预计算baseline（★ 唯一历史物候：2000 年前 ens25 + 2000 年起 observed）
# ======================================================
print("计算baseline均值...")
baseline_he, baseline_ma = {}, {}
for ri in RI_TYPES:
    he_stack, ma_stack = [], []
    for yr in HIS_YEARS:
        suf = hist_pheno_tag(yr)
        he, ma = read_pair(PHENO_ANN / f"{ri}_HE_{yr}_{suf}.tif", PHENO_ANN / f"{ri}_MA_{yr}_{suf}.tif", ri)
        if he is not None:
            he_stack.append(he); ma_stack.append(ma)
    baseline_he[ri] = nanmean_list(he_stack)
    baseline_ma[ri] = nanmean_list(ma_stack)
    print(f"  {ri}: {len(he_stack)} 年  HE DOY {np.nanmin(baseline_he[ri]):.0f}~{np.nanmax(baseline_he[ri]):.0f}")


def draw_figure(fname_stem, fut_he, fut_ma):
    # ── 构建图 ──
    # 布局：3行 × 5列
    # 列：ri1地图 | ri2地图 | ri1直方图 | ri2直方图 | colorbar
    # ── 布局：3行×4列地图+直方图，底部留colorbar空间 ──
    SPACER_W = 0.08         # 地图块↔直方图块 间距
    MAP_HIST_WSPACE = 0.20  # 块内两图间距
    fig = plt.figure(figsize=(240/25.4, 185/25.4))
    gs = gridspec.GridSpec(
        3, 5,
        figure=fig,
        left=0.075, right=0.985,
        top=0.93, bottom=0.13,   # bottom留出colorbar空间
        hspace=0.35, wspace=MAP_HIST_WSPACE,
        width_ratios=[1.4, 1.4, SPACER_W, 1.0, 1.0],
    )

    scenarios = ["baseline", "ssp126", "ssp585"]
    panel_labels = list("abcdefghijkl")
    label_idx = 0
    all_map_axes = []   # 收集地图ax，用于放置colorbar
    hist_axes = []      # 收集直方图ax，循环后统一纵轴

    for row_i, scen in enumerate(scenarios):
        if scen == "baseline":
            he1 = baseline_he["ri1"]; ma1 = baseline_ma["ri1"]
            he2 = baseline_he["ri2"]; ma2 = baseline_ma["ri2"]
        else:
            he1 = fut_he[scen]["ri1"]; ma1 = fut_ma[scen]["ri1"]
            he2 = fut_he[scen]["ri2"]; ma2 = fut_ma[scen]["ri2"]

        # ri1地图
        ax_m1 = fig.add_subplot(gs[row_i, 0])
        draw_map(ax_m1, he1, ylabel=ROW_LABELS[scen],
                 show_lon=(row_i == 2), show_lat=True)
        if row_i == 0:
            ax_m1.set_title(RI_LABELS["ri1"], fontsize=FS_TITLE, pad=12)
        ax_m1.text(-0.12, 1.16, f"({panel_labels[label_idx]})",  # ★ 上移到1.16
                   transform=ax_m1.transAxes, fontsize=FS_PANEL,
                   fontweight="bold", va="top")
        all_map_axes.append(ax_m1)
        label_idx += 1

        # ri2地图
        ax_m2 = fig.add_subplot(gs[row_i, 1])
        draw_map(ax_m2, he2, show_lon=(row_i == 2), show_lat=False)
        if row_i == 0:
            ax_m2.set_title(RI_LABELS["ri2"], fontsize=FS_TITLE, pad=12)
        ax_m2.text(-0.08, 1.16, f"({panel_labels[label_idx]})",  # ★ 上移
                   transform=ax_m2.transAxes, fontsize=FS_PANEL,
                   fontweight="bold", va="top")
        all_map_axes.append(ax_m2)
        label_idx += 1

        # ri1直方图（完全不加标题）
        ax_h1 = fig.add_subplot(gs[row_i, 3])
        draw_hist(ax_h1, filling_dur(he1, ma1), scen, show_ylabel=True, show_xlabel=(row_i == 2))
        hist_axes.append(ax_h1)
        if row_i == 0:
            ax_h1.set_title(RI_LABELS["ri1"], fontsize=FS_TITLE, pad=12)
        ax_h1.text(-0.24, 1.16, f"({panel_labels[label_idx]})",
                   transform=ax_h1.transAxes, fontsize=FS_PANEL,
                   fontweight="bold", va="top")
        label_idx += 1

        # ri2直方图（完全不加标题）
        ax_h2 = fig.add_subplot(gs[row_i, 4])
        draw_hist(ax_h2, filling_dur(he2, ma2), scen, show_ylabel=False, show_xlabel=(row_i == 2))
        ax_h2.tick_params(labelleft=False)
        hist_axes.append(ax_h2)
        if row_i == 0:
            ax_h2.set_title(RI_LABELS["ri2"], fontsize=FS_TITLE, pad=12)
        ax_h2.text(-0.08, 1.16, f"({panel_labels[label_idx]})",
                   transform=ax_h2.transAxes, fontsize=FS_PANEL,
                   fontweight="bold", va="top")
        label_idx += 1

    # ── 统一所有直方图纵轴（0 ~ 全局最大频率）──
    if hist_axes:
        ymax = max(ax.get_ylim()[1] for ax in hist_axes)
        for ax in hist_axes:
            ax.set_ylim(0, ymax)

    # ── 共用colorbar：水平放置在左侧两地图正下方（居中）──
    sm = ScalarMappable(cmap=DOY_CMAP, norm=norm_global)
    sm.set_array([])
    pos_l = gs[2, 0].get_position(fig)
    pos_r = gs[2, 1].get_position(fig)
    cbar_ax = fig.add_axes([
        pos_l.x0 + pos_l.width * 0.15,
        pos_l.y0 - 0.075,
        (pos_r.x1 - pos_l.x0) * 0.70,
        0.016,
    ])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Heading DOY", fontsize=FS_CBLAB, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_CBTICK)

    # 保存
    fname = OUT_PATH / fname_stem
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{fname}.svg", bbox_inches="tight")
    plt.close(fig)



# ======================================================
# ★ 全国汇总（[1013] 统一口径：D 内按 SUMMARY_WEIGHT；集合行格网 = 基准与两个情景集合场都有值；
#    成员行 = 同一组格网中该成员有值处）
# ======================================================
stat_rows = []
def add_stats(name, fut_he, fut_ma, kref=None):
    """kref：集合行的格网（成员在其上再取有值处）；返回本次使用的格网"""
    ks = {}
    for ri in RI_TYPES:
        w = S.weights(ri)
        bh, bm = baseline_he[ri], baseline_ma[ri]
        arrs = [bh, bm] + [fut_he[s][ri] for s in SSPS] + [fut_ma[s][ri] for s in SSPS]
        k = S.cells(ri, *arrs)
        if kref is not None:
            k &= kref[ri]
        ks[ri] = k
        if not k.any():
            continue
        wm = lambda x: S.wmean(x, w, k)
        for ssp in SSPS:
            fh, fm = fut_he[ssp][ri], fut_ma[ssp][ri]
            stat_rows.append(dict(member=name, ri=ri, ssp=ssp, n_cells=int(k.sum()),
                                  HE_base=wm(bh), HE_fut=wm(fh), HE_change=wm(fh) - wm(bh),
                                  MA_base=wm(bm), MA_fut=wm(fm), MA_change=wm(fm) - wm(bm),
                                  GF_base=wm(bm - bh), GF_fut=wm(fm - fh), GF_change=wm(fm - fh) - wm(bm - bh),
                                  GF_change_pct=S.rel(wm(fm - fh), wm(bm - bh))))
    return ks

# ======================================================
# 逐成员读取 → 作图；同时累积集合平均
# ======================================================
total = len(CROP_MODELS) * len(GCMS)
print(f"\n开始绘制 {total} 张成员图 + 1 张集合平均图...")
ens_he = {ssp: {ri: [] for ri in RI_TYPES} for ssp in SSPS}
ens_ma = {ssp: {ri: [] for ri in RI_TYPES} for ssp in SSPS}
members = {}
count = 0
for model in CROP_MODELS:
    for gcm in GCMS:
        count += 1
        print(f"  [{count}/{total}] {model} × {gcm}")
        fut_he, fut_ma = {}, {}
        for ssp in SSPS:
            fut_he[ssp], fut_ma[ssp] = {}, {}
            for ri in RI_TYPES:
                d = PHENO_FUT / model / gcm / ssp
                he_stack, ma_stack = [], []
                for yr in FUT_YEARS:
                    he, ma = read_pair(d / f"{model}_{gcm}_{ssp}_{ri}_HE_{yr}_corrected.tif",
                                       d / f"{model}_{gcm}_{ssp}_{ri}_MA_{yr}_corrected.tif", ri)
                    if he is not None:
                        he_stack.append(he); ma_stack.append(ma)
                fut_he[ssp][ri] = nanmean_list(he_stack)
                fut_ma[ssp][ri] = nanmean_list(ma_stack)
                if fut_he[ssp][ri] is not None:
                    ens_he[ssp][ri].append(fut_he[ssp][ri]); ens_ma[ssp][ri].append(fut_ma[ssp][ri])
        draw_figure(f"{model}_{gcm}", fut_he, fut_ma)
        members[f"{model}_{gcm}"] = (fut_he, fut_ma)

# ★ 集合平均图
e_he = {ssp: {ri: nanmean_list(ens_he[ssp][ri]) for ri in RI_TYPES} for ssp in SSPS}
e_ma = {ssp: {ri: nanmean_list(ens_ma[ssp][ri]) for ri in RI_TYPES} for ssp in SSPS}
draw_figure("ensemble_mean", e_he, e_ma)
K_ENS = add_stats("ensemble_mean", e_he, e_ma)
for name, (fh_, fm_) in members.items():
    add_stats(name, fh_, fm_, kref=K_ENS)

# ★ 统计表与要点
import pandas as pd
st = pd.DataFrame(stat_rows)
st.to_csv(OUT_PATH / "phenology_summary.csv", index=False, float_format="%.4f")
mem = st[st.member != "ensemble_mean"]
pd.set_option("display.width", 200)
print("\n全国汇总（统一格网域，权重 SUMMARY_WEIGHT）：集合平均 | 成员中位数 [范围]")
for (ri, ssp), g in mem.groupby(["ri", "ssp"]):
    e = st[(st.member == "ensemble_mean") & (st.ri == ri) & (st.ssp == ssp)].iloc[0]
    f = lambda c: f"{e[c]:+.1f} | {g[c].median():+.1f} [{g[c].min():+.1f}, {g[c].max():+.1f}]"
    print(f"  {ri} {ssp}: 抽穗 {e.HE_base:.0f}→Δ {f('HE_change')} d；成熟 {e.MA_base:.0f}→Δ {f('MA_change')} d；"
          f"灌浆期 {e.GF_base:.1f} d→Δ {f('GF_change')} d")
print(f"[saved] {OUT_PATH / 'phenology_summary.csv'}")

print(f"\n全部完成！共输出 {total + 1} 张图")
print(f"输出目录: {OUT_PATH}")
