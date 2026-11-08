import numpy as np
import pandas as pd
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

from config_1004 import (CROP_MODELS as _CM, GCMS as _G, SSPS as _S, RI_TYPES as _RI,
                         HIS_P, FUT_P, TAG, pred_dir, fig_dir, support_file, area_file, DIAG_WEIGHT)
PRED_DIR  = pred_dir()
OUT_PATH  = fig_dir("uncertainty", thresh=False)
USE_CHANGE = True
WEIGHT_BY_AREA = (DIAG_WEIGHT == "area")
OUT_PATH.mkdir(parents=True, exist_ok=True)

CROP_MODELS = _CM
GCMS        = _G
SSPS        = _S
RI_TYPES    = _RI
TARGETS     = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]
DISPLAY_NAME = {"BRR": "BRR", "MRR": "MRR", "HRR": "HRR", "CGR": "CGR",
                "CD": "CD", "Acontent": "AC", "PC": "PC"}

N_A = len(CROP_MODELS)
N_B = len(GCMS)
N_C = len(SSPS)
N_D = len(RI_TYPES)

SOURCE_LABELS = [
    "GGCM", "GCM", "SSP", "RS",
    "GGCM×GCM", "GGCM×SSP", "GGCM×RS",
    "GCM×SSP",  "GCM×RS",   "SSP×RS",
    "GGCM×GCM×SSP", "GGCM×GCM×RS",
    "GGCM×SSP×RS",  "GCM×SSP×RS",
    "GGCM×GCM×SSP×RS",
]

SOURCE_CLASS = {
    "GGCM": "main",   "GCM": "main",   "SSP": "main",   "RS": "main",
    "GGCM×GCM": "two-way", "GGCM×SSP": "two-way", "GGCM×RS": "two-way",
    "GCM×SSP":  "two-way", "GCM×RS":   "two-way", "SSP×RS":  "two-way",
    "GGCM×GCM×SSP": "three-way", "GGCM×GCM×RS": "three-way",
    "GGCM×SSP×RS":  "three-way", "GCM×SSP×RS":  "three-way",
    "GGCM×GCM×SSP×RS": "four-way",
}

COLOR_MAP = {
    "GGCM": "#08519C",  "GCM": "#2171B5",  "SSP": "#4292C6",  "RS": "#9ECAE1",
    "GGCM×GCM": "#99000D",  "GGCM×SSP": "#CB181D",  "GGCM×RS": "#EF3B2C",
    "GCM×SSP":  "#FB6A4A",  "GCM×RS":   "#FC9272",  "SSP×RS":  "#FCBBA1",
    "GGCM×GCM×SSP": "#4A1486",  "GGCM×GCM×RS": "#6A51A3",
    "GGCM×SSP×RS":  "#9E9AC8",  "GCM×SSP×RS":  "#BCBDDC",
    "GGCM×GCM×SSP×RS": "#525252",
}

BAR_HEIGHT       = 0.72
YTICK_FONTSIZE   = 10
XTICK_FONTSIZE   = 10
VALUE_FONTSIZE   = 10
TITLE_FONTSIZE   = 14
PANEL_FONTSIZE   = 15
PANEL_Y          = 1.08
XLABEL_FONTSIZE  = 15
LEGEND_FONTSIZE  = 12
LEGEND_Y         = -0.025
XLIM_MAX         = 40
HSPACE           = 0.19
WSPACE           = 1.1
VALUE_THRESHOLD  = 0.2
VALUE_OFFSET     = 0.30

def read_tif(path):
    if not Path(path).exists():
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == np.float32(ds.nodata)] = np.nan
    return arr


def _anova_3factor_masked(arr_5d):
    """
    对一批像元做3因子分解（GGCM×GCM×SSP），RS维度已被压掉。
    arr_5d: (N_A, N_B, N_C, n_pix)   ← n_pix 个单季像元
    返回 7个SS分量 (each shape (n_pix,)) 和 SS_total (n_pix,)
    """
    mu   = np.nanmean(arr_5d, axis=(0, 1, 2))
    mu_A = np.nanmean(arr_5d, axis=(1, 2))
    mu_B = np.nanmean(arr_5d, axis=(0, 2))
    mu_C = np.nanmean(arr_5d, axis=(0, 1))

    SS_A = N_B*N_C * np.nansum((mu_A - mu)**2, axis=0)
    SS_B = N_A*N_C * np.nansum((mu_B - mu)**2, axis=0)
    SS_C = N_A*N_B * np.nansum((mu_C - mu)**2, axis=0)

    mu_AB = np.nanmean(arr_5d, axis=2)
    mu_AC = np.nanmean(arr_5d, axis=1)
    mu_BC = np.nanmean(arr_5d, axis=0)

    th_AB = mu_AB - mu_A[:, np.newaxis] - mu_B[np.newaxis, :] + mu
    th_AC = mu_AC - mu_A[:, np.newaxis] - mu_C[np.newaxis, :] + mu
    th_BC = mu_BC - mu_B[:, np.newaxis] - mu_C[np.newaxis, :] + mu

    SS_AB = N_C * np.nansum(th_AB**2, axis=(0, 1))
    SS_AC = N_B * np.nansum(th_AC**2, axis=(0, 1))
    SS_BC = N_A * np.nansum(th_BC**2, axis=(0, 1))

    SS_total = np.nansum((arr_5d - mu)**2, axis=(0, 1, 2))

    SS_ABC = SS_total - SS_A - SS_B - SS_C - SS_AB - SS_AC - SS_BC
    SS_ABC = np.where(SS_ABC < 0, 0.0, SS_ABC)

    return {
        "GGCM": SS_A, "GCM": SS_B, "SSP": SS_C,
        "GGCM×GCM": SS_AB, "GGCM×SSP": SS_AC, "GCM×SSP": SS_BC,
        "GGCM×GCM×SSP": SS_ABC,
    }, SS_total


def _anova_4factor_masked(arr_6d):
    """
    对一批像元做4因子分解。
    arr_6d: (N_A, N_B, N_C, N_D, n_pix)   ← n_pix 个双季像元
    返回 15个SS分量 (each (n_pix,)) 和 SS_total (n_pix,)
    """
    mu   = np.nanmean(arr_6d, axis=(0, 1, 2, 3))
    mu_A = np.nanmean(arr_6d, axis=(1, 2, 3))
    mu_B = np.nanmean(arr_6d, axis=(0, 2, 3))
    mu_C = np.nanmean(arr_6d, axis=(0, 1, 3))
    mu_D = np.nanmean(arr_6d, axis=(0, 1, 2))

    SS_A = N_B*N_C*N_D * np.nansum((mu_A - mu)**2, axis=0)
    SS_B = N_A*N_C*N_D * np.nansum((mu_B - mu)**2, axis=0)
    SS_C = N_A*N_B*N_D * np.nansum((mu_C - mu)**2, axis=0)
    SS_D = N_A*N_B*N_C * np.nansum((mu_D - mu)**2, axis=0)

    mu_AB = np.nanmean(arr_6d, axis=(2, 3)); mu_AC = np.nanmean(arr_6d, axis=(1, 3))
    mu_AD = np.nanmean(arr_6d, axis=(1, 2)); mu_BC = np.nanmean(arr_6d, axis=(0, 3))
    mu_BD = np.nanmean(arr_6d, axis=(0, 2)); mu_CD = np.nanmean(arr_6d, axis=(0, 1))

    th_AB = mu_AB - mu_A[:, np.newaxis] - mu_B[np.newaxis, :] + mu
    th_AC = mu_AC - mu_A[:, np.newaxis] - mu_C[np.newaxis, :] + mu
    th_AD = mu_AD - mu_A[:, np.newaxis] - mu_D[np.newaxis, :] + mu
    th_BC = mu_BC - mu_B[:, np.newaxis] - mu_C[np.newaxis, :] + mu
    th_BD = mu_BD - mu_B[:, np.newaxis] - mu_D[np.newaxis, :] + mu
    th_CD = mu_CD - mu_C[:, np.newaxis] - mu_D[np.newaxis, :] + mu

    SS_AB = N_C*N_D * np.nansum(th_AB**2, axis=(0, 1))
    SS_AC = N_B*N_D * np.nansum(th_AC**2, axis=(0, 1))
    SS_AD = N_B*N_C * np.nansum(th_AD**2, axis=(0, 1))
    SS_BC = N_A*N_D * np.nansum(th_BC**2, axis=(0, 1))
    SS_BD = N_A*N_C * np.nansum(th_BD**2, axis=(0, 1))
    SS_CD = N_A*N_B * np.nansum(th_CD**2, axis=(0, 1))

    mu_ABC = np.nanmean(arr_6d, axis=3); mu_ABD = np.nanmean(arr_6d, axis=2)
    mu_ACD = np.nanmean(arr_6d, axis=1); mu_BCD = np.nanmean(arr_6d, axis=0)

    th_ABC = (mu_ABC - mu_AB[:, :, np.newaxis] - mu_AC[:, np.newaxis, :]
              - mu_BC[np.newaxis, :, :] + mu_A[:, np.newaxis, np.newaxis]
              + mu_B[np.newaxis, :, np.newaxis] + mu_C[np.newaxis, np.newaxis, :] - mu)
    th_ABD = (mu_ABD - mu_AB[:, :, np.newaxis] - mu_AD[:, np.newaxis, :]
              - mu_BD[np.newaxis, :, :] + mu_A[:, np.newaxis, np.newaxis]
              + mu_B[np.newaxis, :, np.newaxis] + mu_D[np.newaxis, np.newaxis, :] - mu)
    th_ACD = (mu_ACD - mu_AC[:, :, np.newaxis] - mu_AD[:, np.newaxis, :]
              - mu_CD[np.newaxis, :, :] + mu_A[:, np.newaxis, np.newaxis]
              + mu_C[np.newaxis, :, np.newaxis] + mu_D[np.newaxis, np.newaxis, :] - mu)
    th_BCD = (mu_BCD - mu_BC[:, :, np.newaxis] - mu_BD[:, np.newaxis, :]
              - mu_CD[np.newaxis, :, :] + mu_B[:, np.newaxis, np.newaxis]
              + mu_C[np.newaxis, :, np.newaxis] + mu_D[np.newaxis, np.newaxis, :] - mu)

    SS_ABC = N_D * np.nansum(th_ABC**2, axis=(0, 1, 2))
    SS_ABD = N_C * np.nansum(th_ABD**2, axis=(0, 1, 2))
    SS_ACD = N_B * np.nansum(th_ACD**2, axis=(0, 1, 2))
    SS_BCD = N_A * np.nansum(th_BCD**2, axis=(0, 1, 2))

    SS_total = np.nansum((arr_6d - mu)**2, axis=(0, 1, 2, 3))
    SS_ABCD = (SS_total - SS_A - SS_B - SS_C - SS_D
               - SS_AB - SS_AC - SS_AD - SS_BC - SS_BD - SS_CD
               - SS_ABC - SS_ABD - SS_ACD - SS_BCD)
    SS_ABCD = np.where(SS_ABCD < 0, 0.0, SS_ABCD)

    return {
        "GGCM": SS_A, "GCM": SS_B, "SSP": SS_C, "RS": SS_D,
        "GGCM×GCM": SS_AB, "GGCM×SSP": SS_AC, "GGCM×RS": SS_AD,
        "GCM×SSP": SS_BC, "GCM×RS": SS_BD, "SSP×RS": SS_CD,
        "GGCM×GCM×SSP": SS_ABC, "GGCM×GCM×RS": SS_ABD,
        "GGCM×SSP×RS": SS_ACD, "GCM×SSP×RS": SS_BCD,
        "GGCM×GCM×SSP×RS": SS_ABCD,
    }, SS_total


def anova_4factor(arr_6d):
    """
    分区处理不平衡的RS因子（早/晚稻种植区不完全重叠）。
    输入: arr_6d  (N_A, N_B, N_C, N_D, n_lat, n_lon)

    像元分三类（按各成员是否有效）:
      A. 双季像元: ri1和ri2都有完整的 5×5×2×2=100 个有效值
                   → 四因子分解(15项)，逐像元归一化到100%
      B. 单季像元: 只有一季(ri1或ri2)有完整的 5×5×2=50 个有效值
                   → 三因子分解(GGCM×GCM×SSP, 7项)，含RS的8项记为0
                     逐像元归一化到100%
      C. 无效像元: 两季都不完整 → 全NaN

    返回: dict {15个来源: frac_map(n_lat,n_lon) 已逐像元归一化到100%},
          season_map(n_lat,n_lon): 1=单季, 2=双季, 0=无效
    """
    n_lat, n_lon = arr_6d.shape[4], arr_6d.shape[5]
    n_pix = n_lat * n_lon
    flat = arr_6d.reshape(N_A, N_B, N_C, N_D, n_pix)

    valid_ri1 = ~np.any(np.isnan(flat[:, :, :, 0]), axis=(0, 1, 2))
    valid_ri2 = ~np.any(np.isnan(flat[:, :, :, 1]), axis=(0, 1, 2))

    is_double = valid_ri1 & valid_ri2
    is_single = (valid_ri1 ^ valid_ri2)

    frac_flat = {src: np.full(n_pix, np.nan, dtype=np.float64)
                 for src in SOURCE_LABELS}
    season_flat = np.zeros(n_pix, dtype=np.int8)

    if np.any(is_double):
        sub = flat[:, :, :, :, is_double]
        SS, SS_tot = _anova_4factor_masked(sub)
        with np.errstate(invalid="ignore", divide="ignore"):
            tot = np.zeros_like(SS_tot)
            for src in SOURCE_LABELS:
                tot = tot + SS[src]
            for src in SOURCE_LABELS:
                frac_flat[src][is_double] = np.where(
                    tot > 1e-12, SS[src] / tot * 100, np.nan)
        season_flat[is_double] = 2

    if np.any(is_single):
        sub = flat[:, :, :, :, is_single]
        v1 = valid_ri1[is_single]
        arr_season = np.where(v1[np.newaxis, np.newaxis, np.newaxis, :],
                              sub[:, :, :, 0], sub[:, :, :, 1])
        SS3, SS_tot3 = _anova_3factor_masked(arr_season)
        with np.errstate(invalid="ignore", divide="ignore"):
            tot3 = np.zeros_like(SS_tot3)
            for src in SS3:
                tot3 = tot3 + SS3[src]
            for src in SOURCE_LABELS:
                if src in SS3:
                    frac_flat[src][is_single] = np.where(
                        tot3 > 1e-12, SS3[src] / tot3 * 100, np.nan)
                else:
                    frac_flat[src][is_single] = np.where(
                        tot3 > 1e-12, 0.0, np.nan)
        season_flat[is_single] = 1

    fracs = {src: frac_flat[src].reshape(n_lat, n_lon) for src in SOURCE_LABELS}
    season_map = season_flat.reshape(n_lat, n_lon)
    return fracs, season_map


def spatial_mean(frac_map, w=None):
    """对有效像元（非nan）取空间均值；w 为面积权重（[1013]）"""
    ok = ~np.isnan(frac_map)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
        return float(np.sum(frac_map[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
    v = frac_map[ok]
    return float(np.mean(v)) if len(v) > 0 else np.nan


def area_in_domain(ri):
    a = read_tif(area_file(ri)); d = read_tif(support_file(ri))
    return np.where(np.isfinite(d) & np.isfinite(a) & (a > 0), a, 0.0).astype(np.float64)

AREA_D = {ri: area_in_domain(ri) for ri in RI_TYPES}


def draw_barh(ax, values_dict, panel_label, title, xlim=None, show_xaxis=True):
    """水平条形图，按贡献大小降序排列。
    show_xaxis=False → 隐藏x轴刻度标签（上排子图），与下排共用一条轴。
    """
    labels = list(values_dict.keys())
    vals   = [values_dict[l] for l in labels]

    order  = np.argsort(vals)[::-1]
    labels = [labels[i] for i in order]
    vals   = [vals[i]   for i in order]
    colors = [COLOR_MAP[l] for l in labels]

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, vals, color=colors, height=BAR_HEIGHT,
                    edgecolor="white", linewidth=0.25)

    for bar, v in zip(bars, vals):
        if np.isfinite(v) and v > VALUE_THRESHOLD:
            ax.text(v + VALUE_OFFSET, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", va="center", ha="left",
                    fontsize=VALUE_FONTSIZE, color="#333333", clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=YTICK_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=4)
    ax.tick_params(labelsize=XTICK_FONTSIZE, length=2.5)
    ax.invert_yaxis()

    if xlim is not None:
        ax.set_xlim(0, xlim)

    if show_xaxis:
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines["bottom"].set_visible(True)

    ax.text(-0.22, PANEL_Y, panel_label,
            transform=ax.transAxes, fontsize=PANEL_FONTSIZE,
            fontweight="bold", va="top")


print("开始4因子ANOVA不确定性分析 (GGCM × GCM × SSP × RS)...")

results     = {}
export_rows = []

for target in TARGETS:
    print(f"  {target} ...", end=" ", flush=True)

    arr_6d_list = []
    for model in CROP_MODELS:
        row_b = []
        for gcm in GCMS:
            row_c = []
            for ssp in SSPS:
                row_d = []
                for ri in RI_TYPES:
                    fp  = PRED_DIR / ssp / f"{model}_{gcm}_{ssp}_{ri}_{target}_{FUT_P}.tif"
                    arr = read_tif(fp)
                    if arr is None:
                        arr = np.full((72, 126), np.nan, dtype=np.float32)
                    elif USE_CHANGE:
                        b = read_tif(PRED_DIR / "historical" / f"{gcm}_historical_{ri}_{target}_{HIS_P}.tif")
                        arr = arr - b if b is not None else np.full_like(arr, np.nan)
                    sp = read_tif(support_file(ri))
                    if sp is not None:
                        arr = np.where(np.isfinite(sp), arr, np.nan).astype(np.float32)
                    row_d.append(arr)
                row_c.append(np.stack(row_d, axis=0))
            row_b.append(np.stack(row_c, axis=0))
        arr_6d_list.append(np.stack(row_b, axis=0))
    arr_6d = np.stack(arr_6d_list, axis=0)

    fracs, season_map = anova_4factor(arr_6d)
    if WEIGHT_BY_AREA:
        v1 = ~np.any(np.isnan(arr_6d[:, :, :, 0]), axis=(0, 1, 2))
        v2 = ~np.any(np.isnan(arr_6d[:, :, :, 1]), axis=(0, 1, 2))
        wmap = np.where(v1, AREA_D[RI_TYPES[0]], 0.0) + np.where(v2, AREA_D[RI_TYPES[1]], 0.0)
    else:
        wmap = None
    mean_fracs = {src: spatial_mean(fmap, wmap) for src, fmap in fracs.items()}
    _tot = sum(v for v in mean_fracs.values() if np.isfinite(v))
    if _tot and np.isfinite(_tot):
        mean_fracs = {s: (v / _tot * 100 if np.isfinite(v) else v)
                      for s, v in mean_fracs.items()}
    results[target] = mean_fracs

    n_double = int(np.sum(season_map == 2))
    n_single = int(np.sum(season_map == 1))
    total_check = sum(v for v in mean_fracs.values() if np.isfinite(v))
    print(f"sum={total_check:.1f}%  [双季{n_double} 单季{n_single} 像元]")

    for src, val in mean_fracs.items():
        export_rows.append({
            "target":          target,
            "source":          src,
            "class":           SOURCE_CLASS[src],
            "uncertainty_pct": round(val, 3),
        })

all_fracs = {}
for src in SOURCE_LABELS:
    vals_src = [results[t][src] for t in TARGETS
                if t in results and np.isfinite(results[t][src])]
    all_fracs[src] = float(np.mean(vals_src)) if vals_src else np.nan
_tot = sum(v for v in all_fracs.values() if np.isfinite(v))
if _tot and np.isfinite(_tot):
    all_fracs = {s: (v / _tot * 100 if np.isfinite(v) else v)
                 for s, v in all_fracs.items()}
results["All"] = all_fracs
for src, val in all_fracs.items():
    export_rows.append({
        "target": "All", "source": src,
        "class": SOURCE_CLASS[src], "uncertainty_pct": round(val, 3),
    })

DISPLAY_NAME["All"] = "All"
panel_order  = ["All"] + TARGETS
panel_labels = list("abcdefgh")

fig = plt.figure(figsize=(200 / 25.4, 185 / 25.4))
gs  = gridspec.GridSpec(
    2, 4, figure=fig,
    left=0.12, right=0.99,
    top=0.95, bottom=0.09,
    hspace=HSPACE, wspace=WSPACE,
)

xlim = XLIM_MAX

for idx, (panel_name, plabel) in enumerate(zip(panel_order, panel_labels)):
    row_i = idx // 4
    ax = fig.add_subplot(gs[row_i, idx % 4])
    draw_barh(ax, results[panel_name], f"({plabel})", DISPLAY_NAME[panel_name],
              xlim=xlim, show_xaxis=(row_i == 1))

fig.text(0.555, 0.04, "Share of variance among simulations (%)",
         ha="center", va="center", fontsize=XLABEL_FONTSIZE, fontweight="bold")


fname = OUT_PATH / f"uncertainty_anova_4factor_{TAG}{'_change' if USE_CHANGE else '_level'}"
fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{fname}.pdf", bbox_inches="tight")
fig.savefig(f"{fname}.svg", bbox_inches="tight")
plt.close(fig)
print(f"\n  → figure saved: {fname.name}.png / .pdf / .svg")

df   = pd.DataFrame(export_rows)
wide = df.pivot_table(index="target", columns="source",
                      values="uncertainty_pct").reset_index()

wide["MAIN_sum"]     = wide[["GGCM","GCM","SSP","RS"]].sum(axis=1)
wide["TWOWAY_sum"]   = wide[["GGCM×GCM","GGCM×SSP","GGCM×RS",
                              "GCM×SSP","GCM×RS","SSP×RS"]].sum(axis=1)
wide["THREEWAY_sum"] = wide[["GGCM×GCM×SSP","GGCM×GCM×RS",
                              "GGCM×SSP×RS","GCM×SSP×RS"]].sum(axis=1)
wide["FOURWAY"]      = wide["GGCM×GCM×SSP×RS"]

df.to_csv(OUT_PATH / "uncertainty_4factor_long.csv",  index=False)
wide.to_csv(OUT_PATH / "uncertainty_4factor_wide.csv", index=False)

print(f"\n✓ 完成！输出目录: {OUT_PATH}")
print(f"  图    : uncertainty_anova_4factor.png / .pdf / .svg")
print(f"  数据  : uncertainty_4factor_long.csv / _wide.csv")
