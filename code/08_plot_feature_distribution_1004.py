# ============================================================
# 特征分布对比图（补充图 S18）
# ★ 1004：时段、路径、支撑区均来自 config_1004；新增 TSR；CO2 由文件计算
# ★ [1013] 统一口径：
#   · 预测输入 = 模型实际吃进去的"逐年 × 逐成员"灌浆期气候（PRED_MODE = annual），不再用集合多年均值
#     （DIST_INPUT = "periodmean" 可切回旧口径作对照）；与 29 号表 S10 同一批输入
#   · 只取统一格网域 D；每个"格网 × 年 × 成员"等权（config.DIAG_WEIGHT，与 S10 相同；两季合并），均值线为同口径均值
#   · 外推比例表已移到 29 号（表 S10），本脚本不再输出 extrapolation_check.csv，避免两套数字
# 7个特征 × KDE图；数据源：训练数据 + 基准 + SSP126 + SSP585
# ============================================================
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import rasterio
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2,   "ytick.major.size": 2,
})

# ======================================================
# 路径
# ======================================================
from config_1004 import (CLIM_HIS, CLIM_FUT, HIS_P, FUT_P, HIS_LABEL, FUT_LABEL, HIS_YEARS, FUT_YEARS,
                         FIGURES, TAG, FACE_XLSX, CROP_MODELS, GCMS, support_file, area_file, co2_mean, co2_series,
                         DIAG_WEIGHT)
DIST_INPUT = "annual"      # ★ "annual"：逐年 × 逐成员（与预测、S10 一致）；"periodmean"：集合多年均值（旧口径）
N_SAMPLE   = 40000         # 每个时段抽样的点数（两季合并；用于 KDE）
HIS_DIR  = CLIM_HIS / "ensemble"
FUT_DIR  = CLIM_FUT / "ensemble"
FACE_XLS = FACE_XLSX
OUT_PATH = FIGURES / f"feature_distribution_{TAG}"
APPLY_EXCLUDE = False      # ★ 训练集已用全部记录（不再排除）；若仍按旧名单排除，改为 True
OUT_PATH.mkdir(parents=True, exist_ok=True)

# ======================================================
# FACE数据过滤
# ======================================================
df_raw = pd.read_excel(FACE_XLS)
df = (df_raw[~df_raw["Exp_ID"].isin(exclude)] if APPLY_EXCLUDE else df_raw).copy().reset_index(drop=True)
print(f"FACE数据: {len(df)} 条")

# ======================================================
# 配置
# ======================================================
FEATURES = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]
FEAT_LABELS = {
    "CO2":   "CO₂ (ppm)",
    "Tavg":  "T$_{avg}$ (°C)",
    "Tdif":  "T$_{dif}$ (°C)",
    "CTSR":  "CTSR (MJ m$^{-2}$)",
    "TSR":   "TSR (MJ m$^{-2}$ d$^{-1}$)",
    "Rain":  "Rain (mm)",
    "RHavg": "RH$_{avg}$ (%)",
}
RASTER_LABEL = {
    "Tavg":  "mean_temp_degC",
    "Tdif":  "mean_tmpdiff_degC",
    "CTSR":  "cumul_solar_MJ_m2",
    "TSR":   "mean_daily_solar_MJ_m2",
    "Rain":  "cumul_precip_mm",
    "RHavg": "mean_humidity_pct",
}

# 颜色方案
COLORS = {
    "FACE":    "#555555",
    "历史":    "#2166AC",
    "SSP126":  "#4DAF4A",
    "SSP585":  "#D73027",
}
LABELS_ORDER = ["FACE", "历史", "SSP126", "SSP585"]
LABEL_DISPLAY = {
    "FACE":   f"Training data (n={len(df)})",
    "历史":   f"Baseline ({HIS_LABEL})",            # [1105] Historical → Baseline
    "SSP126": f"SSP1-2.6 ({FUT_LABEL})",
    "SSP585": f"SSP5-8.5 ({FUT_LABEL})",
}

# ★ CO2：直接从 ISIMIP3b CO2 文件计算时段均值（不再手填）
CO2_VALS = {"历史":   co2_mean("historical", HIS_YEARS),
            "SSP126": co2_mean("ssp126", FUT_YEARS),
            "SSP585": co2_mean("ssp585", FUT_YEARS)}
print("CO2 时段均值:", {k: round(v, 1) for k, v in CO2_VALS.items()})

# ======================================================
# 读取预测输入（统一格网域 D；按汇总权重抽样，最终口径为等权）
# ======================================================
def rd(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == ds.nodata] = np.nan
    return arr

DOM = {}
for ri in ["ri1", "ri2"]:
    a_ = rd(area_file(ri)); d_ = np.isfinite(rd(support_file(ri))) & np.isfinite(a_) & (a_ > 0)
    DOM[ri] = (d_, np.where(d_, 1.0 if DIAG_WEIGHT == "cell" else a_, 0.0))

def annual_files(src, ri):
    if src == "历史":
        return [CLIM_HIS / "annual" / f"{g}_historical_{ri}_{HIS_P}_annual.nc" for g in GCMS]
    s_ = "ssp126" if src == "SSP126" else "ssp585"
    return [CLIM_FUT / "annual" / s_ / m / g / f"{m}_{g}_{s_}_{ri}_{FUT_P}_annual.nc"
            for m in CROP_MODELS for g in GCMS]

def mean_files(src, ri, label):
    if src == "历史":
        return [HIS_DIR / f"ensemble_historical_{ri}_{label}_{HIS_P}_mean.tif"]
    s_ = "ssp126" if src == "SSP126" else "ssp585"
    return [FUT_DIR / f"ensemble_{s_}_{ri}_{label}_{FUT_P}_mean.tif"]

rng = np.random.default_rng(0)
raster_data, raster_mean = {}, {}
for src in ["历史", "SSP126", "SSP585"]:
    vals = {f: [] for f in RASTER_LABEL}; wts = []
    for ri in ["ri1", "ri2"]:
        dom, w2 = DOM[ri]
        if DIST_INPUT == "annual":
            for fp in annual_files(src, ri):
                if not fp.exists():
                    print(f"  [缺] {fp}"); continue
                with xr.open_dataset(fp) as ds:
                    A = {f: ds[lab].values[:, dom] for f, lab in RASTER_LABEL.items()}      # 年 × 格点
                ww = np.repeat(w2[dom][None, :], next(iter(A.values())).shape[0], axis=0).ravel()
                X = np.stack([A[f].ravel() for f in RASTER_LABEL], 1)
                ok = np.isfinite(X).all(1)
                for j, f in enumerate(RASTER_LABEL): vals[f].append(X[ok, j])
                wts.append(ww[ok])
        else:
            X = np.stack([rd(mean_files(src, ri, lab)[0])[dom] for lab in RASTER_LABEL.values()], 1)
            ok = np.isfinite(X).all(1)
            for j, f in enumerate(RASTER_LABEL): vals[f].append(X[ok, j])
            wts.append(w2[dom][ok])
    if not wts:
        continue
    w_all = np.concatenate(wts)
    idx = rng.choice(len(w_all), size=min(N_SAMPLE, len(w_all)), replace=True, p=w_all / w_all.sum())
    for f in RASTER_LABEL:
        v = np.concatenate(vals[f])
        raster_data[f"{src}_{f}"] = v[idx]                                  # 抽样 → KDE
        raster_mean[f"{src}_{f}"] = float(np.sum(v * w_all) / w_all.sum())  # 全部输入值的精确均值
    print(f"  {src}: {len(w_all):,} 个输入值（{DIST_INPUT}），抽样 {len(idx):,}")

CO2_RANGE = {"历史": co2_series("historical", HIS_YEARS),
             "SSP126": co2_series("ssp126", FUT_YEARS), "SSP585": co2_series("ssp585", FUT_YEARS)}

# ======================================================
# 绘图：2行3列，每格一个特征
# ======================================================
fig, axes = plt.subplots(2, 4, figsize=(220/25.4, 120/25.4))          # ★ 7 个特征 + 1 格图例
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18,
                    hspace=0.50, wspace=0.38)

panel_labels = list("abcdefg")
axes[1, 3].axis("off")

for idx, feat in enumerate(FEATURES):
    ax = axes[idx // 4, idx % 4]

    # 收集各来源数据
    datasets = {}
    if feat == "CO2":
        datasets["FACE"] = df["CO2"].dropna().values
        # CO2为标量，用竖线+文字表示
        for src, val in CO2_VALS.items():
            datasets[src] = np.array([val])
    else:
        datasets["FACE"] = df[feat].dropna().values
        for src in ["历史", "SSP126", "SSP585"]:
            datasets[src] = raster_data.get(f"{src}_{feat}", np.array([]))

    # 确定x范围
    all_vals = np.concatenate([v for v in datasets.values() if len(v) > 1])
    p1, p99 = np.percentile(all_vals, [0.5, 99.5])
    xpad = (p99 - p1) * 0.05
    xmin, xmax = p1 - xpad, p99 + xpad

    # KDE曲线
    x_grid = np.linspace(xmin, xmax, 300)
    max_density = 0

    for src in LABELS_ORDER:
        v = datasets[src]
        if len(v) < 2:
            continue
        color = COLORS[src]
        lw = 1.5 if src == "FACE" else 1.0
        ls = "--" if src == "FACE" else "-"
        alpha = 1.0

        try:
            kde = gaussian_kde(v, bw_method="scott")
            dens = kde(x_grid)
            max_density = max(max_density, dens.max())
            ax.plot(x_grid, dens, color=color, linewidth=lw,
                    linestyle=ls, alpha=alpha,
                    label=LABEL_DISPLAY[src], zorder=3)
            ax.fill_between(x_grid, dens, alpha=0.08, color=color, zorder=2)
        except Exception:
            pass

    # FACE范围阴影（训练范围）
    face_v = datasets["FACE"]
    if len(face_v) > 1:
        fmin, fmax = face_v.min(), face_v.max()
        ax.axvspan(fmin, fmax, alpha=0.06, color=COLORS["FACE"],
                   zorder=1, label="_nolegend_")
        ax.axvline(fmin, color=COLORS["FACE"], linewidth=0.5,
                   linestyle=":", alpha=0.7, zorder=4)
        ax.axvline(fmax, color=COLORS["FACE"], linewidth=0.5,
                   linestyle=":", alpha=0.7, zorder=4)

    # CO2特殊处理：竖线
    if feat == "CO2":
        ax.cla()
        face_v = datasets["FACE"]
        try:
            kde = gaussian_kde(face_v, bw_method="scott")
            x_grid2 = np.linspace(face_v.min()*0.95, 800, 300)
            dens = kde(x_grid2)
            ax.plot(x_grid2, dens, color=COLORS["FACE"], linewidth=1.5,
                    linestyle="--", label=LABEL_DISPLAY["FACE"])
            ax.fill_between(x_grid2, dens, alpha=0.08, color=COLORS["FACE"])
        except Exception:
            pass
        for src, val in CO2_VALS.items():
            color = COLORS[src]
            if DIST_INPUT == "annual":                     # 逐年 CO2 的范围（预测实际使用的值）
                ax.axvspan(min(CO2_RANGE[src]), max(CO2_RANGE[src]), color=color, alpha=0.15, zorder=1)
            ax.axvline(val, color=color, linewidth=1.2,
                       label=LABEL_DISPLAY[src], zorder=5)   # [1105] 图例不再附 CO₂ 浓度
        ax.set_xlim(330, 800)

    # 统计标注（均值线）
    for src in ["历史", "SSP126", "SSP585"]:
        v = datasets[src]
        if len(v) > 1 and feat != "CO2":
            ax.axvline(raster_mean[f"{src}_{feat}"], color=COLORS[src], linewidth=0.6,
                       linestyle="--", alpha=0.6, zorder=4)

    ax.set_xlabel(FEAT_LABELS[feat], fontsize=6.5, labelpad=2)
    ax.set_ylabel("Density", fontsize=6, labelpad=2)
    ax.tick_params(labelsize=5.5, length=2)
    # [1105] 面板号改在全部画完后按各轴实际外沿（含刻度标签）定位，见下方，避免压到纵轴刻度

    if feat != "CO2":
        ax.set_xlim(xmin, xmax)

# 共享图例
handles, labels_leg = axes[0, 0].get_legend_handles_labels()
# 去掉重复
seen = set()
h_uniq, l_uniq = [], []
for h, l in zip(handles, labels_leg):
    if l not in seen and not l.startswith("_"):
        seen.add(l); h_uniq.append(h); l_uniq.append(l)

axes[1, 3].legend(h_uniq, l_uniq, loc="center", fontsize=6.5, handlelength=1.5,
                  handletextpad=0.4, frameon=False)                 # ★ 图例放在空白格

# [1105] 面板号：放在每个子图“含刻度标签的外沿”左上方（纵轴刻度再宽也不会重合）
fig.canvas.draw(); _rr = fig.canvas.get_renderer()
for idx, feat in enumerate(FEATURES):
    ax = axes[idx // 4, idx % 4]
    bb = ax.get_tightbbox(_rr).transformed(fig.transFigure.inverted())
    pos = ax.get_position()
    fig.text(bb.x0, pos.y1 + 0.015, f"({panel_labels[idx]})", fontsize=7, fontweight="bold", ha="left", va="bottom")

# [1013] 外推比例统一由 29 号输出（表 S10，逐年 × 逐成员）；此处不再重复计算
fname = OUT_PATH / "feature_distribution"
fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{fname}.pdf", bbox_inches="tight")
fig.savefig(f"{fname}.svg", bbox_inches="tight")   # [1108d] 补出 SVG（Word 用 SVG 插图）
plt.close(fig)
print(f"[saved] {fname}.png")
print("完成！")
