"""
SHAP Analysis — Rice Quality Indicators (Blending / VotingRegressor)
====================================================================
两档分析：
  • 定级三性状 HRR / CD / PC   → 三情景(Hist/SSP126/SSP585)三面板
        (a) mean|SHAP| 分组条形  (b) ΔSHAP 热图(future−hist)  (c) CO2 vs Tavg 散点
  • 其余性状 BRR / MRR / CGR / PCA → 仅历史，两面板
        (a) mean|SHAP| 条形       (b) beeswarm 方向图
  • Acontent → 仅历史，两面板；与预测写法一致按籼稻占比 f 加权：
        SHAP = f·SHAP(TYPE=1) + (1−f)·SHAP(TYPE=2)，只展示气候因子（不做籼/粳二分）

关键实现：
  - SHAP 按 VotingRegressor 的反比-RMSE 权重加权(与预测口径一致)，非简单平均
  - 仅对树模型用 TreeExplainer；断言树学习器覆盖 >50% 集成权重，避免静默偏差
  - 配色并入 paper_palette（缺失则回退 CVD 安全蓝/橙/红）
  - SHAP 在“格网预测输入”上计算(历史/未来特征矩阵 CSV)，非训练数据

USAGE:  python 19_shap_analysis_1004.py
[1004] 运行顺序 19（依赖 18）——模型、特征、时段来自 config_1004；特征含 TSR
  · 特征顺序以模型自身 feature_names_in_ 为准（与 04 预测一致），SHAP 结果再映射回展示顺序
  · 另输出 shap_mean_abs.csv（mean|SHAP| 与 mean SHAP，按 指标×季节×情景），供论文取数
[1013] 统一口径：格网 = 统一格网域 D（18 号已限定）；mean|SHAP| 与 mean SHAP 按汇总权重平均
       （config.DIAG_WEIGHT：定稿为逐格等权）。缓存行数与特征表不一致时自动重算。
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import shap
import joblib
from pathlib import Path

from config_1004 import (MODEL_DIR, RI_TYPES as _RI, HIS_LABEL, FUT_LABEL, out_dir, fig_dir, DIAG_WEIGHT)

FEAT_DIR   = out_dir("shap_features", thresh=False)
RI_TYPES   = _RI
def feat_csv(period, ri):
    return FEAT_DIR / f"features_{period}_{ri}.csv"

OUT_DIR    = fig_dir("shap", thresh=False)
ARR_DIR    = OUT_DIR / "shap_arrays"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARR_DIR.mkdir(parents=True, exist_ok=True)

GRADING   = ["HRR", "CD", "PC"]
HIST_ONLY = ["BRR", "MRR", "CGR", "PCA"]
ACONTENT  = "Acontent"

TARGET_LABELS = {
    "BRR": "Brown rice rate (%)", "MRR": "Milled rice rate (%)",
    "HRR": "Head rice rate (%)",  "CGR": "Chalky grain rate (%)",
    "CD":  "Chalkiness degree (%)","Acontent": "Amylose content (%)",
    "PC":  "Protein content (%)", "PCA": "Protein content albumin (%)",
}

FEATURES_BASE = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]
def features_for(target):
    return FEATURES_BASE

DISPLAY_FEATS = FEATURES_BASE

SCEN_ENTRIES = [
    (f"Baseline ({HIS_LABEL})", "hist",   "historical"),
    (f"SSP1-2.6 ({FUT_LABEL})", "ssp126", "ssp126"),
    (f"SSP5-8.5 ({FUT_LABEL})", "ssp585", "ssp585"),
]

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper_palette as PP
    COLORS = {"hist": PP.SSP.get("baseline", "#4C4C4C"),
              "ssp126": PP.SSP.get("ssp126", "#3B78B0"),
              "ssp585": PP.SSP.get("ssp585", "#E08214")}
except Exception:
    COLORS = {"hist": "#4C4C4C", "ssp126": "#3B78B0", "ssp585": "#E08214"}

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
})

FS_SUP    = 24
FS_TITLE  = 23
FS_LABEL  = 19
FS_TICK   = 19
FS_LEGEND = 19
FS_ANNOT  = 19

SHAP_ROWS = []

_feat_cache, _w_cache = {}, {}
def load_feat(csv_path, feats):
    key = (str(csv_path), tuple(feats))
    if key in _feat_cache:
        return _feat_cache[key]
    df = pd.read_csv(csv_path)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"{csv_path.name} 缺少列: {missing}")
    sub = df.dropna(subset=feats).reset_index(drop=True)
    out = sub[feats]
    _feat_cache[key] = out
    if DIAG_WEIGHT == "cell" or "w" not in sub.columns:
        _w_cache[key] = np.ones(len(sub))
    else:
        _w_cache[key] = sub["w"].to_numpy(float)
    return out


def wts(csv_path, feats):
    return _w_cache[(str(csv_path), tuple(feats))]


def wmean0(a, w):
    """按行（格网）加权的列均值（权重见 wts）"""
    return (np.asarray(a) * w[:, None]).sum(0) / w.sum()

def ensemble_shap(vr, X_df, feats, cache_key):
    cache = ARR_DIR / f"sv_{cache_key}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(X_df):
            print(f"    [cache] {cache.name}")
            return arr
        print(f"    [cache 过期] {cache.name}: {arr.shape[0]} 行 ≠ 特征表 {len(X_df)} 行 → 重算")

    n_est = len(vr.estimators_)
    w = getattr(vr, "weights", None)
    w = np.ones(n_est) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()

    mfeats = list(getattr(vr, "feature_names_in_", feats))
    Xm = X_df[mfeats]
    total = np.zeros((len(X_df), len(mfeats)), dtype=float)
    used_w = 0.0
    for est, wi in zip(vr.estimators_, w):
        try:
            sv = shap.TreeExplainer(est).shap_values(Xm)
            total += wi * np.asarray(sv)
            used_w += wi
        except Exception as e:
            print(f"    skip {type(est).__name__} (non-tree?): {e}")
    assert used_w > 0.5, f"{cache_key}: 树学习器仅覆盖 {used_w:.2f} 集成权重(<0.5)，需用 PermutationExplainer 回退"
    result = total / used_w
    result = result[:, [mfeats.index(f) for f in feats]]
    np.save(cache, result)
    return result

def _order_by_importance(*ma_arrays):
    """按平均 |SHAP| 升序（beeswarm/条形从下到上由小到大）返回 DISPLAY_FEATS 的索引顺序"""
    idx = [FEATURES_BASE.index(f) for f in DISPLAY_FEATS]
    score = np.mean([np.asarray(a)[idx] for a in ma_arrays], axis=0)
    return [idx[i] for i in np.argsort(score)]

def panel_importance_grouped(ax, ma_dict, order, title, show_legend=True):
    y = np.arange(len(order)); bh = 0.24
    for i, (lbl, key) in enumerate([("Baseline", "hist"),
                                    ("SSP1-2.6", "ssp126"),
                                    ("SSP5-8.5", "ssp585")]):
        vals = [ma_dict[key][fi] for fi in order]
        ax.barh(y + bh * (i - 1), vals, height=bh * 0.9,
                color=COLORS[key], alpha=0.9, label=lbl)
    ax.set_yticks(y); ax.set_yticklabels([FEATURES_BASE[i] for i in order], fontsize=FS_TICK)
    ax.set_xlabel("mean |SHAP|", fontsize=FS_LABEL, labelpad=1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.xaxis.grid(True, ls="--", lw=0.35, alpha=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK)
    if show_legend:
        ax.legend(loc="lower right", fontsize=FS_LEGEND, handlelength=0.9, handletextpad=0.3)

def panel_importance_single(ax, ma, order, title, color):
    y = np.arange(len(order))
    ax.barh(y, [ma[fi] for fi in order], height=0.62, color=color, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels([FEATURES_BASE[i] for i in order], fontsize=FS_TICK)
    ax.set_xlabel("mean |SHAP|", fontsize=FS_LABEL, labelpad=1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.xaxis.grid(True, ls="--", lw=0.35, alpha=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK)

def panel_delta_heatmap(ax, ms_dict, order, title, show_cbar=True, vmax=None):
    d126 = np.array([ms_dict["ssp126"][fi] - ms_dict["hist"][fi] for fi in order])
    d585 = np.array([ms_dict["ssp585"][fi] - ms_dict["hist"][fi] for fi in order])
    heat = np.vstack([d126, d585])
    if vmax is None:
        vmax = max(np.abs(heat).max() * 1.05, 1e-6)
    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([FEATURES_BASE[i] for i in order], rotation=40, ha="right", fontsize=FS_TICK)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SSP1-2.6\n−Baseline", "SSP5-8.5\n−Baseline"], fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    for r in range(2):
        for c in range(len(order)):
            v = heat[r, c]
            ax.text(c, r, f"{v:+.2f}", ha="center", va="center", rotation=90,
                    fontsize=FS_ANNOT,
                    color="white" if abs(v) > vmax * 0.5 else "#333")
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cb.ax.tick_params(labelsize=FS_TICK); cb.set_label("ΔSHAP", fontsize=FS_LABEL)

def panel_co2_tavg(ax, sv_dict, feats, title, show_legend=True):
    ci, ti = feats.index("CO2"), feats.index("Tavg")
    rng = np.random.default_rng(42)
    for lbl, key in [("Baseline", "hist"), ("SSP1-2.6", "ssp126"), ("SSP5-8.5", "ssp585")]:
        sv = sv_dict[key]; idx = rng.choice(len(sv), min(400, len(sv)), replace=False)
        ax.scatter(sv[idx, ci], sv[idx, ti], s=4, alpha=0.5, color=COLORS[key],
                   label=lbl, linewidths=0, rasterized=True)
    ax.axhline(0, color="#aaa", lw=0.4, ls="--"); ax.axvline(0, color="#aaa", lw=0.4, ls="--")
    ax.set_xlabel("SHAP — CO2", fontsize=FS_LABEL, labelpad=1)
    ax.set_ylabel("SHAP — Tavg", fontsize=FS_LABEL, labelpad=1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FS_TICK)
    if show_legend:
        ax.legend(fontsize=FS_LEGEND, markerscale=2.5, handletextpad=0.3, loc="upper left")

def panel_beeswarm(ax, sv, X_df, order, title, max_pts=2000):
    rng = np.random.default_rng(0)
    n = sv.shape[0]
    if n > max_pts:
        sel = rng.choice(n, max_pts, replace=False); sv = sv[sel]; Xv = X_df.values[sel]
    else:
        Xv = X_df.values
    sc = None
    for yi, fi in enumerate(order):
        s = sv[:, fi]; fv = Xv[:, fi].astype(float)
        lo, hi = np.nanpercentile(fv, [5, 95])
        c = np.clip((fv - lo) / (hi - lo + 1e-9), 0, 1)
        jit = (rng.random(len(s)) - 0.5) * 0.55
        sc = ax.scatter(s, yi + jit, c=c, cmap="coolwarm", s=3, alpha=0.6,
                        linewidths=0, rasterized=True, vmin=0, vmax=1)
    ax.axvline(0, color="#888", lw=0.5, ls="--")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([FEATURES_BASE[i] for i in order], fontsize=FS_TICK)
    ax.set_xlabel("SHAP value", fontsize=FS_LABEL, labelpad=1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FS_TICK)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=FS_TICK)
    cb.set_label("feature value", fontsize=FS_LABEL)

def load_model(target):
    p = MODEL_DIR / f"{target}_Blending.joblib"
    if not p.exists():
        print(f"  ✗ 模型缺失: {p}"); return None
    vr = joblib.load(p)
    print(f"  模型载入: {len(vr.estimators_)} base learners")
    return vr

def run_grading(target, ri):
    print(f"\n[GRADING] {target} · {ri}")
    vr = load_model(target)
    if vr is None: return None
    feats = features_for(target)
    sv_dict, ma, ms = {}, {}, {}
    for _, key, period in SCEN_ENTRIES:
        X = load_feat(feat_csv(period, ri), feats); w = wts(feat_csv(period, ri), feats)
        print(f"  SHAP {key}: {len(X)} px")
        sv = ensemble_shap(vr, X, feats, f"{target}_{ri}_{key}")
        sv_dict[key] = sv
        ma[key] = wmean0(np.abs(sv), w); ms[key] = wmean0(sv, w)
    order = _order_by_importance(ma["hist"], ma["ssp126"], ma["ssp585"])
    for key in ma:
        for i, f in enumerate(feats):
            SHAP_ROWS.append(dict(trait=target, ri=ri, scenario=key, feature=f,
                                  mean_abs_shap=float(ma[key][i]), mean_shap=float(ms[key][i])))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.22, wspace=0.52)
    panel_importance_grouped(axes[0], ma, order, "(a) Feature importance")
    panel_delta_heatmap(axes[1], ms, order, "(b) ΔSHAP (future−baseline)")
    panel_co2_tavg(axes[2], sv_dict, feats, "(c) CO2 vs Tavg SHAP")
    stem = OUT_DIR / f"shap_{target}_{ri}_3scen"
    for ext in ("svg", "pdf"): fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {stem.name}.svg/.pdf/.tiff")
    return sv_dict, ma, ms, order

def run_hist_only(target, ri):
    print(f"\n[HIST] {target} · {ri}")
    vr = load_model(target)
    if vr is None: return
    feats = features_for(target)
    X = load_feat(feat_csv("historical", ri), feats); w = wts(feat_csv("historical", ri), feats)
    sv = ensemble_shap(vr, X, feats, f"{target}_{ri}_hist")
    ma, ms = wmean0(np.abs(sv), w), wmean0(sv, w)
    for i, f in enumerate(feats):
        SHAP_ROWS.append(dict(trait=target, ri=ri, scenario="hist", feature=f,
                              mean_abs_shap=float(ma[i]), mean_shap=float(ms[i])))
    order = _order_by_importance(ma)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.82, bottom=0.20, wspace=0.55)
    panel_importance_single(axes[0], ma, order, "(a) Feature importance", COLORS["hist"])
    panel_beeswarm(axes[1], sv, X, order, "(b) Direction (beeswarm)")
    stem = OUT_DIR / f"shap_{target}_{ri}_hist"
    for ext in ("svg", "pdf"): fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {stem.name}.svg/.pdf/.tiff")

def ensemble_shap_f(vr, X_df, f, cache_key):
    """SHAP = f·SHAP(TYPE=1) + (1−f)·SHAP(TYPE=2)，返回气候特征部分"""
    feats_t = FEATURES_BASE + ["TYPE"]
    sv1 = ensemble_shap(vr, X_df.assign(TYPE=1)[feats_t], feats_t, f"{cache_key}_T1")
    sv2 = ensemble_shap(vr, X_df.assign(TYPE=2)[feats_t], feats_t, f"{cache_key}_T2")
    n = len(FEATURES_BASE)
    return f[:, None] * sv1[:, :n] + (1.0 - f[:, None]) * sv2[:, :n]


def run_acontent_fweighted(ri):
    print(f"\n[HIST · f-weighted] {ACONTENT} · {ri}")
    vr = load_model(ACONTENT)
    if vr is None: return
    D = load_feat(feat_csv("historical", ri), FEATURES_BASE + ["f_indica"])
    w = wts(feat_csv("historical", ri), FEATURES_BASE + ["f_indica"])
    X, f = D[FEATURES_BASE], D["f_indica"].to_numpy(dtype=float)
    sv = ensemble_shap_f(vr, X, f, f"{ACONTENT}_{ri}_hist")
    ma, ms = wmean0(np.abs(sv), w), wmean0(sv, w)
    for i, ft in enumerate(FEATURES_BASE):
        SHAP_ROWS.append(dict(trait=ACONTENT, ri=ri, scenario="hist", feature=ft,
                              mean_abs_shap=float(ma[i]), mean_shap=float(ms[i])))
    order = _order_by_importance(ma)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.82, bottom=0.20, wspace=0.55)
    panel_importance_single(axes[0], ma, order, "(a) Feature importance", COLORS["hist"])
    panel_beeswarm(axes[1], sv, X, order, "(b) Direction (beeswarm)")
    stem = OUT_DIR / f"shap_{ACONTENT}_{ri}_hist"
    for ext in ("svg", "pdf"): fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {stem.name}.svg/.pdf/.tiff")


def run_acontent_by_type(ri):
    print(f"\n[HIST · by TYPE] {ACONTENT} · {ri}")
    vr = load_model(ACONTENT)
    if vr is None: return
    feats = features_for(ACONTENT)
    X_all = load_feat(feat_csv("historical", ri), feats)
    groups = {1: "indica (high-AC)", 2: "japonica (low-AC)"}

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.09, hspace=0.55, wspace=0.55)
    for r, (tv, gname) in enumerate(groups.items()):
        Xg = X_all[X_all["TYPE"] == tv].reset_index(drop=True)
        if len(Xg) < 20:
            print(f"  ! TYPE=={tv} 样本过少({len(Xg)})，跳过"); 
            axes[r,0].set_axis_off(); axes[r,1].set_axis_off(); continue
        print(f"  {gname}: {len(Xg)} px")
        sv = ensemble_shap(vr, Xg, feats, f"{ACONTENT}_{ri}_hist_type{tv}")
        ma = np.abs(sv).mean(0)
        order = _order_by_importance(ma[:len(FEATURES_BASE)])
        panel_importance_single(axes[r, 0], ma, order,
                                f"({'ac'[r]}) Importance — {gname}", COLORS["hist"])
        panel_beeswarm(axes[r, 1], sv[:, :len(FEATURES_BASE)],
                       Xg[FEATURES_BASE], order, f"({'bd'[r]}) Direction — {gname}")
    stem = OUT_DIR / f"shap_{ACONTENT}_{ri}_hist_byTYPE"
    for ext in ("svg", "pdf"): fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {stem.name}.svg/.pdf/.tiff")

def make_grading_summary(grading_out, ri):
    """定级三性状汇总大图（3 行 × 3 面板），供正文主图。"""
    print(f"\n[SUMMARY · {ri}] 定级三性状汇总大图 ...")
    fig = plt.figure(figsize=(14.0, 13.0))
    gs = gridspec.GridSpec(3, 3, figure=fig, left=0.12, right=0.975,
                           top=0.90, bottom=0.16, hspace=0.66, wspace=0.50,
                           width_ratios=[1.25, 1.0, 1.0])
    col_headers = ["SHAP-importance", "ΔSHAP", "SHAP interaction"]
    vmax_all = 1e-6
    for t in GRADING:
        _, _, ms, order = grading_out[t]
        d126 = np.array([ms["ssp126"][fi] - ms["hist"][fi] for fi in order])
        d585 = np.array([ms["ssp585"][fi] - ms["hist"][fi] for fi in order])
        vmax_all = max(vmax_all, np.abs(np.vstack([d126, d585])).max())
    vmax_all *= 1.05
    top_axes = []
    for row, t in enumerate(GRADING):
        sv_dict, ma, ms, order = grading_out[t]
        ax0 = fig.add_subplot(gs[row, 0]); ax1 = fig.add_subplot(gs[row, 1]); ax2 = fig.add_subplot(gs[row, 2])
        panel_importance_grouped(ax0, ma, order, f"({chr(97+row*3)})", show_legend=False)
        panel_delta_heatmap(ax1, ms, order, f"({chr(97+row*3+1)})", show_cbar=False, vmax=vmax_all)
        panel_co2_tavg(ax2, sv_dict, features_for(t), f"({chr(97+row*3+2)})", show_legend=False)
        p0 = ax0.get_position()
        fig.text(0.030, (p0.y0 + p0.y1) / 2, t, ha="center", va="center",
                 rotation=90, fontsize=FS_TITLE, fontweight="bold")
        if row == 0:
            top_axes = [ax0, ax1, ax2]
    for ax, htxt in zip(top_axes, col_headers):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.050, htxt, ha="center", va="bottom",
                 fontsize=FS_TITLE, fontweight="bold")
    xc0 = 0.5 * (top_axes[0].get_position().x0 + top_axes[0].get_position().x1)
    xc1 = 0.5 * (top_axes[1].get_position().x0 + top_axes[1].get_position().x1)
    xc2 = 0.5 * (top_axes[2].get_position().x0 + top_axes[2].get_position().x1)
    y_leg = 0.075
    scen = [("Baseline", "hist"), ("SSP1-2.6", "ssp126"), ("SSP5-8.5", "ssp585")]
    bar_h = [Patch(facecolor=COLORS[k], label=l) for l, k in scen]
    fig.legend(handles=bar_h, loc="center", bbox_to_anchor=(xc0, y_leg), ncol=3,
               fontsize=FS_LEGEND, frameon=False, handlelength=1.1,
               handletextpad=0.4, columnspacing=1.2)
    dot_h = [Line2D([], [], marker="o", linestyle="", markersize=8,
                    color=COLORS[k], label=l) for l, k in scen]
    fig.legend(handles=dot_h, loc="center", bbox_to_anchor=(xc2, y_leg), ncol=3,
               fontsize=FS_LEGEND, frameon=False, handlelength=1.0,
               handletextpad=0.3, columnspacing=1.2)
    cax = fig.add_axes([xc1 - 0.10, y_leg - 0.006, 0.20, 0.016])
    sm = ScalarMappable(norm=Normalize(-vmax_all, vmax_all), cmap="RdBu_r"); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("ΔSHAP", fontsize=FS_LABEL); cb.ax.tick_params(labelsize=FS_TICK)
    stem = OUT_DIR / f"shap_grading_summary_{ri}"
    for ext in ("svg", "pdf"): fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {stem.name}.svg/.pdf/.tiff")


if __name__ == "__main__":
    print("=" * 60)
    print("SHAP: 定级三性状(三情景) + 其余(历史) + Acontent(历史·按 f 加权)")
    print("循环: ri1 / ri2")
    print("=" * 60)

    for ri in RI_TYPES:
        print(f"\n{'#'*60}\n#  {ri}\n{'#'*60}")
        grading_out = {}
        for t in GRADING:
            r = run_grading(t, ri)
            if r is not None: grading_out[t] = r

        for t in HIST_ONLY:
            run_hist_only(t, ri)

        run_acontent_fweighted(ri)

        if len(grading_out) == len(GRADING):
            make_grading_summary(grading_out, ri)

    pd.DataFrame(SHAP_ROWS).to_csv(OUT_DIR / "shap_mean_abs.csv", index=False, float_format="%.5f")
    print("  ✓ shap_mean_abs.csv")
    print("\n完成。输出目录:", OUT_DIR)
