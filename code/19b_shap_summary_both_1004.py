"""
19b_shap_summary_both_1004.py —— Supplementary Fig. S20：定级三性状 SHAP 汇总，两季合为一张图
====================================================================================
上半 a–i = 单季/早稻（ri1），下半 j–r = 晚稻（ri2）；每季三行（HRR / CD / PC）× 三列
（mean|SHAP| 分组条形、ΔSHAP 热图、CO₂ vs Tavg 散点），版式与 19 号的 shap_grading_summary_{ri} 相同。

  · 复用 19_shap_analysis_1004.py 的数据加载、加权 SHAP 与面板函数（importlib 导入，不改 19 号）
  · SHAP 数组优先读 19 号的缓存 shap_arrays/sv_{trait}_{ri}_{key}.npy；缺失或行数不符时按 19 号同样的方式重算
  · 六个 ΔSHAP 热图共用一个色阶（全图统一 vmax），底部一条色标；散点每期随机取 400 格点（seed 42，同 19 号）
  · 汇总权重同 19 号（config.DIAG_WEIGHT，定稿为逐格等权）

USAGE:  cd code_1004b && python figs_1020/19b_shap_summary_both_1004.py
        （先跑过 19 号则直接用缓存，几秒出图；没跑过会先算 SHAP）
输出:   fig_dir("shap")/shap_grading_summary_both.svg / .pdf / .tiff
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(Path.cwd()))

# ---- 导入 19 号脚本（文件名以数字开头，用 importlib）；19 号的主流程在 __main__ 保护下，不会被触发
_cands = [HERE / "19_shap_analysis_1004.py", HERE.parent / "19_shap_analysis_1004.py", Path.cwd() / "19_shap_analysis_1004.py"]
_p19 = next((p for p in _cands if p.exists()), None)
if _p19 is None:
    raise SystemExit("找不到 19_shap_analysis_1004.py（应与本脚本同目录或在上级目录）")
_spec = importlib.util.spec_from_file_location("shap19", _p19)
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)

GRADING = S.GRADING                      # ["HRR", "CD", "PC"]
RI = list(S.RI_TYPES)                    # ["ri1", "ri2"]
RI_LAB = {"ri1": "Single/early rice", "ri2": "Late rice"}
assert RI == ["ri1", "ri2"], f"RI_TYPES = {RI}，本脚本按 ri1 在上、ri2 在下排版"


def grading_data(target, ri):
    """与 19 号 run_grading 相同的数据准备，但不画单图：返回 sv_dict, ma, ms, order"""
    vr = S.load_model(target)
    if vr is None:
        raise SystemExit(f"模型缺失: {target}")
    feats = S.features_for(target)
    sv_dict, ma, ms = {}, {}, {}
    for _, key, period in S.SCEN_ENTRIES:
        X = S.load_feat(S.feat_csv(period, ri), feats); w = S.wts(S.feat_csv(period, ri), feats)
        sv = S.ensemble_shap(vr, X, feats, f"{target}_{ri}_{key}")      # 缓存命中则直接读
        sv_dict[key] = sv
        ma[key] = S.wmean0(np.abs(sv), w); ms[key] = S.wmean0(sv, w)
    order = S._order_by_importance(ma["hist"], ma["ssp126"], ma["ssp585"])
    return sv_dict, ma, ms, order


print("=" * 60); print("S20: SHAP grading summary, both seasons in one figure"); print("=" * 60)
DATA = {(ri, t): grading_data(t, ri) for ri in RI for t in GRADING}

# ---- 全图统一的 ΔSHAP 色阶
vmax_all = 1e-6
for (ri, t), (_, _, ms, order) in DATA.items():
    d = np.array([[ms[s][fi] - ms["hist"][fi] for fi in order] for s in ("ssp126", "ssp585")])
    vmax_all = max(vmax_all, float(np.abs(d).max()))
vmax_all *= 1.05

# ---- 版式：6 行（ri1: HRR/CD/PC；ri2: HRR/CD/PC）× 3 列；两季之间留一条空行放季名
fig = plt.figure(figsize=(14.0, 26.5))
gs = gridspec.GridSpec(7, 3, figure=fig, left=0.12, right=0.975, top=0.955, bottom=0.085,
                       hspace=0.70, wspace=0.50, width_ratios=[1.25, 1.0, 1.0],
                       height_ratios=[1, 1, 1, 0.28, 1, 1, 1])
col_headers = ["SHAP-importance", "ΔSHAP", "SHAP interaction"]
ROWS = [(0, "ri1", "HRR"), (1, "ri1", "CD"), (2, "ri1", "PC"), (4, "ri2", "HRR"), (5, "ri2", "CD"), (6, "ri2", "PC")]
first_axes = {}
k = 0                                                   # 面板字母计数 a–r
for grow, ri, t in ROWS:
    sv_dict, ma, ms, order = DATA[(ri, t)]
    ax0 = fig.add_subplot(gs[grow, 0]); ax1 = fig.add_subplot(gs[grow, 1]); ax2 = fig.add_subplot(gs[grow, 2])
    S.panel_importance_grouped(ax0, ma, order, f"({chr(97 + k)})", show_legend=False)
    S.panel_delta_heatmap(ax1, ms, order, f"({chr(97 + k + 1)})", show_cbar=False, vmax=vmax_all)
    S.panel_co2_tavg(ax2, sv_dict, S.features_for(t), f"({chr(97 + k + 2)})", show_legend=False)
    k += 3
    p0 = ax0.get_position()
    fig.text(0.030, (p0.y0 + p0.y1) / 2, t, ha="center", va="center", rotation=90,
             fontsize=S.FS_TITLE, fontweight="bold")
    if grow in (0, 4):
        first_axes[ri] = [ax0, ax1, ax2]

# ---- 列标 + 季名（每季上方各一组）
for ri, axes in first_axes.items():
    for ax, htxt in zip(axes, col_headers):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.022, htxt, ha="center", va="bottom",
                 fontsize=S.FS_TITLE, fontweight="bold")
    xl = first_axes[ri][0].get_position().x0; xr = first_axes[ri][2].get_position().x1
    fig.text((xl + xr) / 2, first_axes[ri][0].get_position().y1 + 0.046, RI_LAB[ri],
             ha="center", va="bottom", fontsize=S.FS_SUP, fontweight="bold")

# ---- 底部图例：条形色块 / ΔSHAP 色标 / 散点圆点，各居中于对应列下方
top = first_axes["ri1"]
xc0, xc1, xc2 = [0.5 * (a.get_position().x0 + a.get_position().x1) for a in top]
y_leg = 0.040
scen = [("Baseline", "hist"), ("SSP1-2.6", "ssp126"), ("SSP5-8.5", "ssp585")]
fig.legend(handles=[Patch(facecolor=S.COLORS[kk], label=l) for l, kk in scen], loc="center",
           bbox_to_anchor=(xc0, y_leg), ncol=3, fontsize=S.FS_LEGEND, frameon=False,
           handlelength=1.1, handletextpad=0.4, columnspacing=1.2)
fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=8, color=S.COLORS[kk], label=l) for l, kk in scen],
           loc="center", bbox_to_anchor=(xc2, y_leg), ncol=3, fontsize=S.FS_LEGEND, frameon=False,
           handlelength=1.0, handletextpad=0.3, columnspacing=1.2)
cax = fig.add_axes([xc1 - 0.10, y_leg - 0.004, 0.20, 0.009])
sm = ScalarMappable(norm=Normalize(-vmax_all, vmax_all), cmap="RdBu_r"); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("ΔSHAP", fontsize=S.FS_LABEL); cb.ax.tick_params(labelsize=S.FS_TICK)

stem = S.OUT_DIR / "shap_grading_summary_both"
for ext in ("svg", "pdf"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
fig.savefig(f"{stem}.tiff", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {stem.name}.svg/.pdf/.tiff  （a–i 单季/早稻，j–r 晚稻；ΔSHAP 统一色阶 ±{vmax_all:.2f}）")
