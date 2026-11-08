# ============================================================
# figS_co2_interaction.py —— 补充材料图：CO₂ 效应随灌浆期温度的变化（PC、HRR）
#   数据：check_co2_response.py 的输出（同目录）
#     check_co2_interaction_cells.csv  每个格网的 CO₂ 效应（随情景 CO₂ − 固定 CO₂，相对基准，pp）与未来 Tavg
#     check_co2_fixedwindow.csv        实际窗口 vs 固定窗口（增温较小）的 CO₂ 效应
#   a/b（d/e）单季/早稻、晚稻：按格网未来灌浆期 Tavg 分箱（2 °C）的箱线图，蓝 = SSP1-2.6，橙 = SSP5-8.5；
#       箱 = 四分位，须 = 10–90 百分位；格网数 < 15 的分箱不画；ρ = Spearman 秩相关
#   c（f）实际窗口（实色）vs 窗口固定在基准期（斜线，同 Fig. 2b）的 CO₂ 效应
#   用法：python figs_1020/figS_co2_interaction.py [--traits PC,HRR] [--out ...]
# ============================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_common as F                                            # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

F.setup()
F.FONT_SCALE = 1.2                             # [1105] 补充图文字放大 20%（最小 6 pt），与正文图可读性一致
p = argparse.ArgumentParser()
p.add_argument("--traits", default="PC")                 # [1105] SI 图 S14 只用 PC；加 --traits PC,HRR 可出两行版
p.add_argument("--cells", default=str(HERE / "check_co2_interaction_cells.csv"))
p.add_argument("--fixedwindow", default=str(HERE / "check_co2_fixedwindow.csv"))
p.add_argument("--out", default=None)
a = p.parse_args()
OUT = Path(a.out) if a.out else F.default_out()
TRAITS = [t.strip() for t in a.traits.split(",") if t.strip()]

cells = pd.read_csv(a.cells)
fw = pd.read_csv(a.fixedwindow) if Path(a.fixedwindow).exists() else pd.DataFrame()
BINS = [-np.inf, 22, 24, 26, 28, 30, np.inf]
BLAB = ["<22", "22–24", "24–26", "26–28", "28–30", "≥30"]
cells["bin"] = pd.cut(cells["Tavg"], BINS, labels=BLAB)
COL = F.SSP_COL
MINN = 15
HATCH = "//////"
TNAME = {"PC": "PC", "HRR": "HRR"}

W = 180.0
ncol = 3 if not fw.empty else 2
RH = 44.0                                                         # 每行高度（mm）
H = RH * len(TRAITS) + 12.0
fig, axs = plt.subplots(len(TRAITS), ncol, figsize=(W * F.MM, H * F.MM),
                        gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.8][:ncol], wspace=0.32, hspace=0.62))
axs = np.atleast_2d(axs)
letters = iter("abcdefghijkl")
for i, t in enumerate(TRAITS):
    ylims = []
    for j, ri in enumerate(F.RIS):
        ax = axs[i, j]
        d = cells[(cells.trait == t) & (cells.ri == ri)]
        used = [b for b in BLAB if (d["bin"] == b).sum() >= MINN * 2 or
                any(((d["bin"] == b) & (d.ssp == s)).sum() >= MINN for s in F.SSPS)]
        for k, s in enumerate(F.SSPS):
            ds = d[d.ssp == s]
            for x, b in enumerate(used):
                v = ds.loc[ds["bin"] == b, "co2_effect_pp"].dropna().values
                if v.size < MINN:
                    continue
                q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
                xc = x + (k - 0.5) * 0.36
                ax.add_patch(plt.Rectangle((xc - 0.15, q25), 0.30, q75 - q25, facecolor=COL[s], edgecolor=F.INK,
                                           linewidth=0.4, alpha=0.9, zorder=3))
                ax.plot([xc - 0.15, xc + 0.15], [q50, q50], color=F.INK, lw=0.8, zorder=4)
                ax.plot([xc, xc], [q10, q25], color=F.INK, lw=0.5, zorder=2)
                ax.plot([xc, xc], [q75, q90], color=F.INK, lw=0.5, zorder=2)
                ylims += [q10, q90]
            r_ = ds["Tavg"].corr(ds["co2_effect_pp"], method="spearman")
            ax.text(0.03, 0.96 - 0.10 * k, f"ρ = {r_:.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6, color=COL[s])
        ax.axhline(0, color=F.INK, lw=0.5, zorder=1)
        ax.set_xticks(range(len(used))); ax.set_xticklabels(used, fontsize=5.8)
        ax.set_xlim(-0.6, len(used) - 0.4)
        ax.tick_params(axis="x", length=0, pad=2); ax.tick_params(axis="y", labelsize=6, length=1.5, pad=1)
        ax.set_xlabel("Future grain-filling Tavg (°C)", fontsize=6.5, labelpad=1.5)
        if j == 0:
            ax.set_ylabel(f"CO₂ effect on {TNAME[t]} (pp)", fontsize=6.5, labelpad=1)
        ax.set_title(F.RI_LAB[ri], fontsize=7, fontweight="bold", pad=3)
        ax.grid(axis="y", color="0.93", lw=0.4, zorder=0); ax.set_axisbelow(True)
        ax.text(-0.16 if j == 0 else -0.12, 1.06, next(letters), transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="bottom")
    lo, hi = min(ylims), max(ylims)
    pad = 0.08 * (hi - lo)
    for j in range(2):
        axs[i, j].set_ylim(lo - pad, hi + pad)                   # 两季同一纵轴
    if ncol == 3:
        ax = axs[i, 2]
        keys = [(ri, s) for ri in F.RIS for s in F.SSPS]
        for x, (ri, s) in enumerate(keys):
            r = fw[(fw.trait == t) & (fw.ri == ri) & (fw.ssp == s)].set_index("window")
            ax.bar(x - 0.19, r.loc["actual window", "co2_effect_pp"], 0.36, color=COL[s], edgecolor=F.INK,
                   linewidth=0.4, zorder=3)
            ax.bar(x + 0.19, r.loc["fixed window", "co2_effect_pp"], 0.36, facecolor="white", edgecolor=COL[s],
                   hatch=HATCH, linewidth=0.6, zorder=3)
        ax.axhline(0, color=F.INK, lw=0.5, zorder=1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{'S/E' if ri == 'ri1' else 'Late'}\n{F.SSP_LAB[s].replace('SSP', '')}" for ri, s in keys],
                           fontsize=5.6)
        ax.tick_params(axis="x", length=0, pad=2); ax.tick_params(axis="y", labelsize=6, length=1.5, pad=1)
        ax.set_ylabel("CO₂ effect (pp)", fontsize=6.5, labelpad=1)
        ax.set_title("Grain-filling window", fontsize=7, fontweight="bold", pad=3)
        ax.grid(axis="y", color="0.93", lw=0.4, zorder=0); ax.set_axisbelow(True)
        ax.text(-0.22, 1.06, next(letters), transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left",
                va="bottom")
hdl = [Patch(facecolor=COL["ssp126"], edgecolor=F.INK, linewidth=0.4, label="SSP1-2.6"),
       Patch(facecolor=COL["ssp585"], edgecolor=F.INK, linewidth=0.4, label="SSP5-8.5")]
if ncol == 3:
    hdl += [Patch(facecolor="0.45", edgecolor=F.INK, linewidth=0.4, label="Projected window"),
            Patch(facecolor="white", edgecolor="0.45", hatch=HATCH, linewidth=0.6,
                  label="Window fixed at baseline (less warming)")]
fig.legend(handles=hdl, loc="lower center", ncol=len(hdl), fontsize=6, handlelength=1.3, columnspacing=1.2,
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.subplots_adjust(left=0.07, right=0.99, top=1 - 6.0 / H, bottom=13.0 / H)
F.save(fig, OUT, "FigS_co2_interaction", tight=False)
