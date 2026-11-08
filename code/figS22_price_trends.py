# -*- coding: utf-8 -*-
"""
figS22_price_trends.py —— 补图 22：中国集贸市场米价 + 稻谷价格（同一张图）
  实线 = 精米（2002–2024）：普通籼米、粳米、优质籼米、优质参考价 = mean(优质籼米, 粳米)
  虚线 = 稻谷（2017–2024）：早籼稻、中晚籼稻、粳稻、稻谷参考价 = 三者均值（csv 里 2025、2026 年不画）
  灰色阴影 = 2020–2024（估值所用均值的时段）；图例括号内为该时段均值（估值实际使用的三个价格）
输入（与脚本同目录或用参数指定）：
  milled_rice_price_trends_plotdata.csv  列: year, ordinary_indica_milled, japonica_milled, HQ_indica_milled, HQ_rice_mean
  rice_price_trends_plotdata.csv         列: year, early indica rice, mid/late indica rice, japonica rice
输出：<out>/price_trend.svg / .pdf / .png（600 dpi）；out 默认 /home/debian/pythonfile/ISIMIP3b/output/figures
      （collect_figs_1105.py 会把 price_trend.* 收为 FigS22_price_trend）
用法：python figS22_price_trends.py [--milled 路径] [--paddy 路径] [--out 目录]
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--milled", default=str(HERE / "milled_rice_price_trends_plotdata.csv"))
ap.add_argument("--paddy", default=str(HERE / "rice_price_trends_plotdata.csv"))
default_out = Path("/home/debian/pythonfile/ISIMIP3b/output/figures")
ap.add_argument("--out", default=str(default_out if default_out.exists() else HERE))
A = ap.parse_args()

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

WIN = (2020, 2024)
YEAR_MAX = 2024
m = pd.read_csv(A.milled).dropna(subset=["ordinary_indica_milled"])
m = m[m.year <= YEAR_MAX]
p = pd.read_csv(A.paddy)
p.columns = [c.strip() for c in p.columns]
PCOL = ["early indica rice", "mid/late indica rice", "japonica rice"]
p = p.dropna(subset=PCOL, how="all")
p = p[p.year <= YEAR_MAX]
p["paddy_mean"] = p[PCOL].mean(axis=1)

def wmean(df, col):
    s = df[(df.year >= WIN[0]) & (df.year <= WIN[1])][col]
    return s.mean()

ref = {"ordinary_indica_milled": wmean(m, "ordinary_indica_milled"),
       "HQ_rice_mean": wmean(m, "HQ_rice_mean"),
       "paddy_mean": wmean(p, "paddy_mean")}
print("2020–2024 means: ordinary %.2f, HQ reference %.2f (ratio %.3f), paddy %.2f; paddy series %s" % (
    ref["ordinary_indica_milled"], ref["HQ_rice_mean"], ref["HQ_rice_mean"] / ref["ordinary_indica_milled"],
    ref["paddy_mean"], ", ".join(f"{c} {wmean(p, c):.2f}" for c in PCOL)))

MILLED = [("ordinary_indica_milled", "Ordinary indica", "#1f77b4", "o"),
          ("japonica_milled", "Japonica", "#ff7f0e", "s"),
          ("HQ_indica_milled", "High-quality indica", "#d62728", "^"),
          ("HQ_rice_mean", "High-quality reference", "#2ca02c", "D")]
PADDY = [("early indica rice", "Early indica", "#6baed6", "o"),
         ("mid/late indica rice", "Mid/late indica", "#08519c", "v"),
         ("japonica rice", "Japonica", "#fdae6b", "s"),
         ("paddy_mean", "Paddy reference", "0.25", None)]

fig, ax = plt.subplots(figsize=(146 / 25.4, 78 / 25.4))
ax.axvspan(WIN[0] - 0.5, WIN[1] + 0.5, color="0.92", zorder=0, lw=0)
ax.text((WIN[0] + WIN[1]) / 2, 6.95, "2020–2024 means\nused in the valuation", ha="center", va="top",
        fontsize=5.8, color="0.35", linespacing=1.1)

h_m, h_p = [], []
for col, lab, c, mk in MILLED:
    ax.plot(m.year, m[col], "-", marker=mk, ms=2.8, lw=1.2, color=c, zorder=3)
    lab2 = f"{lab} ({ref[col]:.2f})" if col in ref else lab
    h_m.append(Line2D([], [], color=c, marker=mk, ms=2.8, lw=1.2, label=lab2))
for col, lab, c, mk in PADDY:
    if mk is None:
        ax.plot(p.year, p[col], ls=(0, (4, 1.6)), lw=1.6, color=c, zorder=4)
        h_p.append(Line2D([], [], color=c, ls=(0, (4, 1.6)), lw=1.6, label=f"{lab} ({ref[col]:.2f})"))
    else:
        ax.plot(p.year, p[col], ls=(0, (3, 1.5)), marker=mk, ms=2.8, lw=1.0, color=c, mfc="white",
                mew=0.8, zorder=3)
        h_p.append(Line2D([], [], color=c, ls=(0, (3, 1.5)), marker=mk, ms=2.8, lw=1.0, mfc="white",
                          mew=0.8, label=lab))

ax.set_xlim(2001.5, YEAR_MAX + 0.5)
ax.set_ylim(1.0, 7.0)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_xlabel("Year")
ax.set_ylabel(r"Rural-market price (yuan kg$^{-1}$)")
ax.grid(axis="y", color="0.88", lw=0.5, zorder=1)

leg1 = ax.legend(handles=h_m, title="Milled rice", loc="upper left", frameon=False, handlelength=2.4,
                 borderaxespad=0.3, labelspacing=0.35, title_fontsize=6.5)
leg1._legend_box.align = "left"
ax.add_artist(leg1)
leg2 = ax.legend(handles=h_p, title="Paddy", loc="lower right", frameon=False, handlelength=2.4,
                 borderaxespad=0.3, labelspacing=0.35, title_fontsize=6.5, ncol=2, columnspacing=1.2)
leg2._legend_box.align = "left"

fig.tight_layout(pad=0.3)
out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "pdf", "png"):
    fig.savefig(out / f"price_trend.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved:", out / "price_trend.{svg,pdf,png}")
