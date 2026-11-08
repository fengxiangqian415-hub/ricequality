import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path.cwd()))
try:
    import paper_palette as PP
    SSP_COL = dict(PP.SSP)
    INK = PP.INK
except Exception:
    SSP_COL = {"baseline": "#7D7D7D", "ssp126": "#2C6FB0", "ssp585": "#E8833A"}
    INK = "#2B2B2B"

MM = 1 / 25.4
SSPS = ["ssp126", "ssp585"]
RIS = ["ri1", "ri2"]
SSP_LAB = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
RI_LAB = {"ri1": "Single/early rice", "ri2": "Late rice"}
RI_SHORT = {"ri1": "Single/\nearly", "ri2": "Late"}
TRAITS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]
TLAB = {"BRR": "BRR", "MRR": "MRR", "HRR": "HRR", "CGR": "CGR", "CD": "CD", "Acontent": "AC", "PC": "PC"}
TNAME = {"BRR": "Brown-rice rate", "MRR": "Milled-rice rate", "HRR": "Head-rice rate",
         "CGR": "Chalky-grain rate", "CD": "Chalkiness degree", "Acontent": "Amylose content",
         "PC": "Protein content"}
QTRAITS = {"HRR", "CD", "PC"}
CO2_STYLE = {"with_CO2": dict(filled=True, label="Scenario CO₂"),
             "fixed_CO2": dict(filled=False, label="CO₂ fixed at 1981–2010 mean")}


def setup():
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.cal": "sans",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": True, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    })


def panel_label(ax, s, x=-0.14, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def args(desc, need_xlsx=True):
    p = argparse.ArgumentParser(description=desc)
    if need_xlsx:
        p.add_argument("--xlsx", default=None, help="paper_data Excel（默认 config 的 excel 目录）")
    p.add_argument("--out", default=None, help="输出目录（默认 output/figures/main_figs_1020）")
    return p


def default_xlsx():
    import config_1004 as C
    return C.out_dir("excel") / f"paper_data_{C.TAG}.xlsx"


def default_out():
    try:
        import config_1004 as C
        return C.FIGURES / "main_figs_1020"
    except Exception:
        return HERE / "main_figs_1020"


def read_sheet(xlsx, name):
    return pd.read_excel(xlsx, sheet_name=name)


FONT_SCALE = 1.0
MIN_FONT_PT = 6.0
IMG_DPI = 600


def scale_fonts(fig, k=None, floor=None):
    """把图里所有文字（含刻度、图例、色条）字号乘以 k，并抬到不低于 floor。先 draw 一次，保证刻度标签已生成。"""
    import matplotlib.text as mtext
    k = FONT_SCALE if k is None else k
    floor = MIN_FONT_PT if floor is None else floor
    if k == 1.0 and not floor:
        return
    fig.canvas.draw()
    seen = set()
    for t in fig.findobj(mtext.Text):
        if id(t) in seen or not t.get_text():
            continue
        seen.add(id(t))
        t.set_fontsize(max(t.get_fontsize() * k, floor))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            if labs:
                axis.set_tick_params(labelsize=labs[0].get_fontsize())


def save(fig, out, stem, tight=True):
    """tight=False：按画布原尺寸输出（版式全部按毫米排好的图用，成图宽正好 = 画布宽）"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scale_fonts(fig)
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=IMG_DPI, **kw)
    fig.savefig(out / f"{stem}.png", dpi=600, **kw)
    print(f"[saved] {out / stem}.png / .pdf / .svg  (FONT_SCALE={FONT_SCALE}, min {MIN_FONT_PT} pt)")
