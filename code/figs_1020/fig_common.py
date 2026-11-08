# ============================================================
# fig_common.py —— 正文新图（1020）公共设置：NC 版式、配色、读表
#   放在 code_1004b 目录下与 config_1004.py / paper_palette.py 同级运行
# ============================================================
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
sys.path.insert(0, str(HERE.parent))                # 放在 code_1004b/figs_1020/ 下时可直接导入 config_1004
sys.path.insert(0, str(Path.cwd()))                 # 或在 code_1004b 目录下运行
try:
    import paper_palette as PP
    SSP_COL = dict(PP.SSP)
    INK = PP.INK
except Exception:                                   # 预览环境没有 paper_palette 时的兜底
    SSP_COL = {"baseline": "#7D7D7D", "ssp126": "#2C6FB0", "ssp585": "#E8833A"}
    INK = "#2B2B2B"

MM = 1 / 25.4                                       # mm → inch
SSPS = ["ssp126", "ssp585"]
RIS = ["ri1", "ri2"]
SSP_LAB = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
RI_LAB = {"ri1": "Single/early rice", "ri2": "Late rice"}
RI_SHORT = {"ri1": "Single/\nearly", "ri2": "Late"}
TRAITS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]           # 全文统一顺序：碾磨 → 外观 → 食味
TLAB = {"BRR": "BRR", "MRR": "MRR", "HRR": "HRR", "CGR": "CGR", "CD": "CD", "Acontent": "AC", "PC": "PC"}
TNAME = {"BRR": "Brown-rice rate", "MRR": "Milled-rice rate", "HRR": "Head-rice rate",
         "CGR": "Chalky-grain rate", "CD": "Chalkiness degree", "Acontent": "Amylose content",
         "PC": "Protein content"}
QTRAITS = {"HRR", "CD", "PC"}
# 含 / 固定 CO₂ 的编码：颜色 = 情景，实心 = 情景 CO₂，空心 = CO₂ 固定（全文统一，不再用蓝/橙区分 CO₂）
CO2_STYLE = {"with_CO2": dict(filled=True, label="Scenario CO₂"),
             "fixed_CO2": dict(filled=False, label="CO₂ fixed at 1981–2010 mean")}


def setup():
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.cal": "sans",                       # 公式里的下标与正文同字体（Arial）
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": True, "text.color": INK, "axes.labelcolor": INK,   # [1108] unicode_minus=True：刻度负号用 U+2212（Arial 有此字形）
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


FONT_SCALE = 1.0          # [1031] 全图文字放大倍数（各脚本在 save 前设置，如 F.FONT_SCALE = 1.15）
MIN_FONT_PT = 6.0         # [1031] 放大后仍低于此值的文字抬到此值（Nature 要求最小 5–7 pt；100% 视图可读）
IMG_DPI = 600             # [1031] 矢量输出中嵌入位图（imshow 照片）的分辨率；默认 fig.dpi=100 会把照片压糊


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
    for ax in fig.axes:                                              # 刻度标签已被上面放大；把 tick 参数同步，防止保存时重建的标签恢复原字号
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            if labs:
                axis.set_tick_params(labelsize=labs[0].get_fontsize())


def save(fig, out, stem, tight=True):
    """tight=False：按画布原尺寸输出（版式全部按毫米排好的图用，成图宽正好 = 画布宽）"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scale_fonts(fig)
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}             # 边距 0.5 mm，成图宽 ≤ 180 mm
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=IMG_DPI, **kw)                     # dpi 只影响嵌入的位图（照片）
    fig.savefig(out / f"{stem}.png", dpi=600, **kw)
    print(f"[saved] {out / stem}.png / .pdf / .svg  (FONT_SCALE={FONT_SCALE}, min {MIN_FONT_PT} pt)")
