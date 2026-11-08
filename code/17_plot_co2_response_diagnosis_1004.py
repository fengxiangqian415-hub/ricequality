# ============================================================
# 17_plot_co2_response_diagnosis_1004.py —— CO2 "低谷/平台" 示意图（训练数据 + 模型；时段 CO2 来自 config_1004）
#   (a) 训练数据 CO2 分布 + 三个时段的 CO2 范围
#   (b)(c) PC、HRR：
#       灰线   试验内配对（同一试验的对照 → 高 CO2）：真实的 CO2 效应
#       黑点   按 CO2 分箱的全部记录均值（试验间比较，混杂了地点/品种差异）
#       红/蓝线 树模型偏依赖曲线（其余气候因子固定为单季/晚稻典型值）—— 有模型文件时绘制
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config_1004 import FACE_XLSX, MODEL_DIR, FIGURES, TAG, HIS_YEARS, FUT_YEARS, HIS_LABEL, FUT_LABEL, co2_series
OUT       = FIGURES / f"co2_diagnosis_{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
TRAITS    = ["PC", "HRR"]
LANG      = "auto"     # "auto"：找到中文字体用中文，否则自动改用英文；也可强制 "zh" / "en"
FONT_FILE = None       # 可选：中文字体文件路径，如 "/home/debian/fonts/simhei.ttf"

# ---- 字体：只使用系统里确实存在的中文字体，避免方框乱码 ----
from matplotlib import font_manager as fm
if FONT_FILE and Path(FONT_FILE).exists():
    fm.fontManager.addfont(FONT_FILE)
CJK_CANDIDATES = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Source Han Sans CN",
                  "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei",
                  "AR PL UMing CN", "AR PL UKai CN", "Noto Sans CJK TC", "Noto Sans CJK JP"]
_installed = {f.name for f in fm.fontManager.ttflist}
if FONT_FILE and Path(FONT_FILE).exists():
    _installed.add(fm.FontProperties(fname=FONT_FILE).get_name())
    CJK_CANDIDATES.insert(0, fm.FontProperties(fname=FONT_FILE).get_name())
CJK = next((f for f in CJK_CANDIDATES if f in _installed), None)
if LANG == "auto":
    LANG = "zh" if CJK else "en"
if LANG == "zh" and not CJK:
    print("⚠️ 未找到中文字体，改用英文标注（安装方法见脚本末尾说明）")
    LANG = "en"
print(f"字体: {CJK if LANG == 'zh' else 'DejaVu Sans'}  语言: {LANG}")

mpl.rcParams.update({"font.family": ([CJK] if LANG == "zh" else []) + ["Arial", "DejaVu Sans"],   # [1108] 字体回退：Arial 缺 ₂ 时用 DejaVu Sans
                     "font.sans-serif": ([CJK] if LANG == "zh" else []) + ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],   # [1106] 与其他图统一
                     "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.right": False, "axes.spines.top": False})

T = {
 "zh": dict(PC="蛋白质含量 PC (%)", HRR="整精米率 HRR (%)", count="训练记录数",
            pa="(a) 训练数据的 CO₂ 分布与三个时段的 CO₂ 范围",
            b1="370–450 ppm\n{n} 条\n(对照组为主)", b2="450–575 ppm\n仅 {n} 条\n模型几乎没有数据", b3="575–600 ppm\n{n} 条\n(FACE 高 CO₂ 组)",
            floor="训练下限\n{v:.0f} ppm", period="{lab} (均值 {mu:.0f} ppm)",
            P=[f"基准 {HIS_LABEL}", f"SSP1-2.6 {FUT_LABEL}", f"SSP5-8.5 {FUT_LABEL}"],
            bins="所有记录按 CO₂ 分箱的均值 (试验间比较)", model="树模型预测 ({lab}典型气候)",
            pair="同一试验: 对照 → 高 CO₂ (真实 CO₂ 效应)", resp="试验内平均响应: {v:+.1f}% / 100 ppm ({n} 组)",
            dip="低谷\n(375–450 ppm)", plat="平台\n(450–575 ppm)",
            title="({p}) {t}: 同一试验内单调下降; 跨试验比较才出现低谷与平台", clim=["单季/早稻", "晚稻"]),
 "en": dict(PC="Protein content, PC (%)", HRR="Head rice rate, HRR (%)", count="Training records",
            pa="(a) CO₂ in the training data and in the three periods",
            b1="370–450 ppm\n{n} records\n(mostly ambient)", b2="450–575 ppm\nonly {n} records\n(data gap)", b3="575–600 ppm\n{n} records\n(FACE elevated)",
            floor="training\nminimum\n{v:.0f} ppm", period="{lab} (mean {mu:.0f} ppm)",
            P=[f"Baseline {HIS_LABEL}", f"SSP1-2.6 {FUT_LABEL}", f"SSP5-8.5 {FUT_LABEL}"],
            bins="All records, mean by CO₂ bin (between experiments)", model="Tree-model prediction ({lab} climate)",
            pair="Same experiment: ambient → elevated CO₂ (true CO₂ effect)", resp="Within-experiment response: {v:+.1f}% per 100 ppm ({n} experiments)",
            dip="dip\n(375–450 ppm)", plat="plateau\n(450–575 ppm)",
            title="({p}) {t}: monotonic within experiments; dip and plateau appear only between experiments",
            clim=["single/early rice", "late rice"]),
}[LANG]
LABEL = {"PC": T["PC"], "HRR": T["HRR"]}

# 三个时段的 CO2（基准、SSP1-2.6 为近似范围；SSP5-8.5 为 ISIMIP3b 逐年序列）
# 三个时段逐年 CO2（ISIMIP3b 文件）→ 阴影 = 年际范围，虚线 = 时段均值
_C = {"his": np.array(co2_series("historical", HIS_YEARS)),
      "126": np.array(co2_series("ssp126", FUT_YEARS)),
      "585": np.array(co2_series("ssp585", FUT_YEARS))}
PERIODS = [(T["P"][0], _C["his"].min(), _C["his"].max(), _C["his"].mean(), "#4D4D4D"),
           (T["P"][1], _C["126"].min(), _C["126"].max(), _C["126"].mean(), "#4DAF4A"),
           (T["P"][2], _C["585"].min(), _C["585"].max(), _C["585"].mean(), "#D73027")]
# [1013] 偏依赖曲线的"典型气候" = 06 号 climate_change_summary.csv 的基准均值（统一格网域）；
#        不再手写数值（手写值来自旧版本，与正文基准不一致）。06 未运行时退回旧值并提示。
def _typical_climate():
    fb = {T["clim"][0]: dict(Tavg=26.0, Tdif=7.9, CTSR=650, TSR=16.7, Rain=196, RHavg=78.5),
          T["clim"][1]: dict(Tavg=22.6, Tdif=8.3, CTSR=640, TSR=14.4, Rain=106, RHavg=74.7)}
    p = FIGURES / f"climate_change_{TAG}" / "climate_change_summary.csv"
    if not p.exists():
        print(f"⚠️ 未找到 {p}，典型气候暂用旧手写值（请先运行 06）"); return fb
    d = pd.read_csv(p)
    out = {}
    for ri, lab in zip(["ri1", "ri2"], T["clim"]):
        sub = d[d.ri == ri].drop_duplicates("var").set_index("var")["baseline"]
        out[lab] = {v: float(sub[v]) for v in ["Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]}
    print("典型气候（06 基准均值）:", {k: {v: round(x, 1) for v, x in c.items()} for k, c in out.items()})
    return out
CLIM = _typical_climate()
CCOL = {T["clim"][0]: "#C0392B", T["clim"][1]: "#2471A3"}

df = pd.read_excel(FACE_XLSX)

# ---- 试验内配对：TYPE + CTSR + Rain 相同视为同一试验（同地点同年同灌浆窗口）----
g = df[["TYPE", "CTSR", "Rain"]].round(2).astype(str).agg("|".join, axis=1)
span = df.groupby(g).CO2.agg(lambda x: x.max() - x.min())
pair_groups = span[span >= 100].index


def shade_periods(ax, ymax=None, text=True):
    for lab, lo, hi, mu, c in PERIODS:
        ax.axvspan(lo, hi, color=c, alpha=0.10, lw=0)
        ax.axvline(mu, color=c, lw=1.6, ls="--")
        if text:
            ax.plot([], [], color=c, lw=6, alpha=0.35, label=T["period"].format(lab=lab, mu=mu))


fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                         gridspec_kw=dict(height_ratios=[0.7, 1, 1], hspace=0.18))
x_co2 = np.arange(330, 731, 5)

# ---------------- (a) 数据密度 ----------------
ax = axes[0]
ax.hist(df.CO2, bins=np.arange(330, 735, 10), color="#7F8C8D", edgecolor="white")
ax.set_ylabel(T["count"])
shade_periods(ax)
cnt = lambda lo, hi: int(((df.CO2 >= lo) & (df.CO2 < hi)).sum())
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
ymx = ax.get_ylim()[1]
ax.text(408, ymx * 0.55, T["b1"].format(n=cnt(370, 450)), ha="center", fontsize=8.5, bbox=BOX)
ax.text(512, ymx * 0.55, T["b2"].format(n=cnt(450, 575)), ha="center", fontsize=8.5, color="#922B21", bbox=BOX)
ax.text(588, ymx * 0.55, T["b3"].format(n=cnt(575, 600)), ha="center", fontsize=8.5, bbox=BOX)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.axvline(df.CO2.min(), color="k", lw=0.8, ls=":")
ax.text(df.CO2.min() + 2, ymx * 0.95, T["floor"].format(v=df.CO2.min()), ha="left", va="top", fontsize=7.5)
ax.set_title("(a)", loc="left", fontsize=11, fontweight="bold", pad=8)   # [1105] 删除图内描述性标题，只留面板字母

# ---------------- (b)(c) 指标 ----------------
for k, t in enumerate(TRAITS):
    ax = axes[k + 1]
    shade_periods(ax, text=False)
    # 试验内配对（灰线）
    slopes = []
    for gg in pair_groups:
        s = df[g == gg][["CO2", t]].dropna()
        lo, hi = s[s.CO2 == s.CO2.min()], s[s.CO2 == s.CO2.max()]
        if len(lo) and len(hi):
            ax.plot([lo.CO2.iloc[0], hi.CO2.iloc[0]], [lo[t].mean(), hi[t].mean()],
                    color="0.55", lw=0.9, alpha=0.7, zorder=2)
            ax.scatter([lo.CO2.iloc[0], hi.CO2.iloc[0]], [lo[t].mean(), hi[t].mean()], s=8, color="0.45", zorder=2)
            slopes.append(np.log(hi[t].mean() / lo[t].mean()) / (hi.CO2.iloc[0] - lo.CO2.iloc[0]) * 100)
    # 分箱均值（黑点）
    bins = np.r_[330, np.arange(380, 461, 20), 500, 575, 600, 660, 730]
    b = pd.cut(df.CO2, bins)
    bm = df.groupby(b)[t].agg(["mean", "count"])
    mids = [iv.mid for iv in bm.index]
    ax.plot(mids, bm["mean"], "o-", color="black", ms=6, lw=1.4, zorder=4, label=T["bins"])
    for xm, (mu, n) in zip(mids, bm.values):
        if n > 0:
            ax.text(xm, mu, f" n={int(n)}", fontsize=7, va="bottom", color="0.25")
    # 模型偏依赖
    fp = MODEL_DIR / f"{t}_Blending.joblib"
    if fp.exists():
        m = joblib.load(fp)
        F = list(getattr(m, "feature_names_in_", ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]))
        for lab, base in CLIM.items():
            X = pd.DataFrame([{**base, "CO2": c, "TYPE": 1} for c in x_co2])
            ax.plot(x_co2, m.predict(X[F]), color=CCOL[lab], lw=2.2, zorder=5, label=T["model"].format(lab=lab))
    ax.plot([], [], color="0.55", lw=0.9, label=T["pair"])
    sl = np.mean(slopes) if slopes else np.nan
    ax.text(0.99, 0.03, T["resp"].format(v=100 * (np.exp(sl) - 1), n=len(slopes)),
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    # 低谷 / 平台标注（标在分箱均值曲线上）
    vv = bm["mean"].values; mm = np.array(mids)
    dip = (mm > 395) & (mm < 455); pla = (mm > 455) & (mm < 580)
    if dip.any():
        i = np.nanargmin(np.where(dip, vv, np.nan))
        ax.annotate(T["dip"], xy=(mm[i], vv[i]), xytext=(mm[i] - 10, vv[i] - 0.18 * np.ptp(ax.get_ylim())),
                    ha="center", fontsize=9, color="#1F618D", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#1F618D"))
    if pla.any():
        i = np.nanargmax(np.where(pla, vv, np.nan))
        ax.annotate(T["plat"], xy=(mm[i], vv[i]), xytext=(mm[i], vv[i] + 0.16 * np.ptp(ax.get_ylim())),
                    ha="center", fontsize=9, color="#922B21", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#922B21"))
    ax.set_ylabel(LABEL[t])
    if k == len(TRAITS) - 1:
        h, l = [], []
        for a_ in axes[1:]:
            hh, ll = a_.get_legend_handles_labels()
            for x_, y_ in zip(hh, ll):
                if y_ not in l:
                    h.append(x_); l.append(y_)
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title(f"({'bc'[k]}) {t}", loc="left",            # [1105] 删除描述性长标题，只留面板字母 + 性状
                 fontsize=11, fontweight="bold", pad=6)
axes[-1].set_xlabel("CO₂ (ppm)")
axes[-1].set_xlim(330, 730)
fig.subplots_adjust(bottom=0.09)
fig.savefig(OUT / "co2_response_diagnosis.png", dpi=250, bbox_inches="tight")
fig.savefig(OUT / "co2_response_diagnosis.pdf", bbox_inches="tight")
print(f"[saved] {OUT / 'co2_response_diagnosis.png'}")

# ------------------------------------------------------------
# 服务器上中文显示为方框时：
#   1) 查看是否有中文字体：  fc-list :lang=zh family
#   2) 没有则安装：         sudo apt install fonts-noto-cjk   （或把 simhei.ttf 放到任意目录，填到 FONT_FILE）
#   3) 清除 matplotlib 缓存：rm -rf ~/.cache/matplotlib
# 不想处理字体时，设 LANG = "en" 直接输出英文版（投稿补充材料推荐英文）
# ------------------------------------------------------------
