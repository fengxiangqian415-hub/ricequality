# ============================================================
# figS_hrr_response.py —— 补充材料图：HRR 对 Tavg、TSR 的累积局部效应（ALE）
#   为什么用 ALE 而不是 PDP：Tavg 与 TSR、CTSR 相关，PDP 会把一个变量推到与其他变量不匹配的组合上（外推）；
#     ALE 只在每个格网实际取值的附近做差分，结果不受相关变量干扰（Apley & Zhu 2020, J. R. Stat. Soc. B 82:1059）
#   输入：18 号导出的格网特征（基准 + 两个情景合在一起，统一格网域 D），MODEL_DIR 的 HRR 模型，训练数据（只取范围）
#   每个面板：一个变量 × 一个稻季；曲线 = ALE（HRR，百分点，均值归零）；
#     下方箱线 = 基准 / SSP1-2.6 / SSP5-8.5 格网取值分布（箱 = 四分位，须 = 5–95 百分位）；灰底 = 训练范围以外
#   另输出各情景相对基准、由该变量单独带来的 HRR 变化（逐格网 ALE 差值的平均，百分点及占基准的 %）
#   用法：cd code_1004b && python figs_1020/figS_hrr_response.py [--trait HRR] [--vars Tavg,TSR] [--out ...]
# ============================================================
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
import fig_common as F                                            # noqa: E402
import config_1004 as C                                           # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

F.setup()
F.FONT_SCALE = 1.2                             # [1105] 补充图文字放大 20%（最小 6 pt），与正文图可读性一致
p = argparse.ArgumentParser()
p.add_argument("--trait", default="HRR")
p.add_argument("--vars", default="Tavg,TSR")
p.add_argument("--face", default=None, help="训练数据 Excel（默认 config.FACE_XLSX；只用于标出训练范围）")
p.add_argument("--bins", type=int, default=25, help="ALE 分箱数（按分位数）")
p.add_argument("--out", default=None)
a = p.parse_args()
OUT = Path(a.out) if a.out else F.default_out()
T = a.trait
VARS = [v.strip() for v in a.vars.split(",") if v.strip()]
VLAB = {"Tavg": "Tavg (°C)", "TSR": r"TSR (MJ m$^{-2}$ d$^{-1}$)", "CTSR": r"CTSR (MJ m$^{-2}$)",
        "Tdif": "Tdif (°C)", "RHavg": "RHavg (%)", "Rain": "Rain (mm)"}
PERS = ["historical"] + list(C.SSPS)
PCOL = {"historical": F.SSP_COL["baseline"], **{s: F.SSP_COL[s] for s in C.SSPS}}
PLAB = {"historical": "Baseline", **F.SSP_LAB}
DEFAULT_FN = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]

FEAT = C.out_dir("shap_features", thresh=False)
X = {(per, ri): pd.read_csv(FEAT / f"features_{per}_{ri}.csv") for per in PERS for ri in C.RI_TYPES}
face = pd.read_excel(a.face or C.FACE_XLSX)
m = joblib.load(C.MODEL_DIR / f"{T}_Blending.joblib")
FN = list(getattr(m, "feature_names_in_", DEFAULT_FN))


def predict(df):
    return np.asarray(m.predict(df[FN]), float)


def ale_1d(Xp, var, nbins):
    """一阶 ALE：分位数分箱；每个点在所在箱的上下界之间做差分，箱内平均后累加，再按样本均值归零"""
    x = Xp[var].values
    z = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    k = np.clip(np.searchsorted(z, x, side="left"), 1, len(z) - 1)
    lo, hi = Xp.copy(), Xp.copy()
    lo[var], hi[var] = z[k - 1], z[k]
    d = predict(hi) - predict(lo)
    K = len(z) - 1
    cnt = np.bincount(k - 1, minlength=K)
    eff = np.bincount(k - 1, weights=d, minlength=K) / np.maximum(cnt, 1)
    acc = np.concatenate([[0.0], np.cumsum(eff)])
    mid = (acc[:-1] + acc[1:]) / 2
    return z, acc - np.sum(mid * cnt) / cnt.sum()


ALE, ROWS = {}, []
for ri in C.RI_TYPES:
    Xp = pd.concat([X[(per, ri)] for per in PERS], ignore_index=True)       # 基准 + 两个情景的全部格网输入
    base_mean = predict(X[("historical", ri)]).mean()
    for var in VARS:
        z, acc = ale_1d(Xp, var, a.bins)
        ALE[(ri, var)] = (z, acc)
        a0 = np.interp(X[("historical", ri)][var].values, z, acc).mean()
        for s in C.SSPS:
            a1 = np.interp(X[(s, ri)][var].values, z, acc).mean()
            ROWS.append(dict(trait=T, ri=ri, ssp=s, var=var,
                             mean_base=round(float(X[("historical", ri)][var].mean()), 2),
                             mean_future=round(float(X[(s, ri)][var].mean()), 2),
                             effect_pp=round(a1 - a0, 2), effect_pct_of_base=round((a1 - a0) / base_mean * 100, 2)))
tab = pd.DataFrame(ROWS)
OUT.mkdir(parents=True, exist_ok=True)
tab.to_csv(OUT / f"FigS_{T.lower()}_response_ALE.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(f"[ALE] 各情景相对基准、由单个变量带来的 {T} 变化（逐格网 ALE 差值的平均）：")
print(tab.pivot_table(index=["ri", "var"], columns="ssp", values=["effect_pp", "effect_pct_of_base"]).round(2)
      .to_string())

# ---------------------------------------------------------------- 图
W = 180.0
H = 62.0
fig = plt.figure(figsize=(W * F.MM, H * F.MM))
panels = [(var, ri) for var in VARS for ri in C.RI_TYPES]
n = len(panels)
L, R, GAP = 14.0, 3.0, 9.0
PW = (W - L - R - GAP * (n - 1)) / n
TOP, CH, BH, BG = 7.0, 30.0, 9.0, 1.2                            # 曲线高、箱线区高、两者间距（mm）
letters = iter("abcdefghijkl")
for i, (var, ri) in enumerate(panels):
    x0 = L + i * (PW + GAP)
    ax = fig.add_axes([x0 / W, 1 - (TOP + CH) / H, PW / W, CH / H])
    bx = fig.add_axes([x0 / W, 1 - (TOP + CH + BG + BH) / H, PW / W, BH / H], sharex=ax)
    z, acc = ALE[(ri, var)]
    lo_, hi_ = face[var].min(), face[var].max()
    xlo, xhi = z.min() - 0.03 * np.ptp(z), z.max() + 0.03 * np.ptp(z)
    for a_ in (ax, bx):
        a_.axvspan(xlo, lo_, color="0.90", lw=0, zorder=0)
        a_.axvspan(hi_, xhi, color="0.90", lw=0, zorder=0)
    ax.plot(z, acc, color=F.INK, lw=1.0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (3, 2)), zorder=1)
    for per in PERS:
        xv = float(X[(per, ri)][var].mean())
        ax.plot(xv, np.interp(xv, z, acc), "o", ms=3.4, mfc=PCOL[per], mec="white", mew=0.5, zorder=4)
    for j, per in enumerate(PERS):                               # 下方：各时段格网取值分布
        v = X[(per, ri)][var].dropna().values
        q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        y = len(PERS) - 1 - j
        bx.plot([q5, q95], [y, y], color=PCOL[per], lw=0.6, zorder=2)
        bx.add_patch(plt.Rectangle((q25, y - 0.32), q75 - q25, 0.64, facecolor=PCOL[per], edgecolor="none", zorder=3))
        bx.plot([q50, q50], [y - 0.32, y + 0.32], color="white", lw=0.8, zorder=4)
    bx.set_ylim(-0.7, len(PERS) - 0.3); bx.set_yticks([])
    for s_ in ("left", "right", "top"):
        bx.spines[s_].set_visible(False)
    ax.set_xlim(xlo, xhi)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=6, length=1.5, pad=1)
    bx.tick_params(axis="x", labelsize=6, length=1.5, pad=1)
    bx.set_xlabel(VLAB.get(var, var), fontsize=6.5, labelpad=1)
    if i % len(C.RI_TYPES) == 0:                                 # 每个变量的第一个面板标纵轴
        ax.set_ylabel(f"ALE of {T} (pp)", fontsize=6.5, labelpad=1)
    ax.set_title(F.RI_LAB[ri], fontsize=7, fontweight="bold", pad=3)
    ax.text(-0.14 if i % len(C.RI_TYPES) == 0 else -0.10, 1.04, next(letters), transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")
for k_, var in enumerate(VARS):                                  # 同一变量两季共用纵轴
    axs_ = [fig.axes[2 * (k_ * len(C.RI_TYPES) + j)] for j in range(len(C.RI_TYPES))]
    lo = min(a_.get_ylim()[0] for a_ in axs_); hi = max(a_.get_ylim()[1] for a_ in axs_)
    for a_ in axs_:
        a_.set_ylim(lo, hi)
hdl = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=PCOL[per], mec="white", label=PLAB[per]) for per in PERS] + \
      [plt.Rectangle((0, 0), 1, 1, color="0.90", label="Outside training range")]
fig.legend(handles=hdl, loc="lower center", ncol=len(hdl), fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.3, columnspacing=1.2)
F.save(fig, OUT, f"FigS_{T.lower()}_response", tight=False)
