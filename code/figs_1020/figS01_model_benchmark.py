# ============================================================
# figS01_model_benchmark.py —— 补图 1：21 种算法的五折 CV 基准（a）+ 加权集成的观测–预测散点（b）
#   [1108f] 新增：原补图 1 由训练流程另行出图（衬线字体，不在 code_1004b）；此脚本从训练输出表直接重绘，
#           与其余图同一版式（Arial、180 mm 宽），保证补图 1、补表 1–2 与 paper_data 工作表同源。
#   输入（--tables，默认 config_1004.MODEL_DIR.parent / "Tables"，即 /home/debian/pythonfile/results/Tables）：
#     r2_matrix.csv / rmse_matrix.csv / mae_matrix.csv   行 = 算法，列 = 性状（BRR … PC；五折 CV 均值）
#     Blending_predictions.csv                           列 Target, Set(Train/Test), Observed, Predicted
#   输出（--out，默认 output/figures/model_skill_1108/）：FigS01_model_benchmark.{svg,pdf,png}
#   用法：python figs_1020/figS01_model_benchmark.py [--tables 目录] [--out 目录]
#   版式与原图一致：a 行按七性状平均 R² 降序；R² 色标 −0.2（蓝）→ 1.0（红），低于 −0.2 取端点色；
#         RMSE、MAE 在每个性状内最小–最大归一（最好 = 红，最差 = 蓝）；星号为 R² 档（*** ≥ 0.85，** ≥ 0.60，* ≥ 0.30）
# ============================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fig_common as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

TRAITS = F.TRAITS
TLAB = F.TLAB
COL_TRAIN, COL_TEST, EDGE_TEST = "#5B8FC9", "#E8833A", "#6B3A10"


def default_tables():
    try:
        import config_1004 as C
        return C.MODEL_DIR.parent / "Tables"
    except Exception:
        return Path("/home/debian/pythonfile/results/Tables")


def default_out():
    try:
        import config_1004 as C
        return C.FIGURES / "model_skill_1108"
    except Exception:
        return HERE / "model_skill_1108"


def load(tables):
    tables = Path(tables)
    mats = {k: pd.read_csv(tables / f"{k.lower()}_matrix.csv", index_col=0) for k in ("R2", "RMSE", "MAE")}
    for k, m in mats.items():
        miss = [t for t in TRAITS if t not in m.columns]
        if miss:
            raise KeyError(f"{k.lower()}_matrix.csv 缺性状列 {miss}；现有列 {list(m.columns)}")
        mats[k] = m[TRAITS].astype(float)
    pred = pd.read_csv(tables / "Blending_predictions.csv")
    return mats, pred


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), R2=1 - np.sum((y - p) ** 2) / ss, RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                MAE=float(np.mean(np.abs(y - p))))


def stars(r2):
    return "***" if r2 >= 0.85 else "**" if r2 >= 0.60 else "*" if r2 >= 0.30 else ""


def plot(mats, pred, out, exts=("svg", "pdf", "png")):
    F.setup()
    order = mats["R2"].mean(axis=1).sort_values(ascending=False).index.tolist()
    R2, RMSE, MAE = (mats[k].loc[order] for k in ("R2", "RMSE", "MAE"))
    nr, nc = len(order), len(TRAITS)
    W, H = 180.0, 193.0
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))
    ax_mm = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])   # (x, y) = 左上角，单位 mm

    # ---------------- a：三张热图
    X0, Y0, HW, CW, GAP = 19.0, 7.0, 50.0, None, 4.5
    RH = 3.75                                                   # 行高 mm
    HH = nr * RH
    cm_r2 = plt.get_cmap("RdBu_r"); nrm_r2 = Normalize(-0.2, 1.0)
    cm_err = plt.get_cmap("RdBu")                               # 0 = 最好（红） → 1 = 最差（蓝）
    blocks = [("R²", R2, lambda v, t: cm_r2(nrm_r2(np.clip(v, -0.2, 1.0)))),
              ("RMSE", RMSE, None), ("MAE", MAE, None)]
    for b, (title, M, colf) in enumerate(blocks):
        x = X0 + b * (HW + GAP)
        ax = ax_mm(x, Y0, HW, HH)
        ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
        for j, t in enumerate(TRAITS):
            col = M[t].values
            if colf is None:
                lo, hi = np.nanmin(col), np.nanmax(col)
                sc = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
            for i in range(nr):
                c = colf(col[i], t) if colf is not None else cm_err(sc[i])
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=c, ec="white", lw=0.6))
                s = stars(R2[t].values[i])
                if s:
                    ax.text(j + 0.5, i + 0.62, s, ha="center", va="center", fontsize=5.2, color="#111111")
        ax.set_xticks(np.arange(nc) + 0.5); ax.set_xticklabels([TLAB[t] for t in TRAITS])
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=1.5, labelsize=6.3)
        ax.set_yticks(np.arange(nr) + 0.5)
        if b == 0:
            ax.set_yticklabels(order, fontsize=6.3); ax.tick_params(axis="y", length=0, pad=2)
        else:
            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # 色条
        cax = ax_mm(x + 6, Y0 + HH + 6.5, HW - 12, 2.4)
        if b == 0:
            sm = ScalarMappable(norm=nrm_r2, cmap=cm_r2)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-0.2, 0.4, 1.0])
            cb.ax.set_xticklabels(["−0.2", "0.4", "1.0"])
        else:
            sm = ScalarMappable(norm=Normalize(0, 1), cmap=cm_err)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
            cb.ax.set_xticklabels(["best", "worst"])
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
        cax.set_title(title, fontsize=7, pad=2, fontweight="bold")
    fig.text(1.0 / W, 1 - 3.0 / H, "a", fontsize=8, fontweight="bold", va="top")

    # ---------------- b：观测–预测散点（集成模型）
    YB = Y0 + HH + 17.0
    fig.text(1.0 / W, 1 - YB / H, "b", fontsize=8, fontweight="bold", va="top")
    PW, PG, PX0 = 30.5, 14.2, 13.5                              # 方形面板边长、间距、左起点（第 4 列右缘 178 mm）
    PY0, PRG = YB + 6.0, 13.0
    tcol = "Target" if "Target" in pred.columns else pred.columns[0]
    for k, t in enumerate(TRAITS):
        r, c = divmod(k, 4)
        ax = ax_mm(PX0 + c * (PW + PG), PY0 + r * (PW + PRG), PW, PW)
        d = pred[pred[tcol].astype(str) == t]
        tr, te = d[d.Set.str.lower() == "train"], d[d.Set.str.lower() == "test"]
        ax.scatter(tr.Observed, tr.Predicted, s=3.0, color=COL_TRAIN, alpha=0.55, lw=0, zorder=2)
        ax.scatter(te.Observed, te.Predicted, s=9.0, color=COL_TEST, ec=EDGE_TEST, lw=0.3, alpha=0.9, zorder=3)
        allv = np.r_[d.Observed.values, d.Predicted.values]
        lo, hi = np.nanmin(allv), np.nanmax(allv); pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, ls=(0, (3, 2)), color="#222222", lw=0.7, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
        ax.set_xlabel(f"Observed {TLAB[t]}", labelpad=1.5); ax.set_ylabel(f"Predicted {TLAB[t]}", labelpad=1.5)
        ax.set_title(TLAB[t], fontsize=7, fontweight="bold", pad=3)
        mte, mtr = metrics(te.Observed, te.Predicted), metrics(tr.Observed, tr.Predicted)
        ax.text(0.04, 0.97, f"Test\nR² = {mte['R2']:.2f}\nRMSE = {mte['RMSE']:.2f}\nMAE = {mte['MAE']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=5.6, linespacing=1.15)
        ax.text(0.97, 0.03, f"Train\nR² = {mtr['R2']:.2f}\nRMSE = {mtr['RMSE']:.2f}\nMAE = {mtr['MAE']:.2f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=5.6, linespacing=1.15)
    # 图例（第 8 格）
    ax = ax_mm(PX0 + 3 * (PW + PG), PY0 + 1 * (PW + PRG), PW, PW); ax.axis("off")
    n_tr = int((pred.Set.str.lower() == "train").sum() / max(pred[tcol].nunique(), 1))
    n_te = int((pred.Set.str.lower() == "test").sum() / max(pred[tcol].nunique(), 1))
    h = [plt.Line2D([], [], marker="o", ls="", ms=3.2, color=COL_TRAIN, alpha=0.8),
         plt.Line2D([], [], marker="o", ls="", ms=4.2, color=COL_TEST, mec=EDGE_TEST, mew=0.4),
         plt.Line2D([], [], ls=(0, (3, 2)), color="#222222", lw=0.8)]
    ax.legend(h, [f"Train (n = {n_tr})", f"Test (n = {n_te})", "1:1 line"], loc="center", fontsize=6.3,
              handlelength=1.8, labelspacing=1.0)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for e in exts:
        fig.savefig(out / f"FigS01_model_benchmark.{e}", dpi=600 if e == "png" else 300)
    plt.close(fig)
    print(f"[saved] {out / 'FigS01_model_benchmark'}.{{{','.join(exts)}}}")
    return order


def main():
    ap = argparse.ArgumentParser(description="Supplementary Fig. 1 from training outputs")
    ap.add_argument("--tables", default=str(default_tables()))
    ap.add_argument("--out", default=str(default_out()))
    a = ap.parse_args()
    mats, pred = load(a.tables)
    order = plot(mats, pred, a.out)
    print("行序（平均 R² 降序）：", ", ".join(order))


if __name__ == "__main__":
    main()
