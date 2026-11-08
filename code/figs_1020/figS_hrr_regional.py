# ============================================================
# figS_hrr_regional.py —— 补充材料图 S13：单季/早稻 HRR 两情景差值的省级来源（10-28 版式）
#   数据：check_hrr_regional.py 输出的 check_hrr_regional.csv（同目录）
#     用到的列：province, area_share_pct, base, ssp126_total_pp, ssp585_total_pp, ssp126_co2_pp, ssp585_co2_pp,
#              gap_total_pp, gap_climate_pp, gap_co2_pp, Tavg_ssp126_future, Tavg_ssp585_future
#   a 各省对全国差值（SSP1-2.6 − SSP5-8.5）的贡献，单位为 HRR 百分点 = 面积份额 × 两情景差值；
#     拆成气候（CO₂ 固定）与 CO₂ 两部分，黑点 = 合计，各省之和 = 全国差值（左下角注明全国值）
#   b 各省 CO₂ 对 HRR 的效应（随情景 CO₂ − 固定 CO₂，HRR 百分点）：蓝 = SSP1-2.6，橙 = SSP5-8.5，两点连线
#     a、b 共用省份行：按 SSP5-8.5 未来灌浆期 Tavg 从高到低排列，"其他省份"放最后并用虚线隔开；
#     SSP5-8.5 灌浆期 Tavg ≥ 30 °C（--hot）的省份整行加灰底，不进图例
#   c 各省 HRR 相对变化（%）对未来灌浆期 Tavg：蓝 / 橙同上，同省两点灰线相连，点面积 ∝ 单季/早稻面积；
#     "其他省份"不画；Tavg ≥ 30 °C（--hot）的范围加灰底，不进图例；省名标在 SSP5-8.5 的点旁（不画训练数据的 90/95 百分位）
#   用法：python figs_1020/figS_hrr_regional.py [--csv ...] [--hot 30] [--label text|all|N] [--two] [--out ...]
#     --two：只画 a 与散点图（不画 b，两栏版式）
# ============================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
import fig_common as F                                            # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

F.setup()
F.FONT_SCALE = 1.2                             # [1105] 补充图文字放大 20%（最小 6 pt），与正文图可读性一致
p = argparse.ArgumentParser()
p.add_argument("--csv", default=str(HERE / "check_hrr_regional.csv"))
p.add_argument("--trait", default="HRR")
p.add_argument("--hot", type=float, default=30.0, help="a、b 整行加灰底的阈值：SSP5-8.5 未来灌浆期 Tavg ≥ 此值（°C）")
p.add_argument("--label", default="text", help="散点图标注哪些省：text = 正文提到的省（默认），all，或前 N 个（按贡献绝对值）")
p.add_argument("--two", action="store_true", help="两栏版式：不画各省 CO₂ 效应（b）")
p.add_argument("--out", default=None)
a = p.parse_args()
OUT = Path(a.out) if a.out else F.default_out()
T = a.trait
S1, S5 = "ssp126", "ssp585"
COL = F.SSP_COL
C_CLIM, C_CO2 = "#A9BCC8", "#4E6A7C"
ROW_HOT = "0.88"                                                  # SSP5-8.5 Tavg ≥ --hot 的省份行
TEXT_PROV = ["Hunan", "Anhui", "Hubei", "Jiangxi", "Guangxi", "Guangdong", "Henan", "Hainan",
             "Heilongjiang", "Jilin", "Guizhou", "Yunnan"]         # 正文 2.2 提到的省

t = pd.read_csv(a.csv)
need = ["province", "area_share_pct", "base", f"{S1}_total_pp", f"{S5}_total_pp", f"{S1}_co2_pp", f"{S5}_co2_pp",
        "gap_total_pp", "gap_climate_pp", "gap_co2_pp", f"Tavg_{S1}_future", f"Tavg_{S5}_future"]
miss = [c for c in need if c not in t.columns]
if miss:
    sys.exit(f"[错误] {a.csv} 缺少列：{miss}\n现有列：{list(t.columns)}")
t["w"] = t["area_share_pct"] / t["area_share_pct"].sum()
BASE_NAT = float((t["w"] * t["base"]).sum())
t["rel1"] = t[f"{S1}_total_pp"] / t["base"] * 100                 # HRR 相对变化（%）
t["rel5"] = t[f"{S5}_total_pp"] / t["base"] * 100
for comp in ("total", "climate", "co2"):                          # HRR 百分点；各省之和 = 全国差值
    t[f"c_{comp}"] = t["w"] * t[f"gap_{comp}_pp"]
t["name"] = t["province"].str.replace(r"Other provinces.*", "Other provinces", regex=True)
other = t["name"] == "Other provinces"
t = pd.concat([t[~other].sort_values(f"Tavg_{S5}_future", ascending=False), t[other]], ignore_index=True)
other = (t["name"] == "Other provinces").values
n = len(t)
y = np.arange(n)
hot = (t[f"Tavg_{S5}_future"].values >= a.hot) & ~other

gap = t[["c_total", "c_climate", "c_co2"]].sum()
nat = {s: float((t["w"] * t[f"{s}_total_pp"]).sum() / BASE_NAT * 100) for s in (S1, S5)}
print(f"全国 {T}：SSP1-2.6 {nat[S1]:+.1f}%，SSP5-8.5 {nat[S5]:+.1f}%；差值 {gap['c_total']:+.2f} pp"
      f"（气候 {gap['c_climate']:+.2f}，CO₂ {gap['c_co2']:+.2f}；CO₂ 占 {gap['c_co2'] / gap['c_total'] * 100:.0f}%）")
print(f"灰底省份（SSP5-8.5 Tavg ≥ {a.hot:g} °C）：{', '.join(t['name'][hot])}")

# ---------------------------------------------------------------- 版式（毫米）
W = 180.0
ROW = 3.0
TOP, BOT = 8.0, 17.0
H = TOP + ROW * n + BOT
NAME_W = 19.0
if a.two:
    AW, BW, G1, G2 = 62.0, 0.0, 0.0, 14.0
else:
    AW, BW, G1, G2 = 40.0, 34.0, 5.0, 13.0
CW = W - (NAME_W + 1.0) - AW - G1 - BW - G2 - 2.5
fig = plt.figure(figsize=(W * F.MM, H * F.MM))
x0 = NAME_W + 1.0
axa = fig.add_axes([x0 / W, BOT / H, AW / W, ROW * n / H])
axb = None
if not a.two:
    axb = fig.add_axes([(x0 + AW + G1) / W, BOT / H, BW / W, ROW * n / H])
axc = fig.add_axes([(x0 + AW + G1 + BW + G2) / W, BOT / H, CW / W, ROW * n / H])


def rows(ax):
    """a、b 共用的行底纹与框线。"""
    ax.set_ylim(n - 0.5, -0.5)
    for yy in y:
        col = ROW_HOT if hot[yy] else ("0.965" if yy % 2 == 0 else "white")
        ax.axhspan(yy - 0.5, yy + 0.5, color=col, lw=0, zorder=0)
    ax.axvline(0, color=F.INK, lw=0.5, zorder=1)
    if other.any():
        ax.axhline(n - 1.5, color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=1)
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.8, length=1.5, pad=1)


# ---------------------------------------------------------------- a：对全国差值的贡献
ax = axa
rows(ax)
ax.barh(y - 0.17, t["c_climate"], 0.32, color=C_CLIM, zorder=3)
ax.barh(y + 0.17, t["c_co2"], 0.32, color=C_CO2, zorder=3)
ax.scatter(t["c_total"], y, s=8, color=F.INK, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(t["name"], fontsize=5.8)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_xlabel(f"Contribution (pp of {T})", fontsize=6.2, labelpad=1.5)
ax.set_title("SSP1-2.6 − SSP5-8.5", fontsize=6.8, fontweight="bold", pad=3, loc="left")
ax.text(0.02, 0.015, f"National {gap['c_total']:+.2f} pp\nclimate {gap['c_climate']:+.2f}, CO₂ {gap['c_co2']:+.2f}".replace("-", "\u2212"),   # [1108] 负号用 U+2212
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, color=F.INK, linespacing=1.15, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# ---------------------------------------------------------------- b：各省 CO₂ 效应
if axb is not None:
    ax = axb
    rows(ax)
    v1, v5 = t[f"{S1}_co2_pp"].values, t[f"{S5}_co2_pp"].values
    for yy, (p1, p5) in enumerate(zip(v1, v5)):
        ax.plot([p1, p5], [yy, yy], color="0.65", lw=0.7, zorder=2)
    ax.scatter(v1, y, s=11, color=COL[S1], edgecolor="white", lw=0.3, zorder=3)
    ax.scatter(v5, y, s=11, color=COL[S5], edgecolor="white", lw=0.3, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel(f"CO₂ effect (pp of {T})", fontsize=6.2, labelpad=1.5)
    ax.set_title(f"CO₂ effect on {T}", fontsize=6.8, fontweight="bold", pad=3, loc="left")

# ---------------------------------------------------------------- c：各省 HRR 变化对灌浆期 Tavg
ax = axc
bt = t[~other].reset_index(drop=True)
x1, x5 = bt[f"Tavg_{S1}_future"].values, bt[f"Tavg_{S5}_future"].values
y1, y5 = bt["rel1"].values, bt["rel5"].values
size = 6 + 90 * bt["w"].values / bt["w"].max()                    # 点面积（pt²）∝ 面积份额
for xa, ya, xb, yb in zip(x1, y1, x5, y5):
    ax.plot([xa, xb], [ya, yb], color="0.72", lw=0.6, zorder=2)
ax.scatter(x1, y1, s=size, color=COL[S1], edgecolor="white", lw=0.3, zorder=3)
ax.scatter(x5, y5, s=size, color=COL[S5], edgecolor="white", lw=0.3, zorder=3)
ax.axhline(0, color=F.INK, lw=0.5, zorder=1)
xmin = min(x1.min(), x5.min()) - 0.6
xmax = max(x1.max(), x5.max()) + 0.6
if xmax > a.hot:                                                  # Tavg ≥ --hot 的范围加灰底，与 a、b 的灰底行对应
    ax.axvspan(max(a.hot, xmin), xmax, color="0.90", lw=0, zorder=0)
ax.set_xlim(xmin, xmax)
lo, hi = min(y1.min(), y5.min()), max(y1.max(), y5.max())
pad = 0.08 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
ax.set_xlabel("Future grain-filling Tavg (°C)", fontsize=6.2, labelpad=1.5)
ax.set_ylabel(f"Change in {T} (%)", fontsize=6.2, labelpad=1)
ax.tick_params(axis="both", labelsize=5.8, length=1.5, pad=1)
for s_ in ("right", "top"):
    ax.spines[s_].set_visible(False)
ax.set_title(f"Provincial change in {T}", fontsize=6.8, fontweight="bold", pad=3, loc="left")


def place_labels(ax, xs, ys, names, which, obstacles, fs=5.2):
    """贪心放置省名：候选位置由近到远、先右后左再上下；不与已放标签、任何点（含自身）或坐标轴外重叠；
    放不下时退到最近的候选并加引线。obstacles = [(x, y, 面积 pt²)]，数据坐标。"""
    fig_ = ax.figure
    fig_.canvas.draw()
    r = fig_.canvas.get_renderer()
    tr = ax.transData
    pt = fig_.dpi / 72.0
    axbb = ax.get_window_extent(r)
    obs = []
    for ox, oy, s in obstacles:
        xd, yd = tr.transform((ox, oy))
        rad = (np.sqrt(s) / 2 + 0.6) * pt
        obs.append((xd - rad, yd - rad, xd + rad, yd + rad))

    def hit(b1, b2):
        return b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]
    placed = []
    for i in which:
        xd, yd = tr.transform((xs[i], ys[i]))
        tx = ax.text(0, 0, names[i], fontsize=fs)
        bb = tx.get_window_extent(r)
        w, h = bb.width, bb.height
        tx.remove()
        own = (np.sqrt(size[i]) / 2 + 1.5)
        cands = []
        for d in (own, own + 3, own + 6, own + 10, own + 15):
            cands += [(d, 0, "left", "center"), (-d, 0, "right", "center"),
                      (0.7 * d, 0.8 * d, "left", "bottom"), (0.7 * d, -0.8 * d, "left", "top"),
                      (-0.7 * d, 0.8 * d, "right", "bottom"), (-0.7 * d, -0.8 * d, "right", "top"),
                      (0, d, "center", "bottom"), (0, -d, "center", "top")]
        choice = None
        for dx, dy, ha, va in cands:
            bx = xd + dx * pt - (w if ha == "right" else (w / 2 if ha == "center" else 0))
            by = yd + dy * pt - (h if va == "top" else (h / 2 if va == "center" else 0))
            box = (bx, by, bx + w, by + h)
            inside = box[0] >= axbb.x0 and box[2] <= axbb.x1 and box[1] >= axbb.y0 and box[3] <= axbb.y1
            if inside and not any(hit(box, o) for o in obs) and not any(hit(box, q) for q in placed):
                choice = (dx, dy, ha, va, box)
                break
        if choice is None:                                        # 放不下：取最近的候选，加引线
            dx, dy, ha, va = cands[0]
            bx = xd + dx * pt
            choice = (dx, dy, ha, va, (bx, yd - h / 2, bx + w, yd + h / 2))
        dx, dy, ha, va, box = choice
        placed.append(box)
        far = np.hypot(dx, dy) > own + 2                        # 离开第一圈即加引线
        ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fs,
                    color=F.INK, ha=ha, va=va, zorder=5,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, shrinkA=0, shrinkB=1.5) if far else None)


names = list(bt["name"])
if a.label == "text":
    which = [i for i, nm in enumerate(names) if nm in TEXT_PROV]
elif a.label == "all":
    which = list(range(len(names)))
else:
    which = list(np.argsort(-np.abs(bt["c_total"].values))[:int(a.label)])
which = sorted(which, key=lambda i: -abs(bt["c_total"].values[i]))  # 贡献大的先放
obst = [(xx, yy, ss) for xx, yy, ss in zip(np.r_[x1, x5], np.r_[y1, y5], np.r_[size, size])]
place_labels(ax, x5, y5, names, which, obst)

# ---------------------------------------------------------------- 面板字母与图例
letters = ["a", "b", "c"] if axb is not None else ["a", "b"]
axa.text(-(NAME_W + 1.0) / AW, 1.0, letters[0], transform=axa.transAxes, fontsize=8, fontweight="bold", ha="left",
         va="bottom")
if axb is not None:
    axb.text(-3.5 / BW, 1.0, letters[1], transform=axb.transAxes, fontsize=8, fontweight="bold", ha="left",
             va="bottom")
axc.text(-9.0 / CW, 1.0, letters[-1], transform=axc.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
hdl = [Line2D([], [], ls="none", marker="o", ms=4, color=COL[S1], label="SSP1-2.6"),
       Line2D([], [], ls="none", marker="o", ms=4, color=COL[S5], label="SSP5-8.5"),
       Patch(color=C_CLIM, label="Climate (CO₂ fixed)"), Patch(color=C_CO2, label="CO₂"),
       Line2D([], [], ls="none", marker="o", ms=3, color=F.INK, label="Total")]
fig.legend(handles=hdl, loc="lower center", ncol=len(hdl), fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.3, columnspacing=0.9)
F.save(fig, OUT, f"FigS_{T.lower()}_regional", tight=False)
