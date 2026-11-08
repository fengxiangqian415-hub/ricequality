from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import fig_common as F

F.setup()
F.FONT_SCALE = 1.15
mpl.rcParams["hatch.linewidth"] = 0.45
p = F.args("Fig. 2 main")
p.add_argument("--ssp", default="ssp585", choices=["ssp126", "ssp585"], help="a 面板地图的情景")
a = p.parse_args()
XLSX = a.xlsx or F.default_xlsx()
OUT = a.out or F.default_out()
SSP_MAP = a.ssp

COL = F.SSP_COL
GROUPS = [("ri1", "ssp126"), ("ri1", "ssp585"), ("ri2", "ssp126"), ("ri2", "ssp585")]
HATCH = "//////"

W = 180.0
ROW2_Y, ROW2_H = 6.0, 88.0
ROW1_Y, ROW1_H = ROW2_Y + ROW2_H + 10.0, 58.0
ROW3_Y, ROW3_H = ROW1_Y + ROW1_H + 10.0, 38.0
H = ROW3_Y + ROW3_H + 7.0
EX, EW = 120.0, 57.0
fig = plt.figure(figsize=(W * F.MM, H * F.MM))


def axmm(x, y, w, h, **kw):
    """以毫米给位置（y = 上沿到图顶的距离）"""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def label(x, y, s):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", ha="left", va="top")


def fmt(v, nd=1):
    return f"{v:+.{nd}f}".replace("-", "−")


def vlegend(x, ytop, h, items, fs=6.0):
    """旋转 90° 的竖排图例（自下而上读），在 (x, ytop, 4 mm, h) 的竖条内居中。
    items = [(Rectangle 的样式 dict, 文字), ...]"""
    axl = axmm(x, ytop, 4.0, h); axl.axis("off")
    axl.set_xlim(0, 4.0); axl.set_ylim(0, h)
    PW, PL, TG, IG = 1.8, 2.4, 0.8, 2.6
    txts = [axl.text(2.0, 0, lab, rotation=90, rotation_mode="anchor", ha="left", va="center", fontsize=fs,
                     multialignment="center", linespacing=1.1)
            for _, lab in items]
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    k = max(F.FONT_SCALE, F.MIN_FONT_PT / fs)
    tl = [t.get_window_extent(rr).height / fig.dpi * 25.4 * k for t in txts]
    yy = (h - (len(items) * (PL + TG) + sum(tl) + IG * (len(items) - 1))) / 2
    for (kw, _), t, L_ in zip(items, txts, tl):
        axl.add_patch(Rectangle((2.0 - PW / 2, yy), PW, PL, clip_on=False, **kw))
        t.set_position((2.0, yy + PL + TG))
        yy += PL + TG + L_ + IG
    return axl


def cf_bars(ax, rows, xlab, label_fn, xlim=None):
    """rows: [(y, 对照值 cf, 完整值 full, 颜色)]；斜线条 0→cf，实色段 cf→full（带白箭头）"""
    hh, hs = 0.66, 0.38
    span = max(max(abs(r[1]), abs(r[2])) for r in rows)
    for y, cf, full, col in rows:
        ax.barh(y, cf, height=hh, left=0, facecolor="white", edgecolor=col, hatch=HATCH, linewidth=0.5,
                zorder=2)
        ax.barh(y, full - cf, height=hs, left=cf, color=col, edgecolor=col, linewidth=0.5, zorder=3)
        eff = full - cf
        if abs(eff) > 0.12 * span:
            s = np.sign(eff); pad = 0.06 * abs(eff)
            ax.annotate("", xy=(full - s * pad, y), xytext=(cf + s * pad, y),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7, mutation_scale=4.0,
                                        shrinkA=0, shrinkB=0), zorder=4)
        lo, hi = min(0, cf, full), max(0, cf, full)
        txt = label_fn(y, cf, full)
        if txt:
            if eff >= 0:
                ax.text(hi + 0.04 * span, y, txt, ha="left", va="center", fontsize=5.8, color=F.INK, zorder=5)
            else:
                ax.text(lo - 0.04 * span, y, txt, ha="right", va="center", fontsize=5.8, color=F.INK, zorder=5)
    ax.axvline(0, color=F.INK, linewidth=0.6, zorder=1)
    lim = xlim or (-0.15 * span if min(min(r[1], r[2]) for r in rows) >= 0 else -1.45 * span,
                   0.15 * span if max(max(r[1], r[2]) for r in rows) <= 0 else 1.45 * span)
    ax.set_xlim(*lim)
    ax.set_xlabel(xlab, fontsize=6.5, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=6, length=1.5, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def row_layout(ax, ys, groups, wmm, row_labels=True, group_labels=True):
    """行标签（情景）与组标签（稻作类型，竖排），组间虚线分隔（同参考图）；wmm = 轴宽（毫米）"""
    ax.set_ylim(max(ys) + 0.7, min(ys) - 0.7)
    for k, (y, (ri, ssp)) in enumerate(zip(ys, groups)):
        if row_labels:
            ax.text(-1.2 / wmm, y, F.SSP_LAB[ssp], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6)
    seps = [(ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1) if groups[i][0] != groups[i + 1][0]]
    for s in seps:
        ax.axhline(s, color="0.75", linewidth=0.5, linestyle=(0, (3, 2)), zorder=0)
    if group_labels:
        for ri in dict.fromkeys(g[0] for g in groups):
            yy = [y for y, g in zip(ys, groups) if g[0] == ri]
            name = {"ri1": "Single/early", "ri2": "Late", "all": "Both"}[ri]
            ax.text(-15.5 / wmm, np.mean(yy), name, transform=ax.get_yaxis_transform(), ha="center", va="center",
                    rotation=90, fontsize=6.5)


dec = F.read_sheet(XLSX, "11_decomp_373").set_index(["ri", "ssp", "var"])
mem = F.read_sheet(XLSX, "31_quality_change_members")
qc = F.read_sheet(XLSX, "02_quality_change").set_index(["trait", "ri", "ssp", "co2"])
sh = F.read_sheet(XLSX, "13_shap")
unc = F.read_sheet(XLSX, "12_uncertainty").set_index("target")

label(0, ROW1_Y - 4.0, "b")
VARS_A = [("days", "ΔDuration (d)", False, "total"), ("Tavg", "ΔTavg (°C)", False, "shift"),
          ("CTSR", "ΔCTSR (%)", True, "shift"), ("TSR", "ΔTSR (%)", True, "shift")]
TITLE_A = {"days": "Grain-filling\nduration"}
ys = [0, 1, 2.4, 3.4]
ax0x, colw, gap = 22.5, 13.5, 7.0
for j, (var, xlab, pct, lab_kind) in enumerate(VARS_A):
    ax = axmm(ax0x + j * (colw + gap), ROW1_Y + 3.0, colw, ROW1_H - 12.0)
    rows = []
    for y, (ri, ssp) in zip(ys, GROUPS):
        r = dec.loc[(ri, ssp, var)]
        base = float(r["baseline"])
        clim = float(r["climate"]) + float(r["base_window"])
        tot = float(r["total"])
        if pct:
            clim, tot = clim / base * 100, tot / base * 100
        rows.append((y, clim, tot, COL[ssp]))
    if lab_kind == "total":
        labs = {y: fmt(full) for y, _, full, _ in rows}
    else:
        labs = {y: fmt(full - cf) for y, cf, full, _ in rows}
    xl = None
    if j == 0:
        sp_ = max(max(abs(r[1]), abs(r[2])) for r in rows)
        xl = (-2.0 * sp_, 0.15 * sp_)
    cf_bars(ax, rows, xlab, lambda y, cf, full, s=labs: s[y].replace("-", "−"), xlim=xl)
    if j == 0:
        ax.set_xticks([-20, -10, 0])
    row_layout(ax, ys, GROUPS, colw, row_labels=(j == 0), group_labels=(j == 0))
    ax.set_title(TITLE_A.get(var, var), fontsize=7, pad=2, linespacing=1.0)
axl = axmm(ax0x, ROW1_Y + ROW1_H + 0.3, 4 * colw + 3 * gap, 4.0); axl.axis("off")
axl.legend(handles=[Patch(facecolor="white", edgecolor="0.35", hatch=HATCH, label="window fixed (baseline)"),
                    Patch(facecolor="0.35", label="window shift"),
                    ], loc="center", ncol=2, fontsize=6, handlelength=1.4, columnspacing=1.0,
           borderaxespad=0, handletextpad=0.4)

label(99.0, ROW1_Y - 4.0, "c")
BX, BW_, BH_, BG = 108.5, 31.0, 22.0, 8.5
pos = {}
xc = 0.0
for ri in F.RIS:
    for ssp in F.SSPS:
        for co2 in ("with_CO2", "fixed_CO2"):
            pos[(ri, ssp, co2)] = xc; xc += 0.78
        xc += 0.18
    xc += 0.55
RI_MID = {ri: np.mean([v for k, v in pos.items() if k[0] == ri]) for ri in F.RIS}
order = ["HRR", "CD", "PC"]
cells = [(0, 0), (0, 1), (1, 0)]
for t, (r, c) in zip(order, cells):
    ax = axmm(BX + c * (BW_ + BG), ROW1_Y + r * (BH_ + 9.0), BW_, BH_)
    for (ri, ssp, co2), x in pos.items():
        v = mem[(mem.trait == t) & (mem.ri == ri) & (mem.ssp == ssp) & (mem.co2 == co2)].rel_change_pct.dropna()
        if v.empty:
            continue
        filled = co2 == "with_CO2"
        bp = ax.boxplot(v.values, positions=[x], widths=0.62, patch_artist=True, whis=1.5, showfliers=True,
                        medianprops=dict(color=F.INK, linewidth=0.7),
                        whiskerprops=dict(color=F.INK, linewidth=0.5), capprops=dict(color=F.INK, linewidth=0.5),
                        flierprops=dict(marker="*", ms=1.7, mec=F.INK, mfc=F.INK, mew=0.25))
        b = bp["boxes"][0]
        b.set(facecolor=COL[ssp] if filled else "white", edgecolor=COL[ssp] if not filled else F.INK,
              linewidth=0.5, hatch=None if filled else HATCH)
        if (t, ri, ssp, co2) in qc.index:
            ax.plot(x, float(qc.loc[(t, ri, ssp, co2), "rel_change_pct"]), marker="^", ms=2.6,
                    mfc="white" if filled else COL[ssp], mec=F.INK, mew=0.4, zorder=5)
    ax.axhline(0, color=F.INK, linewidth=0.5, zorder=0)
    ax.set_xticks([RI_MID["ri1"], RI_MID["ri2"]])
    ax.set_xticklabels(["Single/early", "Late"], fontsize=5.8)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=5.8, length=1.5, pad=1)
    ax.set_xlim(-0.6, xc - 0.55 - 0.18)
    ax.set_title(F.TLAB[t], loc="left", fontsize=6.2, pad=1.5)
    if c == 0:
        ax.set_ylabel("Change (%)", fontsize=6.5, labelpad=1)
    ax.grid(axis="y", color="0.93", linewidth=0.4, zorder=0); ax.set_axisbelow(True)
axl = axmm(BX + 1 * (BW_ + BG), ROW1_Y + BH_ + 9.0, BW_, BH_); axl.axis("off")
hdl = [Patch(facecolor=COL["ssp126"], edgecolor=F.INK, linewidth=0.5, label="SSP1-2.6"),
       Patch(facecolor=COL["ssp585"], edgecolor=F.INK, linewidth=0.5, label="SSP5-8.5"),
       Patch(facecolor="white", edgecolor="0.35", hatch=HATCH, linewidth=0.5, label="CO₂ fixed at baseline"),
       Line2D([], [], marker="^", ls="none", ms=3, mfc="white", mec=F.INK, mew=0.5, label="Ensemble mean"),
       Line2D([], [], marker="*", ls="none", ms=3, mfc=F.INK, mec=F.INK, label="Outlying member")]
axl.legend(handles=hdl, loc="upper left", fontsize=5.9, handlelength=1.3, labelspacing=0.55, borderaxespad=0,
           bbox_to_anchor=(0.02, 1.02))

label(0, ROW2_Y - 4.0, "a")
try:
    import rasterio
    import geopandas as gpd
    import config_1004 as C

    TR3 = ["HRR", "CD", "PC"]
    LIM = {"HRR": 15, "CD": 200, "PC": 10}
    BASE_MIN = {"CD": 2.0}
    LON_VALS = np.arange(73.25, 136.0, 0.5); LAT_VALS = np.arange(53.75, 17.75, -0.5)
    EXTENT = [LON_VALS[0] - .25, LON_VALS[-1] + .25, LAT_VALS[-1] - .25, LAT_VALS[0] + .25]
    INSET = [105, 125, 2, 26]
    W_DIR = C.pred_dir(C.PRED_MODE, "withco2") / "ensemble"
    F_DIR = C.pred_dir(C.PRED_MODE, "fixedco2") / "ensemble"
    china = gpd.read_file(C.CHINA_SHP).to_crs("EPSG:4326")
    poly = gpd.read_file(C.POLY_SHP).to_crs("EPSG:4326")
    nanhai = poly[poly.geometry.apply(lambda g: g.bounds[1] < 25)]
    jd = poly[poly["adchar"] == "JD"] if "adchar" in poly.columns else nanhai
    bx0, by0, bx1, by1 = china.total_bounds

    def rd(path):
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype("float32")
            if ds.nodata is not None:
                arr[arr == np.float32(ds.nodata)] = np.nan
        return arr

    def pct(fut, base, bmin=None):
        ok = np.isfinite(fut) & np.isfinite(base) & (base != 0)
        if bmin is not None:
            ok &= np.abs(base) >= bmin
        return np.where(ok, (fut - base) / np.abs(base) * 100, np.nan)

    DOM = {ri: np.isfinite(rd(C.support_file(ri))) for ri in F.RIS}
    CHG = {}
    for t in TR3:
        for ri in F.RIS:
            bse = np.where(DOM[ri], rd(W_DIR / f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif"), np.nan)
            CHG[(t, ri, "with")] = pct(rd(W_DIR / f"ensemble_{SSP_MAP}_{ri}_{t}_{C.FUT_P}.tif"), bse, BASE_MIN.get(t))
            CHG[(t, ri, "fixed")] = pct(rd(F_DIR / f"ensemble_{SSP_MAP}_{ri}_{t}_{C.FUT_P}.tif"), bse, BASE_MIN.get(t))
    MAP_OK = True
except Exception as e:
    print(f"[c] ⚠️ 读不到预测 tif，c 面板留空：{e}")
    MAP_OK = False


def place_strip_label(axs, txt, lo, hi, mu, yv, L, color, fs=5.2):
    """按最终字号（F.save 时会乘 FONT_SCALE 并抬到 MIN_FONT_PT）实测文字宽度后放置：
    1) 须线外侧、远离 0 的一侧；2) 须线另一侧；两处都要求完全在轴内且不跨 0 线；
    3) 都不行时放到 0 线另一侧并紧贴 0 线（该行那一侧没有数据）。"""
    r = axs.figure.canvas.get_renderer()
    fs_final = max(fs * F.FONT_SCALE, F.MIN_FONT_PT)
    t = axs.text(0, yv, txt, fontsize=fs_final); bb = t.get_window_extent(r); t.remove()
    inv = axs.transData.inverted()
    w = abs(inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0])
    pad = 0.03 * 2 * L
    pref = -1 if mu <= 0 else 1
    cands = []
    for side in (pref, -pref):
        if side > 0:
            x0, x1, xt, ha = hi + pad, hi + pad + w, hi + pad, "left"
        else:
            x0, x1, xt, ha = lo - pad - w, lo - pad, lo - pad, "right"
        cands.append((x0 >= -L and x1 <= L and not (x0 < 0 < x1), xt, ha))
    ok = [c for c in cands if c[0]]
    if ok:
        _, xt, ha = ok[0]
    elif hi <= 0:
        xt, ha = pad, "left"
    elif lo >= 0:
        xt, ha = -pad, "right"
    else:
        _, xt, ha = cands[0]
    axs.text(xt, yv, txt, ha=ha, va="center", fontsize=fs, color=color)


def draw_hist_ref(ax, axs, v_with, v_fixed, L, bins, st, ylabel=True, xlabel=False, title=None, key=None):
    """ax：原实心柱直方图（两种 CO₂ 设定，配色/透明度沿用原图）+ 面积加权中位数虚线 + 右上角环形图
    （情景 CO₂ 下上升/下降的种植面积占比）；
    axs：其上方带独立横轴的成员范围条（参考 Zhang et al. 2026 Fig. 2 的 Contribution 小轴）：
    圆点 = 集合均值的全国相对变化，横线 = 25 个 GGCM × GCM 成员的最小–最大"""
    M_ = "\u2212"
    data = {"with": v_with, "fixed": v_fixed}
    for kind in ("with", "fixed"):
        v = data[kind]; v = v[np.isfinite(v)]; v = np.clip(v, -L, L)
        wts = np.full(v.shape, 100.0 / max(v.size, 1))
        ax.hist(v, bins=bins, weights=wts, color=HCOL[kind], alpha=HALPHA[kind], edgecolor="white",
                linewidth=0.3, zorder=2 if kind == "with" else 3)
        if st.get(kind) is not None:
            med = float(st[kind]["cell_chg_median"])
            ax.axvline(np.clip(med, -L, L), color=HCOL[kind], linewidth=0.8, linestyle=(0, (3, 1.5)), zorder=4)
    ax.axvline(0, color="0.35", linewidth=0.6, linestyle=(0, (4, 2)), zorder=4)
    ax.set_xlim(-L, L); ax.set_xticks([-L, 0, L])
    ax.set_xticklabels([f"≤{M_}{L}", "0", f"≥{L}"])
    ax.tick_params(labelsize=5.8, length=1.5, pad=1)
    if ylabel:
        ax.set_ylabel("Frequency (%)", fontsize=6.3, labelpad=1)
    if xlabel:
        ax.set_xlabel("Change (%)", fontsize=6.5, labelpad=1)
    if st.get("with") is not None:
        dec = float(st["with"]["decrease_share_pct"]); inc = 100.0 - dec
        hmax = {}
        for side_, sl in (("L", slice(0, max(1, int(len(bins) * 0.35)))), ("R", slice(int(len(bins) * 0.65), None))):
            hm = 0.0
            for kind in ("with", "fixed"):
                v = data[kind]; v = np.clip(v[np.isfinite(v)], -L, L)
                cnt, _ = np.histogram(v, bins=bins); cnt = cnt * 100.0 / max(v.size, 1)
                hm = max(hm, float(np.max(cnt[sl])) if cnt[sl].size else 0.0)
            hmax[side_] = hm
        corner = "R" if hmax["R"] <= hmax["L"] else "L"
        box = [0.70, 0.56, 0.22, 0.34] if corner == "R" else [0.14, 0.56, 0.22, 0.34]
        if key in PIE_BOX_OVERRIDE:
            box = PIE_BOX_OVERRIDE[key]
        dax = ax.inset_axes(box)
        dax.pie([inc, dec], colors=[PIE_COL["inc"], PIE_COL["dec"]], startangle=90, counterclock=False,
                wedgeprops=dict(width=0.40, linewidth=0.3, edgecolor="white"))
        dax.set_aspect("equal")
        dax.text(1.02, 0.96, f"{inc:.0f}%", transform=dax.transAxes, ha="left", va="bottom", fontsize=5.0,
                 color=PIE_COL["inc"])
        dx_pt = DEC_LABEL_DX_PT.get(key, 0.0)
        dax.text(-0.02, 0.04, f"{dec:.0f}%", transform=mtransforms.offset_copy(dax.transAxes, fig=fig, x=dx_pt, y=0, units="points"),
                 ha="right", va="top", fontsize=5.0,
                 color=PIE_COL["dec"])
    axs.set_xlim(-L, L); axs.set_ylim(-0.9, 1.8)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    axs.spines["bottom"].set_linewidth(0.5); axs.spines["left"].set_linewidth(0.5)
    axs.set_yticks([1.25, -0.05]); axs.set_yticklabels([])
    axs.tick_params(axis="y", length=1.5, width=0.4)
    axs.set_xticks([-L, 0, L]); axs.set_xticklabels([f"{M_}{L}", "0", f"{L}"])
    axs.tick_params(axis="x", labelsize=5.0, length=1.2, pad=0.8, width=0.4)
    axs.axvline(0, color="0.35", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
    for k, kind in enumerate(("with", "fixed")):
        s_ = st.get(kind)
        if s_ is None:
            continue
        yv = 1.25 - k * 1.3
        lo, hi = np.clip([float(s_["member_min"]), float(s_["member_max"])], -L, L)
        mu = np.clip(float(s_["rel_change_pct"]), -L, L)
        axs.plot([lo, hi], [yv, yv], color=HCOL[kind], linewidth=0.8, solid_capstyle="butt", zorder=2)
        for xe in (lo, hi):
            axs.plot([xe, xe], [yv - 0.25, yv + 0.25], color=HCOL[kind], linewidth=0.6, zorder=2)
        axs.plot(mu, yv, marker="o", markersize=2.6, color=HCOL[kind], markeredgecolor="white",
                 markeredgewidth=0.3, zorder=3)
        txt = fmt(float(s_["rel_change_pct"])) + "%"
        place_strip_label(axs, txt, lo, hi, mu, yv, L, HCOL[kind])
    if title:
        axs.set_title(title, fontsize=7, fontweight="bold", pad=3.0)

MAPW, MAPH = 39.0, 27.5
try:
    import paper_palette as PP
    HCOL = {"with": PP.CO2["with"], "fixed": PP.CO2["without"]}
except Exception:
    HCOL = {"with": "#2C6FB0", "fixed": "#E8833A"}
HALPHA = {"with": 0.55, "fixed": 0.45}
PIE_COL = {"inc": "#8FBF6A", "dec": "#D98BB5"}
PIE_BOX_OVERRIDE = {("PC", "ri2"): [0.05, 0.64, 0.20, 0.32]}
DEC_LABEL_DX_PT = {("PC", "ri2"): 9.0}
hist_axes = {}
strip_axes = []
M = "\u2212"
MX = 9.0
HX = MX + 2 * MAPW + 2.0 + 15.0
HW = 27.5
if MAP_OK:
    MAP_ASPECT = 1 / np.cos(np.deg2rad(36))
    for i, t in enumerate(TR3):
        L = LIM[t]; norm = TwoSlopeNorm(0, -L, L)
        y0 = ROW2_Y + 4.0 + i * (MAPH + 1.5)
        for j, ri in enumerate(F.RIS):
            ax = axmm(MX + j * (MAPW + 1.0), y0, MAPW, MAPH)
            ax.imshow(CHG[(t, ri, "with")], extent=EXTENT, origin="upper", cmap="RdBu_r", norm=norm,
                      interpolation="nearest", zorder=1)
            china.boundary.plot(ax=ax, linewidth=0.22, color=F.INK, zorder=3)
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xlim(bx0 - 0.5, bx1 + 0.5); ax.set_ylim(by0 - 0.5, by1 + 0.5)
            ax.set_aspect(MAP_ASPECT, adjustable="box", anchor="C")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.4)
            iax = ax.inset_axes([0.80, 0.02, 0.18, 0.30])
            iax.imshow(CHG[(t, ri, "with")], extent=EXTENT, origin="upper", cmap="RdBu_r", norm=norm,
                       interpolation="nearest")
            nanhai.boundary.plot(ax=iax, linewidth=0.15, color=F.INK)
            jd.boundary.plot(ax=iax, linewidth=0.3, color=F.INK)
            iax.set_xlim(INSET[0], INSET[1]); iax.set_ylim(INSET[2], INSET[3])
            iax.set_xticks([]); iax.set_yticks([]); iax.set_xlabel(""); iax.set_ylabel("")
            for s in iax.spines.values():
                s.set_linewidth(0.3)
            if i == 0:
                ax.set_title(F.RI_LAB[ri], fontsize=7, fontweight="bold", pad=2)
            if j == 0:
                ax.text(-0.06, 0.5, F.TLAB[t], transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=7, fontweight="bold")
        cax = axmm(MX + 2 * MAPW + 2.5, y0 + 3.0, 1.6, MAPH - 6.0)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, extend="both")
        cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
        cb.set_ticks([-L, 0, L])
        cax.set_title("%", fontsize=5.8, pad=1.5)
        step = max(1, (2 * L) // 40)
        bins = np.arange(-L, L + step, step)
        for j, ri in enumerate(F.RIS):
            x0h = HX + j * (HW + 9.5)
            ax = axmm(x0h, y0 + 8.9, HW, MAPH - 11.6)
            hist_axes.setdefault(t, []).append(ax)
            axs_ = axmm(x0h, y0 + 0.4, HW, 5.6); strip_axes.append(axs_)
            st = {}
            for kind, key in (("with", "with_CO2"), ("fixed", "fixed_CO2")):
                try:
                    st[kind] = qc.loc[(t, ri, SSP_MAP, key)]
                except KeyError:
                    st[kind] = None
            draw_hist_ref(ax, axs_, CHG[(t, ri, "with")], CHG[(t, ri, "fixed")], L, bins, st,
                          ylabel=(j == 0), xlabel=(i == 2), title=(F.RI_LAB[ri] if i == 0 else None), key=(t, ri))
    for axs in hist_axes.values():
        ymax = max(ax_.get_ylim()[1] for ax_ in axs)
        for ax_ in axs:
            ax_.set_ylim(0, ymax)
    fig.canvas.draw(); _rr = fig.canvas.get_renderer()
    _xr = max(a.get_tightbbox(_rr).x1 for a in strip_axes + [a for v in hist_axes.values() for a in v])
    _xr = _xr / fig.dpi * 25.4
    LGX = max(_xr + 1.2 - 2.0 + 1.1, EX + EW - 7.0)
    vlegend(LGX, ROW2_Y + 5.0, 3 * (MAPH + 1.5) - 6.0,
            [(dict(facecolor=HCOL[k], alpha=HALPHA[k], edgecolor="none"), lab)
             for k, lab in (("with", "Scenario CO₂"), ("fixed", "CO₂ fixed at baseline"))])
    vlegend(LGX + 4.0, ROW2_Y + 5.0, 3 * (MAPH + 1.5) - 6.0,
            [(dict(facecolor=PIE_COL[k], edgecolor="none"), lab)
             for k, lab in (("inc", "Area increased (%)"), ("dec", "Area decreased (%)"))])
    fig.text((HX + 2 * HW + 11.0) / W, 1 - (ROW2_Y + 3 * (MAPH + 1.5) + 0.5) / H, "", fontsize=5)

label(0, ROW3_Y - 4.0, "d")
FEATS = ["Tavg", "Tdif", "TSR", "CTSR", "Rain", "RHavg", "CO2"]
FLAB = {"CO2": "CO₂"}
P = sh.pivot_table(index=["trait", "ri", "feature"], columns="scenario", values="mean_shap")
try:
    import config_1004 as C
    d_ = C.out_dir("shap_features", thresh=False)
    NCELL = {ri: len(pd.read_csv(d_ / f"features_historical_{ri}.csv", usecols=[0])) for ri in F.RIS}
except Exception:
    NCELL = {ri: int(qc.xs(ri, level="ri")["n_cells"].iloc[0]) for ri in F.RIS}
WRI = {ri: NCELL[ri] / sum(NCELL.values()) for ri in F.RIS}


def dshap(t, f, ssp):
    return sum(WRI[ri] * (P.loc[(t, ri, f), ssp] - P.loc[(t, ri, f), "hist"]) for ri in F.RIS)


HMW, HMH = 21.0, ROW3_H - 8.0
for k, t in enumerate(["HRR", "CD", "PC"]):
    ax = axmm(14.0 + k * (HMW + 9.5), ROW3_Y + 2.5, HMW, HMH)
    M = np.array([[dshap(t, f, s) for s in F.SSPS] for f in FEATS])
    vmax = float(np.ceil(np.nanmax(np.abs(M)) * 10) / 10)
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i_ in range(M.shape[0]):
        for j_ in range(M.shape[1]):
            v = M[i_, j_]
            ax.text(j_, i_, fmt(v, 2), ha="center", va="center", fontsize=5.4,
                    color="white" if abs(v) > 0.6 * vmax else F.INK)
    ax.axhline(len(FEATS) - 1.5, color="white", linewidth=1.1)
    ax.axvline(0.5, color="white", linewidth=0.9)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels([FLAB.get(f, f) for f in FEATS] if k == 0 else [], fontsize=5.9)
    ax.set_xticks(range(2)); ax.set_xticklabels(["1-2.6", "5-8.5"], fontsize=5.8)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(t, fontsize=7, fontweight="bold", pad=2)
    cax = ax.inset_axes([1.05, 0.0, 0.08, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-vmax, 0, vmax]); cb.ax.tick_params(labelsize=5.3, length=1.2, width=0.4, pad=0.8)
    cb.outline.set_linewidth(0.4)
    if k == 2:
        cb.set_label("ΔSHAP (pp)", fontsize=6.5, rotation=90, labelpad=1)

label(107.0, ROW3_Y - 4.0, "e")
SRC = [("GCM", "GCM"), ("GGCM", "GGCM"), ("SSP", "SSP"), ("RS", "Season"), ("INTER", "Interactions")]
SRC_COL = {"GCM": "#2C6FB0", "GGCM": "#9DC3E6", "SSP": "#E8833A", "RS": "#F7C9A0", "INTER": "#D6D6D6"}
SRC_TXT = {"GCM": "white", "GGCM": F.INK, "SSP": "white", "RS": F.INK, "INTER": F.INK}
unc["INTER"] = unc["TWOWAY_sum"] + unc["THREEWAY_sum"] + unc["FOURWAY"]
E_ROWS = [("All", "All traits"), ("HRR", "HRR"), ("CD", "CD"), ("PC", "PC")]
yse = [0, 1.3, 2.3, 3.3]
ax = axmm(EX, ROW3_Y + 3.5, EW, ROW3_H - 11.0)
for y, (t, lab) in zip(yse, E_ROWS):
    left = 0.0
    for k, _ in SRC:
        v = float(unc.loc[t, k])
        ax.barh(y, v, left=left, height=0.72, color=SRC_COL[k], edgecolor="white", linewidth=0.4, zorder=2)
        if v >= 7:
            ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color=SRC_TXT[k], zorder=3)
        left += v
    ax.text(-1.2 / EW * 100, y, lab, ha="right", va="center", fontsize=6,
            fontweight="bold" if t == "All" else "normal")
ax.axhline((yse[0] + yse[1]) / 2, color="0.75", linewidth=0.5, linestyle=(0, (3, 2)), zorder=0)
ax.set_ylim(max(yse) + 0.6, -0.6)
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelsize=6, length=1.5, pad=1)
ax.set_xlabel("Variance share (%)", fontsize=6.5, labelpad=1.5)
axl = axmm(EX - 9.0, ROW3_Y - 2.2, EW + 9.0, 4.0); axl.axis("off")
axl.legend(handles=[Patch(facecolor=SRC_COL[k], edgecolor="none", label=lab) for k, lab in SRC],
           loc="center", ncol=5, fontsize=6, handlelength=1.1, handleheight=0.9, columnspacing=0.9,
           handletextpad=0.35, borderaxespad=0)

F.save(fig, OUT, "Fig2_main")
