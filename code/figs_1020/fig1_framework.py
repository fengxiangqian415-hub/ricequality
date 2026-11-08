# ============================================================
# fig1_framework.py —— 新 Fig. 1：数据与分析框架（整张矢量图）
#   排版（--layout）：
#     C（默认，1022）：a 稻米品质概念图（优质 vs 劣质；fig1_concept.py）；b 采样点地图 + c Whittaker 气候空间；
#               d 分析流程（模型精度以测试集 R² 范围写在流程框里）
#     B         ：第一排 a 采样点地图（缩小）+ b Whittaker 气候空间（单独面板，图例在图内、群系写全称）；
#               第二排 c 模型精度 + d 输入分布（峰峦图）；第三排 e 分析流程
#     inset    ：a 采样点地图（Whittaker 作插图）  b 模型精度  c 输入覆盖  d 分析流程
#   采样点地图：直接运行同目录 fig1a_sampling_map.py，画进本图（不嵌 PNG）
#   模型精度：箱 = 五折交叉验证逐折 R²，▲ = 独立测试集 R²（95% CI 见 Supplementary Table S3）；
#             默认读训练输出 /home/debian/pythonfile/results/Tables（skill_tables.py），并与表 S3 抄录值核对
#   输入分布：峰峦图 = 训练数据核密度；峰内直线 = 预测输入 5–95% 范围与均值（Excel 30_S10b；训练数据 FACE_XLSX）
#   分析流程：fig1_workflow.py（Supplementary Fig. S17 的浓缩版）
#   用法：python figs_1020/fig1_framework.py [--datadir map.xlsx 所在目录] [--tables 目录 | --skill csv] [--out ...]
# ============================================================
import runpy
import types
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import fig_common as F
from fig1_workflow import draw_workflow
from fig1_concept import concept_height, draw_concept

F.setup()
F.FONT_SCALE = 1.15                            # [1031] 正文图文字放大 15%，最小 6 pt（100% 视图可读）
p = F.args("Fig. 1 framework")
p.add_argument("--datadir", default=None, help="map.xlsx 等输入表所在目录（默认自动查找）")
p.add_argument("--tables", default="/home/debian/pythonfile/results/Tables", help="训练输出表目录")
p.add_argument("--skill", default=None, help="改用整理好的 csv（如 model_skill_tableS3.csv）")
p.add_argument("--face", default=None, help="训练数据 Excel（默认 config 的 FACE_XLSX）")
p.add_argument("--layout", default="C", choices=["C", "B", "inset"],
               help="C = 概念图 + 地图 + Whittaker + 流程；B = 旧版（含 R² 与输入分布）；inset = Whittaker 作地图插图")
a = p.parse_args()
OUT = a.out or F.default_out()

REF = F.HERE / "model_skill_tableS3.csv"
if a.skill:
    sk = pd.read_csv(a.skill)
else:
    from skill_tables import load_skill
    sk = load_skill(a.tables, reference=REF)
sk = sk.sort_values("test_r2", ascending=True)

# ---------------- 版式（毫米） ----------------
W = 180.0
LAY_C = a.layout == "C"
LAY_B = a.layout in ("B", "C")                 # C 沿用 B 的地图、Whittaker 版式
LET = iter("abcde")                            # 面板字母按阅读顺序依次取
WF_H = 71.0                                    # 流程高度（[1108d] 67.6→71.0：d 按放大后的最终字号排版，需多 3.4 mm）
MAP_X, MAP_W = (6.5, 86.0) if LAY_B else (7.5, 111.0)   # 地图轴（左侧留经纬度刻度）
if LAY_C:                                      # [1023] b、c 加虚线框：地图右移、略缩，纬度标签收进框内
    MAP_X, MAP_W = 10.0, 82.5
FR_L, FR_R = 0.6, W - 0.6                      # 各虚线框的左、右外沿（a、b、c、d 对齐）
FR_BR, FR_CL = 94.0, 97.8                      # （C）b 框右沿、c 框左沿
FR_PT, FR_PB = 3.0, 1.8                        # （C）b、c 框内上、下留白（框顶在图框上方 3 mm）
MAP_H = MAP_W * (54 - 17) / (136 - 73) / np.cos(np.deg2rad((17 + 54) / 2))   # 与地图等经纬比例一致
ROW2_Y, ROW2_H = WF_H + 5.0, 37.0              # （B）第二排：模型精度 + 输入分布
MAP_Y = ROW2_Y + ROW2_H + 5.0 if LAY_B else WF_H + 7.0   # 地图排底边
if LAY_C:                                      # （C）没有第二排；地图排之上是概念图
    MAP_Y = WF_H + 5.6
TOP = MAP_Y + (4.5 + MAP_H + 1.0 if LAY_B else MAP_H)   # 地图排顶边（mm，自下而上）
CON_H = concept_height(W)                      # （C）概念图高（含虚线框；随扫描图宽高比自动算）
CON_Y = MAP_Y + 4.5 + MAP_H + FR_PT + 5.8 if LAY_C else 0.0   # （C）概念图底边（下面留出 b、c 的虚线框和面板字母）
H = CON_Y + CON_H + 0.5 if LAY_C else TOP + 3.0
fig = plt.figure(figsize=(W * F.MM, H * F.MM))
fx = lambda mm: mm / W
fy = lambda mm: mm / H
ROW_LET_Y = fy(MAP_Y + 4.5 + MAP_H + FR_PT + 4.2) if LAY_C else 1.0   # 地图排面板字母：虚线框顶边之上 4.2 mm（同 a、d）

# ---------------- （C）a 稻米品质概念图 ----------------
cvA = draw_concept(fig, 0.0, CON_Y, W, H, W=W, H=CON_H, label=next(LET)) if LAY_C else None

# ---------------- a 采样点地图（原生绘制） ----------------
ax_map = fig.add_axes([fx(MAP_X), fy(MAP_Y + 4.5 if LAY_B else MAP_Y), fx(MAP_W), fy(MAP_H)])
NS = runpy.run_path(str(F.HERE / "fig1a_sampling_map.py"), run_name="fig1a_embedded",
                    init_globals=dict(EXT_FIG=fig, EXT_AX=ax_map,
                                      EXT_ARGS=types.SimpleNamespace(fig1=True, datadir=a.datadir, out=None,
                                                                     whittaker=not LAY_B)))
F.setup()                                      # 地图脚本改过 rcParams，恢复本图设置
fig.text(0.0, ROW_LET_Y, next(LET), fontsize=8, fontweight="bold", ha="left", va="top")

# ---------------- （B）Whittaker 气候空间：单独面板 ----------------
#   采样点在年均温（MAT）× 年降水（MAP）空间的位置；底色 = Whittaker 生物群系（缩写直接标在色块上），形状 = 六大稻区
if LAY_B:
    WX = 104.0 if LAY_C else 103.0                 # 与 a 的地图框上下对齐
    ax_w = fig.add_axes([fx(WX), fy(MAP_Y + 4.5), fx(W - WX - (2.2 if LAY_C else 3.0)), fy(MAP_H)])
    try:
        NS["draw_whittaker_panel"](ax_w, fs=6.5, dot=8.5, mk=3.5, xmin=-20, bfs=6.3, rfs=6.2)
    except Exception as e:
        print(f"[Fig. 1b] ⚠️ Whittaker 面板没画成（{e}）")
    fig.text(fx(FR_CL - 0.6) if LAY_C else fx(WX - 11.0), ROW_LET_Y, next(LET), fontsize=8, fontweight="bold",
             ha="left", va="top")

if LAY_C:                                      # [1023] b、c 各加一个虚线框（样式同 a、d；左右外沿与 a、d 对齐）
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    _rr = fig.canvas.get_renderer()
    _mm = lambda bb: [v / fig.dpi * 25.4 for v in (bb.x0, bb.y0, bb.x1, bb.y1)]
    tb_m, tb_w = _mm(ax_map.get_tightbbox(_rr)), _mm(ax_w.get_tightbbox(_rr))   # 含刻度、坐标轴标题
    fy0 = min(tb_m[1], tb_w[1]) - FR_PB                                         # 框底：最低的标签下 1.8 mm
    fy1 = MAP_Y + 4.5 + MAP_H + FR_PT                                           # 框顶：图框上 3 mm
    ov = fig.add_axes([0, 0, 1, 1], zorder=-1); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis("off")
    ov.set_in_layout(False)                     # 覆盖层不计入 tight 边界（成图宽仍 ≤ 180 mm）
    for x0_, x1_ in ((FR_L, FR_BR), (FR_CL, FR_R)):
        ov.add_patch(FancyBboxPatch((x0_, fy0), x1_ - x0_, fy1 - fy0, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="none", ec="#1E3448", lw=0.9, ls=(0, (4.5, 2.8))))
    for nm, tb, (x0_, x1_) in (("b", tb_m, (FR_L, FR_BR)), ("c", tb_w, (FR_CL, FR_R))):
        if tb[0] < x0_ + 0.3 or tb[2] > x1_ - 0.3 or tb[3] > fy1:
            print(f"[Fig. 1{nm}] ⚠️ 内容超出虚线框：{[round(v, 1) for v in tb]} vs 框 x {x0_}–{x1_}, 顶 {fy1:.1f}")
    if fy0 < WF_H + 1.0:
        print(f"[Fig. 1] ⚠️ b、c 虚线框底边（{fy0:.1f} mm）离 d 的字母太近")

# [1022] 排版 C：模型精度（R²）与输入分布两块不再放在 Fig. 1（R² 见流程框与 Table S3；输入覆盖见 Table S10）
SKILL_LET = None
if not LAY_C:
    # ---------------- b 模型精度：箱线图 ----------------
    #   箱 = 集成模型五折交叉验证的逐折 R²（训练集，每个性状 5 折，点 = 各折）；▲ = 独立测试集 R²（n = 228）
    #   7 个性状同一配色（不单独突出 Q 的三个性状）；95% CI 见 Supplementary Table S3
    from matplotlib.patches import Patch, Rectangle
    from skill_tables import read_cv_folds
    try:
        folds = read_cv_folds(Path(a.tables) / "Ensemble_CV_folds_training.csv")
    except Exception as e:
        folds = None
        print(f"[Fig. 1b] ⚠️ 读不到逐折 R²（{e}），b 只画测试集 R²")
    BX0 = 109.0 if LAY_B else 134.0                # （B）d 放右侧：先数据（c），后模型（d）
    ax_b = fig.add_axes([fx(BX0), fy(ROW2_Y + 4.0), fx(W - BX0 - 2.0), fy(ROW2_H - 8.5)]) if LAY_B else \
        fig.add_axes([fx(BX0), fy(TOP - 33.0), fx(W - BX0 - 1.5), fy(24.0)])
    C_LET, D_LET = next(LET), next(LET)
    SKILL_LET, RIDGE_LET = (D_LET, C_LET) if LAY_B else (C_LET, D_LET)
    TR = list(F.TRAITS)
    BOXFILL = "#D9D9D9"
    rng = np.random.default_rng(42)
    for i, t in enumerate(TR):
        fc = BOXFILL
        if folds is not None and (folds.trait == t).any():
            v = folds.loc[folds.trait == t, "r2"].values
            bp = ax_b.boxplot(v, positions=[i], widths=0.58, patch_artist=True, whis=1.5, showfliers=False,
                              medianprops=dict(color=F.INK, linewidth=0.8),
                              whiskerprops=dict(color=F.INK, linewidth=0.6), capprops=dict(color=F.INK, linewidth=0.6))
            bp["boxes"][0].set(facecolor=fc, edgecolor=F.INK, linewidth=0.6)
            ax_b.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=2.0, color="0.30", linewidths=0, zorder=4)
        r = sk[sk.trait == t]
        if not r.empty:
            ax_b.plot(i + 0.42, float(r.test_r2.iloc[0]), marker="^", ms=3.0, color=F.INK, mec=F.INK, zorder=5,
                      ls="none")
    ax_b.set_xticks(range(len(TR)))
    ax_b.set_xticklabels([F.TLAB[t] for t in TR], fontsize=6.3)
    ax_b.tick_params(axis="x", length=0, pad=1.5)
    ax_b.tick_params(axis="y", labelsize=6.3)
    ax_b.set_xlim(-0.6, len(TR) - 0.3); ax_b.set_ylim(0.25, 0.80)
    ax_b.set_ylabel("R²", fontsize=6.8, labelpad=1)
    ax_b.set_xlabel("Rice quality trait", fontsize=6.8, labelpad=1.5)
    ax_b.grid(axis="y", color="0.92", linewidth=0.5, zorder=0); ax_b.set_axisbelow(True)
    ax_b.legend(handles=[Patch(facecolor="white", edgecolor=F.INK, linewidth=0.6, label="Five-fold CV"),
                         Line2D([], [], marker="^", ls="none", ms=3.0, color=F.INK, label="Held-out test")],
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, handlelength=1.2,
                columnspacing=1.0, borderaxespad=0.1)
    fig.text(fx(97.0) if LAY_B else fx(BX0 - 11.0), fy(ROW2_Y + ROW2_H + 3.0) if LAY_B else 1.0, SKILL_LET,
             fontsize=8, fontweight="bold", ha="left", va="top")

    # ---------------- d 输入分布：峰峦图（训练数据）+ 预测输入的范围线 ----------------
    #   每行一个预测因子；横轴 = 在训练数据范围内的位置（0 = 训练最小值，1 = 最大值），各因子放在同一轴上比较
    #   灰色峰 = 1,136 条训练记录的核密度；峰内三条直线 = 预测输入的 5–95% 范围，白竖线 = 均值
    #   （Baseline 1981–2010、SSP1-2.6、SSP5-8.5 2041–2070；Excel 30_S10b，与 Supplementary Table S10 同一组数）
    #   目的：说明预测输入基本都落在训练数据覆盖的范围内（超出比例见 Table S10，图上不另标）
    from scipy.stats import gaussian_kde
    FEATS_D = ["Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg", "CO2"]
    FLAB_D = {"CO2": "CO₂"}
    PCOL = {"baseline": "#5E5E5E", "ssp126": F.SSP_COL["ssp126"], "ssp585": F.SSP_COL["ssp585"]}
    PLAB = {"baseline": "Baseline", "ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
    S10B_COL = {"baseline": "Baseline 1981–2010", "ssp126": "SSP1-2.6 2041–2070", "ssp585": "SSP5-8.5 2041–2070"}


    def _num(sv):
        import re
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(sv))]


    RANGES = {}                                              # (period, feature) -> (mean, p5, p95)
    try:
        import config_1004 as C
        face = pd.read_excel(a.face or C.FACE_XLSX)
        try:
            s10b = F.read_sheet(a.xlsx or F.default_xlsx(), "30_S10b_predictor_means")
            s10b = s10b.set_index(s10b.columns[0])
            s10b.index = s10b.index.map(lambda v: str(v).split(" (")[0].replace("CO₂", "CO2"))
            for pk, col in S10B_COL.items():
                for f in FEATS_D:
                    RANGES[(pk, f)] = tuple(_num(s10b.loc[f, col])[:3])
        except Exception as e:                               # 没有 S10b 时由预测输入抽样现算
            print(f"[Fig. 1d] 读不到 S10b（{e}），改由预测输入抽样计算")
            from predictor_samples import load_samples
            SAMP = load_samples(Path(OUT) / "_cache")
            for pk in S10B_COL:
                for f in FEATS_D:
                    v = np.asarray(SAMP.get(pk, {}).get(f, []), float)
                    if v.size:
                        RANGES[(pk, f)] = (float(v.mean()), *np.percentile(v, [5, 95]))
        RIDGE_OK = True
    except Exception as e:
        RIDGE_OK = False
        print(f"[Fig. 1d] ⚠️ 读不到训练数据（{e}），d 留空")

    if RIDGE_OK:
        if LAY_B:
            ax_d = fig.add_axes([fx(14.0), fy(ROW2_Y + 4.0), fx(72.0), fy(ROW2_H - 9.0)])
        else:
            ax_d = fig.add_axes([fx(BX0 + 2.0), fy(MAP_Y + 6.5), fx(W - BX0 - 4.0), fy(TOP - 48.0 - MAP_Y - 6.5)])
        XG = np.linspace(-0.15, 1.15, 400)
        HPK = 1.25                                               # 峰高（行距 = 1）
        nrow = len(FEATS_D)
        for i, f in enumerate(FEATS_D):
            y0 = nrow - 1 - i
            tv = face[f].dropna().values.astype(float)
            lo, hi = float(tv.min()), float(tv.max())
            z = lambda v: (np.asarray(v, float) - lo) / (hi - lo)
            d = gaussian_kde(z(tv))(XG)
            d = d / d.max() * HPK
            ax_d.fill_between(XG, y0, y0 + d, color="#D9D9D9", lw=0, zorder=3 * i + 1)
            ax_d.plot(XG, y0 + np.where(d > 0.01 * HPK, d, np.nan), color="0.45", lw=0.5, zorder=3 * i + 1)
            for k, pk in enumerate(("baseline", "ssp126", "ssp585")):
                if (pk, f) not in RANGES:
                    continue
                m, p5, p95 = RANGES[(pk, f)]
                yl = y0 + 0.16 + 0.17 * k
                ax_d.plot([z(p5), z(p95)], [yl, yl], color=PCOL[pk], lw=1.3, solid_capstyle="butt", zorder=3 * i + 2)
                ax_d.plot([z(m)] * 2, [yl - 0.07, yl + 0.07], color="white", lw=0.6, zorder=3 * i + 3)
            ax_d.axhline(y0, color="0.75", lw=0.4, zorder=0)
        for xv in (0, 1):
            ax_d.axvline(xv, color="0.55", lw=0.4, ls=(0, (2, 2)), zorder=0)
        ax_d.set_yticks(range(nrow))
        ax_d.set_yticklabels([FLAB_D.get(f, f) for f in FEATS_D[::-1]], fontsize=6.3, va="bottom")
        ax_d.tick_params(axis="y", length=0, pad=1.5)
        ax_d.set_ylim(-0.1, nrow - 1 + HPK + 0.1)
        ax_d.set_xlim(XG[0], XG[-1])
        ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(["min", "max"], fontsize=6.3)
        ax_d.tick_params(axis="x", length=1.5, pad=1)
        ax_d.set_xlabel("Position in training range", fontsize=6.8, labelpad=1)
        ax_d.spines["left"].set_visible(False)
        hdl = [Patch(facecolor="#D9D9D9", edgecolor="0.45", linewidth=0.5, label="Training records")] + \
              [Line2D([], [], color=PCOL[k], lw=1.3, solid_capstyle="butt", label=PLAB[k])
               for k in ("baseline", "ssp126", "ssp585")]
        ax_d.legend(handles=hdl, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.0,
                    handlelength=1.2, columnspacing=0.9, borderaxespad=0.1)
        fig.text(fx(0.0) if LAY_B else fx(BX0 - 11.0), fy(ROW2_Y + ROW2_H + 3.0 if LAY_B else TOP - 39.5), RIDGE_LET,
                 fontsize=8, fontweight="bold", ha="left", va="top")

# ---------------- d 分析流程 ----------------
R2TXT = None
if LAY_C and "test_r2" in sk and sk.test_r2.notna().any():   # （C）R² 面板删去后，流程框里写测试集 R² 范围
    R2TXT = f"test R² {sk.test_r2.min():.2f}–{sk.test_r2.max():.2f}"
cv = draw_workflow(fig, 0.0, 0.0, W, H, W=W, H=WF_H, label=next(LET), skill_ref=SKILL_LET, skill_txt=R2TXT)
if cvA is not None:
    cvA.check()                                # a：照旧在放大前检查

# [1108d] 原来先检查 d、再在 F.save 里统一 ×1.15 放大（最小 6 pt），放大后的文字顶出框、互相贴住（Fig. 1d 互压）。
#         现在先放大，再按最终字号检查 d（必要时缩小，不低于 MIN_FONT_PT；仍放不下的粗体改常规），保存时不再重复放大
_SCALE = (F.FONT_SCALE, F.MIN_FONT_PT)
F.scale_fonts(fig)
cv.check(min_size=F.MIN_FONT_PT)
F.FONT_SCALE, F.MIN_FONT_PT = 1.0, 0.0
F.save(fig, OUT, "Fig1_data_framework", tight=not LAY_C)   # （C）版式按毫米排定，按画布原尺寸 180 mm 宽输出
F.FONT_SCALE, F.MIN_FONT_PT = _SCALE
print(f"[Fig. 1] 文字已在检查前统一放大（×{_SCALE[0]}，最小 {_SCALE[1]} pt），上面一行的 FONT_SCALE=1.0 指保存时不再放大")
