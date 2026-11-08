# ============================================================
# plot_bivariate_loss.py
# 双变量地图: 每格 [优质稻面积损失] × [经济损失], 3×3=9色, 带3×3图例。
# 4 个情形(ssp×ri)各出一张单独的图, 均含九段线。
#
# 每格两变量:
#   经济损失(亿元) = 读 loss_{ri}_{ssp}.tif (compute_production_loss.py 的输出)
#   优质面积损失(km²) = area_weight × [ 1(zone_his==3) − 1(zone_fut==3) ]
# 三档(low/median/high)断点 = 四张图合并后的 33.3/66.7 分位(同色可跨图比较)
# ============================================================
# [1004] 运行顺序 16（依赖 10、11、14）——配置来自 config_1004
# 输入：
#   out_dir("loss")/loss_{ri}_{ssp}.tif                                     （14 经济核算，亿元/格）
#   out_dir("quality_zones")/{PRED_MODE}/zone_{ri}_{baseline|ssp_future}.tif（11 品质分区）
#   out_dir("head_rice_yield")/head_rice_yield_{ri}_{historical|ssp}.tif    （10 整精米产量，t/ha）
#   output/area_weights/area_weight_{ri}_2000_2010.tif                      （km²）
#   phenology_corrected/historical_annual/{ri}_baseline_support_{HIS_P}.tif（共同支撑区）
# 输出（out_dir("loss")，默认 output/loss_1004/）：
#   bivariate_2x2_with_bar.{png,pdf,svg}（柱状图 = 单位面积损失）  bivariate_summary.csv（三个全国量）
#   province_loss.csv（宽表：面积、总损失、US$/ha、数量/等级效应 US$/ha）  province_loss_long.csv（省×情景×季别）
#   province_frac_1km.npz（各 0.5° 格在各省的 1 km 水稻面积占比，供 26 号使用）
# ============================================================
import numpy as np
import rasterio
from pathlib import Path
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import geopandas as gpd
import sys, os
import warnings; warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],"font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold","font.size":7})

# ── 全文统一配色: 同目录放 paper_palette.py ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_palette as PAL_P

# ── 本图专用配色(本地覆盖, 不改 paper_palette → 不影响其它图) ──
BIVAR_LOCAL = [                         # 9阶(参照文献图): 橙(面积损失)×紫(经济损失), 双高=深红, 双低=淡黄
    ["#F3EFA5", "#B9A9C9", "#6A3D9A"],   # 面积低: 淡黄 / 淡紫 / 鲜紫(经济高)
    ["#EAB94F", "#B08A87", "#7A4A93"],   # 面积中
    ["#E8912F", "#C0603C", "#8A2B2B"],   # 面积高: 橙 / 锈红 / 深红(双高, 与右下鲜紫拉开)
]
# BIVAR_LOCAL = [                         # 9阶: 橙(面积损失)×紫(经济损失), 双高=深栗
#     ["#f0f0a8", "#b6a6c9", "#5b2d82"],
#     ["#E4A73C", "#ad8f9c", "#6b3a7a"],
#     ["#e69138", "#b56a45", "#6e2d2d"],
# ]
BAR_C     = {"ssp126": "#2C6FB0", "ssp585": "#E8833A"}   # 柱: 蓝/橙(同 rice_change 直方图)
BAR_ALPHA = {"ssp126": 0.55,      "ssp585": 0.45}         # 透明度(同 rice_change)→ 统一浅色系
MAP_ALPHA = 0.65                                          # 地图/9阶图例 透明度 → 颜色更浅(同浅色系)
PIE_C = ["#E4A73C", "#4C9E8A", "#5b2d82"]            # 饼: 面积→橙, HRY→青绿, 经济→紫(与9阶紫呼应)

# ---------------- 配置 ----------------
from config_1004 import (PRED_MODE, HIS_P, HIS_LABEL, FUT_LABEL, AREA_DIR, HIS_ANN, CHINA_SHP as _CHINA,
                         POLY_SHP as _POLY, out_dir)
LOSS_DIR = out_dir("loss")                                 # loss_{ri}_{ssp}.tif 所在
ZONE_DIR = out_dir("quality_zones") / PRED_MODE
AW_DIR   = AREA_DIR
AW_NAME  = "area_weight_{ri}_2000_2010.tif"                # 1 km 真实种植面积聚合
HRY_DIR  = out_dir("head_rice_yield", thresh=False)                      # 整精米产量 tif(算全国HRY变化)
import config_1004 as _CFG                                   # [1013] 格网域、价格、汇率只在 config 定义
OUT_DIR  = LOSS_DIR;  OUT_DIR.mkdir(parents=True, exist_ok=True)

# 边界 + 九段线(与你采样点图同源: adchar=="JD" 为九段线)
CHINA_SHP  = _CHINA
POLY_SHP   = _POLY   # 九段线在这
NINEDASH_SHP   = POLY_SHP
NINEDASH_FIELD = "adchar"
NINEDASH_VALUE = "JD"

RI_TYPES = ["ri1", "ri2"]
SSPS     = ["ssp126", "ssp585"]
RI_L  = {"ri1":"Single/early rice", "ri2":"Late rice"}   # [1105] 与正文统一为 Single/early
SSP_L = {"ssp126":f"SSP1-2.6\n({FUT_LABEL})", "ssp585":f"SSP5-8.5\n({FUT_LABEL})"}
# 经济损失换算美元的汇率(可改成你采用的年份均值或PPP)
CNY_PER_USD = _CFG.CNY_PER_USD   # 与正文 Methods 4.7、Fig. 6 图注一致

# ============ 字体/尺寸 集中配置(全部可调) ============
FS = {
    # —— 主图元素: 13-15 ——
    "col_title":  15,    # 顶部列标题 (Single/Early rice, Late rice)
    "row_label":  15,    # 左侧行标签 (SSP1-2.6, SSP5-8.5)
    "map_tick":   13,    # 地图经纬度刻度 (°N/°E)
    "bar_tick":   13,    # 柱状图 x 轴省份缩写
    "bar_label":  14,    # 柱状图 y 轴标题
    "bar_title":  15,    # 柱状图标题
    "bar_legend": 10,    # 柱状图图例（[1106] 13→10，与刻度字号协调）
    # —— 小插图: 框小, 必须留小号(否则溢出); 唯饼图数字放大 ——
    "leg_tick":   7.0,   # 9阶图例刻度（数值断点）
    "leg_label":  7.5,   # 9阶图例轴标题
    "scale":      10,    # 比例尺标注
}
FS.update({"box": 8.5})                   # 左上角数值框字号
# [1017] 9 阶图例：正方形色块（按毫米定边长，不随子图长宽比变形）；轴标签用 Low / Medium / High，断点数值写进图注
LEG_CELL_MM  = 4.5                        # 每个色块边长（mm）；整个图例 = 3×3 个
LEG_ORIGIN   = (0.225, 0.21)              # 图例色块左下角（相对子图）：x 增大右移，y 增大上移（西藏/缅北无水稻区）
LEG_X_LABELS = ["Low", "Medium", "High"]  # 经济轴：四幅图合并的三分位
LEG_Y_STYLE  = "status"                   # 优质面积轴标签："status" = Gained/Unchanged/Lost（按格点优质状态，准确）
                                          #               "lmh"    = Low/Medium/High（同一含义：损失由低到高）
LEG_Y_LABELS = {"status": ["Gained", "Unchanged", "Lost"], "lmh": ["Low", "Medium", "High"]}
BOX_RECT    = (0.015, 0.56, 0.46, 0.42)   # 左上角数值框（[1017] 去掉 "vs. baseline" 行后缩短）
# 地图经济轴：
#   "per_ha" = 单位面积损失 US$ ha⁻¹ yr⁻¹（与柱状图一致，格点间可比，默认）
#   "total"  = 每格年均损失 百万美元 yr⁻¹（受格内水稻面积影响）
MAP_ECON = "per_ha"
# 优质面积轴：每格取值只有 −面积 / 0 / +面积 三种（整格升为优质 / 不变 / 降出优质），
#   三分位在此没有意义（大量 0 值时上下断点都为 0），改按符号分三类：Gained / Unchanged / Lost
RANK_BY  = "ssp126"                       # 柱状图与分省表的排序情景（两情景不相加、不平均）
LEG_STYLE = "boundary"                    # 图例经济轴："boundary" 断点数值标在分界处；"class" 每档写 <X / X–Y / >Y

# 省份英文→缩写(柱状图 x 轴); 不在表内则取前两位大写
PROV_ABBR = {"Jiangsu":"JS","Heilongjiang":"HLJ","Jiangxi":"JX","Liaoning":"LN","Guizhou":"GZ",
 "Jilin":"JL","Yunnan":"YN","Zhejiang":"ZJ","Shandong":"SD","Shanghai":"SH","Hunan":"HUN",
 "Hubei":"HB","Guangdong":"GD","Guangxi":"GX","Sichuan":"SC","Anhui":"AH","Henan":"HN",   # [1017] 河南=HN；湖南改 HUN 防重名
 "Fujian":"FJ","Chongqing":"CQ","Hebei":"HE","Beijing":"BJ","Tianjin":"TJ","Shanxi":"SX",
 "Inner Mongolia":"NM","Shaanxi":"SN","Gansu":"GS","Qinghai":"QH","Ningxia":"NX",
 "Xinjiang":"XJ","Tibet":"XZ","Hainan":"HI","Taiwan":"TW"}
def prov_abbr(name): return PROV_ABBR.get(name, str(name)[:2].upper())

# 0.5° 格(与 HRY / loss 一致)
LON = np.arange(73.25,136.0,0.5); LAT = np.arange(53.75,17.75,-0.5)
EXTENT = [LON[0]-0.25, LON[-1]+0.25, LAT[-1]-0.25, LAT[0]+0.25]

# 3×3 双变量色板 PAL[iy][ix]  蓝(经济损失, ix)×橙(优质面积损失, iy), 双高=深棕。
#   原为 紫×橙; 改为蓝×橙以并入全文色族并提升色盲安全性(紫为全文孤立色, 已弃用)。
#   ix=经济损失(→蓝), iy=优质(Premium)面积损失(→橙), 双高=深棕。
PAL = BIVAR_LOCAL
# -------------------------------------

def read_tif(p):
    p = Path(p)
    if not p.exists(): return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64"); nd = ds.nodata
        if nd is not None: a[a == nd] = np.nan
    return a

# ---- 收集四张图的两变量, 求全局三档断点; 同时算每情形三个全国量 ----
data = {}; stats = {}       # stats[(ssp,ri)] = (面积损失[10^3 km²], HRY变化[Mt], 经济损失[亿元])
econ_all, area_all = [], []
for ssp in SSPS:
    for ri in RI_TYPES:
        econ = read_tif(LOSS_DIR / f"loss_{ri}_{ssp}.tif")             # 亿元/格
        zh   = read_tif(ZONE_DIR / f"zone_{ri}_baseline.tif")
        zf   = read_tif(ZONE_DIR / f"zone_{ri}_{ssp}_future.tif")
        aw   = read_tif(AW_DIR   / AW_NAME.format(ri=ri))
        sm   = read_tif(_CFG.support_file(ri))                 # 统一格网域 D
        yh   = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_historical.tif")
        yf   = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_{ssp}.tif")
        if any(x is None for x in (econ, zh, zf, aw)):
            print(f"[skip] {ri} {ssp}: 缺文件"); continue
        valid = np.isfinite(econ) & np.isfinite(aw) & (aw > 0) & (zh > 0) & (zf > 0)
        if sm is not None: valid &= np.isfinite(sm)
        area_loss = aw * ((zh == 3).astype(float) - (zf == 3).astype(float))   # km², 正=丢优质
        if MAP_ECON == "per_ha":
            econ_m = np.where(valid, econ * 1e8 / (aw * 100.0) / CNY_PER_USD, np.nan)   # US$/ha
        else:
            econ_m = econ * 1e8 / CNY_PER_USD / 1e6                                    # 百万美元/格
        data[(ssp, ri)] = (econ_m, area_loss, valid)
        econ_all.append(econ_m[valid]); area_all.append(area_loss[valid])
        # 三个全国量
        area_kkm2 = float(np.nansum(area_loss[valid])) / 1e3               # 10^3 km² (正=优质面积减少)
        if yh is not None and yf is not None:
            vh = valid & np.isfinite(yh) & np.isfinite(yf)
            prod_h = float(np.nansum(aw[vh] * 100.0 * yh[vh]))            # 历史整精米产量(t)
            prod_f = float(np.nansum(aw[vh] * 100.0 * yf[vh]))            # 未来整精米产量(t)
            hry_pct = (prod_f - prod_h) / prod_h * 100 if prod_h > 0 else np.nan   # %变化, 正=增产
        else:
            hry_pct = np.nan
            print(f"[warn] 缺 HRY tif({ri} {ssp}), HRY扇形留空")
        econ_yi = float(np.nansum(econ[valid]))                            # 亿元 (正=损失)
        stats[(ssp, ri)] = (area_kkm2, hry_pct, econ_yi)

# ★ 三个全国量写表（正文引用）
import csv as _csv0
with open(OUT_DIR / "bivariate_summary.csv", "w", newline="", encoding="utf-8-sig") as _f:
    _w = _csv0.writer(_f)
    _w.writerow(["ssp", "ri", "premium_area_loss_1e3km2", "head_rice_production_change_pct", "economic_loss_1e8CNY"])
    for (ssp, ri), (a_, h_, e_) in stats.items():
        _w.writerow([ssp, ri, round(a_, 2), round(h_, 2), round(e_, 2)])
        print(f"[{ssp}/{ri}] 优质面积损失 {a_:.1f}×10³ km²  整精米总产 {h_:+.1f}%  经济损失 {e_:.1f} 亿元")
print(f"[saved] {OUT_DIR / 'bivariate_summary.csv'}")

econ_all = np.concatenate(econ_all); area_all = np.concatenate(area_all)
ex_brk = np.nanpercentile(econ_all, [100/3, 200/3])     # 经济轴：四幅图合并后的三分位断点
ECON_UNIT = r"US\$ ha$^{-1}$ yr$^{-1}$" if MAP_ECON == "per_ha" else r"US\$ M yr$^{-1}$"   # [1106] 上标用 mathtext，Arial 无 ⁻ 字形
_fmt = ((lambda v: f"{v:,.0f}".replace("-", "\u2212")) if MAP_ECON == "per_ha"
        else (lambda v: f"{v:,.2f}".replace("-", "\u2212")))
EX_LAB = [f"<{_fmt(ex_brk[0])}", f"{_fmt(ex_brk[0])}\u2013{_fmt(ex_brk[1])}", f">{_fmt(ex_brk[1])}"]
EY_LAB = ["Gained", "Unchanged", "Lost"]    # 优质稻面积：格点升为优质 / 不变（含始终优质与始终非优质）/ 降出优质
_ex = np.digitize(econ_all, ex_brk); _ey = np.sign(area_all).astype(int) + 1
print(f"经济轴三分位断点（{ECON_UNIT}）: {ex_brk.round(2)}  → 图例 {EX_LAB}")
print(f"  其中最低档里仍为损失(>0)的格点占该档 {100*np.mean(econ_all[_ex == 0] > 0):.0f}%"
      f"；为增值(<0)的占 {100*np.mean(econ_all[_ex == 0] < 0):.0f}%")
print("优质面积轴（按符号）: " + "  ".join(f"{l} {100*np.mean(_ey == i):.0f}%" for i, l in enumerate(EY_LAB)))
with open(OUT_DIR / "bivariate_breaks.csv", "w", newline="", encoding="utf-8-sig") as _f:   # 图注用
    _w = _csv0.writer(_f)
    _w.writerow(["axis", "class", "label", "lower", "upper", "unit", "share_of_cells_pct"])
    lims = [(-np.inf, ex_brk[0]), (ex_brk[0], ex_brk[1]), (ex_brk[1], np.inf)]
    for i, ((lo, hi), lab) in enumerate(zip(lims, EX_LAB)):
        _w.writerow(["economic", i, lab, round(lo, 2), round(hi, 2), ECON_UNIT, round(100*np.mean(_ex == i), 1)])
    for i, lab in enumerate(EY_LAB):
        _w.writerow(["premium_rice_area", i, lab, "", "", "premium status of the cell: gained / unchanged / lost",
                     round(100*np.mean(_ey == i), 1)])
print(f"[saved] {OUT_DIR / 'bivariate_breaks.csv'}")

# 预转 RGBA
PAL_RGBA = np.array([[to_rgba(PAL[iy][ix]) for ix in range(3)] for iy in range(3)])

# 边界 + 南海(九段线 jd_line + 岛礁 nanhai), 沿用你采样点图的读法
china   = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
polygon = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
nanhai  = polygon[polygon.geometry.apply(lambda g: g.bounds[1] < 25)]
jd_line = polygon[polygon[NINEDASH_FIELD] == NINEDASH_VALUE]
lon_min, lat_min, lon_max, lat_max = china.total_bounds
xmin, ymin, xmax, ymax = lon_min, lat_min, lon_max, lat_max
print(f"九段线要素 {len(jd_line)} 条; 南海要素 {len(nanhai)} 个")

NANHAI_EXT = [105, 123, 2, 24]      # 南海小图范围
SCALE_KM = 1000                     # 比例尺长度(km)
LAT_REF  = 35                       # 比例尺参考纬度(算km→经度)

def bivariate_rgb(econ, area, valid):
    ix = np.digitize(econ, ex_brk)                       # 0/1/2：经济轴三分位
    iy = np.sign(np.nan_to_num(area)).astype(int) + 1    # 0/1/2：优质面积 增加/不变/减少
    H, W = econ.shape
    rgb = np.zeros((H, W, 4))        # 默认透明
    for yy in range(3):
        for xx in range(3):
            sel = valid & (ix == xx) & (iy == yy)
            rgb[sel] = PAL_RGBA[yy, xx]
    rgb[..., 3] *= MAP_ALPHA          # 有效格调浅(无效格 alpha=0 不变)
    return rgb

def add_nanhai_inset(ax, rgb, rect=(0.80, 0.02, 0.185, 0.34)):
    """子图内·右下角南海小图(缩小): 同色栅格(含海南) + 九段线 + 岛礁"""
    iax = ax.inset_axes(rect)                       # 相对该子图坐标, 落在地图内
    iax.imshow(rgb, extent=EXTENT, origin="upper", aspect="auto", zorder=1)
    china.boundary.plot(ax=iax, linewidth=0.2, color="black", zorder=5)
    if len(nanhai):  nanhai.boundary.plot(ax=iax, linewidth=0.2, color="black", zorder=5)
    if len(jd_line): jd_line.plot(ax=iax, linewidth=0.4, color="black", zorder=6)
    iax.set_xlim(NANHAI_EXT[0], NANHAI_EXT[1]); iax.set_ylim(NANHAI_EXT[2], NANHAI_EXT[3])
    iax.set_xticks([]); iax.set_yticks([]); iax.set_xlabel(""); iax.set_ylabel("")
    for s in iax.spines.values(): s.set_linewidth(0.5)

def add_legend(ax, origin=LEG_ORIGIN, cell_mm=LEG_CELL_MM):
    """[1017] 子图内·左下角 3×3 双变量图例：正方形色块；x = 经济损失三分位（Low/Medium/High），
    y = 优质面积损失（Gained/Unchanged/Lost，自下而上损失增加）；断点数值见 bivariate_breaks.csv，写进图注"""
    fig = ax.figure
    pos = ax.get_position()                                        # 子图物理尺寸 → 色块保持正方形
    w = 3 * cell_mm / (pos.width * fig.get_figwidth() * 25.4)
    h = 3 * cell_mm / (pos.height * fig.get_figheight() * 25.4)
    lax = ax.inset_axes((origin[0], origin[1], w, h))
    lax.imshow(PAL_RGBA, origin="lower", aspect="auto", alpha=MAP_ALPHA, extent=(0, 3, 0, 3))
    for k in (1, 2):                                               # 色块之间留白线
        lax.axhline(k, color="white", linewidth=0.6); lax.axvline(k, color="white", linewidth=0.6)
    lax.set_xlim(0, 3); lax.set_ylim(0, 3)
    lax.set_xticks([0.5, 1.5, 2.5])
    lax.set_xticklabels(LEG_X_LABELS, fontsize=FS["leg_tick"], rotation=40, ha="right", rotation_mode="anchor")
    lax.set_yticks([0.5, 1.5, 2.5])
    lax.set_yticklabels(LEG_Y_LABELS[LEG_Y_STYLE], fontsize=FS["leg_tick"],
                        rotation=40, ha="right", va="center", rotation_mode="anchor")   # [1017] 纵轴标签同样斜 40°
    lax.tick_params(length=0, pad=2)
    lax.set_xlabel("Economic loss \u2192", fontsize=FS["leg_label"], labelpad=2)
    lax.set_ylabel("Premium-area loss \u2192", fontsize=FS["leg_label"], rotation=90, labelpad=2)   # [1017] 竖排
    for s in lax.spines.values(): s.set_visible(False)

def add_scale_bar(ax, x_left, cy, length_km=SCALE_KM, lat_ref=LAT_REF, n_seg=4, h=0.0040):
    """在 ax 底部画黑白分段比例尺(figure 坐标); 只在需要的子图调用一次。"""
    fig = ax.figure
    deg = length_km / (111.320 * np.cos(np.deg2rad(lat_ref)))
    x_lo, x_hi = ax.get_xlim(); pos = ax.get_position()
    w = deg / (x_hi - x_lo) * pos.width; seg = w / n_seg
    for i in range(n_seg):
        fig.patches.append(Rectangle((x_left + i*seg, cy), seg, h,
                           facecolor="black" if i % 2 == 0 else "white",
                           edgecolor="black", linewidth=0.4,
                           transform=fig.transFigure, zorder=10, figure=fig))
    fig.text(x_left + w + 0.006, cy + h/2, f"{length_km:,} km",
             ha="left", va="center", fontsize=FS["scale"], zorder=10)

def add_value_box(ax, area_kkm2, hry_pct, econ_yi, rect=BOX_RECT):
    """子图内·左上角数值框（替代三等分饼图，#12）：三个全国量直接写数值（相对基准期）。
    Δ Premium rice area（Mha，负 = 减少） Δ Head rice yield（%，正 = 增加） Economic loss（US$ bn yr⁻¹，正 = 损失）"""
    from matplotlib.patches import FancyBboxPatch
    d_area = -area_kkm2 / 10.0                          # 10³ km² → Mha；area_kkm2 正 = 优质面积减少
    loss   = econ_yi / (10.0 * CNY_PER_USD)              # 亿元 → 十亿美元；正 = 损失
    sg = lambda x, nd: f"{x:+.{nd}f}".replace("-", "\u2212") if np.isfinite(x) else "n/a"
    rows = [("\u0394 Premium rice area", f"{sg(d_area, 2)} Mha"),
            ("\u0394 Head rice yield", f"{sg(hry_pct, 1)} %"),
            ("Economic loss", (f"US\\${loss:.2f} bn yr$^{{-1}}$".replace("\\$-", "\\$\u2212")
                               if np.isfinite(loss) else "n/a"))]
    iax = ax.inset_axes(rect); iax.set_xlim(0, 1); iax.set_ylim(0, 1); iax.axis("off")
    iax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.04",
                                 facecolor="white", edgecolor="0.6", linewidth=0.5, alpha=0.85,
                                 transform=iax.transAxes, zorder=1))
    # [1017] 'vs. 1981–2010 baseline' 移到图注
    for k, ((lab, val), col) in enumerate(zip(rows, PIE_C)):
        y = 0.80 - k * 0.30                                    # 每项两行：名称 / 数值（[1017] 无表头后均匀排布）
        iax.add_patch(Rectangle((0.05, y - 0.13), 0.06, 0.18, facecolor=col, edgecolor="none",
                                alpha=MAP_ALPHA, transform=iax.transAxes, zorder=2))
        iax.text(0.15, y + 0.03, lab, ha="left", va="center", fontsize=FS["box"], zorder=3)
        iax.text(0.15, y - 0.095, val, ha="left", va="center", fontsize=FS["box"], fontweight="bold", zorder=3)

# ================= 省级经济损失（底部柱状图用；按 1 km 水稻像元拆分到省）=================
from rasterio.features import rasterize
PROV_FIELD = None    # None=自动探测省名字段; 或手动指定, 如 "NAME"
_cand = ["NAME","name","PR_NAME","pr_name","NAME_1","EN_NAME","ename","PYNAME",
         "省","省名","adname","FCNAME","PROVINCE","Province"]
if PROV_FIELD is None:
    for c in _cand:
        if c in china.columns: PROV_FIELD = c; break
CN2EN = {"北京":"Beijing","天津":"Tianjin","河北":"Hebei","山西":"Shanxi","内蒙古":"Inner Mongolia",
 "辽宁":"Liaoning","吉林":"Jilin","黑龙江":"Heilongjiang","上海":"Shanghai","江苏":"Jiangsu",
 "浙江":"Zhejiang","安徽":"Anhui","福建":"Fujian","江西":"Jiangxi","山东":"Shandong","河南":"Henan",
 "湖北":"Hubei","湖南":"Hunan","广东":"Guangdong","广西":"Guangxi","海南":"Hainan","重庆":"Chongqing",
 "四川":"Sichuan","贵州":"Guizhou","云南":"Yunnan","西藏":"Tibet","陕西":"Shaanxi","甘肃":"Gansu",
 "青海":"Qinghai","宁夏":"Ningxia","新疆":"Xinjiang","台湾":"Taiwan","香港":"Hong Kong","澳门":"Macao"}
def cn2en(s):
    s = str(s)
    for k, v in CN2EN.items():
        if k in s: return v
    return s

# 每个 0.5° 格的损失，按格内 1 km 水稻像元面积落在各省的比例拆分到省。
#   1 km 像元与 area_weight 同源（make_rice_area.py：1 km 物候产品有效 HE 像元，2000–2010 平均），
#   跨省界、沿海格点不再整格归给一个省；省界用 CHINA_SHP（与地图同一文件），同名多要素按省名合并。
#   兜底：极少数格点在省界内没有 1 km 水稻像元 → 按格点中心归入最近省份（打印个数）。
from rasterio.features import rasterize
from pyproj import Transformer
from pheno_common_1004 import OBS_DIR, OBS_PAT
AREA_YEARS = range(2000, 2011)                          # 与 area_weight_{ri}_2000_2010 一致
AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m"


def _pix_area_rows(ds):
    """每行源像元面积（km²），同 make_rice_area.pixel_area_km2"""
    t = ds.transform
    if ds.crs.is_geographic:
        r = np.arange(ds.height)
        la, lb = np.deg2rad(t.f + r * t.e), np.deg2rad(t.f + (r + 1) * t.e)
        return 6371.0088 ** 2 * np.deg2rad(abs(t.a)) * np.abs(np.sin(la) - np.sin(lb))
    try:    f = ds.crs.linear_units_factor[1]
    except Exception: f = 1.0
    return np.full(ds.height, abs(t.a * t.e) * f * f / 1e6)


def prov_area_1km(ri, pg, pidx, K, gtrans, GH, GW):
    """A[k, GH, GW]：0.5° 格内落在第 k 个省的 1 km 水稻面积（km²，多年平均）；k=K 为省界外像元"""
    A = np.zeros((K + 1) * GH * GW); n = 0; cache = {}
    for yr in AREA_YEARS:
        fp = OBS_DIR / OBS_PAT[ri]["HE"].format(year=yr)
        if not fp.exists():
            print(f"    ⚠️ 缺 {fp.name}"); continue
        with rasterio.open(fp) as ds:
            src = ds.read(1).astype(np.float32)
            ok = np.isfinite(src) & (src > 0)
            if ds.nodata is not None: ok &= src != np.float32(ds.nodata)
            key = (tuple(ds.transform), ds.height, ds.width, ds.crs.to_string())
            if key not in cache:                        # 省界栅格化到 1 km 网格（同一网格只做一次）
                pid = rasterize(((g, pidx[i]) for i, g in enumerate(pg.to_crs(ds.crs).geometry)),
                                out_shape=src.shape, transform=ds.transform, fill=K, dtype="int16")
                cache = {key: pid}
            pid = cache[key]
            rr, cc = np.nonzero(ok)
            a = _pix_area_rows(ds)[rr]
            xs, ys = ds.transform * (cc + 0.5, rr + 0.5)
            if not (ds.crs.is_geographic and ds.crs.to_epsg() == 4326):
                xs, ys = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
        col = np.floor((np.asarray(xs) - gtrans.c) / gtrans.a).astype(int)
        row = np.floor((np.asarray(ys) - gtrans.f) / gtrans.e).astype(int)
        inb = (row >= 0) & (row < GH) & (col >= 0) & (col < GW)
        k = pid[rr, cc].astype(int)
        A += np.bincount(((k * GH + row) * GW + col)[inb], weights=a[inb], minlength=(K + 1) * GH * GW)
        n += 1
    return (A / max(n, 1)).reshape(K + 1, GH, GW), n


# ---- 单位面积损失 / 分解 所需常数（与 14 号完全一致）----
P_ORD, P_HQ  = _CFG.P_ORD, _CFG.P_HQ   # 元/kg：普通（中等+普通）/ 优质
UNIT, YI     = 1e5, 1e8            # HRY[t/ha]·area[km²]·price[元/kg] → 元；元 → 亿元
MIN_AREA_KHA = 500                 # 重点省份：播种面积 ≥ 0.5 Mha（ri1+ri2）才单列并参与排名
# 播种面积 < 0.5 Mha 的省份，以及非水稻主产区（训练集无样本，均 < 0.5 Mha）不单列，
#   合并为一行 OTHER；仍计入全国总量，各行加总 = 全国
NON_MAJOR = ["Ningxia", "Shaanxi", "Shanxi", "Shandong", "Hebei", "Beijing", "Tianjin", "Inner Mongolia",
             "Gansu", "Xinjiang", "Qinghai", "Tibet", "Hong Kong", "Macao", "Taiwan"]
OTHER = "Other provinces (<0.5 Mha)"
BAR_TOPN = 10                      # 柱状图显示前 N 个重点省份；None = 全部重点省份（含南方增值省）


def _price(z):
    p = np.full(z.shape, np.nan); p[(z == 1) | (z == 2)] = P_ORD; p[z == 3] = P_HQ
    return p


prov_loss = {ssp: {} for ssp in SSPS}; prov_names = {}; prov_ha = {ssp: {} for ssp in SSPS}
nat_ha = {}; rank_names = []
ref_tif = next((LOSS_DIR/f"loss_{ri}_{ssp}.tif" for ssp in SSPS for ri in RI_TYPES
                if (LOSS_DIR/f"loss_{ri}_{ssp}.tif").exists()), None)
if ref_tif is not None and PROV_FIELD is not None:
    import pandas as _pd
    with rasterio.open(ref_tif) as ds:
        gtrans, GH, GW = ds.transform, ds.height, ds.width
    pg = china[china.geometry.notna() & ~china.geometry.is_empty
               & china.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    pnames = [cn2en(nm) for nm in pg[PROV_FIELD].values]
    U = sorted(set(pnames)); K = len(U); pidx = [U.index(n) for n in pnames]   # 同名要素 → 同一省
    prov_names = {n: n for n in U}
    cpt = gpd.GeoSeries(pg.geometry.values, crs=pg.crs).to_crs(AEA)

    # ---- 1) 每个 0.5° 格在各省的面积占比 frac[k]（1 km 水稻像元；兜底格整格归最近省）----
    FRAC = {}; n_fb = np.zeros(K, int)
    for ri in RI_TYPES:
        A, ny = prov_area_1km(ri, pg, pidx, K, gtrans, GH, GW)
        Ain = A[:K].sum(0)
        frac = np.divide(A[:K], Ain, out=np.zeros_like(A[:K]), where=Ain > 0)
        aw = read_tif(AW_DIR / AW_NAME.format(ri=ri))
        msg = f"  [{ri}] 1 km 水稻面积 {A.sum()/1e4:.3f} Mha（{ny} 年平均），其中省界外像元 {A[K].sum()/max(A.sum(),1e-9):.2%}"
        print(msg + (f"；area_weight 合计 {np.nansum(aw)/1e4:.3f} Mha" if aw is not None else ""))
        need = np.isfinite(aw) & (aw > 0) if aw is not None else np.zeros((GH, GW), bool)
        for ssp in SSPS:
            L = read_tif(LOSS_DIR / f"loss_{ri}_{ssp}.tif")
            if L is not None and L.shape == (GH, GW): need |= np.isfinite(L)
        fb = need & ~(Ain > 0)
        if fb.any():
            rr, cc = np.nonzero(fb)
            xs, ys = gtrans * (cc + 0.5, rr + 0.5)
            pts = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs="EPSG:4326").to_crs(AEA)
            near = np.array([pidx[int(np.argmin(cpt.distance(p).values))] for p in pts])
            frac[near, rr, cc] = 1.0
            np.add.at(n_fb, near, 1)
            print(f"    {int(fb.sum())} 个格点在省界内没有 1 km 水稻像元，整格归入最近省份")
        FRAC[ri] = frac

    # ---- 2) 分省：面积、总损失、数量效应、等级效应（逐格算好后按 frac 拆分）----
    rec = []
    for ri in RI_TYPES:
        aw = read_tif(AW_DIR / AW_NAME.format(ri=ri))
        zh = read_tif(ZONE_DIR / f"zone_{ri}_baseline.tif")
        yh = read_tif(HRY_DIR / f"head_rice_yield_{ri}_historical.tif")
        for ssp in SSPS:
            L  = read_tif(LOSS_DIR / f"loss_{ri}_{ssp}.tif")
            zf = read_tif(ZONE_DIR / f"zone_{ri}_{ssp}_future.tif")
            yf = read_tif(HRY_DIR / f"head_rice_yield_{ri}_{ssp}.tif")
            if any(x is None for x in (L, aw, zh, zf, yh, yf)):
                print(f"[skip] 分省 {ri} {ssp}: 缺文件"); continue
            dom = np.isfinite(L)                                   # = 14 号的有效域
            ph, pf = _price(zh), _price(zf)
            L0   = np.where(dom, L, 0.0)
            qty  = np.nan_to_num(np.where(dom, aw * UNIT * ph * (yh - yf) / YI, 0.0))   # 数量效应（亿元）
            grd  = np.nan_to_num(np.where(dom, aw * UNIT * yf * (ph - pf) / YI, 0.0))   # 等级效应（亿元）
            area = np.nan_to_num(np.where(dom, aw, 0.0))                                # km²
            dmax = float(np.max(np.abs(qty + grd - L0)))
            if dmax > 1e-3: print(f"    ⚠️ {ri} {ssp}: 数量+等级 与 loss 栅格最大差 {dmax:.4f} 亿元/格")
            F = FRAC[ri]
            for k in range(K):
                rec.append(dict(province=U[k], ssp=ssp, ri=ri,
                                area_km2=float((F[k] * area).sum()), loss_yi=float((F[k] * L0).sum()),
                                qty_yi=float((F[k] * qty).sum()), grade_yi=float((F[k] * grd).sum())))
    D = _pd.DataFrame(rec)
    # 省 → 展示分组：面积 < MIN_AREA_KHA 或非主产区 → OTHER（分组写进 npz，26、27 号沿用）
    _area = D[D.ssp == SSPS[0]].groupby("province").area_km2.sum() / 10.0           # kha，ri1+ri2
    GRP = np.array([OTHER if (n in NON_MAJOR or _area.get(n, 0.0) < MIN_AREA_KHA) else n for n in U])
    np.savez_compressed(OUT_DIR / "province_frac_1km.npz", names=np.array(U), groups=GRP,
                        transform=np.array(tuple(gtrans)[:6]), shape=np.array([GH, GW]),
                        **{f"frac_{ri}": FRAC[ri].astype(np.float32) for ri in RI_TYPES})
    D["province"] = D.province.map(dict(zip(U, GRP)))
    D = D.groupby(["province", "ssp", "ri"], as_index=False)[["area_km2", "loss_yi", "qty_yi", "grade_yi"]].sum()
    G = sorted(set(GRP) - {OTHER}) + [OTHER]
    n_fb_g = {g: int(sum(n_fb[k] for k in range(K) if GRP[k] == g)) for g in G}
    prov_names = {n: n for n in G if n != OTHER}                          # 柱状图只用主产区
    num = ["area_km2", "loss_yi", "qty_yi", "grade_yi"]
    D = _pd.concat([D, D.groupby(["province", "ssp"], as_index=False)[num].sum().assign(ri="all")], ignore_index=True)
    N = D.groupby(["ssp", "ri"], as_index=False)[num].sum().assign(province="National")
    D = _pd.concat([D, N], ignore_index=True)
    ha = D.area_km2 * 100.0
    for c in ["loss", "qty", "grade"]:                                   # 亿元 → 元/ha → US$/ha
        D[f"{c}_USD_ha"] = np.where(ha > 0, D[f"{c}_yi"] * YI / ha.where(ha > 0) / CNY_PER_USD, np.nan)
    D["area_kha"] = D.area_km2 / 10.0

    # 一致性检查：各省合计 = 全国（loss 栅格总和）= 14 号
    _ref = LOSS_DIR / "production_loss_summary.csv"
    _summ = _pd.read_csv(_ref) if _ref.exists() else None
    for ssp in SSPS:
        sub = D[(D.ssp == ssp) & (D.ri == "all") & (D.province != "National")]
        ps = float(sub.loss_yi.sum())
        nat = float(sum(np.nansum(read_tif(LOSS_DIR / f"loss_{ri}_{ssp}.tif")) for ri in RI_TYPES))
        msg = f"  [一致性] {ssp}: 各省合计 {ps:.2f} 亿元  栅格全国 {nat:.2f} 亿元"
        ok = abs(ps - nat) <= max(0.05, 1e-4 * abs(nat))
        if _summ is not None:
            r14 = float(_summ[(_summ.ssp == ssp) & (_summ.ri == "ri1+ri2")].loss_yi.iloc[0])
            msg += f"  14号 {r14:.2f} 亿元"; ok &= abs(ps - r14) <= max(0.5, 0.005 * abs(r14))
        print(msg + ("  ✔" if ok else "  ✘ 不一致，请检查"))

    # 长表：省 × 情景 × 季别（ri1 / ri2 / all），含 National 行
    out_long = D[["province", "ssp", "ri", "area_kha", "loss_yi", "qty_yi", "grade_yi",
                  "loss_USD_ha", "qty_USD_ha", "grade_USD_ha"]].round(4)
    out_long = out_long[(out_long.area_kha > 0) | (out_long.loss_yi != 0)]
    out_long.sort_values(["province", "ssp", "ri"]).to_csv(OUT_DIR / "province_loss_long.csv",
                                                           index=False, encoding="utf-8-sig")
    # 宽表（ri1+ri2）：重点省份（≥ 0.5 Mha）按 RANK_BY 情景的单位面积损失排序（两情景不相加、不平均），其后为 OTHER 与 National
    A_ = D[D.ri == "all"].set_index(["province", "ssp"])
    rows = []
    for n in G + ["National"]:
        r = {"province": n}
        for s in SSPS:
            if (n, s) not in A_.index: continue
            x = A_.loc[(n, s)]
            r.update({f"{s}_area_kha": round(x.area_kha, 1), f"{s}_yi": round(x.loss_yi, 3),
                      f"{s}_bnUSD": round(x.loss_yi / (10.0 * CNY_PER_USD), 4),
                      f"{s}_USD_ha": round(x.loss_USD_ha, 1), f"{s}_qty_USD_ha": round(x.qty_USD_ha, 1),
                      f"{s}_grade_USD_ha": round(x.grade_USD_ha, 1)})
        if len(r) == 1 or not r.get(f"{SSPS[0]}_area_kha", 0) > 0: continue
        r["in_ranking"] = bool(n not in ("National", OTHER) and r[f"{SSPS[0]}_area_kha"] >= MIN_AREA_KHA)
        r["fallback_cells"] = n_fb_g.get(n, 0)
        rows.append(r)
    W = _pd.DataFrame(rows)
    W["_o"] = np.where(W.province == "National", 3, np.where(W.province == OTHER, 2, np.where(W.in_ranking, 0, 1)))
    W = W.sort_values(["_o", f"{RANK_BY}_USD_ha"], ascending=[True, False]).drop(columns="_o")
    W.to_csv(OUT_DIR / "province_loss.csv", index=False, encoding="utf-8-sig")
    print(f"[saved] {OUT_DIR/'province_loss.csv'}（宽表）  {OUT_DIR/'province_loss_long.csv'}（长表）  province_frac_1km.npz")

    # 供柱状图使用
    for s in SSPS:
        prov_loss[s] = {n: float(A_.loc[(n, s)].loss_yi) for n in G if n != OTHER and (n, s) in A_.index}
        prov_ha[s]   = {n: float(A_.loc[(n, s)].loss_USD_ha) for n in G if n != OTHER and (n, s) in A_.index}
        nat_ha[s]    = float(A_.loc[("National", s)].loss_USD_ha)
    rank_names = W[W.in_ranking].province.tolist()
    print(f"  合并为 {OTHER}：{', '.join(n for n, g in zip(U, GRP) if g == OTHER and _area.get(n, 0) > 0)}")
    print(f"  单位面积损失（US$/ha，播种面积 ≥ 0.5 Mha 的省份，按 {RANK_BY} 排序）：")
    print(W[W.in_ranking | W.province.isin(["National", OTHER])][["province", f"{SSPS[0]}_area_kha"]
          + [f"{s}_USD_ha" for s in SSPS] + [f"{s}_qty_USD_ha" for s in SSPS]
          + [f"{s}_grade_USD_ha" for s in SSPS]].to_string(index=False))
else:
    print(f"[warn] 省名字段未找到(china.columns={list(china.columns)}) 或缺 loss tif; 柱状图跳过")

# ================= 组合图: 上 2×2 地图 + 下 堆叠柱状图 =================
LON_TICKS = [80, 100, 120]; LAT_TICKS = [20, 30, 40, 50]
lon_fmt = FuncFormatter(lambda v, p: f"{int(round(v))}\u00b0E")
lat_fmt = FuncFormatter(lambda v, p: f"{int(round(v))}\u00b0N")
nrow, ncol = len(SSPS), len(RI_TYPES)

fig = plt.figure(figsize=(180/25.4, 210/25.4), dpi=300)
gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 0.10, 0.72],   # 第3行为空隙
                      hspace=0.06, wspace=0.03,
                      left=0.07, right=0.995, top=0.965, bottom=0.10)
map_ax = [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)]

for i, ssp in enumerate(SSPS):
    for j, ri in enumerate(RI_TYPES):
        ax = map_ax[i][j]
        if (ssp, ri) not in data:
            ax.axis("off"); continue
        econ, area, valid = data[(ssp, ri)]
        rgb = bivariate_rgb(econ, area, valid)
        ax.imshow(rgb, extent=EXTENT, origin="upper", aspect="auto", zorder=1)
        china.boundary.plot(ax=ax, linewidth=0.3, color="black", zorder=5)
        ax.set_xlim(xmin-0.5, xmax+0.5); ax.set_ylim(ymin-0.5, ymax+0.5)
        ax.set_xticks(LON_TICKS); ax.set_yticks(LAT_TICKS)
        ax.xaxis.set_major_formatter(lon_fmt); ax.yaxis.set_major_formatter(lat_fmt)
        ax.tick_params(labelsize=FS["map_tick"], length=2,
                       labelbottom=(i == nrow-1), labelleft=(j == 0))
        ax.set_xlabel(""); ax.set_ylabel("")
        if i == 0: ax.set_title(RI_L[ri], fontsize=FS["col_title"] - 1)   # [1105] 列标题不加粗
        if j == 0: ax.set_ylabel(SSP_L[ssp], fontsize=FS["row_label"])   # [1106] 不加粗
        add_nanhai_inset(ax, rgb)
        if (ssp, ri) in stats:
            add_value_box(ax, *stats[(ssp, ri)])   # 左上角数值框（替代三等分饼图）
add_legend(map_ax[-1][0])

# 左上角(ssp126 single/early)加比例尺
_p0 = map_ax[0][0].get_position()
add_scale_bar(map_ax[0][0], x_left=_p0.x0 + 0.03, cy=_p0.y0 + 0.015)

# --- 底部分组柱状图: 单位面积年均损失（两情景并列，不相加、不平均）---
#   BAR_METRIC = "per_ha"：US$ ha⁻¹ yr⁻¹，播种面积 ≥ 0.5 Mha 的省份，按 RANK_BY 情景排序，虚线 = 全国平均
#   BAR_METRIC = "total" ：十亿美元 yr⁻¹（旧版总量口径，留作对照）
BAR_METRIC = "per_ha"
ax_bar = fig.add_subplot(gs[3, :])
if prov_names:
    if BAR_METRIC == "per_ha":
        val  = prov_ha
        pool = rank_names                                              # 已按 RANK_BY 情景排序
        top  = pool[:BAR_TOPN] if BAR_TOPN else pool
        ylab = "Economic loss\n(US\\$ ha$^{-1}$ yr$^{-1}$)"
        _rl  = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}[RANK_BY]
        ttl  = "Economic loss per hectare"                       # [1017] 前 N 省、排序情景写进图注
        # 图注写明：provinces with ≥ 0.5 Mha sown rice area；两情景为平行情景，不相加
        conv = 1.0
    else:
        val  = prov_loss
        tot  = {n: val[RANK_BY].get(n, 0) for n in prov_names}        # 按单一情景排序
        top  = sorted(tot, key=lambda k: tot[k], reverse=True)[:(BAR_TOPN or len(tot))]
        ylab = "Annual loss\n(billion US\\$ yr$^{-1}$)"
        ttl  = f"Provinces by annual economic loss, ordered by {RANK_BY}"
        conv = 1.0 / (10.0 * CNY_PER_USD)                            # 亿元 → billion US$
    b126 = [val["ssp126"].get(n, 0) * conv for n in top]
    b585 = [val["ssp585"].get(n, 0) * conv for n in top]
    x = np.arange(len(top))
    # ★ 蓝/橙 色盲安全对: SSP1-2.6=蓝, SSP5-8.5=橙。
    BW = 0.38
    # [1106] 若 24b 号已生成 25 成员的分省结果，则画箱线图（成员分布）+ 三角 = 集合平均估计；否则退回柱状图
    _MEM = _CFG.out_dir("sensitivity") / "province_loss_members.csv"
    _mem = None
    if BAR_METRIC == "per_ha" and _MEM.exists():
        import pandas as _pdm
        _mem = _pdm.read_csv(_MEM)
    if _mem is not None:
        BWB = 0.32
        for s_, off in (("ssp126", -BW / 2), ("ssp585", BW / 2)):
            data = [_mem[(_mem.ssp == s_) & (_mem.province == n)].usd_ha.values for n in top]
            bp = ax_bar.boxplot(data, positions=x + off, widths=BWB, patch_artist=True, showfliers=False,
                                whis=(0, 100), manage_ticks=False, zorder=3,
                                boxprops=dict(facecolor=to_rgba(BAR_C[s_], 0.25), edgecolor=BAR_C[s_], linewidth=0.8),
                                medianprops=dict(color=BAR_C[s_], linewidth=1.2),
                                whiskerprops=dict(color=BAR_C[s_], linewidth=0.8),
                                capprops=dict(color=BAR_C[s_], linewidth=0.8))
            # [1106b] 25 个成员以抖动散点画在箱体内（固定种子，可复现）
            _rng = np.random.default_rng(0)
            for xi, v in zip(x + off, data):
                jit = _rng.uniform(-BWB * 0.35, BWB * 0.35, size=len(v))
                ax_bar.scatter(xi + jit, v, s=5, color=BAR_C[s_], alpha=0.7, linewidths=0, zorder=4)
        _hdl = [Rectangle((0, 0), 1, 1, facecolor=to_rgba(BAR_C["ssp126"], 0.25), edgecolor=BAR_C["ssp126"], label="SSP1-2.6"),
                Rectangle((0, 0), 1, 1, facecolor=to_rgba(BAR_C["ssp585"], 0.25), edgecolor=BAR_C["ssp585"], label="SSP5-8.5")]
    else:
        ax_bar.bar(x - BW / 2, b126, width=BW, color=BAR_C["ssp126"], alpha=BAR_ALPHA["ssp126"],
                   edgecolor="white", linewidth=0.3, label="SSP1-2.6", zorder=3)
        ax_bar.bar(x + BW / 2, b585, width=BW, color=BAR_C["ssp585"], alpha=BAR_ALPHA["ssp585"],
                   edgecolor="white", linewidth=0.3, label="SSP5-8.5", zorder=3)
        _hdl = None
    if BAR_METRIC == "per_ha" and _mem is None:                         # 全国平均（虚线）：仅柱状图版保留；[1106b] 箱线图版不画
        for s, lab in (("ssp126", "SSP1-2.6"), ("ssp585", "SSP5-8.5")):
            ax_bar.axhline(nat_ha[s], color=BAR_C[s], linestyle="--", linewidth=0.7, zorder=2,
                           label=f"National mean, {lab}")
    ax_bar.axhline(0, color="0.3", linewidth=0.6, zorder=2)
    ax_bar.set_xlim(-0.6, len(top) - 0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([prov_abbr(n) for n in top], rotation=0, ha="center", fontsize=FS["bar_tick"])
    ax_bar.set_ylabel(ylab, fontsize=FS["bar_label"])   # [1106] 不加粗
    ax_bar.text(0.5, 1.05, ttl, transform=ax_bar.transAxes, ha="center", va="top", fontsize=FS["bar_title"])
    if _hdl is not None:
        ax_bar.legend(handles=_hdl, fontsize=FS["bar_legend"], frameon=False, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=2)   # [1106b] 图例内移，离标题和右边框远一点
    else:
        ax_bar.legend(fontsize=FS["bar_legend"], frameon=False, loc="upper right", ncol=2)
    ax_bar.grid(axis="y", alpha=0.3, zorder=0)
    ax_bar.tick_params(labelsize=FS["bar_tick"], length=2)
    for sp in ("top", "right"): ax_bar.spines[sp].set_visible(False)
else:
    ax_bar.axis("off")

out = OUT_DIR / "bivariate_2x2_with_bar"
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.svg", bbox_inches="tight")     # 矢量, 文字可编辑
print(f"[saved] {out}.png / .pdf / .svg")
print("完成：2×2 地图 + 底部前十省份柱状图（单位面积损失）。")
