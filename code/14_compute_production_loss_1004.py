# ============================================================
# compute_production_loss.py
# 未来气候情景下 优质米 vs 普通米 的产值损失(固定价,历史/未来同价)
#
# 价值(元)/格 = HRY(t/ha) × 面积(km²) × 100(ha/km²) × 1000(kg/t) × 价(元/kg)
#             = HRY × area_km2 × price × 1e5
#   价: 优质区(zone==3)用 P_HQ; 非优质(zone∈{1,2})用 P_ORD
# 损失 = 历史价值 − 未来价值; 并做 数量效应 / 品质(降档)效应 的精确加和分解:
#   ΔV = area·1e5·[ p_his·(HRY_his−HRY_fut) + HRY_fut·(p_his−p_fut) ]
#          └── 数量效应 ──┘   └──── 品质(单价)效应 ────┘
#
# 输入(均 0.5° EPSG:4326 同格):
#   quality_zones/<VER>/zone_{ri}_baseline.tif , zone_{ri}_{ssp}_future.tif   (3/2/1/0)
#   head_rice_yield/head_rice_yield_{ri}_historical.tif , _{ri}_{ssp}.tif       (t/ha)
#   area_weights/area_weight_{ri}_2000_2010.tif   (km²，make_rice_area.py 生成)
#   统一格网域 D（03b；support_file = 支撑区 ∩ 面积>0）  [1013]
#
# 输出:
#   loss/loss_{ri}_{ssp}.tif                每格损失(亿元)
#   loss/production_loss_summary.csv        全国汇总(亿元) + 分解 + 价比敏感性
#   figures/production_loss_map.png/.pdf    2(ssp)×2(ri) 损失地图
# ============================================================
# ------------------------------------------------------------
# [1004] 运行顺序 14（依赖 10、11）——配置来自 config_1004
# 输入：
#   out_dir("quality_zones")/{PRED_MODE}/zone_{ri}_{baseline|ssp_future}.tif
#   out_dir("head_rice_yield")/head_rice_yield_{ri}_{historical|ssp}.tif   (t/ha)
#   output/area_weights/area_weight_{ri}_2000_2010.tif                      (km²)
#   output/analysis_domain_{TAG}/{ri}_analysis_domain.tif（统一格网域 D，03b）
# 输出（out_dir("loss")，默认 output/loss_1004/）：
#   loss_{ri}_{ssp}.tif（亿元/格）  production_loss_summary.csv  production_loss_map.{png,pdf}
# ------------------------------------------------------------
import numpy as np
import rasterio
from pathlib import Path
import csv
import warnings
warnings.filterwarnings("ignore")

# ---------------- 配置 ----------------
from config_1004 import (RI_TYPES, SSPS, PRED_MODE, HIS_P, AREA_DIR, HIS_ANN, CHINA_SHP as _CHINA,
                         out_dir, support_file, P_ORD as _P_ORD, P_HQ as _P_HQ)
ZONE_VER = PRED_MODE                                    # quality_zones 下的子目录（主结果）
ZONE_DIR = out_dir("quality_zones") / ZONE_VER
HRY_DIR  = out_dir("head_rice_yield", thresh=False)
AW_DIR   = AREA_DIR
MASK_DIR = HIS_ANN                                      # 共同支撑区所在目录
OUT_DIR  = out_dir("loss"); OUT_DIR.mkdir(parents=True, exist_ok=True)  # tif/csv/图 都放这

# 采用价(近5年均值, 元/kg, 米口径; 历史=未来同价)
P_ORD = _P_ORD    # 普通米 (非优质区: 中等+普通)   [1013] 价格只在 config_1004 定义
P_HQ  = _P_HQ     # 优质米 (优质区)
# 价比敏感性(优质/普通): 相对损失稳健性
RATIO_SENS = [1.10, 1.14, 1.18]

# 区域支持掩膜: 保证 HRR 预测与品质分区落在同一支持区、逐格对应(与 HRY 脚本同口径)
USE_SUPPORT_MASK = True
UNIT = 1e5               # HRY[t/ha]·area[km²]·price[元/kg] → 元
YI   = 1e8              # 元 → 亿元
# -------------------------------------

def read_tif(p):
    p = Path(p)
    if not p.exists():
        return None, None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64"); nd = ds.nodata
        if nd is not None: a[a == nd] = np.nan
        prof = ds.profile
    return a, prof

def price_of_zone(zone):
    """zone: 3=优质→P_HQ; 1,2=非优质→P_ORD; 0/nan→nan"""
    p = np.full(zone.shape, np.nan)
    p[(zone == 1) | (zone == 2)] = P_ORD
    p[zone == 3] = P_HQ
    return p

def price_of_zone_ratio(zone, ratio):
    p = np.full(zone.shape, np.nan)
    p[(zone == 1) | (zone == 2)] = 1.0
    p[zone == 3] = ratio
    return p

rows = []           # 汇总表
loss_ref_profile = None

for ssp in SSPS:
    for ri in RI_TYPES:
        # 读入
        zh, prof = read_tif(ZONE_DIR / f"zone_{ri}_baseline.tif")
        zf, _    = read_tif(ZONE_DIR / f"zone_{ri}_{ssp}_future.tif")
        yh, _    = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_historical.tif")
        yf, _    = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_{ssp}.tif")
        aw, _    = read_tif(AW_DIR   / f"area_weight_{ri}_2000_2010.tif")   # 1 km 真实种植面积聚合（make_rice_area.py）
        if any(x is None for x in (zh, zf, yh, yf, aw)):
            print(f"[skip] {ri} {ssp}: 缺输入文件"); continue

        # 形状一致性检查(不同版本/网格会在此暴露)
        shapes = {a.shape for a in (zh, zf, yh, yf, aw)}
        if len(shapes) != 1:
            raise ValueError(f"{ri} {ssp}: 栅格形状不一致 {shapes} —— 分区与HRY是否同网格/同预测版本?")

        # 有效域
        valid = (np.isfinite(yh) & np.isfinite(yf) & np.isfinite(aw) & (aw > 0)
                 & (zh > 0) & (zf > 0))
        if USE_SUPPORT_MASK:
            sm_arr, _ = read_tif(support_file(ri))          # [1013] 统一格网域 D
            if sm_arr is not None:
                valid &= np.isfinite(sm_arr)              # 掩膜有值处=支持区(同 HRY 脚本口径)
            else:
                print(f"[warn] 无统一格网域 {support_file(ri)}, 未做筛选")

        ph = price_of_zone(zh); pf = price_of_zone(zf)
        # 每格价值与损失(元)
        v_his = np.where(valid, aw * yh * ph * UNIT, np.nan)
        v_fut = np.where(valid, aw * yf * pf * UNIT, np.nan)
        loss  = v_his - v_fut                                   # >0 = 损失

        # 精确分解
        qty  = np.where(valid, aw * UNIT * ph * (yh - yf), np.nan)          # 数量效应
        qual = np.where(valid, aw * UNIT * yf * (ph - pf), np.nan)          # 品质(降档)效应

        # 保存每格损失(亿元)
        prof_out = prof.copy(); prof_out.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        with rasterio.open(OUT_DIR / f"loss_{ri}_{ssp}.tif", "w", **prof_out) as d:
            d.write((loss / YI).astype("float32"), 1)
        if loss_ref_profile is None: loss_ref_profile = prof_out

        # 全国汇总(亿元)
        Vh = np.nansum(v_his) / YI
        Vf = np.nansum(v_fut) / YI
        L  = np.nansum(loss)  / YI
        Lqty  = np.nansum(qty)  / YI
        Lqual = np.nansum(qual) / YI

        # 优质区面积/产值份额(诊断)
        aw_v = aw[valid]
        hq_area_his = np.nansum(aw[(zh == 3) & valid]); hq_area_fut = np.nansum(aw[(zf == 3) & valid])

        # 价比敏感性(相对损失, 只吃比值; 绝对价位约掉)
        sens = {}
        for r in RATIO_SENS:
            rh = price_of_zone_ratio(zh, r); rf = price_of_zone_ratio(zf, r)
            vh = np.where(valid, aw * yh * rh, np.nan)
            vf = np.where(valid, aw * yf * rf, np.nan)
            sens[r] = 100 * np.nansum(vh - vf) / np.nansum(vh)     # 相对损失 %

        rows.append({
            "ssp": ssp, "ri": ri,
            "V_his_yi": round(Vh, 3), "V_fut_yi": round(Vf, 3),
            "loss_yi": round(L, 3), "loss_pct": round(100*L/Vh, 2) if Vh else np.nan,
            "loss_qty_yi": round(Lqty, 3), "loss_qual_yi": round(Lqual, 3),
            "HQ_area_his_km2": round(hq_area_his, 0), "HQ_area_fut_km2": round(hq_area_fut, 0),
            "rel_loss_ratio1.10": round(sens[1.10], 2),
            "rel_loss_ratio1.14": round(sens[1.14], 2),
            "rel_loss_ratio1.18": round(sens[1.18], 2),
        })
        print(f"[{ssp}/{ri}] 损失={L:.2f}亿元 ({100*L/Vh:.1f}%)  数量={Lqty:.2f} 品质={Lqual:.2f}  "
              f"优质面积 {hq_area_his:.0f}->{hq_area_fut:.0f} km²")

# 汇总 CSV(含 ri1+ri2 合计)
if rows:
    tot = {}
    for ssp in SSPS:
        sub = [r for r in rows if r["ssp"] == ssp]
        if not sub: continue
        agg = {k: round(sum(r[k] for r in sub), 3) for k in
               ["V_his_yi","V_fut_yi","loss_yi","loss_qty_yi","loss_qual_yi"]}
        agg.update(ssp=ssp, ri="ri1+ri2",
                   loss_pct=round(100*agg["loss_yi"]/agg["V_his_yi"], 2) if agg["V_his_yi"] else np.nan)
        rows.append(agg)
    keys = list(rows[0].keys())
    with open(OUT_DIR / "production_loss_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[saved] {OUT_DIR/'production_loss_summary.csv'}")

# ---------------- 损失地图 (2 ssp × 2 ri) ----------------
try:
    import matplotlib as mpl; mpl.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],"font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold","font.size":7})

    # 与 HRY 图一致的地图范围
    LON=np.arange(73.25,136.0,0.5); LAT=np.arange(53.75,17.75,-0.5)
    EXTENT=[LON[0]-0.25,LON[-1]+0.25,LAT[-1]-0.25,LAT[0]+0.25]
    china=gpd.read_file(_CHINA).to_crs("EPSG:4326")
    xmin,ymin,xmax,ymax=china.total_bounds

    # 统一色标(损失, 亿元/格; 用 98 分位对称)
    allv=[]
    for ssp in SSPS:
        for ri in RI_TYPES:
            a,_=read_tif(OUT_DIR/f"loss_{ri}_{ssp}.tif")
            if a is not None: allv.append(a[np.isfinite(a)])
    vmax = float(np.percentile(np.concatenate(allv),98)) if allv else 1.0
    vmax = max(vmax, 1e-6)
    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)   # >0 损失(红) <0 增益(蓝)

    fig,axes=plt.subplots(len(SSPS),len(RI_TYPES),figsize=(180/25.4,150/25.4))
    RI_L={"ri1":"Single/early rice","ri2":"Late rice"}   # [1105] 与正文统一为 Single/early
    SSP_L={"ssp126":"SSP1-2.6","ssp585":"SSP5-8.5"}
    for i,ssp in enumerate(SSPS):
        for j,ri in enumerate(RI_TYPES):
            ax=axes[i,j] if len(SSPS)>1 else axes[j]
            a,_=read_tif(OUT_DIR/f"loss_{ri}_{ssp}.tif")
            if a is not None:
                ax.imshow(a,extent=EXTENT,origin="upper",cmap="RdBu_r",norm=norm,aspect="auto")
            china.boundary.plot(ax=ax,linewidth=0.35,color="black",zorder=5)
            ax.set_xlim(xmin-0.5,xmax+0.5); ax.set_ylim(ymin-0.5,ymax+0.5)
            ax.tick_params(labelsize=5,length=2)
            if i==0: ax.set_title(RI_L[ri],fontsize=7)
            if j==0: ax.set_ylabel(SSP_L[ssp],fontsize=7,fontweight="bold")
    sm=ScalarMappable(cmap="RdBu_r",norm=norm); sm.set_array([])
    cb=fig.colorbar(sm,ax=axes,orientation="horizontal",fraction=0.05,pad=0.08)
    cb.set_label("Production-value loss (100 million yuan per 0.5° cell)  [red = loss]",fontsize=6)
    fig.suptitle("Future HQ-rice production-value loss (fixed price)",fontsize=9,fontweight="bold")
    fig.savefig(OUT_DIR/"production_loss_map.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT_DIR/"production_loss_map.pdf",bbox_inches="tight")
    print(f"[saved] {OUT_DIR/'production_loss_map.png'}")
except Exception as e:
    print(f"[figure skipped] {e}")

print("\n完成。")
