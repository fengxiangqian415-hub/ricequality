import numpy as np
import rasterio
from pathlib import Path
import csv
import warnings
warnings.filterwarnings("ignore")

from config_1004 import (RI_TYPES, SSPS, PRED_MODE, HIS_P, AREA_DIR, HIS_ANN, CHINA_SHP as _CHINA,
                         out_dir, support_file, P_ORD as _P_ORD, P_HQ as _P_HQ)
ZONE_VER = PRED_MODE
ZONE_DIR = out_dir("quality_zones") / ZONE_VER
HRY_DIR  = out_dir("head_rice_yield", thresh=False)
AW_DIR   = AREA_DIR
MASK_DIR = HIS_ANN
OUT_DIR  = out_dir("loss"); OUT_DIR.mkdir(parents=True, exist_ok=True)

P_ORD = _P_ORD
P_HQ  = _P_HQ
RATIO_SENS = [1.10, 1.14, 1.18]

USE_SUPPORT_MASK = True
UNIT = 1e5
YI   = 1e8

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

rows = []
loss_ref_profile = None

for ssp in SSPS:
    for ri in RI_TYPES:
        zh, prof = read_tif(ZONE_DIR / f"zone_{ri}_baseline.tif")
        zf, _    = read_tif(ZONE_DIR / f"zone_{ri}_{ssp}_future.tif")
        yh, _    = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_historical.tif")
        yf, _    = read_tif(HRY_DIR  / f"head_rice_yield_{ri}_{ssp}.tif")
        aw, _    = read_tif(AW_DIR   / f"area_weight_{ri}_2000_2010.tif")
        if any(x is None for x in (zh, zf, yh, yf, aw)):
            print(f"[skip] {ri} {ssp}: 缺输入文件"); continue

        shapes = {a.shape for a in (zh, zf, yh, yf, aw)}
        if len(shapes) != 1:
            raise ValueError(f"{ri} {ssp}: 栅格形状不一致 {shapes} —— 分区与HRY是否同网格/同预测版本?")

        valid = (np.isfinite(yh) & np.isfinite(yf) & np.isfinite(aw) & (aw > 0)
                 & (zh > 0) & (zf > 0))
        if USE_SUPPORT_MASK:
            sm_arr, _ = read_tif(support_file(ri))
            if sm_arr is not None:
                valid &= np.isfinite(sm_arr)
            else:
                print(f"[warn] 无统一格网域 {support_file(ri)}, 未做筛选")

        ph = price_of_zone(zh); pf = price_of_zone(zf)
        v_his = np.where(valid, aw * yh * ph * UNIT, np.nan)
        v_fut = np.where(valid, aw * yf * pf * UNIT, np.nan)
        loss  = v_his - v_fut

        qty  = np.where(valid, aw * UNIT * ph * (yh - yf), np.nan)
        qual = np.where(valid, aw * UNIT * yf * (ph - pf), np.nan)

        prof_out = prof.copy(); prof_out.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        with rasterio.open(OUT_DIR / f"loss_{ri}_{ssp}.tif", "w", **prof_out) as d:
            d.write((loss / YI).astype("float32"), 1)
        if loss_ref_profile is None: loss_ref_profile = prof_out

        Vh = np.nansum(v_his) / YI
        Vf = np.nansum(v_fut) / YI
        L  = np.nansum(loss)  / YI
        Lqty  = np.nansum(qty)  / YI
        Lqual = np.nansum(qual) / YI

        aw_v = aw[valid]
        hq_area_his = np.nansum(aw[(zh == 3) & valid]); hq_area_fut = np.nansum(aw[(zf == 3) & valid])

        sens = {}
        for r in RATIO_SENS:
            rh = price_of_zone_ratio(zh, r); rf = price_of_zone_ratio(zf, r)
            vh = np.where(valid, aw * yh * rh, np.nan)
            vf = np.where(valid, aw * yf * rf, np.nan)
            sens[r] = 100 * np.nansum(vh - vf) / np.nansum(vh)

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

try:
    import matplotlib as mpl; mpl.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"],"font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold","font.size":7})

    LON=np.arange(73.25,136.0,0.5); LAT=np.arange(53.75,17.75,-0.5)
    EXTENT=[LON[0]-0.25,LON[-1]+0.25,LAT[-1]-0.25,LAT[0]+0.25]
    china=gpd.read_file(_CHINA).to_crs("EPSG:4326")
    xmin,ymin,xmax,ymax=china.total_bounds

    allv=[]
    for ssp in SSPS:
        for ri in RI_TYPES:
            a,_=read_tif(OUT_DIR/f"loss_{ri}_{ssp}.tif")
            if a is not None: allv.append(a[np.isfinite(a)])
    vmax = float(np.percentile(np.concatenate(allv),98)) if allv else 1.0
    vmax = max(vmax, 1e-6)
    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig,axes=plt.subplots(len(SSPS),len(RI_TYPES),figsize=(180/25.4,150/25.4))
    RI_L={"ri1":"Single/early rice","ri2":"Late rice"}
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
