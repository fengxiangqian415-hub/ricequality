# ============================================================
# build_migration_stats.py  (v2：稳健迁移指标)
# 面积加权转移矩阵 + 迁移量化（见设计规格 §7，按诊断结论改）
#
# 迁移指标改用【稳健】版：
#   robust_lat = 本期按 Q 排名前 1/3 的格的面积加权平均纬度
#   （该集合恒有 1/3 面积，不会像"未来优质区质心"那样塌成噪声）
# 旧的"优质区质心"仍算，但当未来优质面积 < COLLAPSE_FRAC×基线 时标记不可靠。
#
# 输出（migration_stats/{variant}/）：
#   transition_{ri}_{ssp}.csv   from/to/area_km2/area_pct
#   migration_{ri}_{ssp}.csv    稳健纬度位移 + 质心(带可靠标记)
#   sankey_{ri}_{ssp}.json      节点+流量+迁移指标
# zone: 3=优质 2=中等 1=普通
# ============================================================
# ------------------------------------------------------------
# [1004] 运行顺序 13（依赖 11）——配置来自 config_1004
# 输入：
#   out_dir("quality_zones")/{PRED_MODE}/{zone,Q}_{ri}_{baseline|ssp_future}.tif
#   output/area_weights/area_weight_{ri}_2000_2010.tif
# 输出（out_dir("migration_stats")/{PRED_MODE}/，默认 output/migration_stats_1004/annual/）：
#   transition_{ri}_{ssp}.csv  migration_{ri}_{ssp}.csv  sankey_{ri}_{ssp}.json  class_centroid_{ri}.csv  chord_{ri}.csv
# ------------------------------------------------------------
import numpy as np
import rasterio
from pathlib import Path
import json, csv, math
import warnings
warnings.filterwarnings("ignore")

from config_1004 import RI_TYPES as _RI, SSPS as _SSPS, PRED_MODE, AREA_DIR, out_dir
ZONE_ROOT = out_dir("quality_zones")
WEIGHT    = AREA_DIR
AW_NAME   = "area_weight_{ri}_2000_2010.tif"   # 1 km 真实种植面积聚合
OUT_ROOT  = out_dir("migration_stats")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT  = PRED_MODE                # 主结果（逐年预测再平均）
VARIANTS = [VARIANT]
RI_TYPES = _RI
SSPS     = _SSPS
GRADES   = [3, 2, 1]
GNAME    = {3: "优质", 2: "中等", 1: "普通"}
TOPFRAC       = 1/3     # 稳健指标：前 1/3
COLLAPSE_FRAC = 0.2     # 未来优质<基线20% → 质心判不可靠

def read_tif(p):
    if not Path(p).exists(): return None, None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64"); nd = ds.nodata
        if nd is not None: a[a == nd] = np.nan
        return a, ds.transform

def lonlat(transform, shape):
    H, W = shape
    lon = transform.c + (np.arange(W)+0.5)*transform.a
    lat = transform.f + (np.arange(H)+0.5)*transform.e
    return np.meshgrid(lon, lat)

def haversine(lo1, la1, lo2, la2):
    R=6371.0; p1,p2=math.radians(la1),math.radians(la2)
    dp=math.radians(la2-la1); dl=math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def bearing(lo1,la1,lo2,la2):
    p1,p2=math.radians(la1),math.radians(la2); dl=math.radians(lo2-lo1)
    y=math.sin(dl)*math.cos(p2); x=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y,x))+360)%360

def compass(b):
    return ["N","NE","E","SE","S","SW","W","NW"][int((b+22.5)//45)%8]

def robust_top_lat(Q, w, LAT, valid, topfrac=TOPFRAC):
    """本期按 Q 前 topfrac 的面积加权平均纬度。"""
    m = valid & np.isfinite(Q)
    q=Q[m]; ww=w[m]; la=LAT[m]
    o=np.argsort(q)[::-1]; q,ww,la=q[o],ww[o],la[o]
    cum=np.cumsum(ww)/ww.sum(); sel=cum<=topfrac
    if sel.sum()==0: sel[0]=True
    return float((ww[sel]*la[sel]).sum()/ww[sel].sum())

for variant in VARIANTS:
    for ri in RI_TYPES:
        w, wt = read_tif(WEIGHT / AW_NAME.format(ri=ri))
        zb, _ = read_tif(ZONE_ROOT / variant / f"zone_{ri}_baseline.tif")
        Qb, _ = read_tif(ZONE_ROOT / variant / f"Q_{ri}_baseline.tif")
        if w is None or zb is None or Qb is None: continue
        LON, LAT = lonlat(wt, w.shape)

        for ssp in SSPS:
            zf, _ = read_tif(ZONE_ROOT / variant / f"zone_{ri}_{ssp}_future.tif")
            Qf, _ = read_tif(ZONE_ROOT / variant / f"Q_{ri}_{ssp}_future.tif")
            if zf is None or Qf is None: continue

            valid = (np.isfinite(w) & (w>0) &
                     np.isin(zb, GRADES) & np.isin(zf, GRADES))
            if not valid.any(): continue
            wv=w[valid]; zbv=zb[valid].astype(int); zfv=zf[valid].astype(int)
            lonv=LON[valid]; latv=LAT[valid]; tot=wv.sum()
            out_dir = OUT_ROOT/variant; out_dir.mkdir(parents=True, exist_ok=True)

            # 转移矩阵 + 流量
            rows=[]; links=[]
            for i in GRADES:
                for j in GRADES:
                    m=(zbv==i)&(zfv==j); area=float(wv[m].sum())
                    rows.append([GNAME[i],GNAME[j],round(area,2),round(100*area/tot,2)])
                    if area>0:
                        links.append({"source":f"base_{GNAME[i]}","target":f"fut_{GNAME[j]}",
                                      "value":round(area,2),
                                      "direction":("down" if j<i else "up" if j>i else "same")})
            with open(out_dir/f"transition_{ri}_{ssp}.csv","w",newline="") as f:
                wr=csv.writer(f); wr.writerow(["from","to","area_km2","area_pct"]); wr.writerows(rows)

            base_marg={GNAME[i]:float(wv[zbv==i].sum()) for i in GRADES}
            fut_marg ={GNAME[j]:float(wv[zfv==j].sum()) for j in GRADES}

            # 稳健迁移：前1/3平均纬度
            blat_r=robust_top_lat(Qb,w,LAT,valid)
            flat_r=robust_top_lat(Qf,w,LAT,valid)
            dlat_r=flat_r-blat_r

            # 旧质心（带可靠标记）
            def cen(mask):
                ww=wv[mask]
                if ww.sum()==0: return None,None
                return float((wv[mask]*lonv[mask]).sum()/ww.sum()), float((wv[mask]*latv[mask]).sum()/ww.sum())
            blon,blat=cen(zbv==3); flon,flat=cen(zfv==3)
            reliable = fut_marg["优质"] >= COLLAPSE_FRAC*base_marg["优质"]
            if None not in (blon,flon):
                cdist=haversine(blon,blat,flon,flat); cbrg=bearing(blon,blat,flon,flat)
            else:
                cdist=cbrg=None

            with open(out_dir/f"migration_{ri}_{ssp}.csv","w",newline="") as f:
                wr=csv.writer(f)
                wr.writerow(["metric","baseline","future","delta","note"])
                wr.writerow(["robust_top33_meanlat_degN", round(blat_r,3), round(flat_r,3),
                             round(dlat_r,3), "北移+/南移-"])
                wr.writerow(["premium_area_km2", round(base_marg["优质"],1),
                             round(fut_marg["优质"],1),
                             round(fut_marg["优质"]-base_marg["优质"],1), ""])
                wr.writerow(["premium_centroid_shift_km",
                             "" if cdist is None else round(cdist,1),
                             "" if cbrg is None else compass(cbrg),
                             "", "可靠" if reliable else "★塌缩,质心不可靠"])

            sankey={"variant":variant,"ri":ri,"ssp":ssp,"total_area_km2":round(tot,1),
                "nodes":([{"id":f"base_{GNAME[i]}","grade":GNAME[i],"side":"baseline",
                           "area_km2":round(base_marg[GNAME[i]],1)} for i in GRADES]
                        +[{"id":f"fut_{GNAME[j]}","grade":GNAME[j],"side":"future",
                           "area_km2":round(fut_marg[GNAME[j]],1)} for j in GRADES]),
                "links":links,
                "migration":{
                    "robust_top33_meanlat":{"baseline":round(blat_r,3),"future":round(flat_r,3),
                                            "delta_deg":round(dlat_r,3),
                                            "direction":"北移" if dlat_r>0 else "南移"},
                    "premium_centroid":{
                        "baseline":None if blon is None else {"lon":round(blon,4),"lat":round(blat,4)},
                        "future":  None if flon is None else {"lon":round(flon,4),"lat":round(flat,4)},
                        "shift_km":None if cdist is None else round(cdist,1),
                        "compass": None if cbrg is None else compass(cbrg),
                        "reliable":bool(reliable)},
                }}
            with open(out_dir/f"sankey_{ri}_{ssp}.json","w") as f:
                json.dump(sankey,f,ensure_ascii=False,indent=2)

            print(f"[{variant}/{ri}/{ssp}] 优质 {base_marg['优质']:.0f}→{fut_marg['优质']:.0f} km² "
                  f"(Δ{fut_marg['优质']-base_marg['优质']:+.0f}) | "
                  f"稳健前1/3纬度 {blat_r:.2f}→{flat_r:.2f}°N (Δ{dlat_r:+.2f}, "
                  f"{'北移' if dlat_r>0 else '南移'}) | "
                  f"质心{'可靠' if reliable else '★塌缩不可靠'}")

print(f"\n全部完成！输出: {OUT_ROOT}")

# ============================================================
# 追加：三类(优/中/普)质心 baseline/ssp126/ssp585 → class_centroid_{ri}.csv
# 供 Fig.4 中间"各等级质心迁移"图；并打印各类南/北移，判断哪些类别南移
# ============================================================
def _lonlat(t, shape):
    H,W=shape
    lon=t.c+(np.arange(W)+0.5)*t.a; lat=t.f+(np.arange(H)+0.5)*t.e
    return np.meshgrid(lon,lat)

_GRc=[("Premium",3),("Moderate",2),("Common",1)]
for ri in RI_TYPES:
    w, wt = read_tif(WEIGHT / AW_NAME.format(ri=ri))
    if w is None: continue
    LON, LAT = _lonlat(wt, w.shape)
    def ccent(period, code):
        zt = (ZONE_ROOT/VARIANT/f"zone_{ri}_baseline.tif" if period=="baseline"
              else ZONE_ROOT/VARIANT/f"zone_{ri}_{period}_future.tif")
        z,_ = read_tif(zt)
        if z is None: return (None,None)
        m = np.isfinite(w)&(w>0)&(z==code)
        if w[m].sum()==0: return (None,None)
        return (float((w[m]*LON[m]).sum()/w[m].sum()), float((w[m]*LAT[m]).sum()/w[m].sum()))
    out = OUT_ROOT/VARIANT; out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for gname,code in _GRc:
        cb=ccent("baseline",code)
        for period,label in [("baseline","Baseline"),("ssp126","SSP1-2.6"),("ssp585","SSP5-8.5")]:
            lon,lat=ccent(period,code)
            km = "" if (cb[0] is None or lon is None) else round(haversine(cb[0],cb[1],lon,lat),1)
            dlat = "" if (cb[1] is None or lat is None) else round(lat-cb[1],3)
            rows.append([gname,label,"" if lon is None else round(lon,4),
                         "" if lat is None else round(lat,4),km,dlat])
    with open(out/f"class_centroid_{ri}.csv","w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["grade","scenario","lon","lat","shift_km","dlat_degN"]); wr.writerows(rows)
    # 打印各类 SSP5-8.5 纬度位移，判断南/北移
    print(f"[class_centroid] {ri}:")
    for gname,code in _GRc:
        cb=ccent("baseline",code); cf=ccent("ssp585",code)
        if cb[1] is not None and cf[1] is not None:
            dl=cf[1]-cb[1]
            print(f"    {gname:9s} baseline {cb[1]:.2f}°N -> ssp585 {cf[1]:.2f}°N "
                  f"(Δ{dl:+.2f}, {'南移' if dl<0 else '北移'})")

# ============================================================
# 追加：弦图英文数据(circlize 用) chord_{ri}.csv
# ============================================================
_GEN={"优质":"Premium","中等":"Moderate","普通":"Common"}
for ri in RI_TYPES:
    out=OUT_ROOT/VARIANT
    with open(out/f"chord_{ri}.csv","w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["from","ssp","to","value_1e4km2"])
        for ssp,slab in [("ssp126","SSP1-2.6"),("ssp585","SSP5-8.5")]:
            tp=out/f"transition_{ri}_{ssp}.csv"
            if not tp.exists(): continue
            with open(tp) as g:
                for r in csv.DictReader(g):
                    wr.writerow([_GEN[r["from"]], slab, _GEN[r["to"]], round(float(r["area_km2"])/1e4,4)])
    print(f"[chord csv] chord_{ri}.csv written")
