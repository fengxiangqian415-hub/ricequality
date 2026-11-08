# =====================================================
# q_common_1004.py — shared I/O and Q-index helpers for the 1004b sensitivity analyses
# (#774 PC weight / no-PC, #809 thresholds, #910 member-level economics)
# [1004] paths, periods, thresholds, prices and anchors are taken from config_1004 / step 11 / step 14,
#        so Q, zones and values reproduce the main-text figures exactly.
# =====================================================
import os
from functools import lru_cache
import config_1004 as CFG
import numpy as np
import rasterio

# ---------------- CONFIG（全部来自 config_1004，与正文一致）----------------
OUTDIR = str(CFG.out_dir("sensitivity"))       # output/sensitivity_1004b_fix6266/（22 会收集）
RIS, SSPS = CFG.RI_TYPES, CFG.SSPS
GGCMS, GCMS = CFG.CROP_MODELS, CFG.GCMS
PRED = CFG.pred_dir()                          # v5_1004b/annual（随情景 CO2）
HRY  = CFG.out_dir("head_rice_yield", thresh=False)

# [1013] 锚点、价格、汇率只在 config_1004 定义一次（11/14/16/24/25/27/28 同源）
ANCHORS = CFG.ANCHORS
Q_LOW, Q_HIGH = CFG.Q_LOW, CFG.Q_HIGH          # 见 config（[1031] 0.60 / 0.66：国标三级限值换算，35 号）
P_ORD, P_PREM = CFG.P_ORD, CFG.P_HQ            # CNY/kg：优质区用 P_PREM，中等+普通用 P_ORD
CNY_PER_USD   = CFG.CNY_PER_USD
KM2_TO_HA     = 100.0
# ----------------------------------------

def find(key, ri, ssp=None, model=None, gcm=None, trait=None):
    """显式路径（替代 glob，避免 area_weights 下多个 ri 文件同时匹配、PC 误配 PCA 等问题）"""
    H, F = CFG.HIS_P, CFG.FUT_P
    p = {"q_hist":   lambda: PRED / "historical" / f"{gcm}_historical_{ri}_{trait}_{H}.tif",
         "q_fut":    lambda: PRED / ssp / f"{model}_{gcm}_{ssp}_{ri}_{trait}_{F}.tif",
         "q_ens_hist": lambda: PRED / "ensemble" / f"ensemble_historical_{ri}_{trait}_{H}.tif",
         "q_ens_fut":  lambda: PRED / "ensemble" / f"ensemble_{ssp}_{ri}_{trait}_{F}.tif",
         "hry_hist": lambda: HRY / "historical" / model / gcm / f"{model}_{gcm}_historical_{ri}_HRY_{H}.tif",
         "hry_fut":  lambda: HRY / ssp / model / gcm / f"{model}_{gcm}_{ssp}_{ri}_HRY_{F}.tif",
         "area":     lambda: CFG.area_file(ri),
         "support":  lambda: CFG.support_file(ri)}[key]()
    if not p.exists():
        raise FileNotFoundError(f"[{key}] {p}")
    return str(p)

@lru_cache(maxsize=None)
def read(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a

def check_anchors():
    bad = [k for k, v in ANCHORS.items() if None in v]
    if bad:
        raise ValueError(f"Fill ANCHORS for {bad} with the p1/p99 values used in the main zoning script.")

def desirability(trait, x):
    lo, hi = ANCHORS[trait]
    z = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return z if trait == "HRR" else 1.0 - z           # HRR higher-better; CD, PC lower-better

def q_index(hrr, cd, pc, w=(1.0, 1.0, 1.0)):
    wh, wc, wp = w
    return (wh * desirability("HRR", hrr) + wc * desirability("CD", cd)
            + wp * desirability("PC", pc)) / (wh + wc + wp)

def q_from_limits(hrr, cd, pc, w=(1.0, 1.0, 1.0)):
    """Q value of a grain exactly at given trait limits, e.g. a grade boundary in NY/T 593-2021.
       Use it to translate standard grade limits into Q thresholds (#809/#1706)."""
    return float(q_index(np.array(hrr, float), np.array(cd, float), np.array(pc, float), w))

def classify(q, lo=Q_LOW, hi=Q_HIGH):
    """2 = premium, 1 = moderate, 0 = common, nan outside support."""
    c = np.where(q >= hi, 2.0, np.where(q >= lo, 1.0, 0.0))
    c[~np.isfinite(q)] = np.nan
    return c

def area_and_mask(ri):
    a = read(find("area", ri=ri)).copy()
    try:
        s = read(find("support", ri=ri))
        a[~(np.isfinite(s) & (s > 0))] = np.nan
    except FileNotFoundError:
        print(f"  (no support mask found for {ri}; using area>0 only)")
    a[~(np.isfinite(a) & (a > 0))] = np.nan
    return a

def traits_hist(ri, gcm):
    return [read(find("q_hist", ri=ri, gcm=gcm, trait=t)) for t in ("HRR", "CD", "PC")]

def traits_fut(ri, ssp, model, gcm):
    return [read(find("q_fut", ri=ri, ssp=ssp, model=model, gcm=gcm, trait=t)) for t in ("HRR", "CD", "PC")]

def traits_ens(ri, ssp=None):
    """集合平均性状（11 号分区与 Fig. 5 的口径：先对成员平均性状，再算 Q 并分级）"""
    if ssp is None:
        return [read(find("q_ens_hist", ri=ri, trait=t)) for t in ("HRR", "CD", "PC")]
    return [read(find("q_ens_fut", ri=ri, ssp=ssp, trait=t)) for t in ("HRR", "CD", "PC")]

def shares(cls, area):
    ok = np.isfinite(cls) & np.isfinite(area)
    tot = area[ok].sum()
    return {g: 100.0 * area[ok & (cls == v)].sum() / tot for g, v in (("premium", 2), ("moderate", 1), ("common", 0))}, tot

def wquantile(x, w, q):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, x))

