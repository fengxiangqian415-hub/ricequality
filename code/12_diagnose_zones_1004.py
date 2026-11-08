from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

from config_1004 import RI_TYPES as RI, SSPS, PRED_MODE, Q_HIGH as _QH, out_dir, area_file, thresholds_file, \
    support_file, SUMMARY_WEIGHT
import json
_TH = json.load(open(thresholds_file()))["thresholds"] if thresholds_file().exists() else {}
ZDIR  = out_dir("quality_zones") / PRED_MODE
OUT   = ZDIR.parent / "diagnosis"; OUT.mkdir(parents=True, exist_ok=True)
LAT = np.arange(53.75, 17.75, -0.5); LON = np.arange(73.25, 136.0, 0.5)
LA, LO = np.meshgrid(LAT, LON, indexing="ij")
REGIONS = {
    "东北 NE":        (LA >= 40) & (LO >= 118),
    "华北/西北 N":    (LA >= 33) & ~((LA >= 40) & (LO >= 118)),
    "长江流域 YZ":    (LA >= 27) & (LA < 33),
    "华南西南 S/SW":  (LA < 27),
}


def rd(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(float)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a[a == ds.nodata] = np.nan
    return a


def wq(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(np.searchsorted(c, q), len(x) - 1)]) for q in qs]


rows_q, rows_t, rows_p, rows_r = [], [], [], []
for ri in RI:
    w = rd(area_file(ri))
    w = np.where(np.isfinite(rd(support_file(ri))), w, np.nan)
    wc = np.where(np.isfinite(w) & (w > 0), 1.0, np.nan) if SUMMARY_WEIGHT == "cell" else w
    Qb = rd(ZDIR / f"Q_{ri}_baseline.tif")
    db = {t: rd(ZDIR / f"d_{t}_{ri}_baseline.tif") for t in ("HRR", "CD", "PC")}
    for ssp in SSPS:
        Qf = rd(ZDIR / f"Q_{ri}_{ssp}_future.tif")
        df_ = {t: rd(ZDIR / f"d_{t}_{ri}_{ssp}_future.tif") for t in ("HRR", "CD", "PC")}
        ok = np.isfinite(Qb) & np.isfinite(Qf) & np.isfinite(w) & (w > 0)
        ww, qb, qf = w[ok], Qb[ok], Qf[ok]
        A = ww.sum(); wwc = wc[ok]

        P = [0.1, 0.25, 0.5, 0.75, 0.9]
        rows_q.append(dict(ri=ri, ssp=ssp, **{f"base_p{int(p*100)}": v for p, v in zip(P, wq(qb, wwc, P))},
                           **{f"fut_p{int(p*100)}": v for p, v in zip(P, wq(qf, wwc, P))},
                           dQ_mean=float(np.sum(wwc * (qf - qb)) / wwc.sum())))

        for qh in np.round(np.arange(0.60, 0.745, 0.01), 2):
            b, f = ww[qb >= qh].sum(), ww[qf >= qh].sum()
            rows_t.append(dict(ri=ri, ssp=ssp, Q_HIGH=qh, base_premium_pct=100 * b / A,
                               fut_premium_pct=100 * f / A, premium_kept_pct=100 * f / b if b else np.nan,
                               premium_area_loss_Mha=(b - f) / 1e4))

        lo, hi = wq(qb, ww, [1 / 3, 2 / 3])
        rows_p.append(dict(ri=ri, ssp=ssp, Q_LOW_p33=lo, Q_HIGH_p67=hi,
                           base_premium_pct=100 * ww[qb >= hi].sum() / A,
                           fut_premium_pct=100 * ww[qf >= hi].sum() / A,
                           fut_common_pct=100 * ww[qf < lo].sum() / A,
                           premium_area_loss_Mha=(ww[qb >= hi].sum() - ww[qf >= hi].sum()) / 1e4))

        Q_HIGH = _TH.get(ri, {}).get("Q_HIGH", _QH)
        for rn, rm in REGIONS.items():
            m = ok & rm
            if w[m].sum() <= 0:
                continue
            wm = lambda a: float(np.sum(a[m] * wc[m]) / wc[m].sum())
            rows_r.append(dict(ri=ri, ssp=ssp, region=rn, area_Mha=w[m].sum() / 1e4,
                               base_Q=wm(Qb), dQ=wm(Qf - Qb),
                               base_premium_pct=100 * w[m & (Qb >= Q_HIGH)].sum() / w[m].sum(),
                               fut_premium_pct=100 * w[m & (Qf >= Q_HIGH)].sum() / w[m].sum(),
                               **{f"d{t}": wm(df_[t] - db[t]) for t in ("HRR", "CD", "PC")}))

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
for name, rows in (("Q_distribution", rows_q), ("threshold_sensitivity", rows_t),
                   ("baseline_percentile_zones", rows_p), ("regional", rows_r)):
    d = pd.DataFrame(rows); d.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    print(f"\n===== {name} =====")
    if name == "threshold_sensitivity":
        print(d[d.Q_HIGH.isin([0.60, 0.62, 0.64, 0.66, 0.67, 0.68, 0.70, 0.72])].round(2).to_string(index=False))
    else:
        print(d.round(3).to_string(index=False))
print(f"\n输出: {OUT}")
