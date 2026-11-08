import numpy as np
import pandas as pd

import config_1004 as C
from pheno_common_1004 import read_tif, save_tif

LAB = "mean_temp_degC"
OUT = C.domain_file("ri1").parent
OUT.mkdir(parents=True, exist_ok=True)


def wm(x, w, m):
    """汇总平均（权重按 config.SUMMARY_WEIGHT）"""
    if C.SUMMARY_WEIGHT == "cell":
        w = np.where(np.isfinite(w), 1.0, np.nan)
    m = m & np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


rows = []
for ri in C.RI_TYPES:
    sup = np.isfinite(read_tif(C.raw_support_file(ri)))
    area = read_tif(C.area_file(ri))
    d0 = sup & np.isfinite(area) & (area > 0)

    nb = np.zeros(d0.shape, int)
    for g in C.GCMS:
        a = read_tif(C.CLIM_HIS / g / f"{g}_historical_{ri}_{LAB}_{C.HIS_P}_mean.tif")
        nb += np.isfinite(a)
    nf = {}
    for s in C.SSPS:
        n = np.zeros(d0.shape, int)
        for m in C.CROP_MODELS:
            for g in C.GCMS:
                p = C.CLIM_FUT / s / m / g / f"{m}_{g}_{s}_{ri}_{LAB}_{C.FUT_P}_mean.tif"
                if p.exists():
                    n += np.isfinite(read_tif(p))
        nf[s] = n
    n_all = len(C.CROP_MODELS) * len(C.GCMS)
    complete = (nb == len(C.GCMS))
    for s in C.SSPS:
        complete &= (nf[s] == n_all)
    nmin = np.minimum.reduce([nf[s] for s in C.SSPS])

    dom = d0
    save_tif(np.where(dom, 1.0, np.nan).astype(np.float32), C.domain_file(ri))
    save_tif(np.where(d0, nmin, np.nan).astype(np.float32), OUT / f"{ri}_n_future_members_min.tif")

    tb = read_tif(C.CLIM_HIS / "ensemble" / f"ensemble_historical_{ri}_{LAB}_{C.HIS_P}_mean.tif")
    A = lambda m: float(np.nansum(np.where(m, area, 0.0))) / 1e4
    r = dict(ri=ri,
             support_cells=int(sup.sum()), support_area_Mha=A(sup),
             D0_cells=int(d0.sum()), D0_area_Mha=A(d0),
             complete_cells=int((d0 & complete).sum()), complete_area_Mha=A(d0 & complete),
             incomplete_area_pct=100 * (1 - A(d0 & complete) / A(d0)) if A(d0) else np.nan,
             domain_cells=int(dom.sum()), domain_area_Mha=A(dom),
             baseline_Tavg_D0=wm(tb, area, d0), baseline_Tavg_complete=wm(tb, area, d0 & complete))
    for s in C.SSPS:
        r[f"{s}_mean_members"] = wm(nf[s].astype(float), area, d0)
        r[f"{s}_area_pct_lt{n_all}"] = 100 * A(d0 & (nf[s] < n_all)) / A(d0) if A(d0) else np.nan
        r[f"{s}_area_pct_zero_members"] = 100 * A(d0 & (nf[s] == 0)) / A(d0) if A(d0) else np.nan
    rows.append(r)
    print(f"\n[{ri}] 原始支撑区 {r['support_cells']} 格网 {r['support_area_Mha']:.2f} Mha；"
          f"面积>0 后 {r['D0_cells']} 格网 {r['D0_area_Mha']:.2f} Mha")
    print(f"      成员齐全（基准 5 GCM + 两情景 25 成员）：{r['complete_cells']} 格网 "
          f"{r['complete_area_Mha']:.2f} Mha（缺成员面积 {r['incomplete_area_pct']:.1f}%）")
    for s in C.SSPS:
        print(f"      {s}: 平均成员数 {r[f'{s}_mean_members']:.1f}/{n_all}；"
              f"<{n_all} 成员面积 {r[f'{s}_area_pct_lt{n_all}']:.1f}%；无成员 {r[f'{s}_area_pct_zero_members']:.1f}%")
    print(f"      基准灌浆期均温（集合）：全部格网 {r['baseline_Tavg_D0']:.2f} °C；"
          f"成员齐全格网 {r['baseline_Tavg_complete']:.2f} °C")
    print(f"      → 统一格网域 D：{r['domain_cells']} 格网 {r['domain_area_Mha']:.2f} Mha "
          f"  [saved] {C.domain_file(ri)}")

pd.DataFrame(rows).round(4).to_csv(OUT / "domain_diagnostics.csv", index=False)
print(f"\n[saved] {OUT / 'domain_diagnostics.csv'}")
print("注：缺成员格网的集合平均来自该格网有值的成员（补充材料说明缺值格网占比）")
