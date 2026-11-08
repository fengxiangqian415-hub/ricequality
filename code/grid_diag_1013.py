import numpy as np
import pandas as pd

import config_1004 as C
from pheno_common_1004 import read_tif, load_mask, OUT_HIS

OBS_YEARS = range(2000, 2015)
AREA_YEARS = range(2000, 2011)
LAB = "mean_temp_degC"


def fin(p):
    return np.isfinite(read_tif(p)) if p.exists() else None


rows = []
for ri in C.RI_TYPES:
    M = load_mask(ri)
    area = read_tif(C.area_file(ri))
    a0 = np.where(np.isfinite(area), area, 0.0)
    A = a0 > 0
    S = fin(C.raw_support_file(ri))
    D = fin(C.domain_file(ri))
    Mha = lambda m: float(a0[m].sum()) / 1e4

    P = {y: fin(OUT_HIS / f"{ri}_HE_{y}_observed.tif") for y in OBS_YEARS}
    P = {y: v for y, v in P.items() if v is not None}
    yrs = sorted(P)
    U_area = np.logical_or.reduce([P[y] for y in yrs if y in AREA_YEARS]) if yrs else None
    U_all = np.logical_or.reduce([P[y] for y in yrs]) if yrs else None
    nyrs_obs = np.sum([P[y] for y in yrs], axis=0) if yrs else None

    n = np.zeros(M.shape, int)
    for yr in C.HIS_YEARS:
        tag = C.hist_pheno_tag(yr)
        he, ma = C.HIS_ANN / f"{ri}_HE_{yr}_{tag}.tif", C.HIS_ANN / f"{ri}_MA_{yr}_{tag}.tif"
        if he.exists() and ma.exists():
            h, m_ = read_tif(he), read_tif(ma)
            n += (np.isfinite(h) & np.isfinite(m_) & (h < m_))
    n_ens = np.zeros(M.shape, int)
    for yr in C.HIS_YEARS:
        if yr >= 2000: continue
        he = C.HIS_ANN / f"{ri}_HE_{yr}_ens25.tif"
        if he.exists(): n_ens += np.isfinite(read_tif(he))
    clim = fin(C.CLIM_HIS / "ensemble" / f"ensemble_historical_{ri}_{LAB}_{C.HIS_P}_mean.tif")

    print(f"\n==================== {ri} ====================")
    print(f"面积文件：{C.area_file(ri)}")
    print(f"mask 文件：{C.MASK_DIR / f'{ri}_mask_0.5deg.tif'}")
    print(f"1 km 观测缓存年份：{yrs[0] if yrs else '-'}–{yrs[-1] if yrs else '-'}（{len(yrs)} 年）")

    def line(name, m):
        r = dict(ri=ri, set=name, cells=int(m.sum()), area_Mha=round(Mha(m), 3))
        rows.append(r); print(f"  {name:<44s} {r['cells']:>6d} 格网  {r['area_Mha']:>7.2f} Mha")

    print("  ---- 各集合 ----")
    line("全部 面积>0（1 km，2000–2010 平均）", A)
    line("mask", M)
    if yrs:
        line(f"1 km 观测 {yrs[0]} 年（最早一年）", P[yrs[0]])
        line("1 km 观测 2000–2010 任一年（并集）", U_area)
        line(f"1 km 观测 {yrs[0]}–{yrs[-1]} 任一年（并集）", U_all)
    if S is not None: line("原始支撑区（01：mask ∩ 物候≥25 年）", S)
    if D is not None: line("统一格网域 D（03b：支撑区 ∩ 面积>0）", D)

    print("  ---- mask 和 1 km 逐年分布的关系 ----")
    if yrs:
        for y in yrs:
            same = int((M & P[y]).sum()); mo = int((M & ~P[y]).sum()); po = int((~M & P[y]).sum())
            print(f"    {y}: 交 {same:>5d}  仅 mask {mo:>5d}  仅 1 km {po:>5d}"
                  + ("   ← mask 与这一年完全一致" if mo == 0 and po == 0 else ""))
        for nm, U in (("2000–2010 并集", U_area), ("全部年份并集", U_all)):
            print(f"    {nm}: 交 {int((M & U).sum())}  仅 mask {int((M & ~U).sum())}  仅 1 km {int((~M & U).sum())}")

    print("  ---- 现在的 D 各步丢掉了什么 ----")
    line("面积>0 但不在 mask（被 mask 丢掉）", A & ~M)
    line("mask 内但面积=0（03b 丢掉）", M & ~A)
    if S is not None:
        line("mask 内、面积>0，但物候 <25 年（01 丢掉）", M & A & ~S)
    print(f"    D 覆盖全部面积 {100 * Mha(D) / Mha(A):.1f}%" if D is not None and Mha(A) else "")

    print("  ---- 如果取并集：新增格网的数据是否已有 ----")
    X = A & ~M
    if X.any():
        line("新增候选（面积>0 但不在 mask）", X)
        line("  其中 基准物候 ≥25/30 年（不套 mask 计）", X & (n >= C.SUPPORT_MIN_YEARS))
        line("  其中 1981–1999 有 ens25 物候（≥1 年）", X & (n_ens > 0))
        if clim is not None:
            line("  其中 已有基准气候（02 已提取）", X & clim)
        h = np.bincount(n[X], minlength=len(C.HIS_YEARS) + 1)
        print("    新增候选的基准物候有效年数分布（年数:格网数）：",
              {k: int(v) for k, v in enumerate(h) if v})
        if nyrs_obs is not None:
            h2 = np.bincount(nyrs_obs[X].astype(int), minlength=len(yrs) + 1)
            print("    新增候选的 1 km 观测出现年数分布（年数:格网数）：",
                  {k: int(v) for k, v in enumerate(h2) if v})
    else:
        print("    面积>0 的格网全部在 mask 内，取并集不会新增格网")

out = C.OUTPUT / f"analysis_domain_{C.TAG}" / "grid_diag_1013.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\n[saved] {out}")
