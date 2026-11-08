import numpy as np
import pandas as pd

import config_1004 as C
import stats_common_1004 as S

H_DIR, F_DIR, HP, FP = C.CLIM_HIS, C.CLIM_FUT, C.HIS_P, C.FUT_P
OUT = C.OUTPUT / f"decomposition_373_{C.TAG}"
OUT.mkdir(parents=True, exist_ok=True)
VARS = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "CTSR": "cumul_solar_MJ_m2",
        "TSR": "mean_daily_solar_MJ_m2", "days": "gf_days", "Rain": "cumul_precip_mm",
        "RHavg": "mean_humidity_pct"}
REL = {"CTSR", "TSR", "days", "Rain"}


def ens(ri, lab, ssp=None):
    """集合平均场：A、A′（ssp=None）或 B、C"""
    if ssp is None:
        return dict(A=S.read(H_DIR / "ensemble" / f"ensemble_historical_{ri}_{lab}_{HP}_mean.tif"),
                    A2=S.read(H_DIR / "fixed_window" / "ensemble" / f"ensemble_historical_{ri}_fixedwindow_{lab}_{HP}_mean.tif"))
    return dict(B=S.read(F_DIR / "ensemble" / f"ensemble_{ssp}_{ri}_{lab}_{FP}_mean.tif"),
                C=S.read(F_DIR / "fixed_window" / "ensemble" / f"ensemble_{ssp}_{ri}_fixedwindow_{lab}_{FP}_mean.tif"))


def member(ri, ssp, m, g, lab):
    return dict(A=S.read(H_DIR / g / f"{g}_historical_{ri}_{lab}_{HP}_mean.tif"),
                A2=S.read(H_DIR / "fixed_window" / g / f"{g}_historical_{ri}_fixedwindow_{lab}_{HP}_mean.tif"),
                B=S.read(F_DIR / ssp / m / g / f"{m}_{g}_{ssp}_{ri}_{lab}_{FP}_mean.tif"),
                C=S.read(F_DIR / "fixed_window" / g / f"{g}_{ssp}_{ri}_fixedwindow_{lab}_{FP}_mean.tif"))


def terms(A, A2, B, Cc, v):
    r = dict(A=A, A_fixed=A2, B=B, C=Cc, total=B - A, climate=Cc - A2, shift=B - Cc, base_window=A2 - A)
    if v in REL:
        r.update(total_pct=S.rel(B, A), climate_pct=S.rel(Cc, A2), shift_pct=S.rel(B, Cc))
    return r


mem_rows, sum_rows = [], []
for ri in C.RI_TYPES:
    w = S.weights(ri)
    a_km2 = S.area(ri)
    for v, lab in VARS.items():
        e = {**ens(ri, lab), **{s: ens(ri, lab, s) for s in C.SSPS}}
        k06 = S.cells(ri, e["A"], *[e[s]["B"] for s in C.SSPS])
        k = k06 & np.isfinite(e["A2"])
        for s in C.SSPS:
            k &= np.isfinite(e[s]["C"])
        if (k06 & ~k).any():
            print(f"  ⚠️ [{ri}/{v}] {int((k06 & ~k).sum())} 个格网缺固定窗口 A′/C，已从分解中剔除（与 06 的格网不再完全相同）")
        wm = lambda a: S.wmean(a, w, k)
        area_mha = float(np.nansum(a_km2[k])) / 1e4
        A, A2 = wm(e["A"]), wm(e["A2"])
        for s in C.SSPS:
            cen = terms(A, A2, wm(e[s]["B"]), wm(e[s]["C"]), v)
            mrs = []
            for m in C.CROP_MODELS:
                for g in C.GCMS:
                    d = member(ri, s, m, g, lab)
                    if any(x is None for x in d.values()):
                        continue
                    km = k & np.isfinite(d["A"]) & np.isfinite(d["A2"]) & np.isfinite(d["B"]) & np.isfinite(d["C"])
                    if not km.any():
                        continue
                    mw = lambda a: S.wmean(a, w, km)
                    r = dict(ri=ri, ssp=s, model=m, gcm=g, var=v,
                             coverage_pct=100 * float(np.nansum(w[km]) / np.nansum(w[k])),
                             **terms(mw(d["A"]), mw(d["A2"]), mw(d["B"]), mw(d["C"]), v))
                    mem_rows.append(r); mrs.append(r)
            M = pd.DataFrame(mrs)
            row = dict(ri=ri, ssp=s, var=v, n_cells=int(k.sum()), area_Mha=round(area_mha, 3),
                       baseline=cen["A"], future=cen["B"], total=cen["total"], climate=cen["climate"],
                       shift=cen["shift"], base_window=cen["base_window"],
                       shift_share_pct=100 * cen["shift"] / cen["total"] if cen["total"] else np.nan,
                       n_members=len(M),
                       member_min_coverage_pct=round(M.coverage_pct.min(), 1) if len(M) else np.nan)
            if v in REL:
                row.update(total_pct=cen["total_pct"], climate_pct=cen["climate_pct"], shift_pct=cen["shift_pct"])
            if len(M):
                for c in ["total", "climate", "shift", "base_window"]:
                    row[f"{c}_members"] = S.mrange(M[c])
                if v == "Tavg":
                    row["shift_share_pct_members"] = S.mrange(100 * M["shift"] / M["total"], "{:.0f}")
                if v in REL:
                    for c in ["total_pct", "climate_pct", "shift_pct"]:
                        row[f"{c}_members"] = S.mrange(M[c], "{:+.1f}")
            sum_rows.append(row)
        print(f"[{ri}/{v}] 格网 {int(k.sum())}  面积 {area_mha:.2f} Mha  基准 {A:.2f}")

pd.DataFrame(mem_rows).to_csv(OUT / "members.csv", index=False, float_format="%.4f")
summ = pd.DataFrame(sum_rows)
summ.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 30)
show = lambda df, cols: print(df[[c for c in cols if c in df.columns]].round(2).to_string(index=False))
print(f"\n===== 灌浆期均温分解（°C；集合平均，{'面积加权' if C.SUMMARY_WEIGHT == 'area' else '逐格等权'}；成员中位数 [最小, 最大]）=====")
show(summ[summ["var"] == "Tavg"], ["ri", "ssp", "baseline", "total", "climate", "shift", "base_window",
                                   "shift_share_pct", "total_members", "climate_members", "shift_members"])
print("\n===== CTSR = TSR × 天数：相对变化（%，B vs A）=====")
show(summ[summ["var"].isin(["CTSR", "TSR", "days"])], ["ri", "ssp", "var", "baseline", "total", "total_pct",
                                                       "total_pct_members"])
print("\n===== 其他变量（绝对变化）=====")
show(summ[summ["var"].isin(["Tdif", "Rain", "RHavg"])], ["ri", "ssp", "var", "baseline", "total", "climate", "shift"])
print(f"\n输出: {OUT}")
