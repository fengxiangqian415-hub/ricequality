import os, itertools
import numpy as np
import pandas as pd
import q_common_1004 as C

SCHEMES = {
    "main (1:1:1)":   (1.0, 1.0, 1.0),
    "no PC":          (1.0, 1.0, 0.0),
    "PC x0.5":        (1.0, 1.0, 0.5),
    "PC x2":          (1.0, 1.0, 2.0),
}
STANDARD_SETS = {
}
ENSEMBLE_Q = "mean_traits"

def member_qs(ri, w):
    """baseline Q per GCM (5) and future Q per member (25) for each SSP."""
    qh = {g: C.q_index(*C.traits_hist(ri, g), w=w) for g in C.GCMS}
    qf = {s: {(m, g): C.q_index(*C.traits_fut(ri, s, m, g), w=w)
              for m in C.GGCMS for g in C.GCMS} for s in C.SSPS}
    return qh, qf

def main():
    C.check_anchors(); os.makedirs(C.OUTDIR, exist_ok=True)
    area = {ri: C.area_and_mask(ri) for ri in C.RIS}
    cache = {}
    for name, w in SCHEMES.items():
        cache[name] = {ri: member_qs(ri, w) for ri in C.RIS}
        print("Q computed:", name)

    def base_mean(name, ri):
        if ENSEMBLE_Q == "mean_traits":
            return C.q_index(*C.traits_ens(ri), w=SCHEMES[name])
        return np.nanmean(np.stack(list(cache[name][ri][0].values())), axis=0)

    def fut_mean(name, ri, s):
        if ENSEMBLE_Q == "mean_traits":
            return C.q_index(*C.traits_ens(ri, s), w=SCHEMES[name])
        return np.nanmean(np.stack(list(cache[name][ri][1][s].values())), axis=0)

    ref = "main (1:1:1)"
    pooled_q = np.concatenate([base_mean(ref, ri).ravel() for ri in C.RIS])
    pooled_a = np.concatenate([area[ri].ravel() for ri in C.RIS])
    ok = np.isfinite(pooled_q) & np.isfinite(pooled_a)
    share_prem = pooled_a[ok & (pooled_q >= C.Q_HIGH)].sum() / pooled_a[ok].sum()
    share_comm = pooled_a[ok & (pooled_q < C.Q_LOW)].sum() / pooled_a[ok].sum()

    rows = []
    for name in SCHEMES:
        pq = np.concatenate([base_mean(name, ri).ravel() for ri in C.RIS])
        thr = {"fixed": (C.Q_LOW, C.Q_HIGH),
               "matched": (C.wquantile(pq, pooled_a, share_comm),
                           C.wquantile(pq, pooled_a, 1 - share_prem))}
        thr.update({f"standard: {k}": v for k, v in STANDARD_SETS.items()})
        for mode, (lo, hi) in thr.items():
            for ri in C.RIS:
                qh, qf = cache[name][ri]; a = area[ri]
                qb = base_mean(name, ri)
                sb, tot = C.shares(C.classify(qb, lo, hi), a)
                rows.append(dict(scheme=name, threshold=mode, q_low=round(lo, 4), q_high=round(hi, 4),
                                 ri=ri, period="baseline", premium_ens=sb["premium"],
                                 common_ens=sb["common"], premium_med=np.nan, premium_min=np.nan,
                                 premium_max=np.nan, area_km2=tot))
                for s in C.SSPS:
                    qe = fut_mean(name, ri, s)
                    se, _ = C.shares(C.classify(qe, lo, hi), a)
                    mem = [C.shares(C.classify(q, lo, hi), a)[0]["premium"] for q in qf[s].values()]
                    rows.append(dict(scheme=name, threshold=mode, q_low=round(lo, 4), q_high=round(hi, 4),
                                     ri=ri, period=s, premium_ens=se["premium"], common_ens=se["common"],
                                     premium_med=float(np.median(mem)), premium_min=float(np.min(mem)),
                                     premium_max=float(np.max(mem)), area_km2=tot))
        print("classified:", name)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUTDIR, "q_sensitivity_long.csv"), index=False)

    chk = []
    for (sch, thr, ri), g in df.groupby(["scheme", "threshold", "ri"], sort=False):
        b = g[g.period == "baseline"].premium_ens.iloc[0]
        p126 = g[g.period == "ssp126"].iloc[0]; p585 = g[g.period == "ssp585"].iloc[0]
        chk.append(dict(scheme=sch, threshold=thr, ri=ri, base_premium=round(b, 2),
                        ssp126_premium=round(p126.premium_ens, 2), ssp585_premium=round(p585.premium_ens, 2),
                        contracts_both=(p126.premium_ens < b) and (p585.premium_ens < b),
                        ssp585_lower_ens=p585.premium_ens < p126.premium_ens,
                        ssp585_lower_member_median=p585.premium_med < p126.premium_med))
    ck = pd.DataFrame(chk)
    ck.to_csv(os.path.join(C.OUTDIR, "q_sensitivity_checks.csv"), index=False)
    print(ck.to_string(index=False))

if __name__ == "__main__":
    main()
