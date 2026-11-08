# =====================================================
# 24_econ_members_1004.py — #910 accounting table with ensemble ranges
# [1004] 运行顺序 24（依赖 04、10、11）；价格、汇率、阈值、支撑区与 14/16 号一致
#   年均值：HRY 为时段多年均值 × 固定价格，结果即年均总产值（不是 30 年累计）
#
# For each of the 25 GGCM × GCM members and each rice type:
#   baseline value V_h = Σ a · HRY_h(member) · p(class_h)       class_h from baseline Q of that GCM
#   future   value V_f = Σ a · HRY_f(member) · p(class_f)       class_f from that member's future Q
#   output effect  = Σ a · p_h · (HRY_h − HRY_f)
#   grade effect   = Σ a · HRY_f · (p_h − p_f)                  (sums exactly to V_h − V_f)
# National = ri1 + ri2 of the same member. Reported as median [min–max] across members,
# next to the ensemble-mean estimate used in the main text.
#
# Notes
# * The member median need not equal the ensemble-mean estimate (classification is nonlinear);
#   report the ensemble-mean value as the central estimate and the member range as spread.
# * BASELINE = "member" pairs each member with its own baseline HRY (recommended).
#   BASELINE = "common" uses one ensemble-mean baseline for all members, which matches the
#   current production_loss_summary.csv (V_his identical for both SSPs).
# * Values are mean annual gross head-rice value; broken rice is not valued (see Limitations).
#
# Output: OUTDIR/econ_members_long.csv and econ_accounting_table.csv
# =====================================================
import os
import numpy as np
import pandas as pd
import q_common_1004 as C

BASELINE = "member"          # or "common"
W = (1.0, 1.0, 1.0)          # main Q weights

def price(cls):
    p = np.where(cls == 2, C.P_PREM, C.P_ORD) * 1000.0   # CNY/t
    p[~np.isfinite(cls)] = np.nan
    return p

def value_terms(a_km2, hry_h, cls_h, hry_f, cls_f):
    a = a_km2 * C.KM2_TO_HA
    ph, pf = price(cls_h), price(cls_f)
    ok = np.isfinite(a) & np.isfinite(hry_h) & np.isfinite(hry_f) & np.isfinite(ph) & np.isfinite(pf)
    s = lambda x: float(np.nansum(np.where(ok, x, 0.0))) / 1e8          # → 1e8 CNY (亿元)
    Vh = s(a * hry_h * ph); Vf = s(a * hry_f * pf)
    out = s(a * ph * (hry_h - hry_f)); grade = s(a * hry_f * (ph - pf))
    return dict(V_hist=Vh, V_fut=Vf, output_effect=out, grade_effect=grade, net_loss=Vh - Vf)

def main():
    C.check_anchors(); os.makedirs(C.OUTDIR, exist_ok=True)
    rows = []
    for ri in C.RIS:
        a = C.area_and_mask(ri)
        cls_h_gcm = {g: C.classify(C.q_index(*C.traits_hist(ri, g), w=W)) for g in C.GCMS}
        hry_h_mem = {(m, g): C.read(C.find("hry_hist", ri=ri, model=m, gcm=g))
                     for m in C.GGCMS for g in C.GCMS}
        if BASELINE == "common":
            hry_h_common = np.nanmean(np.stack(list(hry_h_mem.values())), axis=0)
            qh_common = np.nanmean(np.stack([C.q_index(*C.traits_hist(ri, g), w=W) for g in C.GCMS]), axis=0)
            cls_h_common = C.classify(qh_common)
        for s in C.SSPS:
            for m in C.GGCMS:
                for g in C.GCMS:
                    hry_f = C.read(C.find("hry_fut", ri=ri, ssp=s, model=m, gcm=g))
                    cls_f = C.classify(C.q_index(*C.traits_fut(ri, s, m, g), w=W))
                    if BASELINE == "common":
                        hry_h, cls_h = hry_h_common, cls_h_common
                    else:
                        hry_h, cls_h = hry_h_mem[(m, g)], cls_h_gcm[g]
                    rows.append(dict(ssp=s, ri=ri, model=m, gcm=g,
                                     **value_terms(a, hry_h, cls_h, hry_f, cls_f)))
            print("done", ri, s)

    df = pd.DataFrame(rows)
    nat = (df.groupby(["ssp", "model", "gcm"], as_index=False)
             [["V_hist", "V_fut", "output_effect", "grade_effect", "net_loss"]].sum())
    nat["ri"] = "ri1+ri2"
    df = pd.concat([df, nat], ignore_index=True)
    df["net_loss_pct"] = 100 * df.net_loss / df.V_hist
    df.to_csv(os.path.join(C.OUTDIR, "econ_members_long.csv"), index=False)

    usd = lambda x: x * 0.1 / C.CNY_PER_USD             # 1e8 CNY → billion US$
    tab = []
    for (s, ri), g in df.groupby(["ssp", "ri"], sort=False):
        r = dict(ssp=s, ri=ri, n_members=len(g))
        for col in ["V_hist", "V_fut", "output_effect", "grade_effect", "net_loss"]:
            v = usd(g[col])
            r[col + "_bnUSD"] = f"{v.median():.2f} [{v.min():.2f}–{v.max():.2f}]"
        p = g.net_loss_pct
        r["net_loss_pct"] = f"{p.median():.1f} [{p.min():.1f}–{p.max():.1f}]"
        r["members_with_net_loss"] = int((g.net_loss > 0).sum())
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(C.OUTDIR, "econ_accounting_table.csv"), index=False)
    print(tab.to_string(index=False))

if __name__ == "__main__":
    main()
