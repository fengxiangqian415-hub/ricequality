# ============================================================
# 31_annual_classification_1031.py —— 汇总顺序敏感性：先分级再平均（逐年、逐成员） vs 先平均再分级（正文）
#   回应审稿：I[E(Q) ≥ c] ≠ E[I(Q ≥ c)]；"每年"损失是否等于逐年先分级再平均的期望
#   三种口径（同一阈值 Q_LOW/Q_HIGH、同一锚点、同一面积权重、同一格网域 D）：
#     ens     正文口径：集合平均性状 → Q → 分级（11 号 zone 栅格；此处由 ensemble tif 重算，结果应与 11 一致）
#     member  成员口径：每个成员 30 年平均性状 → Q → 分级，再对成员取中位数/范围（Table S8 口径）
#     year    逐年口径：每个成员每一年的预测性状 → Q → 分级 → 优质指示 I；
#             期望优质占比 = Σ a·mean_{m,y} I / Σ a；期望单价 p̄ = P_ORD + (P_HQ − P_ORD)·P(premium)
#             产值 V_m = Σ a·HRY_m·p̄_m（HRY 为该成员时段均值，10 号；逐年产量未缓存，见说明）
#   基准：历史逐年预测按 5 个 GCM（基准物候各成员共用，02 号说明）；成员 (GGCM, GCM) 的基准 = 对应 GCM
#   输入：pred_dir()/{scen}/annual_nc/{name}_{scen}_{ri}_{period}_annual_pred.nc（04 号，含 HRR、CD、PC 逐年）
#         head_rice_yield（10 号）、面积权重、统一格网域 D
#   输出：out_dir("sensitivity")/annual_classification.csv、annual_classification_members.csv
#   用法：python 31_annual_classification_1031.py            （约数分钟；逐年文件 25 成员 × 2 情景 × 2 季）
#   [1108f] 优质占比统一用“有效面积”分母（与 11 号分区、23 号敏感性、SI 补表 5/7 一致）：
#           没有有效分级的格点（该成员该格缺预测、Q 为 NaN）不进分母。原来 (cls == 2).astype(float) 把 NaN 记成 0，
#           等于把缺预测格点算成“非优质”，成员占比被覆盖率拉低（晚稻 SSP5-8.5 成员上限 36.9% → 应为 38.7%）。
# ============================================================
import numpy as np
import pandas as pd
import xarray as xr

import config_1004 as CFG
import q_common_1004 as C

PRED = CFG.pred_dir()
HP, FP = CFG.HIS_P, CFG.FUT_P
P_ORD_T, P_HQ_T = C.P_ORD * 1000.0, C.P_PREM * 1000.0          # CNY/t
USD = C.CNY_PER_USD
TRAITS = ("HRR", "CD", "PC")


def annual_traits(name, scen, ri, period):
    fp = PRED / scen / "annual_nc" / f"{name}_{scen}_{ri}_{period}_annual_pred.nc"   # 04 号写在 {mode}/{scen}/annual_nc/
    if not fp.exists():
        raise FileNotFoundError(fp)
    with xr.open_dataset(fp) as ds:
        return {t: ds[t].values.astype("float64") for t in TRAITS}        # (year, lat, lon)


def premium_prob(tr):
    """逐年 Q → 优质指示 → 年均 P(premium)，以及逐年 Q 的时段均值（供核对）"""
    q = C.q_index(tr["HRR"], tr["CD"], tr["PC"])
    ind = (q >= C.Q_HIGH).astype("float64")
    ind[~np.isfinite(q)] = np.nan
    return np.nanmean(ind, axis=0), np.nanmean(q, axis=0)


def wshare(x, a):
    ok = np.isfinite(x) & np.isfinite(a)
    return 100.0 * float(np.sum(x[ok] * a[ok]) / np.sum(a[ok]))


def prem_ind(cls):
    """[1108f] 优质指示：1 = 优质，0 = 中等/普通，NaN = 无有效分级（缺预测）→ wshare 中不进分母（有效面积口径）"""
    cls = np.asarray(cls, dtype="float64")
    ind = (cls == 2).astype("float64")
    ind[~np.isfinite(cls)] = np.nan
    return ind


def value(a_km2, hry, p_t):
    a = a_km2 * C.KM2_TO_HA
    ok = np.isfinite(a) & np.isfinite(hry) & np.isfinite(p_t)
    return float(np.sum(a[ok] * hry[ok] * p_t[ok])) / 1e8                # 亿元


def hry_path(ri, ssp=None, model=None, gcm=None):
    if ssp is None:
        return C.read(C.find("hry_hist", ri=ri, model=model, gcm=gcm))
    return C.read(C.find("hry_fut", ri=ri, ssp=ssp, model=model, gcm=gcm))


def hry_ens(ri, scen):
    p = C.HRY / f"head_rice_yield_{ri}_{scen}.tif"
    return C.read(str(p)) if p.exists() else None


rows, mrows = [], []
for ri in C.RIS:
    a = C.area_and_mask(ri)
    # ---- 基准：5 个 GCM 的逐年
    Ph, Qh = {}, {}
    for gcm in C.GCMS:
        Ph[gcm], Qh[gcm] = premium_prob(annual_traits(gcm, "historical", ri, HP))
    P_hist_year = np.nanmean(np.stack(list(Ph.values())), axis=0)         # E_{gcm,y} I
    ens_h = C.traits_ens(ri)
    cls_ens_h = C.classify(C.q_index(*ens_h))
    share_ens_h = wshare(prem_ind(cls_ens_h), a)
    share_mem_h = [wshare(prem_ind(C.classify(C.q_index(*C.traits_hist(ri, g)))), a) for g in C.GCMS]
    share_year_h = wshare(P_hist_year, a)
    hry_eh = hry_ens(ri, "historical")

    for ssp in C.SSPS:
        P_mem, S_mem_cls, V = [], [], []
        for m in C.GGCMS:
            for g in C.GCMS:
                Pf, _ = premium_prob(annual_traits(f"{m}_{g}", ssp, ri, FP))
                P_mem.append(Pf)
                cls_mf = C.classify(C.q_index(*C.traits_fut(ri, ssp, m, g)))
                S_mem_cls.append(wshare(prem_ind(cls_mf), a))
                # 成员产值：期望单价（逐年分级）× 成员时段均值 HRY
                hry_h, hry_f = hry_path(ri, None, m, g), hry_path(ri, ssp, m, g)
                p_h = P_ORD_T + (P_HQ_T - P_ORD_T) * Ph[g]
                p_f = P_ORD_T + (P_HQ_T - P_ORD_T) * Pf
                Vh, Vf = value(a, hry_h, p_h), value(a, hry_f, p_f)
                ok = np.isfinite(a) & np.isfinite(hry_h) & np.isfinite(hry_f) & np.isfinite(p_h) & np.isfinite(p_f)
                aa = a * C.KM2_TO_HA
                grade = float(np.sum(aa[ok] * hry_f[ok] * (p_h[ok] - p_f[ok]))) / 1e8
                yield_ = float(np.sum(aa[ok] * p_h[ok] * (hry_h[ok] - hry_f[ok]))) / 1e8
                V.append(dict(ri=ri, ssp=ssp, model=m, gcm=g, share_prem_year_pct=wshare(Pf, a),
                              share_prem_member_pct=S_mem_cls[-1], V_hist_yi=Vh, V_fut_yi=Vf,
                              net_loss_yi=Vh - Vf, grade_effect_yi=grade, yield_effect_yi=yield_))
        mrows += V
        P_fut_year = np.nanmean(np.stack(P_mem), axis=0)
        ens_f = C.traits_ens(ri, ssp)
        cls_ens_f = C.classify(C.q_index(*ens_f))
        dfm = pd.DataFrame(V)
        # 正文口径的等级效应（集合平均 HRY，确定等级）供并排
        hry_ef = hry_ens(ri, ssp)
        if hry_eh is not None and hry_ef is not None:
            p_eh = np.where(cls_ens_h == 2, P_HQ_T, P_ORD_T).astype(float); p_eh[~np.isfinite(cls_ens_h)] = np.nan
            p_ef = np.where(cls_ens_f == 2, P_HQ_T, P_ORD_T).astype(float); p_ef[~np.isfinite(cls_ens_f)] = np.nan
            ok = np.isfinite(a) & np.isfinite(hry_eh) & np.isfinite(hry_ef) & np.isfinite(p_eh) & np.isfinite(p_ef)
            aa = a * C.KM2_TO_HA
            grade_ens = float(np.sum(aa[ok] * hry_ef[ok] * (p_eh[ok] - p_ef[ok]))) / 1e8
        else:
            grade_ens = np.nan
        q = lambda s, k: float(np.percentile(s, k))
        rows.append(dict(
            ri=ri, ssp=ssp,
            base_prem_ens=share_ens_h, base_prem_member_median=float(np.median(share_mem_h)),
            base_prem_member_min=min(share_mem_h), base_prem_member_max=max(share_mem_h), base_prem_year_expected=share_year_h,
            fut_prem_ens=wshare(prem_ind(cls_ens_f), a),
            fut_prem_member_median=float(dfm.share_prem_member_pct.median()),
            fut_prem_member_min=dfm.share_prem_member_pct.min(), fut_prem_member_max=dfm.share_prem_member_pct.max(),
            fut_prem_year_expected=wshare(P_fut_year, a),
            fut_prem_year_member_median=float(dfm.share_prem_year_pct.median()),
            fut_prem_year_member_min=dfm.share_prem_year_pct.min(), fut_prem_year_member_max=dfm.share_prem_year_pct.max(),
            grade_loss_ens_busd=grade_ens / 10.0 / USD,                       # 亿元 → 10 亿美元
            grade_loss_year_mean_busd=dfm.grade_effect_yi.mean() / 10.0 / USD,
            grade_loss_year_median_busd=dfm.grade_effect_yi.median() / 10.0 / USD,
            grade_loss_year_min_busd=dfm.grade_effect_yi.min() / 10.0 / USD,
            grade_loss_year_max_busd=dfm.grade_effect_yi.max() / 10.0 / USD,
            net_loss_year_mean_busd=dfm.net_loss_yi.mean() / 10.0 / USD,
            net_loss_year_median_busd=dfm.net_loss_yi.median() / 10.0 / USD,
            n_members_net_loss=int((dfm.net_loss_yi > 0).sum())))
        print(f"[{ri} {ssp}] 优质占比  基准 ens {share_ens_h:.1f} | 成员中位 {np.median(share_mem_h):.1f} | 逐年期望 {share_year_h:.1f}"
              f"   未来 ens {rows[-1]['fut_prem_ens']:.1f} | 成员中位 {rows[-1]['fut_prem_member_median']:.1f}"
              f" | 逐年期望 {rows[-1]['fut_prem_year_expected']:.1f}"
              f"   等级损失(10亿美元) ens {rows[-1]['grade_loss_ens_busd']:.2f} | 逐年成员均值 {rows[-1]['grade_loss_year_mean_busd']:.2f}"
              f" [{rows[-1]['grade_loss_year_min_busd']:.2f}, {rows[-1]['grade_loss_year_max_busd']:.2f}]")

out = pd.DataFrame(rows); mem = pd.DataFrame(mrows)
# 全国 = 两季同成员相加
nat = mem.groupby(["ssp", "model", "gcm"])[["V_hist_yi", "V_fut_yi", "net_loss_yi", "grade_effect_yi", "yield_effect_yi"]].sum().reset_index()
for ssp in C.SSPS:
    d = nat[nat.ssp == ssp]
    print(f"[national {ssp}] 逐年口径 等级损失 均值 {d.grade_effect_yi.mean()/10/USD:.2f}，中位 {d.grade_effect_yi.median()/10/USD:.2f}"
          f" [{d.grade_effect_yi.min()/10/USD:.2f}, {d.grade_effect_yi.max()/10/USD:.2f}] 10亿美元；"
          f"净损失 均值 {d.net_loss_yi.mean()/10/USD:.2f}，{int((d.net_loss_yi>0).sum())}/25 成员为净损失")
    rows.append(dict(ri="national", ssp=ssp, grade_loss_year_mean_busd=d.grade_effect_yi.mean() / 10 / USD,
                     grade_loss_year_median_busd=d.grade_effect_yi.median() / 10 / USD,
                     grade_loss_year_min_busd=d.grade_effect_yi.min() / 10 / USD,
                     grade_loss_year_max_busd=d.grade_effect_yi.max() / 10 / USD,
                     net_loss_year_mean_busd=d.net_loss_yi.mean() / 10 / USD,
                     net_loss_year_median_busd=d.net_loss_yi.median() / 10 / USD,
                     n_members_net_loss=int((d.net_loss_yi > 0).sum())))
out = pd.DataFrame(rows)
OUT = CFG.out_dir("sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "annual_classification.csv", index=False, float_format="%.3f")
mem.to_csv(OUT / "annual_classification_members.csv", index=False, float_format="%.4f")
print(f"\n[saved] {OUT / 'annual_classification.csv'}\n        {OUT / 'annual_classification_members.csv'}")
print("说明：逐年口径的产值用成员时段均值 HRY × 逐年分级的期望单价；逐年产量未缓存，故产量—等级的年内协变未计入。")
