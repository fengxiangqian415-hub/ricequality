# =====================================================
# 24b_province_members_1004.py —— [1106] 分省单位面积经济损失的 25 成员分布（供 Fig. 4 底部箱线图）
#   运行顺序：10（逐成员 HRY）→ 11 → 14 → 16（生成 province_frac_1km.npz）→ 本脚本 → 再跑 16 出箱线图
#   口径与 24 号完全一致（BASELINE="member"：每个成员配自己的基准 HRY；基准分级用该 GCM 的基准 Q；
#   未来分级用该成员自己的 Q；价格、汇率、阈值、支撑区来自 config_1004）。
#   分省拆分用 16 号保存的 province_frac_1km.npz（每个 0.5° 格在各省的 1 km 水稻面积占比），
#   省份分组（<0.5 Mha 并入 Other provinces）沿用 npz 里的 groups。
#   输出：OUTDIR(sensitivity)/province_loss_members.csv
#        列：ssp, model, gcm, province, area_kha, loss_yi(亿元), usd_ha(US$ ha⁻¹ yr⁻¹)，含 National 行（ri1+ri2 合并）
# =====================================================
import os
import numpy as np
import pandas as pd
import q_common_1004 as C
import config_1004 as CFG

W = (1.0, 1.0, 1.0)
NPZ = CFG.out_dir("loss") / "province_frac_1km.npz"

def price(cls):
    p = np.where(cls == 2, C.P_PREM, C.P_ORD) * 1000.0   # CNY/t
    p[~np.isfinite(cls)] = np.nan
    return p

def main():
    C.check_anchors(); os.makedirs(C.OUTDIR, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    names, groups = [str(x) for x in z["names"]], [str(x) for x in z["groups"]]
    K = len(names)
    rows = []
    for ri in C.RIS:
        F = z[f"frac_{ri}"].astype("float64")                     # (K, H, W)
        a = C.area_and_mask(ri)                                    # km²，支撑区外 NaN
        if F.shape[1:] != a.shape:
            raise SystemExit(f"[{ri}] province_frac 网格 {F.shape[1:]} 与面积栅格 {a.shape} 不一致")
        a_ha = a * C.KM2_TO_HA
        cls_h_gcm = {g: C.classify(C.q_index(*C.traits_hist(ri, g), w=W)) for g in C.GCMS}
        hry_h_mem = {(m, g): C.read(C.find("hry_hist", ri=ri, model=m, gcm=g))
                     for m in C.GGCMS for g in C.GCMS}
        for s in C.SSPS:
            for m in C.GGCMS:
                for g in C.GCMS:
                    hry_f = C.read(C.find("hry_fut", ri=ri, ssp=s, model=m, gcm=g))
                    cls_f = C.classify(C.q_index(*C.traits_fut(ri, s, m, g), w=W))
                    hry_h, cls_h = hry_h_mem[(m, g)], cls_h_gcm[g]
                    ph, pf = price(cls_h), price(cls_f)
                    ok = np.isfinite(a_ha) & np.isfinite(hry_h) & np.isfinite(hry_f) & np.isfinite(ph) & np.isfinite(pf)
                    loss = np.where(ok, a_ha * (hry_h * ph - hry_f * pf), 0.0)      # CNY，正 = 损失
                    area = np.where(ok, a_ha, 0.0)
                    Lk = (F * loss[None]).sum(axis=(1, 2))                          # 各省损失（CNY）
                    Ak = (F * area[None]).sum(axis=(1, 2))                          # 各省面积（ha）
                    for k in range(K):
                        rows.append(dict(ssp=s, ri=ri, model=m, gcm=g, province=names[k], group=groups[k],
                                         loss_cny=float(Lk[k]), area_ha=float(Ak[k])))
            print("done", ri, s)
    df = pd.DataFrame(rows)
    # 省 → 展示分组（Other provinces），两季合并；再加 National
    g = (df.groupby(["ssp", "model", "gcm", "group"], as_index=False)[["loss_cny", "area_ha"]].sum()
           .rename(columns={"group": "province"}))
    nat = df.groupby(["ssp", "model", "gcm"], as_index=False)[["loss_cny", "area_ha"]].sum()
    nat["province"] = "National"
    out = pd.concat([g, nat], ignore_index=True)
    out["area_kha"] = out.area_ha / 1e3
    out["loss_yi"] = out.loss_cny / 1e8
    out["usd_ha"] = out.loss_cny / out.area_ha / C.CNY_PER_USD
    out = out[["ssp", "model", "gcm", "province", "area_kha", "loss_yi", "usd_ha"]]
    fp = os.path.join(C.OUTDIR, "province_loss_members.csv")
    out.to_csv(fp, index=False)
    print(f"[saved] {fp}")
    # 摘要：全国与前几省的中位数 [min, max]
    summ = (out.groupby(["province", "ssp"]).usd_ha
              .agg(median="median", lo="min", hi="max", n="count").round(1).reset_index())
    print(summ[summ.province.isin(["National", "Liaoning", "Jiangsu", "Hunan", "Hainan"])].to_string(index=False))

if __name__ == "__main__":
    main()
