import argparse
import numpy as np
import pandas as pd

import config_1004 as CFG
import q_common_1004 as C

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1000)
p.add_argument("--rmse", default="7.84,5.89,0.95", help="HRR,CD,PC 测试集 RMSE（百分点）")
p.add_argument("--ntest", type=int, default=228)
p.add_argument("--resid", default=None, help="可选：逐记录测试残差 CSV，列 HRR,CD,PC（用于性状间相关）")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
RMSE = np.array([float(v) for v in args.rmse.split(",")])
rng = np.random.default_rng(args.seed)
TR = ("HRR", "CD", "PC")
P_ORD_T, P_HQ_T = C.P_ORD * 1000.0, C.P_PREM * 1000.0
USD = C.CNY_PER_USD

if args.resid:
    R = pd.read_csv(args.resid)[list(TR)].dropna()
    CORR = np.corrcoef(R.values.T)
    print("残差相关（测试集）：\n", pd.DataFrame(CORR, index=TR, columns=TR).round(2))
else:
    CORR = np.eye(3)
    print("未提供残差表：性状间误差按独立处理")
L = np.linalg.cholesky(CORR)


def draw(shape, scale):
    """相关正态误差：shape = (..., ) 每个性状一个数组，scale = 3 个标准差"""
    z = rng.standard_normal((3,) + shape)
    z = np.tensordot(L, z, axes=(1, 0))
    return [z[i] * scale[i] for i in range(3)]


def hry(ri, scen):
    pth = C.HRY / f"head_rice_yield_{ri}_{scen}.tif"
    return C.read(str(pth)) if pth.exists() else None


def metrics(a, th, tf, hry_h, hry_f):
    """给定扰动后的性状 → 优质占比（基准/未来）、优质面积损失、等级效应"""
    ch, cf = C.classify(C.q_index(*th)), C.classify(C.q_index(*tf))
    ok = np.isfinite(a) & np.isfinite(ch) & np.isfinite(cf)
    A = a[ok].sum()
    sh, sf = 100 * a[ok & (ch == 2)].sum() / A, 100 * a[ok & (cf == 2)].sum() / A
    loss_mha = (a[ok & (ch == 2)].sum() - a[ok & (cf == 2)].sum()) / 1e4
    g = np.nan
    if hry_h is not None and hry_f is not None:
        ph = np.where(ch == 2, P_HQ_T, P_ORD_T); pf = np.where(cf == 2, P_HQ_T, P_ORD_T)
        k = ok & np.isfinite(hry_f)
        g = float(np.sum(a[k] * C.KM2_TO_HA * hry_f[k] * (ph[k] - pf[k]))) / 1e8 / 10 / USD
    return sh, sf, loss_mha, g


rows = []
for ri in C.RIS:
    a = C.area_and_mask(ri)
    th0 = C.traits_ens(ri)
    hh = hry(ri, "historical")
    shape = th0[0].shape
    for ssp in C.SSPS:
        tf0 = C.traits_ens(ri, ssp)
        hf = hry(ri, ssp)
        base = metrics(a, th0, tf0, hh, hf)
        for struct in ("iid", "cell", "common", "cell+common"):
            res = []
            for _ in range(args.n):
                eh = [np.zeros(shape) for _ in TR]; ef = [np.zeros(shape) for _ in TR]
                if struct == "iid":
                    eh, ef = draw(shape, RMSE), draw(shape, RMSE)
                if struct in ("cell", "cell+common"):
                    e = draw(shape, RMSE); eh = [x + y for x, y in zip(eh, e)]; ef = [x + y for x, y in zip(ef, e)]
                if struct in ("common", "cell+common"):
                    d = draw((), RMSE / np.sqrt(args.ntest))
                    eh = [x + y for x, y in zip(eh, d)]; ef = [x + y for x, y in zip(ef, d)]
                th = [t + e for t, e in zip(th0, eh)]; tf = [t + e for t, e in zip(tf0, ef)]
                res.append(metrics(a, th, tf, hh, hf))
            r = np.array(res, float)
            q = lambda j, k: float(np.nanpercentile(r[:, j], k))
            rows.append(dict(ri=ri, ssp=ssp, error=struct,
                             base_prem_point=base[0], base_prem_p50=q(0, 50), base_prem_p5=q(0, 5), base_prem_p95=q(0, 95),
                             fut_prem_point=base[1], fut_prem_p50=q(1, 50), fut_prem_p5=q(1, 5), fut_prem_p95=q(1, 95),
                             loss_mha_point=base[2], loss_mha_p50=q(2, 50), loss_mha_p5=q(2, 5), loss_mha_p95=q(2, 95),
                             grade_busd_point=base[3], grade_busd_p50=q(3, 50), grade_busd_p5=q(3, 5), grade_busd_p95=q(3, 95),
                             frac_fut_below_base=float(np.mean(r[:, 1] < r[:, 0]))))
            print(f"[{ri} {ssp} {struct:12s}] 优质占比 基准 {base[0]:.1f} → MC {q(0,50):.1f} [{q(0,5):.1f}, {q(0,95):.1f}]；"
                  f"未来 {base[1]:.1f} → {q(1,50):.1f} [{q(1,5):.1f}, {q(1,95):.1f}]；"
                  f"优质面积损失 {base[2]:.2f} → {q(2,50):.2f} [{q(2,5):.2f}, {q(2,95):.2f}] Mha；"
                  f"等级效应 {base[3]:.2f} → {q(3,50):.2f} [{q(3,5):.2f}, {q(3,95):.2f}] 10亿美元；"
                  f"未来<基准 的抽样占比 {np.mean(r[:,1] < r[:,0]):.3f}")
OUT = CFG.out_dir("sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "quality_error_mc.csv", index=False, float_format="%.3f")
print(f"\n[saved] {OUT / 'quality_error_mc.csv'}")
