# ============================================================
# 36_yield_only_valuation_1106.py —— “只算产量”与“计入品质”的产值变化对比（SI S5.4、补图 23、补表 10；1108f 编号）
#   问题：若像多数气候影响评估那样只按稻谷产量 × 稻谷价格估值，会得到什么？计入整精米率和等级后又是什么？
#   三个估值层级（同一支撑区、同一面积权重、同一 25 成员）：
#     T1 稻谷产量（paddy-only）     ΔV = Σ a·(Y_f − Y_h)·P_paddy            Y = GGCM 稻谷产量（与 10 号同一 NetCDF）
#     T2 整精米产量，单一精米价     ΔV = Σ a·(HRY_f − HRY_h)·P_ord          HRY = 10 号逐成员整精米产量
#     T3 整精米产量 + 等级两档价     ΔV = Σ a·(HRY_f·p_f − HRY_h·p_h)        = 正文主核算（14 号集合平均 / 24 号逐成员）
#   价格：P_paddy = 2020–2024 年农村市场稻谷价（早籼、中晚籼、粳三类均值 = 2.72 元/kg；--paddy-price 可改，如 2.69）
#         P_ord / P_prem = 5.23 / 5.96 元/kg（与正文一致，config_1004）；汇率 7 元/美元
#   输出（out_dir("sensitivity")）：
#     valuation_tiers_members.csv   逐成员 × ssp × ri（T1/T2 逐成员；T3 来自 24 号 econ_members_long.csv）
#     valuation_tiers_summary.csv   集合平均（T3 来自 14 号 production_loss_summary.csv）+ 成员中位数 [范围]  → 补表 13
#     paddy_production_members.csv  逐成员稻谷总产（Mt）
#   图（figures/valuation_1106/）：FigS23_valuation.{png,pdf,svg}（终版只比较两种口径：稻谷产量 × 稻谷价 vs 整精米产量 × 等级价）
#     [1108f] 补图编号按首次引用重排：原补图 25 → 补图 23；同目录旧名 FigS25_valuation.*（含四口径旧图）自动删除
#     同时写出作图源数据：FigS23_valuation_source_bars.csv（条形 = 集合平均）、FigS23_valuation_source_members.csv（散点 = 25 成员）
# ============================================================
import warnings, os, sys, argparse
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_1004 as CFG
import q_common_1004 as C

ap = argparse.ArgumentParser()
ap.add_argument("--paddy-price", type=float, default=2.72, help="稻谷价 CNY/kg（2020–2024 三类均值 2.72）")
A = ap.parse_args()
P_PADDY = A.paddy_price * 1000.0            # CNY/t
P_ORD, P_PREM = CFG.P_ORD * 1000.0, CFG.P_HQ * 1000.0
USD = lambda cny: cny / CFG.CNY_PER_USD / 1e9   # CNY → billion US$

LON_VALS = np.arange(73.25, 136.0, 0.5); LAT_VALS = np.arange(53.75, 17.75, -0.5)
def read_yield_nc(fp, var, years):
    ds = xr.open_dataset(fp, decode_times=False); da = ds[var].astype("float32")
    fv = ds[var].attrs.get("_FillValue", 1e20); yrs = (ds["time"].values + 1601).astype(int)
    lon_idx = np.array([np.argmin(np.abs(da.lon.values - v)) for v in LON_VALS])
    lat_idx = np.array([np.argmin(np.abs(da.lat.values - v)) for v in LAT_VALS])
    d = da.values[:, lat_idx, :][:, :, lon_idx].copy()
    d[np.abs(d) > 1e10] = np.nan; d[d == fv] = np.nan; d[d < 0] = np.nan
    idx = [i for i, y in enumerate(yrs) if y in set(years)]
    ds.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 海洋/无数据格点全 NaN 的提示
        return np.nanmean(d[idx], axis=0) if idx else None
def yfile(m, g, scen, ri, ssp=None):
    d = CFG.YIELD_DIR / ("historical" if scen == "historical" else "future") / m / g / ("" if scen == "historical" else ssp)
    fs = list(d.glob(f"*{'historical' if scen=='historical' else ssp}*yield-{ri}-firr*.nc"))
    return fs[0] if fs else None

def main():
    C.check_anchors(); os.makedirs(C.OUTDIR, exist_ok=True)
    rows, prod = [], []
    for ri in C.RIS:
        a = C.area_and_mask(ri); a_ha = a * C.KM2_TO_HA
        for m in C.GGCMS:
            for g in C.GCMS:
                fh = yfile(m, g, "historical", ri)
                if fh is None: continue
                Yh = read_yield_nc(fh, f"yield-{ri}-firr", CFG.HIS_YEARS)
                HRYh = C.read(C.find("hry_hist", ri=ri, model=m, gcm=g))
                for s in C.SSPS:
                    ff = yfile(m, g, "future", ri, s)
                    if ff is None: continue
                    Yf = read_yield_nc(ff, f"yield-{ri}-firr", CFG.FUT_YEARS)
                    HRYf = C.read(C.find("hry_fut", ri=ri, ssp=s, model=m, gcm=g))
                    ok = np.isfinite(a_ha) & np.isfinite(Yh) & np.isfinite(Yf)
                    Ph, Pf = np.nansum(a_ha[ok] * Yh[ok]), np.nansum(a_ha[ok] * Yf[ok])       # t
                    ok2 = ok & np.isfinite(HRYh) & np.isfinite(HRYf)
                    Hh, Hf = np.nansum(a_ha[ok2] * HRYh[ok2]), np.nansum(a_ha[ok2] * HRYf[ok2])
                    rows.append(dict(ssp=s, ri=ri, model=m, gcm=g,
                                     paddy_h_Mt=Ph / 1e6, paddy_f_Mt=Pf / 1e6,
                                     head_h_Mt=Hh / 1e6, head_f_Mt=Hf / 1e6,
                                     T1_paddy_V0_bnUSD=USD(Ph * P_PADDY), T1_paddy_dV_bnUSD=USD((Pf - Ph) * P_PADDY),
                                     T2_head_ord_V0_bnUSD=USD(Hh * P_ORD), T2_head_ord_dV_bnUSD=USD((Hf - Hh) * P_ORD)))
            print("done", ri, m)
    df = pd.DataFrame(rows)
    # T3：24 号逐成员（亿元，正 = 损失）→ bn US$，正 = 增值
    em = pd.read_csv(os.path.join(C.OUTDIR, "econ_members_long.csv"))
    em = em[em.ri.isin(C.RIS)].copy()
    ycol = next((c for c in ("output_effect", "yield_effect", "hry_effect") if c in em.columns), None)
    if ycol is None:                                   # 老版 24 号没有产量效应列：用 净损失 − 等级效应 反推
        em["output_effect"] = em.net_loss - em.grade_effect; ycol = "output_effect"
    print("24 号列名：", list(em.columns), "| 产量效应列 =", ycol)
    em = em.rename(columns={"V_hist": "T3_V0_yi", ycol: "T3_yield_loss_yi",
                            "grade_effect": "T3_grade_loss_yi", "net_loss": "T3_net_loss_yi"})
    yi2bn = lambda x: x * 1e8 / CFG.CNY_PER_USD / 1e9
    for c in ["T3_V0_yi", "T3_yield_loss_yi", "T3_grade_loss_yi", "T3_net_loss_yi"]:
        em[c[:-3] + "_bnUSD"] = yi2bn(em[c])      # 只换末尾的 _yi（"T3_yield" 里也含 "_yi"，不能用 replace）
    em["T3_V0_bnUSD"] = em.T3_V0_bnUSD
    em["T3_fixedgrade_dV_bnUSD"] = -em.T3_yield_loss_bnUSD           # 固定历史等级的整精米产值变化
    em["T3_full_dV_bnUSD"] = -em.T3_net_loss_bnUSD                   # 计入等级
    em["T3_grade_dV_bnUSD"] = -em.T3_grade_loss_bnUSD
    df = df.merge(em[["ssp", "ri", "model", "gcm", "T3_V0_bnUSD", "T3_fixedgrade_dV_bnUSD", "T3_grade_dV_bnUSD", "T3_full_dV_bnUSD"]],
                  on=["ssp", "ri", "model", "gcm"], how="left")
    nat = df.groupby(["ssp", "model", "gcm"], as_index=False)[[c for c in df.columns if c.endswith("Mt") or c.endswith("bnUSD")]].sum()
    nat["ri"] = "ri1+ri2"
    df = pd.concat([df, nat], ignore_index=True)
    for t in ["T1_paddy", "T2_head_ord"]:
        df[f"{t}_dV_pct"] = 100 * df[f"{t}_dV_bnUSD"] / df[f"{t}_V0_bnUSD"]
    df["T3_full_dV_pct"] = 100 * df.T3_full_dV_bnUSD / df.T3_V0_bnUSD
    df.to_csv(os.path.join(C.OUTDIR, "valuation_tiers_members.csv"), index=False)

    # 集合平均：T1/T2 线性 → 成员均值；T3 用 14 号集合平均场（正文数字）
    pl = pd.read_csv(CFG.out_dir("loss") / "production_loss_summary.csv")
    pl["ri"] = pl.ri.replace({"ri1+ri2": "ri1+ri2"})
    summ = []
    for (s, ri), g in df.groupby(["ssp", "ri"], sort=False):
        r = dict(ssp=s, ri=ri, n_members=len(g),
                 paddy_baseline_Mt=g.paddy_h_Mt.mean(), paddy_future_Mt=g.paddy_f_Mt.mean(),
                 paddy_change_pct=100 * (g.paddy_f_Mt.mean() / g.paddy_h_Mt.mean() - 1),
                 head_baseline_Mt=g.head_h_Mt.mean(), head_future_Mt=g.head_f_Mt.mean())
        for t in ["T1_paddy", "T2_head_ord"]:
            v = g[f"{t}_dV_bnUSD"]; r[f"{t}_V0_bnUSD"] = g[f"{t}_V0_bnUSD"].mean()
            r[f"{t}_dV_bnUSD_ensmean"] = v.mean(); r[f"{t}_dV_members"] = f"{v.median():.2f} [{v.min():.2f}, {v.max():.2f}]"
            r[f"{t}_members_gain"] = int((v > 0).sum())
        q = pl[(pl.ssp == s) & (pl.ri == ri)]
        if len(q):
            q = q.iloc[0]; yi = lambda v: v * 1e8 / CFG.CNY_PER_USD / 1e9
            r.update(T3_V0_bnUSD=yi(q.V_his_yi), T3_fixedgrade_dV_bnUSD_ensmean=-yi(q.loss_qty_yi),
                     T3_grade_dV_bnUSD_ensmean=-yi(q.loss_qual_yi), T3_full_dV_bnUSD_ensmean=-yi(q.loss_yi),
                     T3_full_dV_pct_ensmean=-q.loss_pct)
        for t in ["T3_fixedgrade", "T3_grade", "T3_full"]:
            v = g[f"{t}_dV_bnUSD"].dropna()
            r[f"{t}_dV_members"] = f"{v.median():.2f} [{v.min():.2f}, {v.max():.2f}]"
        r["T3_members_net_loss"] = int((g.T3_full_dV_bnUSD < 0).sum())
        summ.append(r)
    S = pd.DataFrame(summ); S.to_csv(os.path.join(C.OUTDIR, "valuation_tiers_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["ssp", "ri", "paddy_baseline_Mt", "paddy_change_pct", "T1_paddy_dV_bnUSD_ensmean", "T1_paddy_dV_members",
            "T1_paddy_members_gain", "T3_full_dV_bnUSD_ensmean", "T3_full_dV_members", "T3_members_net_loss"]   # [1108d] 论文只用这两种口径
    print(S[cols].round(2).to_string(index=False))
    print(f"\n价格：稻谷 {A.paddy_price} 元/kg；普通/优质精米 {CFG.P_ORD}/{CFG.P_HQ} 元/kg；{CFG.CNY_PER_USD} 元/US$")
    plot(S, df)

def plot(S, df):
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    out = CFG.FIGURES / "valuation_1106"; out.mkdir(parents=True, exist_ok=True)
    COL = {"ssp126": CFG.__dict__.get("C_SSP126", "#2C6FB0"), "ssp585": CFG.__dict__.get("C_SSP585", "#E8833A")}
    try:
        import paper_palette as PP; COL = {k: PP.SSP[k] for k in COL}
    except Exception: pass
    # [1108c] 横向条形：估值口径作为行，标签只在最左侧写一次（原竖排横轴标签在三个面板重复、行数不齐、Step 3a/3 顺序别扭）
    # [1108d] 只比较两种口径：只算稻谷产量 vs 计入碾米与品质等级（主核算）；删去“单一价格”与“基期等级价”两行
    tiers = [("T1_paddy", "Paddy yield\n× paddy price"),
             ("T3_full", "Head-rice yield\n× grade prices")]
    panels = [("ri1+ri2", "National"), ("ri1", "Single/early rice"), ("ri2", "Late rice")]
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 44 / 25.4), sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(tiers))[::-1].astype(float)          # 第一种口径在最上
    hh = 0.36
    for ax, (ri, title) in zip(axes, panels):
        for k, s in enumerate(C.SSPS):
            off = (0.5 - k) * hh                               # SSP1-2.6 在上、SSP5-8.5 在下
            g = df[(df.ssp == s) & (df.ri == ri)]; r = S[(S.ssp == s) & (S.ri == ri)].iloc[0]
            for i, (t, _) in enumerate(tiers):
                v = g[f"{t}_dV_bnUSD"].dropna().values
                mean = r.get(f"{t}_dV_bnUSD_ensmean", np.nan)
                ax.barh(y[i] + off, mean, height=hh * 0.92, color=COL[s], alpha=0.75, edgecolor="none", zorder=3,
                        label={"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}[s] if i == 0 else None)
                ax.scatter(v, y[i] + off + rng.uniform(-hh * 0.3, hh * 0.3, len(v)), s=4, color=COL[s], alpha=0.55,
                           linewidths=0, zorder=4)
        ax.axvline(0, color="0.3", lw=0.6, zorder=2)
        ax.set_title(title, fontsize=7.5, pad=3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelsize=6, length=2, pad=1.5)
        ax.tick_params(axis="y", length=0, pad=3)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab for _, lab in tiers], fontsize=6.3, linespacing=1.1)
    axes[0].set_ylim(-0.6, len(tiers) - 0.4)
    # [1108] US\$ 需转义，否则 $ 被当作公式开始
    axes[1].set_xlabel(r"Change in annual production value (billion US\$ yr$^{-1}$)", fontsize=6.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.07))
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.04, 1.04, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    fig.tight_layout(w_pad=1.0)
    for old in out.glob("FigS25_valuation.*"):                 # [1108f] 旧编号（含 1106 四口径旧图）不再保留，避免收错
        old.unlink(); print("[removed stale]", old.name)
    for e in ("png", "pdf", "svg"):
        fig.savefig(out / f"FigS23_valuation.{e}", dpi=300 if e == "png" else None, bbox_inches="tight")
    print("[saved]", out / "FigS23_valuation.*")
    # [1108f] 作图源数据（只含图中两种口径）
    lab = {"T1_paddy": "paddy_yield_x_paddy_price", "T3_full": "head_rice_yield_x_grade_prices"}
    bars, mem = [], []
    for ri, title in panels:
        for s in C.SSPS:
            g = df[(df.ssp == s) & (df.ri == ri)]; r = S[(S.ssp == s) & (S.ri == ri)].iloc[0]
            for t, _ in tiers:
                v = g[f"{t}_dV_bnUSD"].dropna()
                bars.append(dict(panel=title, ri=ri, ssp=s, valuation=lab[t],
                                 ensemble_mean_bnUSD_yr=r.get(f"{t}_dV_bnUSD_ensmean", np.nan),
                                 member_median=v.median(), member_min=v.min(), member_max=v.max(), n_members=len(v)))
                for _, x in g.iterrows():
                    mem.append(dict(panel=title, ri=ri, ssp=s, valuation=lab[t], model=x.model, gcm=x.gcm,
                                    dV_bnUSD_yr=x[f"{t}_dV_bnUSD"]))
    pd.DataFrame(bars).to_csv(out / "FigS23_valuation_source_bars.csv", index=False, float_format="%.4f")
    pd.DataFrame(mem).to_csv(out / "FigS23_valuation_source_members.csv", index=False, float_format="%.4f")
    print("[saved]", out / "FigS23_valuation_source_{bars,members}.csv")

if __name__ == "__main__":
    main()
