# ============================================================
# check_co2_response.py —— 核对品质模型的 CO₂ 与气候响应（只读：不改模型、不改预测，只出诊断图和表）
#   用途：说明单季/早稻 PC（SSP1-2.6 下 CO₂ 效应为正）与 HRR（CO₂ 效应 SSP1-2.6 大于 SSP5-8.5）的来源
#   另附 CGR、CD 作对照（这两个模型已对 CO₂ 加 +1 单调约束）
#   [1027] 训练数据只作为整体使用（训练范围、CO₂ 分布），不单独拿出任何子集
#
#   A 偏依赖曲线（PDP）：输入 = 18 号导出的基准期格网特征（features_historical_{ri}.csv，统一格网域 D）；
#     每次只改一个变量（CO₂ / Tavg / TSR），其余变量保持各格网原值，全部格网的预测取平均（逐格等权）
#     圆点 = 基准 / SSP1-2.6 / SSP5-8.5 时段均值在曲线上的位置；灰底 = 训练数据范围以外；CO₂ 列另画训练数据的 CO₂ 分布
#   B 模型的 CO₂ 效应（与 04 的固定 CO₂ 对照同一定义）：未来格网输入下
#        [F(CO₂ 未来) − F(CO₂ 基准)] / 基准预测均值 × 100（百分点）
#     对照：paper_data 02_quality_change 的 co2_effect_pp（逐年、逐成员预测的正式结果，有 --xlsx 时读取）
#   检验 CO₂ × 温度交互（PC、HRR）：
#   D 模型：在不同 Tavg 下，CO₂ 从基准升到 SSP1-2.6 / SSP5-8.5 时段均值带来的变化（二维偏依赖：CO₂ × Tavg，
#     其余变量保持基准格网原值；每季随机取 ≤ 400 个格网）。曲线随 Tavg 升高而上移 = 越热，CO₂ 的稀释越弱
#   E 正式预测：每个格网的 CO₂ 效应 =（随情景 CO₂ − 固定 CO₂）/ 基准 × 100（04 集合平均 tif，与 Fig. 2a 同一定义），
#     对该格网的未来 Tavg 作图，按 1 °C 分箱取均值；并给 Spearman 相关
#   F 固定窗口（增温较小的情景）：未来气候套用基准期平均灌浆窗口（03 号组 C，5 GCM 集合平均；同一批格网），
#     不含物候提前带来的额外增温（Fig. 2b 斜线条）；同样的 CO₂ 增量下，比较固定窗口与实际窗口的 CO₂ 效应
#     固定窗口下 CO₂ 稀释更强 = 增温越小、稀释越强，与 D、E 相互印证
#
#   输出（本脚本目录）：check_co2_response.png、check_co2_response_effects.csv、
#                      check_co2_interaction.png、check_co2_interaction.csv、check_co2_interaction_cells.csv、
#                      check_co2_fixedwindow.csv
#   用法：cd code_1004b && python figs_1020/check_co2_response.py [--xlsx paper_data.xlsx] [--face 训练数据.xlsx]
# ============================================================
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
import config_1004 as C

p = argparse.ArgumentParser()
p.add_argument("--xlsx", default=None, help="paper_data Excel（读 co2_effect_pp 作对照；默认 config 的 excel 目录）")
p.add_argument("--face", default=None, help="训练数据 Excel（默认 config.FACE_XLSX）")
p.add_argument("--traits", default="HRR,PC,CGR,CD")
a = p.parse_args()

TRAITS = [t.strip() for t in a.traits.split(",") if t.strip()]
VARS = ["CO2", "Tavg", "TSR"]
VLAB = {"CO2": "CO₂ (ppm)", "Tavg": "Tavg (°C)", "TSR": r"TSR (MJ m$^{-2}$ d$^{-1}$)"}
RI_LAB = {"ri1": "Single/early", "ri2": "Late"}
RI_LS = {"ri1": "-", "ri2": (0, (3, 1.6))}
PER_COL = {"historical": "#7D7D7D", "ssp126": "#2C6FB0", "ssp585": "#E8833A"}
PER_LAB = {"historical": "Baseline", "ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
INK = "#2B2B2B"
DEFAULT_FN = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]

# ---------------------------------------------------------------- 数据
FEAT = C.out_dir("shap_features", thresh=False)
X = {}
for per in ["historical"] + list(C.SSPS):
    for ri in C.RI_TYPES:
        f = FEAT / f"features_{per}_{ri}.csv"
        if not f.exists():
            raise SystemExit(f"缺 {f}，先运行 18_export_shap_features_1004.py")
        X[(per, ri)] = pd.read_csv(f)
CO2_BASE = {ri: float(X[("historical", ri)]["CO2"].mean()) for ri in C.RI_TYPES}   # = 04 固定 CO₂ 的取值
face = pd.read_excel(a.face or C.FACE_XLSX)

xlsx = Path(a.xlsx) if a.xlsx else C.out_dir("excel") / f"paper_data_{C.TAG}.xlsx"
proj = None
if xlsx.exists():
    q = pd.read_excel(xlsx, sheet_name="02_quality_change")
    proj = q[q.co2 == "with_CO2"].set_index(["trait", "ri", "ssp"])["co2_effect_pp"]
else:
    print(f"[提示] 没找到 {xlsx}，B 部分不列正式结果的 co2_effect_pp")


def load_model(t):
    m = joblib.load(C.MODEL_DIR / f"{t}_Blending.joblib")
    fn = list(getattr(m, "feature_names_in_", DEFAULT_FN))
    return m, fn


def predict(m, fn, df):
    miss = [c for c in fn if c not in df.columns]
    if miss:
        raise SystemExit(f"输入缺列 {miss}（模型特征 {fn}）")
    return np.asarray(m.predict(df[fn]), float)


def pdp(m, fn, base, var, grid):
    """base 的每一行把 var 换成 grid 中的每个值，预测后对行取平均"""
    big = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    big[var] = np.tile(grid, len(base))
    y = predict(m, fn, big).reshape(len(base), len(grid))
    return y.mean(axis=0)


# ---------------------------------------------------------------- 计算
EFF, CURVES = [], {}
for t in TRAITS:
    m, fn = load_model(t)
    # A 偏依赖
    for var in VARS:
        lo_ = min(face[var].min(), *(X[k][var].min() for k in X))
        hi_ = max(face[var].max(), *(X[k][var].max() for k in X))
        if var == "CO2":
            grid = np.arange(330.0, 731.0, 5.0)
        else:
            pad = 0.08 * (hi_ - lo_)
            grid = np.linspace(lo_ - pad, hi_ + pad, 70)
        for ri in C.RI_TYPES:
            CURVES[(t, var, ri)] = (grid, pdp(m, fn, X[("historical", ri)], var, grid))
    # B 模型的 CO₂ 效应（与正式结果对照）
    for ri in C.RI_TYPES:
        base_mean = predict(m, fn, X[("historical", ri)]).mean()
        for ssp in C.SSPS:
            xf = X[(ssp, ri)]
            fx = xf.copy(); fx["CO2"] = CO2_BASE[ri]
            d_model = (predict(m, fn, xf).mean() - predict(m, fn, fx).mean()) / base_mean * 100
            EFF.append(dict(trait=t, ri=ri, ssp=ssp, co2_base=round(CO2_BASE[ri], 1),
                            co2_future=round(float(xf["CO2"].mean()), 1),
                            model_co2_effect_pp=round(d_model, 2),
                            projection_co2_effect_pp=(round(float(proj.loc[(t, ri, ssp)]), 2)
                                                      if proj is not None and (t, ri, ssp) in proj.index else np.nan)))

eff = pd.DataFrame(EFF)
eff.to_csv(HERE / "check_co2_response_effects.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n[B] CO₂ 效应（百分点，相对基准均值）：model = 本脚本用集合平均特征算；projection = 正式结果")
print(eff.to_string(index=False))

# ---------------------------------------------------------------- 图
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
                     "font.size": 6.5, "axes.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "legend.frameon": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
nr = len(TRAITS)
fig, axs = plt.subplots(nr, 4, figsize=(180 / 25.4, (42 * nr + 26) / 25.4),
                        gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1.05], wspace=0.42, hspace=0.55))
axs = np.atleast_2d(axs)
for i, t in enumerate(TRAITS):
    for j, var in enumerate(VARS):
        ax = axs[i, j]
        tr_lo, tr_hi = face[var].min(), face[var].max()
        for ri in C.RI_TYPES:
            g, y = CURVES[(t, var, ri)]
            ax.plot(g, y, color=INK, lw=1.0, ls=RI_LS[ri], zorder=3, label=RI_LAB[ri])
            for per in ["historical"] + list(C.SSPS):                   # 各时段均值在曲线上的位置
                xv = float(X[(per, ri)][var].mean())
                ax.plot(xv, np.interp(xv, g, y), "o", ms=3.2, mfc=PER_COL[per], mec="white", mew=0.4, zorder=4)
        g = CURVES[(t, var, C.RI_TYPES[0])][0]
        ax.axvspan(g.min(), tr_lo, color="0.88", lw=0, zorder=0)
        ax.axvspan(tr_hi, g.max(), color="0.88", lw=0, zorder=0)
        ax.set_xlim(g.min(), g.max())
        if var == "CO2":                                                 # 训练数据 CO₂ 分布 + 单一 CO₂ 值数据集
            ah = ax.twinx(); ah.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
            ah.hist(face.CO2, bins=np.arange(330, 735, 10), color="0.80", edgecolor="white", lw=0.3)
            ah.set_ylim(0, ah.get_ylim()[1] * 3.2); ah.set_yticks([]); ah.spines["right"].set_visible(False)
        ax.set_xlabel(VLAB[var], labelpad=1)
        if j == 0:
            ax.set_ylabel(f"{t} (model)", labelpad=1)
        ax.tick_params(labelsize=5.8, pad=1)
        if i == 0:
            ax.set_title({"CO2": "Partial dependence on CO₂", "Tavg": "on Tavg", "TSR": "on TSR"}[var],
                         fontsize=6.8, pad=3)
    # CO₂ 效应对比条形
    ax = axs[i, 3]
    sub = eff[eff.trait == t]
    xs = np.arange(len(sub))
    ax.bar(xs - 0.2, sub.model_co2_effect_pp, 0.38, color=[PER_COL[s] for s in sub.ssp], label="Model (this check)")
    if sub.projection_co2_effect_pp.notna().any():
        ax.bar(xs + 0.2, sub.projection_co2_effect_pp, 0.38, facecolor="white",
               edgecolor=[PER_COL[s] for s in sub.ssp], lw=0.7, label="Projection (paper data)")
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{'S/E' if r.ri == 'ri1' else 'L'}\n{'1-2.6' if r.ssp == 'ssp126' else '5-8.5'}"
                        for r in sub.itertuples()], fontsize=5.5)
    ax.tick_params(axis="x", length=0, pad=1.5); ax.tick_params(axis="y", labelsize=5.8, pad=1)
    ax.set_ylabel("CO₂ effect (pp)", labelpad=1)
    if i == 0:
        ax.set_title("CO₂ effect: model vs projection", fontsize=6.8, pad=3)
h = [Line2D([], [], color=INK, lw=1.0, ls=RI_LS[ri], label=RI_LAB[ri]) for ri in C.RI_TYPES] + \
    [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=PER_COL[p_], mec="white", label=PER_LAB[p_])
     for p_ in ["historical"] + list(C.SSPS)] + \
    [Patch(color="0.88", label="Outside training range"),
     Patch(color="0.80", label="Training records (CO₂)"),
     Patch(color="0.45", label="CO₂ effect, model (this check)"),
     Patch(facecolor="white", edgecolor="0.45", label="CO₂ effect, projection (paper data)")]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.07, right=0.99, top=1 - 5.0 / (42 * nr + 26), bottom=24.0 / (42 * nr + 26))   # [1027] 图例不压横轴标题
out = HERE / "check_co2_response.png"
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"\n[saved] {out}\n        {HERE / 'check_co2_response_effects.csv'}")

# ================================================================ [1027] CO₂ × 温度交互（PC、HRR）
INT_TRAITS = [t for t in ["PC", "HRR"] if t in TRAITS]
tmin = min(float(X[k]["Tavg"].min()) for k in X); tmax = max(float(X[k]["Tavg"].max()) for k in X)
T_GRID = np.arange(np.floor(tmin) - 1.0, np.ceil(tmax) + 1.01, 0.5)

# ---- D 模型：不同 Tavg 下的 CO₂ 效应（二维偏依赖）
INT_CURVES, INT_ROWS = {}, []
for t in INT_TRAITS:
    m, fn = load_model(t)
    for ri in C.RI_TYPES:
        allb = X[("historical", ri)]
        base = allb.sample(min(400, len(allb)), random_state=0)
        ref = predict(m, fn, allb).mean()                        # 基准预测均值（相对变化的分母）

        def surf(co2):
            b = base.copy(); b["CO2"] = co2
            return pdp(m, fn, b, "Tavg", T_GRID)

        y0 = surf(CO2_BASE[ri])
        for ssp in C.SSPS:
            d = (surf(float(X[(ssp, ri)]["CO2"].mean())) - y0) / ref * 100
            INT_CURVES[(t, ri, ssp)] = d
            for per, lab in (("historical", "at baseline Tavg"), (ssp, "at future Tavg")):
                tv = float(X[(per, ri)]["Tavg"].mean())
                INT_ROWS.append(dict(trait=t, ri=ri, ssp=ssp, where=lab, Tavg=round(tv, 2),
                                     co2_effect_pp=round(float(np.interp(tv, T_GRID, d)), 2)))
inter = pd.DataFrame(INT_ROWS)
inter.to_csv(HERE / "check_co2_interaction.csv", index=False, encoding="utf-8-sig")
print("\n[D] 模型：同一 CO₂ 增量在基准 Tavg 与未来 Tavg 下的效应（pp）——未来 Tavg 下更高 = 升温削弱稀释")
print(inter.pivot_table(index=["trait", "ri", "ssp"], columns="where", values="co2_effect_pp").round(2).to_string())

# ---- E 正式预测：格网 CO₂ 效应 vs 格网未来 Tavg
CELLS = []
try:
    import rasterio
    from rasterio.transform import rowcol

    WD = C.pred_dir(C.PRED_MODE, "withco2") / "ensemble"
    FD = C.pred_dir(C.PRED_MODE, "fixedco2") / "ensemble"

    def rd(path):
        with rasterio.open(path) as ds_:
            arr = ds_.read(1).astype("float64")
            if ds_.nodata is not None:
                arr[arr == ds_.nodata] = np.nan
            return arr, ds_.transform

    for t in INT_TRAITS:
        for ri in C.RI_TYPES:
            bse, tf = rd(WD / f"ensemble_historical_{ri}_{t}_{C.HIS_P}.tif")
            for ssp in C.SSPS:
                w_, _ = rd(WD / f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif")
                f_, _ = rd(FD / f"ensemble_{ssp}_{ri}_{t}_{C.FUT_P}.tif")
                xf = X[(ssp, ri)]
                rr, cc = rowcol(tf, xf["lon"].values, xf["lat"].values)
                rr, cc = np.asarray(rr), np.asarray(cc)
                ok = (rr >= 0) & (rr < bse.shape[0]) & (cc >= 0) & (cc < bse.shape[1])
                b_ = bse[rr[ok], cc[ok]]
                e_ = np.where(np.abs(b_) > 0, (w_[rr[ok], cc[ok]] - f_[rr[ok], cc[ok]]) / b_ * 100, np.nan)
                CELLS.append(pd.DataFrame(dict(trait=t, ri=ri, ssp=ssp, Tavg=xf["Tavg"].values[ok], co2_effect_pp=e_)))
except Exception as e:                                          # 没有 04 的 tif 时只出 D
    print(f"[E] ⚠️ 读不到 04 集合平均 tif，跳过格网分析：{e}")
cells = pd.concat(CELLS, ignore_index=True).dropna() if CELLS else pd.DataFrame()
if not cells.empty:
    cells.to_csv(HERE / "check_co2_interaction_cells.csv", index=False, encoding="utf-8-sig")
    print("\n[E] 正式预测：格网 CO₂ 效应与格网未来 Tavg 的 Spearman 相关（正 = 越热的格网 CO₂ 稀释越弱）")
    rho = cells.groupby(["trait", "ri", "ssp"])[["Tavg", "co2_effect_pp"]].apply(
        lambda d: pd.Series(dict(n=len(d), rho=d["Tavg"].corr(d["co2_effect_pp"], method="spearman"),
                                 mean_pp=d["co2_effect_pp"].mean(),
                                 share_positive=(d["co2_effect_pp"] > 0).mean()))).round(3)
    print(rho.to_string(), flush=True)

# ---- F 固定窗口（增温较小的情景）：同一批格网、同样的 CO₂ 增量，比较固定窗口与实际窗口的 CO₂ 效应
CLIM_LAB = {"Tavg": "mean_temp_degC", "Tdif": "mean_tmpdiff_degC", "CTSR": "cumul_solar_MJ_m2",
            "TSR": "mean_daily_solar_MJ_m2", "Rain": "cumul_precip_mm", "RHavg": "mean_humidity_pct"}   # 与 18 号相同
XFW, FW_ROWS = {}, []
try:
    import rasterio
    from rasterio.transform import rowcol

    FWD = C.CLIM_FUT / "fixed_window" / "ensemble"
    for ri in C.RI_TYPES:
        for ssp in C.SSPS:
            xa = X[(ssp, ri)]
            xw = xa.copy()
            for k, lab in CLIM_LAB.items():
                with rasterio.open(FWD / f"ensemble_{ssp}_{ri}_fixedwindow_{lab}_{C.FUT_P}_mean.tif") as ds_:
                    arr = ds_.read(1).astype("float64")
                    if ds_.nodata is not None:
                        arr[arr == ds_.nodata] = np.nan
                    rr, cc = rowcol(ds_.transform, xa["lon"].values, xa["lat"].values)
                xw[k] = arr[np.asarray(rr), np.asarray(cc)]
            XFW[(ssp, ri)] = xw.dropna(subset=list(CLIM_LAB)).reset_index(drop=True)
    # [1105] 增温幅度 dTavg 同时给出面积加权值（与正文/05 号 SUMMARY_WEIGHT="area" 同口径）；等权值仅作对照
    def _aw(ri, df_):
        with rasterio.open(C.area_file(ri)) as ds_:
            arr = ds_.read(1).astype("float64")
            if ds_.nodata is not None:
                arr[arr == ds_.nodata] = np.nan
            rr, cc = rowcol(ds_.transform, df_["lon"].values, df_["lat"].values)
        w_ = arr[np.asarray(rr), np.asarray(cc)]
        return np.where(np.isfinite(w_) & (w_ > 0), w_, 0.0)
    def _wmean(v, w):
        return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(v))
    for t in INT_TRAITS:
        m, fn = load_model(t)
        for ri in C.RI_TYPES:
            xh = X[("historical", ri)]
            wh = _aw(ri, xh)
            base_mean = predict(m, fn, xh).mean()
            for ssp in C.SSPS:
                for win, xx in (("actual window", X[(ssp, ri)]), ("fixed window", XFW[(ssp, ri)])):
                    xb = xx.copy(); xb["CO2"] = CO2_BASE[ri]
                    pw, pb = predict(m, fn, xx).mean(), predict(m, fn, xb).mean()
                    wx = _aw(ri, xx)
                    FW_ROWS.append(dict(trait=t, ri=ri, ssp=ssp, window=win, n_cells=len(xx),
                                        dTavg_areaw=round(_wmean(xx["Tavg"].values, wx) - _wmean(xh["Tavg"].values, wh), 2),
                                        dTavg=round(float(xx["Tavg"].mean() - xh["Tavg"].mean()), 2),
                                        change_actual_co2_pct=round((pw - base_mean) / base_mean * 100, 2),
                                        change_fixed_co2_pct=round((pb - base_mean) / base_mean * 100, 2),
                                        co2_effect_pp=round((pw - pb) / base_mean * 100, 2)))
except Exception as e:                                          # 没有 03 号固定窗口气候时只出 D、E
    print(f"[F] ⚠️ 读不到固定窗口气候（{C.CLIM_FUT}/fixed_window/ensemble），跳过：{e}")
fw = pd.DataFrame(FW_ROWS)
if not fw.empty:
    fw.to_csv(HERE / "check_co2_fixedwindow.csv", index=False, encoding="utf-8-sig")
    print("\n[F] 固定窗口（增温较小）vs 实际窗口：dTavg_areaw = 面积加权增温（°C，与正文同口径）；dTavg = 等权对照；co2_effect_pp = 同样 CO₂ 增量带来的变化"
          "（固定窗口下更负 = 增温越小、稀释越强）")
    print(fw.pivot_table(index=["trait", "ri", "ssp"], columns="window", values=["dTavg_areaw", "dTavg", "co2_effect_pp"])
          .round(2).to_string())

# ---- 图
nr = len(INT_TRAITS)
ncol = 4 if not fw.empty else 3
fig, axs = plt.subplots(nr, ncol, figsize=(180 / 25.4, (46 * nr + 14) / 25.4),
                        gridspec_kw=dict(wspace=0.42, hspace=0.6, width_ratios=[1.1, 1, 1, 1.0][:ncol]))
axs = np.atleast_2d(axs)
tr_lo, tr_hi = face["Tavg"].min(), face["Tavg"].max()
for i, t in enumerate(INT_TRAITS):
    ax = axs[i, 0]
    for ri in C.RI_TYPES:
        for ssp in C.SSPS:
            d = INT_CURVES[(t, ri, ssp)]
            ax.plot(T_GRID, d, color=PER_COL[ssp], lw=1.0, ls=RI_LS[ri], zorder=3)
            for per, mk in (("historical", "o"), (ssp, "o")):
                tv = float(X[(per, ri)]["Tavg"].mean())
                ax.plot(tv, np.interp(tv, T_GRID, d), mk, ms=3.4, mec=PER_COL[ssp], mew=0.7, zorder=4,
                        mfc="white" if per == "historical" else PER_COL[ssp])
            if (ssp, ri) in XFW:                                 # 固定窗口（增温较小）的位置
                tv = float(XFW[(ssp, ri)]["Tavg"].mean())
                ax.plot(tv, np.interp(tv, T_GRID, d), "s", ms=3.0, mec=PER_COL[ssp], mew=0.7, mfc="white", zorder=4)
    ax.axvspan(T_GRID.min(), tr_lo, color="0.88", lw=0, zorder=0)
    ax.axvspan(tr_hi, T_GRID.max(), color="0.88", lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(T_GRID.min(), T_GRID.max())
    ax.set_xlabel("Tavg (°C)", labelpad=1); ax.set_ylabel(f"{t}: CO₂ effect (pp)", labelpad=1)
    ax.tick_params(labelsize=5.8, pad=1)
    if i == 0:
        ax.set_title("Model: CO₂ effect at each Tavg", fontsize=6.8, pad=3)
    for j, ri in enumerate(C.RI_TYPES):
        ax = axs[i, j + 1]
        if cells.empty:
            ax.axis("off"); continue
        for ssp in C.SSPS:
            d = cells[(cells.trait == t) & (cells.ri == ri) & (cells.ssp == ssp)]
            ax.scatter(d.Tavg, d.co2_effect_pp, s=1.5, color=PER_COL[ssp], alpha=0.25, lw=0, zorder=2)
            bins = np.arange(np.floor(d.Tavg.min()), np.ceil(d.Tavg.max()) + 1, 1.0)
            g = d.groupby(pd.cut(d.Tavg, bins), observed=True).co2_effect_pp.agg(["mean", "count"])
            g = g[g["count"] >= 10]
            ax.plot([iv.mid for iv in g.index], g["mean"], "-o", ms=2.2, lw=1.0, color=PER_COL[ssp], zorder=4)
            r_ = d["Tavg"].corr(d["co2_effect_pp"], method="spearman")
            ax.text(0.98, 0.04 + 0.09 * C.SSPS.index(ssp), f"{PER_LAB[ssp]}: ρ = {r_:+.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=PER_COL[ssp], zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.2))
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlabel("Future Tavg of grid cell (°C)", labelpad=1)
        ax.tick_params(labelsize=5.8, pad=1)
        if j == 0:
            ax.set_ylabel("CO₂ effect in projection (pp)", labelpad=1)
        if i == 0:
            ax.set_title(f"Projection, {RI_LAB[ri].lower()} rice", fontsize=6.8, pad=3)
    if ncol == 4:                                                # F 实际窗口（实色）vs 固定窗口（斜线，同 Fig. 2b）
        ax = axs[i, 3]
        sub = fw[fw.trait == t]
        keys = [(ri, ssp) for ri in C.RI_TYPES for ssp in C.SSPS]
        for k_, (ri, ssp) in enumerate(keys):
            a_ = sub[(sub.ri == ri) & (sub.ssp == ssp) & (sub.window == "actual window")].iloc[0]
            f_ = sub[(sub.ri == ri) & (sub.ssp == ssp) & (sub.window == "fixed window")].iloc[0]
            ax.bar(k_ - 0.2, a_.co2_effect_pp, 0.38, color=PER_COL[ssp], zorder=2)
            ax.bar(k_ + 0.2, f_.co2_effect_pp, 0.38, facecolor="white", edgecolor=PER_COL[ssp], hatch="//////",
                   lw=0.6, zorder=2)
        ax.axhline(0, color=INK, lw=0.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{'S/E' if ri == 'ri1' else 'L'}\n{'1-2.6' if ssp == 'ssp126' else '5-8.5'}"
                            for ri, ssp in keys], fontsize=5.5)           # 增温幅度见 check_co2_fixedwindow.csv
        ax.tick_params(axis="x", length=0, pad=1.5); ax.tick_params(axis="y", labelsize=5.8, pad=1)
        ax.set_ylabel("CO₂ effect (pp)", labelpad=1)
        if i == 0:
            ax.set_title("Actual vs fixed window", fontsize=6.8, pad=3)
h = [Line2D([], [], color=INK, lw=1.0, ls=RI_LS[ri], label=RI_LAB[ri]) for ri in C.RI_TYPES] + \
    [Line2D([], [], color=PER_COL[s], lw=1.0, label=f"CO₂ raised to {PER_LAB[s]} level") for s in C.SSPS] + \
    [Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec="0.4", label="at baseline Tavg"),
     Line2D([], [], ls="none", marker="o", ms=3.4, mfc="0.4", mec="0.4", label="at future Tavg"),
     Patch(color="0.88", label="Outside training range")]
if ncol == 4:
    h += [Line2D([], [], ls="none", marker="s", ms=3.0, mfc="white", mec="0.4",
                 label="at future Tavg, window fixed at baseline"),
          Patch(color="0.45", label="Actual window"),
          Patch(facecolor="white", edgecolor="0.45", hatch="//////", label="Window fixed at baseline (less warming)")]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=(0.20 if nr == 1 else 0.15) if ncol == 4 else
                    (0.17 if nr == 1 else 0.13))
for k_, ax_ in enumerate(axs.flat):                              # [1027] 面板字母（逐行：a–d PC，e–h HRR）
    if ax_.axison:
        ax_.text(-0.24, 1.07, "abcdefghijkl"[k_], transform=ax_.transAxes, fontsize=8, fontweight="bold",
                 ha="left", va="bottom")
out2 = HERE / "check_co2_interaction.png"
fig.savefig(out2, dpi=300)
print(f"\n[saved] {out2}\n        {HERE / 'check_co2_interaction.csv'}\n        {HERE / 'check_co2_interaction_cells.csv'}"
      + (f"\n        {HERE / 'check_co2_fixedwindow.csv'}" if not fw.empty else ""))
