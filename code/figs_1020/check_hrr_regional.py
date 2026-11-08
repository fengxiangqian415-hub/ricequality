import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
import fig_common as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

F.setup()
p = argparse.ArgumentParser()
p.add_argument("--xlsx", default=None)
p.add_argument("--face", default=None)
p.add_argument("--ri", default="ri1")
p.add_argument("--trait", default="HRR")
a = p.parse_args()
XLSX = a.xlsx or F.default_xlsx()
RI, T = a.ri, a.trait
S1, S5 = "ssp126", "ssp585"

attr = F.read_sheet(XLSX, "39_prov_attr_co2_long")
drv = F.read_sheet(XLSX, "38_prov_drivers_long")
shp = F.read_sheet(XLSX, "39_prov_attr_shap_long")
try:
    import config_1004 as C
    face = pd.read_excel(a.face or C.FACE_XLSX)
    TQ = face["Tavg"].quantile([0.90, 0.95]).to_dict()
except Exception as e:
    print(f"[提示] 读不到训练数据，不标训练范围：{e}"); TQ = {}

A = attr[(attr.variable == T) & (attr.ri == RI)].pivot_table(index="province", columns=["ssp", "component"],
                                                              values="value")
Wt = attr[(attr.variable == T) & (attr.ri == RI)].groupby("province").weight_sum.first()
nat = A.loc["National"] if "National" in A.index else None
A = A.drop(index=[i for i in A.index if i == "National"])
w = Wt.reindex(A.index) / Wt.reindex(A.index).sum()
D = drv[(drv.ri == RI)]
base_T = D[D.variable == T].groupby("province").baseline.first()
rows = []
for pv in A.index:
    r = dict(province=pv, area_share_pct=100 * w[pv], base=base_T.get(pv, np.nan))
    for s in (S1, S5):
        for comp in ("total", "climate", "co2"):
            r[f"{s}_{comp}_pp"] = A.loc[pv, (s, comp)]
    for comp in ("total", "climate", "co2"):
        r[f"gap_{comp}_pp"] = A.loc[pv, (S1, comp)] - A.loc[pv, (S5, comp)]
        r[f"contrib_{comp}_pp"] = w[pv] * r[f"gap_{comp}_pp"]
    for v in ("Tavg", "TSR", "GF_days"):
        dv = D[(D.variable == v) & (D.province == pv)].set_index("ssp")
        if len(dv):
            r[f"{v}_base"] = dv["baseline"].iloc[0]
            for s in (S1, S5):
                r[f"{v}_{s}_future"] = dv.loc[s, "future"]; r[f"d{v}_{s}"] = dv.loc[s, "delta"]
    sv = shp[(shp.variable == T) & (shp.ri == RI) & (shp.province == pv)]
    for s in (S1, S5):
        for c in ("Tavg", "TSR", "CO2"):
            q = sv[(sv.ssp == s) & (sv.component == c)]
            r[f"dSHAP_{c}_{s}"] = q["value"].iloc[0] if len(q) else np.nan
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("contrib_total_pp")
tab.to_csv(HERE / "check_hrr_regional.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
show = ["province", "area_share_pct", f"{S1}_total_pp", f"{S5}_total_pp", f"{S1}_co2_pp", f"{S5}_co2_pp",
        f"{S1}_climate_pp", f"{S5}_climate_pp", "contrib_total_pp", "contrib_climate_pp", "contrib_co2_pp",
        "Tavg_base", f"Tavg_{S5}_future", f"dSHAP_Tavg_{S5}", f"dSHAP_TSR_{S5}"]
print(f"[1] {T}（{F.RI_LAB[RI]}）各省变化（百分点）与对全国差值（SSP1-2.6 − SSP5-8.5）的贡献，按贡献排序：")
print(tab[[c for c in show if c in tab.columns]].round(2).to_string(index=False))
tot = tab[["contrib_total_pp", "contrib_climate_pp", "contrib_co2_pp"]].sum()
print(f"\n各省贡献之和：总 {tot.iloc[0]:+.2f}，气候 {tot.iloc[1]:+.2f}，CO₂ {tot.iloc[2]:+.2f} 百分点"
      + (f"（全国：{nat[(S1, 'total')] - nat[(S5, 'total')]:+.2f}）" if nat is not None else ""))
if TQ:
    print(f"训练数据 Tavg：90 百分位 {TQ[0.9]:.1f} °C，95 百分位 {TQ[0.95]:.1f} °C")

W, H = 180.0, 78.0
fig = plt.figure(figsize=(W * F.MM, H * F.MM))
axa = fig.add_axes([24 / W, 14 / H, 52 / W, 56 / H])
axb = fig.add_axes([100 / W, 14 / H, 76 / W, 56 / H])
y = np.arange(len(tab))
CCOL = {"climate": "#9DB4C0", "co2": "#4E6A7C"}
for comp in ("climate", "co2"):
    v = tab[f"contrib_{comp}_pp"].values
    left = np.where(v < 0, 0, 0)
    axa.barh(y + (-0.18 if comp == "climate" else 0.18), v, 0.34, color=CCOL[comp], zorder=3)
axa.plot(tab["contrib_total_pp"].values, y, "o", ms=2.6, color=F.INK, zorder=4)
axa.axvline(0, color=F.INK, lw=0.5)
axa.set_yticks(y); axa.set_yticklabels(tab.province.str.replace("Other provinces (<0.5 Mha)", "Other"), fontsize=5.6)
axa.invert_yaxis()
axa.tick_params(axis="y", length=0, pad=1.5); axa.tick_params(axis="x", labelsize=5.8, length=1.5, pad=1)
axa.set_xlabel(f"Contribution to national gap (pp)\n(SSP1-2.6 − SSP5-8.5; negative = larger decline under SSP1-2.6)",
               fontsize=5.8, labelpad=1.5)
axa.set_title(f"{T}, {F.RI_LAB[RI].lower()}", fontsize=7, fontweight="bold", pad=3)
axa.legend(handles=[Patch(color=CCOL["climate"], label="Climate (CO₂ fixed)"), Patch(color=CCOL["co2"], label="CO₂"),
                    Line2D([], [], ls="none", marker="o", ms=3, color=F.INK, label="Total")],
           fontsize=5.4, loc="upper right", frameon=False, handlelength=1.0)
axa.text(-0.42, 1.03, "a", transform=axa.transAxes, fontsize=8, fontweight="bold")

rel = lambda pv, s: tab.set_index("province").loc[pv, f"{s}_total_pp"] / tab.set_index("province").loc[pv, "base"] * 100
if TQ:
    axb.axvspan(TQ[0.95], 40, color="0.90", lw=0, zorder=0)
    axb.axvline(TQ[0.9], color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=0)
xs_all, LABS = [], []
for _, r in tab.iterrows():
    if not np.isfinite(r.get(f"Tavg_{S1}_future", np.nan)):
        continue
    x1, x5 = r[f"Tavg_{S1}_future"], r[f"Tavg_{S5}_future"]
    y1, y5 = r[f"{S1}_total_pp"] / r["base"] * 100, r[f"{S5}_total_pp"] / r["base"] * 100
    sz = 6 + 120 * r["area_share_pct"] / 100
    axb.plot([x1, x5], [y1, y5], color="0.65", lw=0.6, zorder=2)
    axb.scatter([x1], [y1], s=sz, color=F.SSP_COL[S1], edgecolor="white", lw=0.3, zorder=3)
    axb.scatter([x5], [y5], s=sz, color=F.SSP_COL[S5], edgecolor="white", lw=0.3, zorder=3)
    if r["area_share_pct"] >= 2.5 and "Other" not in r["province"]:
        LABS.append([x5 + 0.15, y5, r["province"]])
    xs_all += [x1, x5]
yr_ = np.ptp([l_[1] for l_ in LABS]) if LABS else 1.0
for _ in range(60):
    moved = False
    for i_ in range(len(LABS)):
        for j_ in range(i_ + 1, len(LABS)):
            a_, b_ = LABS[i_], LABS[j_]
            if abs(a_[0] - b_[0]) < 1.3 and abs(a_[1] - b_[1]) < 0.045 * yr_:
                d_ = 0.0225 * yr_ * (1 if a_[1] >= b_[1] else -1)
                a_[1] += d_; b_[1] -= d_; moved = True
    if not moved:
        break
for x_, y_, s_ in LABS:
    axb.text(x_, y_, s_, fontsize=4.8, va="center", color="0.25", zorder=4)
axb.axhline(0, color=F.INK, lw=0.5)
axb.set_xlim(min(xs_all) - 0.8, max(xs_all) + 1.6)
axb.set_xlabel("Future grain-filling Tavg (°C)", fontsize=6.3, labelpad=1)
axb.set_ylabel(f"Change in {T} (%)", fontsize=6.3, labelpad=1)
axb.tick_params(labelsize=5.8, length=1.5, pad=1)
axb.set_title("Provinces: SSP1-2.6 → SSP5-8.5", fontsize=7, fontweight="bold", pad=3)
hb = [Line2D([], [], ls="none", marker="o", ms=4, color=F.SSP_COL[S1], label="SSP1-2.6"),
      Line2D([], [], ls="none", marker="o", ms=4, color=F.SSP_COL[S5], label="SSP5-8.5")]
if TQ:
    hb += [Patch(color="0.90", label=f"Above training 95th percentile ({TQ[0.95]:.1f} °C)")]
axb.legend(handles=hb, fontsize=5.4, loc="upper left", frameon=False, handletextpad=0.3)
axb.text(-0.12, 1.03, "b", transform=axb.transAxes, fontsize=8, fontweight="bold")
out = HERE / "check_hrr_regional.png"
fig.savefig(out, dpi=300)
print(f"\n[saved] {out}\n        {HERE / 'check_hrr_regional.csv'}")
