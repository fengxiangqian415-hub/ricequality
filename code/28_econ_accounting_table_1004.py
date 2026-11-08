import numpy as np
import pandas as pd
import config_1004 as C

CNY_PER_USD = C.CNY_PER_USD
OUT = C.out_dir("loss")
F14 = OUT / "production_loss_summary.csv"
F24 = C.out_dir("sensitivity") / "econ_members_long.csv"
for f, step in ((F14, "14_compute_production_loss_1004.py"), (F24, "24_econ_members_1004.py")):
    if not f.exists():
        raise SystemExit(f"✘ 缺 {f}，请先运行 {step}")

usd = lambda yi: yi / 10.0 / CNY_PER_USD
RI_LAB = {"ri1": "Single/early rice", "ri2": "Late rice", "ri1+ri2": "National"}
SSP_LAB = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
QTY = [("baseline", "Baseline value"), ("future", "Future value"), ("output", "Output effect"),
       ("grade", "Grade-price effect"), ("net", "Net change")]

E = pd.read_csv(F14)
E = E[E.ri.isin(RI_LAB)]
ens = pd.DataFrame({"ssp": E.ssp, "ri": E.ri,
                    "baseline": usd(E.V_his_yi), "future": usd(E.V_fut_yi),
                    "output": -usd(E.loss_qty_yi), "grade": -usd(E.loss_qual_yi),
                    "net": -usd(E.loss_yi)})
chk = (ens.output + ens.grade - ens.net).abs().max()
print(f"[核对] 集合估计：产量效应 + 等级价格效应 − 净变化 最大差 {chk:.2e} 十亿美元")

M = pd.read_csv(F24)
mem = pd.DataFrame({"ssp": M.ssp, "ri": M.ri, "model": M.model, "gcm": M.gcm,
                    "baseline": usd(M.V_hist), "future": usd(M.V_fut),
                    "output": -usd(M.output_effect), "grade": -usd(M.grade_effect),
                    "net": -usd(M.net_loss)})

rows = []
for (ssp, ri), g in mem.groupby(["ssp", "ri"]):
    e = ens[(ens.ssp == ssp) & (ens.ri == ri)]
    for q, _ in QTY:
        rows.append(dict(ssp=ssp, ri=ri, quantity=q,
                         ensemble=float(e[q].iloc[0]) if len(e) else np.nan,
                         member_median=g[q].median(), member_min=g[q].min(), member_max=g[q].max(),
                         n_members=len(g)))
N = pd.DataFrame(rows)
N.round(3).to_csv(OUT / "econ_accounting_numeric.csv", index=False, encoding="utf-8-sig")

f2 = lambda x: f"{x:.2f}".replace("-", "−")
fs = lambda x: f"{x:+.2f}".replace("-", "−")
txt = []
for ssp in C.SSPS:
    tab = []
    for ri in ["ri1", "ri2", "ri1+ri2"]:
        r = {"Rice type": RI_LAB[ri]}
        for q, lab in QTY:
            x = N[(N.ssp == ssp) & (N.ri == ri) & (N.quantity == q)]
            if x.empty: continue
            x = x.iloc[0]
            ff = f2 if q in ("baseline", "future") else fs
            r[f"{lab} (US$ bn yr-1)"] = f"{ff(x.ensemble)} [{ff(x.member_min)}, {ff(x.member_max)}]"
        e = ens[(ens.ssp == ssp) & (ens.ri == ri)].iloc[0]
        g = mem[(mem.ssp == ssp) & (mem.ri == ri)]
        r["Net change (% of baseline)"] = f"{100 * e.net / e.baseline:.1f}".replace("-", "−")
        r["Members with net decrease"] = f"{int((g.net < 0).sum())}/{len(g)}"
        tab.append(r)
    T = pd.DataFrame(tab)
    T.to_csv(OUT / f"econ_accounting_{ssp}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(f"\n==== {SSP_LAB[ssp]}：集合估计 [25 成员最小, 最大]，十亿美元 yr⁻¹（负 = 减少）====")
    print(T.to_string(index=False))

    nat = N[(N.ssp == ssp) & (N.ri == "ri1+ri2")].set_index("quantity")
    g = mem[(mem.ssp == ssp) & (mem.ri == "ri1+ri2")]
    o, gr, ne = nat.loc["output"], nat.loc["grade"], nat.loc["net"]
    txt.append(
        f"{SSP_LAB[ssp]}: the national mean annual head-rice value changed by {fs(ne.ensemble)} billion US$ yr-1 "
        f"(25-member range {fs(ne.member_min)} to {fs(ne.member_max)}), the sum of an output effect of "
        f"{fs(o.ensemble)} ({fs(o.member_min)} to {fs(o.member_max)}) and a grade-price effect of "
        f"{fs(gr.ensemble)} ({fs(gr.member_min)} to {fs(gr.member_max)}); "
        f"{int((g.net < 0).sum())} of {len(g)} members projected a net decrease.")
    ratio = abs(ne.ensemble) / max(abs(o.ensemble), abs(gr.ensemble), 1e-9)
    if np.sign(o.ensemble) != np.sign(gr.ensemble) and ratio < 0.5:
        txt.append(f"  Note: the net change is {100*ratio:.0f}% of the larger opposing component, so it is "
                   f"sensitive to rounding and member spread; report two decimals and the range.")

txt.append("\nCentral values are ensemble estimates (ensemble-mean HRY; grades from ensemble-mean traits); "
           "ranges are minima and maxima across the 25 GGCM x GCM members, each graded separately. "
           "Scenarios are computed independently and are never summed or averaged.")
(OUT / "econ_accounting_text.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n" + "\n".join(txt))
print(f"\n[saved] {OUT}/econ_accounting_ssp126.csv / econ_accounting_ssp585.csv / "
      f"econ_accounting_numeric.csv / econ_accounting_text.txt")
