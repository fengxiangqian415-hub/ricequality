import argparse, json, os, subprocess, sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "figs_1020"))
import config_1004 as C

TRAITS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]
TLAB = {"Acontent": "AC"}
REF = json.loads(r"""{"table1":{"LightGBM":{"BRR":[0.58,2.33],"MRR":[0.52,3.09],"HRR":[0.45,7.9],"CGR":[0.57,13.09],"CD":[0.53,6.05],"Acontent":[0.41,2.9],"PC":[0.63,0.94]},"XGBoost":{"BRR":[0.54,2.44],"MRR":[0.48,3.22],"HRR":[0.43,8.04],"CGR":[0.56,13.36],"CD":[0.53,6.1],"Acontent":[0.38,2.96],"PC":[0.56,1.02]},"GBDT":{"BRR":[0.53,2.45],"MRR":[0.48,3.22],"HRR":[0.42,8.11],"CGR":[0.56,13.29],"CD":[0.52,6.11],"Acontent":[0.38,2.98],"PC":[0.57,1.01]},"CatBoost":{"BRR":[0.54,2.44],"MRR":[0.47,3.25],"HRR":[0.43,8.06],"CGR":[0.55,13.47],"CD":[0.53,6.1],"Acontent":[0.38,2.97],"PC":[0.56,1.03]},"KNN":{"BRR":[0.49,2.56],"MRR":[0.39,3.49],"HRR":[0.39,8.32],"CGR":[0.5,14.16],"CD":[0.48,6.37],"Acontent":[0.28,3.2],"PC":[0.51,1.08]},"RF":{"BRR":[0.49,2.58],"MRR":[0.4,3.49],"HRR":[0.36,8.53],"CGR":[0.46,14.86],"CD":[0.44,6.64],"Acontent":[0.32,3.11],"PC":[0.4,1.19]},"SVR":{"BRR":[0.5,2.53],"MRR":[0.37,3.56],"HRR":[0.35,8.63],"CGR":[0.4,15.55],"CD":[0.41,6.82],"Acontent":[0.31,3.13],"PC":[0.51,1.08]},"ET":{"BRR":[0.37,2.86],"MRR":[0.25,3.89],"HRR":[0.27,9.19],"CGR":[0.31,16.74],"CD":[0.34,7.21],"Acontent":[0.26,3.25],"PC":[0.25,1.34]},"AdaBoost":{"BRR":[0.28,3.05],"MRR":[0.27,3.84],"HRR":[0.29,9.04],"CGR":[0.35,16.18],"CD":[0.28,7.53],"Acontent":[0.21,3.35],"PC":[0.35,1.24]},"DecisionTree":{"BRR":[0.34,2.9],"MRR":[0.38,3.54],"HRR":[0.16,9.8],"CGR":[0.29,16.8],"CD":[0.13,8.23],"Acontent":[0.09,3.58],"PC":[0.52,1.07]},"LassoLARS":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.14,9.97],"CGR":[0.13,18.8],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.05,1.5]},"Lasso":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.14,9.97],"CGR":[0.13,18.8],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.05,1.5]},"BayesianRidge":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.14,9.97],"CGR":[0.13,18.79],"CD":[0.19,7.98],"Acontent":[0.13,3.52],"PC":[0.05,1.51]},"ElasticNet":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.14,9.97],"CGR":[0.13,18.8],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.05,1.5]},"Ridge":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.13,9.97],"CGR":[0.13,18.8],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.05,1.51]},"Linear":{"BRR":[0.24,3.13],"MRR":[0.12,4.22],"HRR":[0.13,9.97],"CGR":[0.13,18.8],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.05,1.51]},"PLS":{"BRR":[0.24,3.14],"MRR":[0.11,4.23],"HRR":[0.13,9.98],"CGR":[0.13,18.83],"CD":[0.19,7.99],"Acontent":[0.13,3.52],"PC":[0.04,1.51]},"Huber":{"BRR":[0.24,3.14],"MRR":[0.11,4.23],"HRR":[0.12,10.07],"CGR":[0.11,19.04],"CD":[0.13,8.28],"Acontent":[0.11,3.55],"PC":[0.04,1.51]},"OMP":{"BRR":[0.15,3.32],"MRR":[0.03,4.42],"HRR":[0.04,10.47],"CGR":[0.08,19.36],"CD":[0.11,8.34],"Acontent":[0.1,3.57],"PC":[0.0,1.54]},"DummyMean":{"BRR":[-0.01,3.61],"MRR":[-0.0,4.5],"HRR":[-0.0,10.73],"CGR":[-0.01,20.2],"CD":[-0.01,8.88],"Acontent":[-0.01,3.79],"PC":[-0.01,1.55]},"PAR":{"BRR":[-0.5,4.4],"MRR":[-0.66,5.77],"HRR":[-0.04,10.94],"CGR":[-0.18,21.73],"CD":[-0.16,9.46],"Acontent":[-0.74,4.97],"PC":[-1.54,2.45]}},"table2":{"BRR":{"cv_mean":0.56,"cv_sd":0.07,"test_r2":[0.51,0.4,0.64],"test_rmse":[2.97,2.29,3.65],"test_mae":[1.87,1.57,2.17]},"MRR":{"cv_mean":0.49,"cv_sd":0.08,"test_r2":[0.43,0.32,0.53],"test_rmse":[3.48,2.99,3.95],"test_mae":[2.54,2.23,2.86]},"HRR":{"cv_mean":0.46,"cv_sd":0.11,"test_r2":[0.5,0.41,0.58],"test_rmse":[7.84,6.98,8.73],"test_mae":[5.86,5.19,6.56]},"CGR":{"cv_mean":0.53,"cv_sd":0.05,"test_r2":[0.63,0.53,0.72],"test_rmse":[12.82,11.24,14.34],"test_mae":[9.46,8.36,10.54]},"CD":{"cv_mean":0.51,"cv_sd":0.07,"test_r2":[0.62,0.46,0.74],"test_rmse":[5.89,4.88,6.9],"test_mae":[4.0,3.46,4.57]},"Acontent":{"cv_mean":0.39,"cv_sd":0.04,"test_r2":[0.43,0.32,0.52],"test_rmse":[2.85,2.46,3.24],"test_mae":[2.01,1.76,2.29]},"PC":{"cv_mean":0.64,"cv_sd":0.03,"test_r2":[0.7,0.6,0.77],"test_rmse":[0.95,0.83,1.09],"test_mae":[0.71,0.62,0.79]}},"figS1b":{"BRR":{"test":[0.51,2.97,1.87],"train":[0.75,1.83,1.23]},"MRR":{"test":[0.43,3.48,2.54],"train":[0.68,2.55,1.8]},"HRR":{"test":[0.5,7.84,5.86],"train":[0.69,5.94,4.43]},"CGR":{"test":[0.63,12.82,9.46],"train":[0.73,10.47,7.84]},"CD":{"test":[0.62,5.89,4.0],"train":[0.72,4.73,3.22]},"Acontent":{"test":[0.43,2.85,2.01],"train":[0.62,2.33,1.66]},"PC":{"test":[0.7,0.95,0.71],"train":[0.84,0.62,0.47]}}}""")
MINUS = "−"


def ts(p):
    try:
        return datetime.datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except FileNotFoundError:
        return "—"


def f2(x):
    return f"{x:.2f}".replace("-", MINUS)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), float(np.sqrt(np.mean((y - p) ** 2))),
            float(np.mean(np.abs(y - p))))


def find_dirs(roots, name):
    skip = {"rice_quality_prediction", "annual_nc", "__pycache__", ".git", "ALL_FIGS_1105", "ALL_DATA_1105",
            "ALL_FIGS_1108", "ALL_DATA_1108", "historical", "future", "ssp126", "ssp585"}
    hits = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if d not in skip and not d.startswith(".")]
            if dp.count(os.sep) - str(r).count(os.sep) > 7:
                dn[:] = []
            if name in fn:
                hits.append(Path(dp))
    return sorted(set(hits))


def read_tables(d):
    d = Path(d); out = {}
    for k in ("r2", "rmse", "mae"):
        p = d / f"{k}_matrix.csv"
        out[k] = pd.read_csv(p, index_col=0) if p.exists() else None
    p = d / "Blending_predictions.csv"
    out["pred"] = pd.read_csv(p) if p.exists() else None
    return out


def table1_diffs(tb):
    if tb["r2"] is None or tb["rmse"] is None:
        return None, []
    bad = []
    for m, row in REF["table1"].items():
        for t, (r2, rmse) in row.items():
            try:
                a, b = float(tb["r2"].loc[m, t]), float(tb["rmse"].loc[m, t])
            except KeyError:
                bad.append(f"表1 {m}/{t}: 训练输出无此格"); continue
            if round(a, 2) != r2 or round(b, 2) != rmse:
                bad.append(f"表1 {m}/{TLAB.get(t, t)}: SI {f2(r2)} ({rmse:.2f}) → 此目录 {f2(a)} ({b:.2f})")
    return len(bad), bad


def scatter_metrics(pred):
    rows = {}
    for t in TRAITS:
        d = pred[pred.Target.astype(str) == t]
        for st in ("Train", "Test"):
            s = d[d.Set.astype(str).str.lower() == st.lower()]
            if len(s):
                rows[(t, st)] = (len(s),) + metrics(s.Observed, s.Predicted)
    return rows


def scatter_diffs(pred):
    if pred is None:
        return None, []
    sm = scatter_metrics(pred); bad = []
    for t, v in REF["figS1b"].items():
        for st, key in (("Test", "test"), ("Train", "train")):
            if (t, st) not in sm:
                bad.append(f"散点 {t}/{st}: 缺"); continue
            n, r2, rmse, mae = sm[(t, st)]
            new = [round(float(r2), 2), round(float(rmse), 2), round(float(mae), 2)]
            if new != v[key]:
                bad.append(f"补图1b/表2 {TLAB.get(t, t)} {st}: SI R²/RMSE/MAE {v[key]} → 此目录 {new}")
    return len(bad), bad


def diagnose(roots):
    print("== 训练输出候选目录（含 r2_matrix.csv 或 Blending_predictions.csv）")
    cands = sorted(set(find_dirs(roots, "r2_matrix.csv") + find_dirs(roots, "Blending_predictions.csv")))
    default = Path(C.MODEL_DIR).parent / "Tables"
    if default.exists() and default not in cands:
        cands.insert(0, default)
    res = []
    for d in cands:
        tb = read_tables(d)
        n1, b1 = table1_diffs(tb); n2, b2 = scatter_diffs(tb["pred"])
        tag = "= SI 版" if (n1 == 0 and n2 == 0) else ("≠ SI 版" if (n1 or n2) else "无法比对")
        print(f"\n  [{tag}] {d}" + ("   ← config 默认（21 号读这里）" if d == default else ""))
        for f in ("r2_matrix.csv", "Blending_predictions.csv", "Test_bootstrap_CI95.csv", "Ensemble_CV_folds_training.csv"):
            if (d / f).exists():
                print(f"      {f:34s} {ts(d / f)}")
        print(f"      与 SI 不同：表1 {n1 if n1 is not None else '—'} 格；散点/表2 {n2 if n2 is not None else '—'} 项")
        for b in (b1 + b2)[:12]:
            print("        · " + b)
        if len(b1 + b2) > 12:
            print(f"        · …（共 {len(b1 + b2)} 项）")
        res.append((d, tag))
    print("\n== 模型文件（*_Blending.joblib）")
    mdirs = find_dirs(roots, "PC_Blending.joblib")
    if Path(C.MODEL_DIR).exists() and Path(C.MODEL_DIR) not in mdirs:
        mdirs.insert(0, Path(C.MODEL_DIR))
    for md in mdirs:
        fs = sorted(Path(md).glob("*_Blending*.joblib"))
        print(f"  {md}" + ("   ← config.MODEL_DIR（04 号预测用这里）" if Path(md) == Path(C.MODEL_DIR) else ""))
        for f in fs:
            print(f"      {f.name:32s} {ts(f)}")
    print("\n== 预测栅格（04 号输出）与工作簿")
    pdir = C.pred_dir()
    tifs = sorted(pdir.rglob("*.tif"))[:2000] if pdir.exists() else []
    if tifs:
        mt = [f.stat().st_mtime for f in tifs]
        fmt = lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M")
        print(f"  {pdir}: {len(tifs)} 个 tif（抽查）  最早 {fmt(min(mt))}  最晚 {fmt(max(mt))}")
    xl = C.out_dir("excel") / f"paper_data_{C.TAG}.xlsx"
    print(f"  {xl}: {ts(xl)}")
    if xl.exists():
        try:
            sc = pd.read_excel(xl, sheet_name="01_ensemble_scatter")
            b = sc[(sc.trait == "BRR") & (sc.set == "Test")].iloc[0]
            print(f"    工作簿 01_ensemble_scatter BRR Test R² = {b.R2:.4f}（SI 版 0.51）")
        except Exception as e:
            print("    （读工作簿失败：", e, "）")
    print("\n== 判定")
    print("  终版 = 与 MODEL_DIR 里 *_Blending.joblib 同一次训练写出的 Tables（时间最接近、且早于全部预测栅格）。")
    print("  · 若模型文件晚于预测栅格：现存模型不是生成预测时那一版，先查清楚再同步（不要直接 --write）。")
    print("  · 确认后：python sync_model_skill_1108.py --use <终版 Tables 目录> --write")
    return res


def read_cv_folds(d):
    p = Path(d) / "Ensemble_CV_folds_training.csv"
    if not p.exists():
        return None
    try:
        import skill_tables as ST
        g = ST.read_cv_folds(p)
        return g.groupby("trait").r2.agg(mean="mean", sd=lambda v: v.std(ddof=1))
    except Exception as e:
        print("  ⚠ 读 Ensemble_CV_folds_training.csv 失败：", e); return None


def bootstrap(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); y, p = np.asarray(y, float), np.asarray(p, float); k = len(y)
    out = np.array([metrics(y[i], p[i]) for i in (rng.integers(0, k, k) for _ in range(n))])
    return np.percentile(out, [2.5, 97.5], axis=0)


def write_all(d, outdir):
    d = Path(d); outdir.mkdir(parents=True, exist_ok=True)
    tb = read_tables(d)
    if tb["r2"] is None or tb["pred"] is None:
        sys.exit(f"{d} 缺 r2_matrix.csv 或 Blending_predictions.csv")
    R2, RM = tb["r2"][TRAITS].astype(float), tb["rmse"][TRAITS].astype(float)
    order = R2.mean(axis=1).sort_values(ascending=False).index
    t1 = pd.DataFrame({"Algorithm": order})
    for t in TRAITS:
        t1[TLAB.get(t, t)] = [f"{f2(R2.loc[m, t])} ({RM.loc[m, t]:.2f})" for m in order]
    t1.to_csv(outdir / "SI_Table1_benchmark.csv", index=False, encoding="utf-8-sig")
    sm = scatter_metrics(tb["pred"]); cv = read_cv_folds(d)
    t2, ins, s3 = [], [], []
    for t in TRAITS:
        dd = tb["pred"][tb["pred"].Target.astype(str) == t]
        te = dd[dd.Set.astype(str).str.lower() == "test"]
        lo, hi = bootstrap(te.Observed, te.Predicted)
        ntr, r2tr, rmtr, matr = sm[(t, "Train")]; nte, r2, rm, ma = sm[(t, "Test")]
        cvs = f"{cv.loc[t, 'mean']:.2f} ± {cv.loc[t, 'sd']:.2f}" if cv is not None and t in cv.index else "（缺 Ensemble_CV_folds_training.csv）"
        t2.append({"Trait": TLAB.get(t, t), "n (train / test)": f"{ntr} / {nte}", "Ensemble five-fold CV R² (mean ± SD)": cvs,
                   "Test R² (95% CI)": f"{r2:.2f} ({lo[0]:.2f}–{hi[0]:.2f})",
                   "Test RMSE (95% CI)": f"{rm:.2f} ({lo[1]:.2f}–{hi[1]:.2f})",
                   "Test MAE (95% CI)": f"{ma:.2f} ({lo[2]:.2f}–{hi[2]:.2f})"})
        ins.append(dict(trait=TLAB.get(t, t), train_R2=round(r2tr, 2), train_RMSE=round(rmtr, 2), train_MAE=round(matr, 2),
                        test_R2=round(r2, 2), test_RMSE=round(rm, 2), test_MAE=round(ma, 2)))
        s3.append(dict(trait=t, cv_r2_mean=round(cv.loc[t, "mean"], 2) if cv is not None else np.nan,
                       cv_r2_sd=round(cv.loc[t, "sd"], 2) if cv is not None else np.nan, test_r2=round(r2, 2),
                       test_r2_lo=round(lo[0], 2), test_r2_hi=round(hi[0], 2), test_rmse=round(rm, 2), test_mae=round(ma, 2)))
    pd.DataFrame(t2).to_csv(outdir / "SI_Table2_ensemble.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ins).to_csv(outdir / "figS1b_insets.csv", index=False)
    pd.DataFrame(s3).to_csv(HERE / "figs_1020" / "model_skill_tableS3.csv", index=False, float_format="%.2f")
    lines = [f"终版训练输出：{d}", f"生成：{datetime.datetime.now():%Y-%m-%d %H:%M}", ""]
    n1, b1 = table1_diffs(tb); n2, b2 = scatter_diffs(tb["pred"])
    for t in TRAITS:
        r = REF["table2"][t]; row = next(x for x in s3 if x["trait"] == t)
        for k, new in (("test_r2", [float(row["test_r2"]), float(row["test_r2_lo"]), float(row["test_r2_hi"])]),):
            if [round(v, 2) for v in new] != r[k]:
                b2.append(f"表2 {TLAB.get(t, t)} 检验 R² (95% CI): SI {r[k]} → {new}")
        if cv is not None and t in cv.index and (round(cv.loc[t, "mean"], 2), round(cv.loc[t, "sd"], 2)) != (r["cv_mean"], r["cv_sd"]):
            b2.append(f"表2 {TLAB.get(t, t)} CV: SI {r['cv_mean']} ± {r['cv_sd']} → {cv.loc[t, 'mean']:.2f} ± {cv.loc[t, 'sd']:.2f}")
    lines += ["与现 SI 相同 ✔（只需重跑 21 号，工作表即与 SI 同源）"] if not (b1 or b2) else \
             [f"与现 SI 不同 {len(b1) + len(b2)} 处（改补表 1–2、补图 1；正文引用的检验 R² 见最后一行）："] + ["  " + x for x in b1 + b2]
    te = {x["trait"]: x["test_R2"] for x in ins}
    lines += ["", "正文 Results 第一段检验 R²（按降序）： " + "; ".join(f"{k}, {v:.2f}" for k, v in sorted(te.items(), key=lambda kv: -kv[1])),
              f"各性状检验 R² 范围：{min(te.values()):.2f}–{max(te.values()):.2f}（SI 版 0.43–0.70）",
              "注：表 2 的 95% CI 为本脚本对 228 条检验记录 1,000 次 bootstrap（seed 42）的 2.5/97.5 百分位；"
              "若训练流程的 Test_bootstrap_CI95.csv 与终版同批，可改用其中数值（下方并列）。"]
    p = d / "Test_bootstrap_CI95.csv"
    if p.exists():
        lines += ["", f"训练流程 Test_bootstrap_CI95.csv（{ts(p)}）：", pd.read_csv(p).to_string(index=False)]
    (outdir / "diff_vs_SI.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    py = sys.executable
    print("\n== 重跑 21 号（--bench-tables）"); subprocess.run([py, str(HERE / "21_export_paper_data_1004.py"), "--bench-tables", str(d)], check=True, cwd=HERE)
    print("\n== 出补图 1"); subprocess.run([py, str(HERE / "figs_1020" / "figS01_model_benchmark.py"), "--tables", str(d), "--out", str(outdir)], check=True, cwd=HERE)
    print(f"\n完成：{outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="*", default=[str(Path(C.MODEL_DIR).parent.parent), "/home/debian/projects"])
    ap.add_argument("--use", default=None, help="确认的终版 Tables 目录")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.use and a.write:
        write_all(a.use, C.FIGURES / "model_skill_1108")
    else:
        diagnose(a.search)
        if a.use:
            print("\n（只给了 --use，没有 --write：未写任何文件）")


if __name__ == "__main__":
    main()
