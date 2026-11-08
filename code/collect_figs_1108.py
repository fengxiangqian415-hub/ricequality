# ============================================================
# collect_figs_1108.py —— 终版图包：正文图 1–4 + 补图 1–25（1108f 编号 = 按首次引用重排后的定稿 SI）
#   用法（code_1004b 目录下）：python collect_figs_1108.py [--out 目录] [--ext svg,pdf,png,tiff] [--with-extra]
#   输出：output/figures/ALL_FIGS_1108/
#           Fig1_*…Fig4_*、FigS01_*…FigS25_*（每图同名 svg/pdf/png[/tiff]，与 Word 内嵌的是同一次导出）
#           _index.csv    每个文件 ← 来源路径、修改时间、生成脚本
#           _missing.txt  只列真正缺的图（空文件 = 齐全）
#           _extra/       --with-extra 时放备查图（SHAP 分图、CO₂ 诊断等，不进 SI）
#   编号对照（1105 → 1108f）：S14→S17（CO₂ 交互）、S15→S14（CGR）、S16→S15（CD）、S17→S16（PC）、
#                            S25→S23（估值对比）、S23→S24（流程图）、S24→S25（相关筛选）；其余不变
#   来源：每项按候选顺序取第一个存在的文件组；
#     补图 1  = figs_1020/figS01_model_benchmark.py 的重绘版（sync_model_skill_1108.py --write 之后），否则 static_figs/（SI 现版）
#     补图 23 = 36 号两口径终版 FigS23_valuation.*（旧名 FigS25_valuation.* 只在 36 号未重跑时作后备，并给出提示）
#     补图 24、25 = static_figs/（流程示意图、训练流程的相关筛选图；从定稿 SI 原样导出）
# ============================================================
import argparse, csv, datetime, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_1004 as C

p = argparse.ArgumentParser()
p.add_argument("--out", default=None)
p.add_argument("--ext", default="svg,pdf,png,tiff")
p.add_argument("--with-extra", action="store_true")
a = p.parse_args()
OUT = Path(a.out) if a.out else C.FIGURES / "ALL_FIGS_1108"
if OUT.exists():
    shutil.rmtree(OUT)                                    # 每次全新生成，避免旧编号文件残留
OUT.mkdir(parents=True)
EXTS = [e.strip().lstrip(".") for e in a.ext.split(",") if e.strip()]

CODE = Path(__file__).resolve().parent
STATIC = CODE / "static_figs"
F = C.FIGURES
MAIN = F / "main_figs_1020"
QUAL = F / f"rice_quality_{C.TAG}_{C.PRED_MODE}"
CLIM = F / f"climate_change_{C.TAG}"
PHEN = F / f"phenology_{C.TAG}"
SHAP = C.fig_dir("shap", thresh=False)
VAL = F / "valuation_1106"

# (终版文件名, [(源目录, 源文件名), …按优先级], 生成脚本)
ITEMS = [
    ("Fig1_data_framework",          [(MAIN, "Fig1_data_framework")],                  "figs_1020/fig1_framework.py"),
    ("Fig2_main",                    [(MAIN, "Fig2_main")],                            "figs_1020/fig2_main.py"),
    ("Fig3_zones",                   [(C.fig_dir("fig4_migration"), "fig4_grid")],     "15_plot_fig4_grid_1004.py"),
    ("Fig4_bivariate_loss",          [(C.out_dir("loss"), "bivariate_2x2_with_bar")],  "16_plot_bivariate_loss_1004.py"),
    ("FigS01_model_benchmark",       [(F / "model_skill_1108", "FigS01_model_benchmark"), (STATIC, "FigS01_model_benchmark")],
                                     "figs_1020/figS01_model_benchmark.py（后备：static_figs）"),
    ("FigS02_feature_distribution",  [(F / f"feature_distribution_{C.TAG}", "feature_distribution")], "08_plot_feature_distribution_1004.py"),
    ("FigS03_phenology",             [(PHEN, "CYGMA1p74_gfdl-esm4")],                  "07_plot_phenology_1004.py"),
    ("FigS04_Tavg",                  [(CLIM, "Tavg")],                                 "06_plot_climate_change_1004.py"),
    ("FigS05_Tdif",                  [(CLIM, "Tdif")],                                 "06_plot_climate_change_1004.py"),
    ("FigS06_CTSR",                  [(CLIM, "CTSR")],                                 "06_plot_climate_change_1004.py"),
    ("FigS07_TSR",                   [(CLIM, "TSR")],                                  "06_plot_climate_change_1004.py"),
    ("FigS08_Rain",                  [(CLIM, "Rain")],                                 "06_plot_climate_change_1004.py"),
    ("FigS09_RHavg",                 [(CLIM, "RHavg")],                                "06_plot_climate_change_1004.py"),
    ("FigS10_HRR",                   [(QUAL, "HRR")],                                  "09_plot_rice_quality_change_1004.py"),
    ("FigS11_BRR",                   [(QUAL, "BRR")],                                  "09_plot_rice_quality_change_1004.py"),
    ("FigS12_MRR",                   [(QUAL, "MRR")],                                  "09_plot_rice_quality_change_1004.py"),
    ("FigS13_hrr_regional",          [(MAIN, "FigS_hrr_regional")],                    "figs_1020/figS_hrr_regional.py"),
    ("FigS14_CGR",                   [(QUAL, "CGR")],                                  "09_plot_rice_quality_change_1004.py"),
    ("FigS15_CD",                    [(QUAL, "CD")],                                   "09_plot_rice_quality_change_1004.py"),
    ("FigS16_PC",                    [(QUAL, "PC")],                                   "09_plot_rice_quality_change_1004.py"),
    ("FigS17_co2_interaction",       [(MAIN, "FigS_co2_interaction")],                 "figs_1020/figS_co2_interaction.py"),
    ("FigS18_AC",                    [(QUAL, "Acontent")],                             "09_plot_rice_quality_change_1004.py"),
    ("FigS19_uncertainty",           [(C.fig_dir("uncertainty", thresh=False), f"uncertainty_anova_4factor_{C.TAG}_change")],
                                     "20_uncertainty_1004.py"),
    ("FigS20_shap_summary",          [(SHAP, "shap_grading_summary_both")],            "19b_shap_summary_both_1004.py"),
    ("FigS21_head_rice_yield",       [(C.fig_dir("head_rice_yield", thresh=False), "head_rice_yield")], "10_calc_head_rice_yield_1004.py"),
    ("FigS22_price_trend",           [(F, "price_trend"), (CODE, "price_trend")],      "figS22_price_trends.py"),
    ("FigS23_valuation",             [(VAL, "FigS23_valuation"), (VAL, "FigS25_valuation")], "36_yield_only_valuation_1106.py"),
    ("FigS24_workflow",              [(STATIC, "FigS24_workflow")],                    "static_figs（示意图，另行绘制）"),
    ("FigS25_correlation",           [(STATIC, "FigS25_correlation")],                 "static_figs（训练流程相关筛选图）"),
]
EXTRA = [
    ("Xtra_FigS_hrr_response",       [(MAIN, "FigS_hrr_response")],                    "figs_1020/figS_hrr_response.py"),
    ("Xtra_co2_response_diagnosis",  [(F / f"co2_diagnosis_{C.TAG}", "co2_response_diagnosis")], "17_plot_co2_response_diagnosis_1004.py"),
]
for t in ("HRR", "CD", "PC"):
    for ri in C.RI_TYPES:
        EXTRA.append((f"Xtra_shap_{t}_{ri}", [(SHAP, f"shap_{t}_{ri}_3scen")], "19_shap_analysis_1004.py"))

rows, missing, notes, n = [], [], [], 0


def collect(items, dest, optional=False):
    global n
    dest.mkdir(parents=True, exist_ok=True)
    for dst, cands, script in items:
        hit = None
        for d, stem in cands:
            if any((Path(d) / f"{stem}.{e}").exists() for e in EXTS):
                hit = (Path(d), stem); break
        if hit is None:
            msg = f"{dst:30s} <- " + " | ".join(f"{Path(d) / s}.*" for d, s in cands) + f"   （{script}）"
            if optional:
                notes.append("备查图未找到（不影响图包）：" + msg.split(" <- ")[0].strip())
            else:
                missing.append(msg)
            continue
        d, stem = hit
        if dst == "FigS23_valuation" and stem == "FigS25_valuation":
            notes.append("FigS23_valuation 取的是旧名 FigS25_valuation.*：请先重跑 36 号（1108f 版），确认是两口径终版")
        if dst == "FigS01_model_benchmark" and d == STATIC:
            notes.append("FigS01 用的是 static_figs（SI 现版）；终版训练输出确认并 sync --write 后会自动换成重绘版")
        for e in EXTS:
            src = d / f"{stem}.{e}"
            if src.exists():
                shutil.copy2(src, dest / f"{dst}.{e}"); n += 1
                st = src.stat()
                rows.append([f"{dest.name if dest != OUT else ''}/{dst}.{e}".lstrip("/"), str(src), st.st_size,
                             datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"), script])


collect(ITEMS, OUT)
if a.with_extra:
    collect(EXTRA, OUT / "_extra", optional=True)
with open(OUT / "_index.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["file", "source", "bytes", "mtime", "script"]); w.writerows(rows)
(OUT / "_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print(f"[collect 1108f] 复制 {n} 个文件 → {OUT}")
print(f"[collect 1108f] 缺 {len(missing)} 项" + ("（见 _missing.txt）：" if missing else "（齐全）"))
for m in missing:
    print("    ", m)
for x in notes:
    print("  ⚠", x)
