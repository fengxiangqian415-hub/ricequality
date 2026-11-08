# ============================================================
# collect_data_1108.py —— 终版数据包（与 collect_figs_1108.py 同一编号；1108f）
#   用法（code_1004b 目录下，先跑 collect_figs_1108.py）：python collect_data_1108.py [--with-tif]
#   输出：output/ALL_DATA_1108/ 及 ALL_DATA_1108.zip
#     00_paper_data/   paper_data_{TAG}.xlsx（21 号：正文与 SI 全部数字）
#     01_figs/         = output/figures/ALL_FIGS_1108/ 的逐字节拷贝（与 Word 内嵌、图包顶层是同一次导出）
#     02_fig_source/   按图分文件夹的源数据（只放 csv/json/txt；不再混放图片，避免旧版图混入）
#     03_tables_si/    SI 表的数据来源
#     04_logs/         运行日志
#     05_code/         脚本快照（含 figs_1020、static_figs）
#     MANIFEST.csv / README_1108.md
#   [1108f] 与 1105 版的区别：文件夹按定稿编号命名（S14→S17 等，见 collect_figs_1108.py 表头）；
#           补图 23（估值对比）的源数据 = 36 号两口径终版的 FigS23_valuation_source_*.csv；
#           补图 1 的源数据 = model_skill_1108/（sync_model_skill_1108.py --write 输出）；补图 22 = 两份价格 csv
# ============================================================
import sys, shutil, csv, datetime, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_1004 import OUTPUT, FIGURES, TAG, PRED_MODE, out_dir, fig_dir

ap = argparse.ArgumentParser(); ap.add_argument("--with-tif", action="store_true"); A = ap.parse_args()
CODE = Path(__file__).resolve().parent
DEST = OUTPUT / "ALL_DATA_1108"
FIGS = FIGURES / "ALL_FIGS_1108"
if not FIGS.exists():
    sys.exit(f"先运行 collect_figs_1108.py（找不到 {FIGS}）")
if DEST.exists(): shutil.rmtree(DEST)
DEST.mkdir(parents=True)

TAB = {".csv", ".xlsx", ".json", ".txt", ".md"}
TIF = {".tif"} if A.with_tif else set()
FIG = {".png", ".pdf", ".svg", ".tiff"}

# (来源目录, 目标子目录, 扩展名, 文件名过滤, 对应图/表说明)
SRC = [
    (out_dir("excel"),                               "00_paper_data",                         TAB,       None, "正文与 SI 全部数字（21 号）"),
    (FIGS,                                           "01_figs",                               FIG | {".csv", ".txt"}, None, "正文图 1–4、补图 1–25（终版编号）"),
    (FIGURES / "main_figs_1020",                     "02_fig_source/Fig1_Fig2_S13_S17",       TAB,       None, "Fig. 1/2、S13、S17 的中间表"),
    (CODE / "figs_1020",                             "02_fig_source/Fig2_S13_S17_checks",     {".csv", ".txt"}, None, "Fig. 2 成员范围、S13 省级 HRR、S17 CO₂ 交互的核对表；model_skill_tableS3.csv"),
    (FIGURES / "model_skill_1108",                   "02_fig_source/S01_model_skill",         TAB,       None, "补图 1、补表 1–2（sync_model_skill_1108.py）"),
    (OUTPUT / f"decomposition_373_{TAG}",            "02_fig_source/Fig2b_decomposition",     TAB,       None, "Fig. 2b 暴露分解"),
    (fig_dir("shap", thresh=False),                  "02_fig_source/Fig2d_S20_shap",          TAB,       None, "Fig. 2d ΔSHAP、S20"),
    (out_dir("shap_features", thresh=False),         "02_fig_source/Fig2d_S20_shap",          TAB,       None, "SHAP 输入特征"),
    (fig_dir("uncertainty", thresh=False),           "02_fig_source/Fig2e_S19_uncertainty",   TAB,       None, "Fig. 2e / S19 方差分解"),
    (out_dir("quality_zones") / PRED_MODE,           "02_fig_source/Fig3_zones",              TAB | TIF, None, "Fig. 3 分区、member_zone_share"),
    (out_dir("quality_zones") / "diagnosis",         "02_fig_source/Fig3_zones",              TAB,       None, "分区诊断（12 号）"),
    (out_dir("migration_stats") / PRED_MODE,         "02_fig_source/Fig3_transitions",        TAB,       None, "Fig. 3 转移矩阵（13 号）"),
    (out_dir("loss"),                                "02_fig_source/Fig4_loss",               TAB | TIF, None, "Fig. 4 双变量图、分省损失"),
    (out_dir("head_rice_yield", thresh=False),       "02_fig_source/Fig4_S21_head_rice_yield", TAB | {".tif"}, None, "整精米产量（S21、经济核算输入）"),
    (fig_dir("head_rice_yield", thresh=False),       "02_fig_source/Fig4_S21_head_rice_yield", TAB,      None, ""),
    (FIGURES / f"climate_change_{TAG}",              "02_fig_source/S4-S9_climate",           TAB,       None, "S4–S9 气候要素"),
    (FIGURES / f"phenology_{TAG}",                   "02_fig_source/S3_phenology",            TAB,       None, "S3 物候"),
    (FIGURES / f"feature_distribution_{TAG}",        "02_fig_source/S2_feature_distribution", TAB,       None, "S2 特征分布"),
    (FIGURES / f"rice_quality_{TAG}_{PRED_MODE}",    "02_fig_source/S10-S18_quality_change",  TAB,       None, "S10–S12、S14–S16、S18 品质变化"),
    (CODE,                                           "02_fig_source/S22_price_trend",         {".csv"},  "price_trends_plotdata", "S22 价格序列（figS22_price_trends.py 的输入）"),
    (FIGURES / "valuation_1106",                     "02_fig_source/S23_valuation",           {".csv"},  "FigS23_valuation_source", "S23 两口径估值（36 号；条形 = 集合平均，散点 = 成员）"),
    (FIGURES / f"co2_diagnosis_{TAG}",               "02_fig_source/co2_diagnosis",           TAB,       None, "CO₂ 响应诊断（备查）"),
    (out_dir("sensitivity"),                         "03_tables_si/sensitivity_econ",         TAB,       None, "SI 表 3–13 的来源：阈值、PC 权重、价比、成员经济核算、逐年分类（31）、MC 扰动（32）、ROC（33/34）、国标换算（35）、估值对比（36）"),
    (OUTPUT / f"analysis_domain_{TAG}",              "03_tables_si/analysis_domain",          TAB,       None, "统一格网域"),
    (OUTPUT / f"consistency_audit_{TAG}",            "03_tables_si/consistency_audit",        TAB,       None, "30 号一致性审计"),
]
LOGS = sorted(d for d in CODE.glob("logs_*") if d.is_dir())

rows = []; n = 0; missing = []
def copy_dir(src, sub, exts, pat, note):
    global n
    src = Path(src)
    if not src.exists():
        missing.append(f"{sub} <- {src}"); return
    files = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in exts and (pat is None or pat in f.name)]
    d = DEST / sub; d.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, d / f.name); n += 1
        st = f.stat()
        rows.append([f"{sub}/{f.name}", str(f), st.st_size,
                     datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"), note])
    print(f"  {sub:44s} <- {len(files):4d}  ({src})")

for src, sub, exts, pat, note in SRC:
    copy_dir(src, sub, exts, pat, note)
for lg in LOGS:
    copy_dir(lg, f"04_logs/{lg.name}", {".log"}, None, "运行日志")

# 代码快照
cd = DEST / "05_code"; cd.mkdir()
for f in list(CODE.glob("*.py")) + list(CODE.glob("*.sh")) + list(CODE.glob("*.md")) + list(CODE.glob("*.csv")):
    shutil.copy2(f, cd / f.name); n += 1
for sub, pats in (("figs_1020", ("*.py", "*.csv")), ("static_figs", ("*",))):
    (cd / sub).mkdir()
    for pat in pats:
        for f in (CODE / sub).glob(pat):
            if f.is_file():
                shutil.copy2(f, cd / sub / f.name); n += 1

with open(DEST / "MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["dest", "source", "bytes", "mtime", "used_by"]); w.writerows(rows)

(DEST / "README_1108.md").write_text(f"""# ALL_DATA_1108（{datetime.date.today()}；编号 = 定稿 SI 1108f）

## 正文 / SI 数字的来源
`00_paper_data/paper_data_{TAG}.xlsx`（21 号）。关键工作表：01_model_benchmark / 01_ensemble_scatter（补图 1、补表 1–2）、
02_quality_change、03_co2_effect、04_zone_area、07_head_rice_yield、08_econ_loss、09_climate_change、12_uncertainty、13_shap、
15_thresholds、16_zone_member_share、41_annual_classification（补表 5；31 号，有效面积口径）、42_quality_error_mc（补表 12）、
43_valuation_tiers（补表 10）。

## 图 ↔ 脚本 ↔ 源数据（01_figs 与图包顶层逐字节相同）
| 图 | 脚本 | 源数据（02_fig_source/） |
|---|---|---|
| Fig. 1 | figs_1020/fig1_framework.py | Fig1_Fig2_S13_S17 |
| Fig. 2 | figs_1020/fig2_main.py（d：19/19b；e：20） | Fig2b_decomposition、Fig2d_S20_shap、Fig2e_S19_uncertainty |
| Fig. 3 | 15 | Fig3_zones、Fig3_transitions |
| Fig. 4 | 16（+24b） | Fig4_loss、Fig4_S21_head_rice_yield |
| 补图 1 | figs_1020/figS01_model_benchmark.py（后备 static_figs） | S01_model_skill |
| 补图 2 | 08 | S2_feature_distribution |
| 补图 3 | 07 | S3_phenology |
| 补图 4–9 | 06 | S4-S9_climate |
| 补图 10–12、14–16、18 | 09 | S10-S18_quality_change |
| 补图 13 | figs_1020/figS_hrr_regional.py | Fig2_S13_S17_checks/check_hrr_regional.csv |
| 补图 17 | figs_1020/figS_co2_interaction.py | Fig2_S13_S17_checks/check_co2_*.csv |
| 补图 19 | 20 | Fig2e_S19_uncertainty |
| 补图 20 | 19b | Fig2d_S20_shap |
| 补图 21 | 10 | Fig4_S21_head_rice_yield |
| 补图 22 | figS22_price_trends.py | S22_price_trend |
| 补图 23 | 36（两口径终版） | S23_valuation |
| 补图 24 | static_figs（流程示意图） | — |
| 补图 25 | static_figs（训练流程相关筛选） | — |
| 补表 1–2 | 训练输出 + sync_model_skill_1108.py | S01_model_skill |
| 补表 3–13 | 23–36 | 03_tables_si/sensitivity_econ |

## 未收集
逐成员预测栅格、逐年 NetCDF、1 km 面积栅格（体积大，见 export_tif_1108.py 的 GeoTIFF 存档）。
""", encoding="utf-8")

zp = shutil.make_archive(str(DEST), "zip", DEST.parent, DEST.name)
size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1024**2
print(f"\n共 {n} 个文件，{size:.1f} MB -> {DEST}\n打包：{zp}")
if missing:
    print("未找到（对应脚本还没跑或目录名不同）：\n  " + "\n  ".join(missing))
