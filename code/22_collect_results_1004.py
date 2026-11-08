import shutil
from pathlib import Path
from config_1004 import OUTPUT, FIGURES, TAG, PRED_MODE, thresh_tag, out_dir, fig_dir

DEST = OUTPUT / f"RESULTS_{TAG}{thresh_tag()}"
if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)

TABLE_EXT = {".csv", ".xlsx", ".json", ".txt"}
FIG_EXT   = {".png", ".pdf", ".svg", ".tiff", ".tif"}

SOURCES = [
    (out_dir("excel"),                                  "00_paper_data",           TABLE_EXT, False),
    (FIGURES / f"climate_change_{TAG}",                 "06_climate_change",       FIG_EXT | TABLE_EXT, False),
    (FIGURES / f"phenology_{TAG}",                      "07_phenology",            FIG_EXT | TABLE_EXT, False),
    (FIGURES / f"feature_distribution_{TAG}",           "08_feature_distribution", FIG_EXT | TABLE_EXT, False),
    (FIGURES / f"rice_quality_{TAG}_{PRED_MODE}",       "09_quality_change",       FIG_EXT | TABLE_EXT, False),
    (out_dir("head_rice_yield", thresh=False),          "10_head_rice_yield",      {".tif"}, False),
    (fig_dir("head_rice_yield", thresh=False),          "10_head_rice_yield",      FIG_EXT, False),
    (out_dir("quality_zones") / PRED_MODE,              "11_quality_zones",        {".tif"} | TABLE_EXT, False),
    (out_dir("quality_zones") / "diagnosis",            "12_zone_diagnosis",       TABLE_EXT, False),
    (out_dir("migration_stats") / PRED_MODE,            "13_migration",            TABLE_EXT, False),
    (out_dir("loss"),                                   "14_16_economic_loss",     FIG_EXT | TABLE_EXT, False),
    (fig_dir("fig4_migration"),                         "15_fig4",                 FIG_EXT, False),
    (FIGURES / f"co2_diagnosis_{TAG}",                  "17_co2_diagnosis",        FIG_EXT, False),
    (fig_dir("shap", thresh=False),                     "19_shap",                 FIG_EXT | TABLE_EXT, False),
    (fig_dir("uncertainty", thresh=False),              "20_uncertainty",          FIG_EXT | TABLE_EXT, False),
    (OUTPUT / f"decomposition_373_{TAG}",               "05_decomposition_373",    TABLE_EXT, False),
    (out_dir("sensitivity"),                            "23_24_sensitivity_econ",  TABLE_EXT, False),
    (OUTPUT / f"analysis_domain_{TAG}",                 "03b_analysis_domain",     {".tif"} | TABLE_EXT, False),
    (OUTPUT / f"consistency_audit_{TAG}",               "30_consistency_audit",    TABLE_EXT, False),
]

n_tot, missing = 0, []
for src, sub, exts, rec in SOURCES:
    if not src.exists():
        missing.append(str(src)); continue
    files = [f for f in (src.rglob("*") if rec else src.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    d = DEST / sub; d.mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(f, d / f.name)
    n_tot += len(files)
    print(f"  {sub:26s} ← {len(files):4d} 个文件  ({src.name})")

shutil.copy2(Path(__file__).with_name("config_1004.py"), DEST / "config_1004_used.py")
_stale = DEST / "08_feature_distribution" / "extrapolation_check.csv"
if _stale.exists():
    _stale.unlink(); print("  已排除旧口径文件 extrapolation_check.csv（由 29 号表 S10 取代）")
zp = shutil.make_archive(str(DEST), "zip", DEST.parent, DEST.name)
size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1024**2
print(f"\n共 {n_tot} 个文件，{size:.1f} MB → {DEST}\n打包：{zp}")
if missing:
    print("未找到（对应步骤可能未运行）：\n  " + "\n  ".join(missing))
