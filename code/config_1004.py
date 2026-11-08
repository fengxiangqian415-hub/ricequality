import os
from pathlib import Path

TAG = "1004b"
HIS = (1981, 2010)
FUT = (2041, 2070)
SSPS = ["ssp126", "ssp585"]
RI_TYPES = ["ri1", "ri2"]
CROP_MODELS = ["pDSSAT", "CYGMA1p74", "ISAM", "LDNDC", "LPJ-GUESS"]
GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]

HIS_P, FUT_P = f"{HIS[0]}_{HIS[1]}", f"{FUT[0]}_{FUT[1]}"
HIS_LABEL, FUT_LABEL = f"{HIS[0]}–{HIS[1]}", f"{FUT[0]}–{FUT[1]}"
HIS_YEARS = list(range(HIS[0], HIS[1] + 1))
FUT_YEARS = list(range(FUT[0], FUT[1] + 1))

SUPPORT_MIN_YEARS = int(0.85 * len(HIS_YEARS))

Q_THRESH_MODE  = "fixed"
Q_LOW, Q_HIGH  = 0.60, 0.66
Q_PCT          = (1 / 3, 2 / 3)
Q_POOL_SEASONS = True


def thresh_tag():
    if Q_THRESH_MODE == "fixed":
        return "" if (Q_LOW, Q_HIGH) == (0.63, 0.68) else f"_fix{int(round(Q_LOW * 100))}{int(round(Q_HIGH * 100))}"
    tag = f"_q{int(round(Q_PCT[0] * 100))}" if abs(Q_PCT[0] + Q_PCT[1] - 1) < 1e-6 else \
          f"_q{int(round(Q_PCT[0] * 100))}-{int(round(Q_PCT[1] * 100))}"
    return tag + ("" if Q_POOL_SEASONS else "s")

PRED_MODE   = "annual"
CO2_VARIANT = "withco2"

SUMMARY_WEIGHT  = "area"
DIAG_WEIGHT     = "cell"
CO2_CF_BASELINE = "withco2"

ANCHORS     = {"HRR": (23.38, 73.77), "CD": (0.31, 44.59), "PC": (4.50, 12.18)}
ANCHOR_SIGN = {"HRR": +1, "CD": -1, "PC": -1}
P_ORD, P_HQ = 5.23, 5.96
CNY_PER_USD = 7.0

BASE      = Path(os.environ.get("RICEQ_BASE", "/home/debian/pythonfile/ISIMIP3b"))
OUTPUT    = BASE / "output"
FIGURES   = OUTPUT / "figures"
PHENO_DIR = BASE / "phenology_corrected"
HIS_ANN   = PHENO_DIR / "historical_annual"
PHENO_FUT = PHENO_DIR / "future"
AREA_DIR  = OUTPUT / "area_weights"
CO2_DIR   = BASE / "co2"
YIELD_DIR = BASE / "yield"
MASK_DIR  = BASE / "masks"

MODEL_DIR = Path(os.environ.get("RICEQ_MODEL_DIR", "/home/debian/pythonfile/results/Models"))
F_TIF     = Path(os.environ.get("RICEQ_F_TIF", "/home/debian/pythonfile/china/f_indica_fraction_0p5deg.tif"))
FACE_XLSX = Path(os.environ.get("RICEQ_TRAIN_XLSX", "/home/debian/pythonfile/FACE_V1002.xlsx"))
CHINA_SHP = Path(os.environ.get("RICEQ_CHINA_SHP", "/home/debian/pythonfile/china/china.shp"))
POLY_SHP  = Path(os.environ.get("RICEQ_POLY_SHP", "/home/debian/pythonfile/china/Polygon.shp"))

CO2_FILES = {"historical": CO2_DIR / "co2_historical_annual_1850_2014.txt",
             "ssp126":     CO2_DIR / "co2_ssp126_annual_2015_2100.txt",
             "ssp585":     CO2_DIR / "co2_ssp585_annual_2015_2100.txt"}

CLIM_HIS = OUTPUT / f"historical_{HIS_P}_v4"
CLIM_FUT = OUTPUT / f"future_{FUT_P}"

PRED_ROOT = OUTPUT / "rice_quality_prediction" / f"v5_{TAG}"


def pred_dir(mode=None, variant=None):
    """品质预测目录：withco2 → PRED_ROOT/mode；其他方案 → PRED_ROOT/variant/mode"""
    mode = mode or PRED_MODE
    variant = variant or CO2_VARIANT
    return PRED_ROOT / mode if variant == "withco2" else PRED_ROOT / variant / mode


def _suffix():
    return TAG if CO2_VARIANT == "withco2" else f"{TAG}_{CO2_VARIANT}"


def out_dir(name, thresh=True):
    """下游结果目录：output/{name}_{TAG}[_{CO2_VARIANT}][阈值后缀]；与阈值无关的结果（如整精米产量）用 thresh=False"""
    return OUTPUT / f"{name}_{_suffix()}{thresh_tag() if thresh else ''}"


def fig_dir(name, thresh=True):
    return FIGURES / f"{name}_{_suffix()}{thresh_tag() if thresh else ''}"


def thresholds_file(mode=None):
    """11 写出的实际阈值（12、作图等读取）"""
    return out_dir("quality_zones") / (mode or PRED_MODE) / "thresholds.json"


def raw_support_file(ri):
    """01 生成的基准支撑区（有效物候年数 ≥ 门槛）；只有 01、03b 直接使用"""
    return HIS_ANN / f"{ri}_baseline_support_{HIS_P}.tif"


def domain_file(ri):
    return OUTPUT / f"analysis_domain_{TAG}" / f"{ri}_analysis_domain.tif"


def support_file(ri):
    """★ 全流程唯一格网域 D(ri)（03b 生成）。文件不存在时直接报错，避免各脚本静默退回不同口径"""
    p = domain_file(ri)
    if not p.exists():
        raise FileNotFoundError(f"缺少统一格网域 {p}；请先运行 03b_make_analysis_domain_1004.py")
    return p


def nyears_file(ri):
    return HIS_ANN / f"{ri}_baseline_nyears_{HIS_P}.tif"


def area_file(ri):
    return AREA_DIR / f"area_weight_{ri}_2000_2010.tif"


def hist_pheno_tag(yr):
    """唯一历史物候：2000 年前为 25 成员平均校正，2000 年起为观测"""
    return "ens25" if yr < 2000 else "observed"


def co2_series(scen, years):
    import pandas as pd
    d = pd.read_csv(CO2_FILES[scen], sep=r"\s+", header=None, names=["year", "ppm"])
    d = d.set_index(d.year.astype(int)).ppm
    return [float(d[y]) for y in years]


def co2_mean(scen, years):
    import numpy as np
    return float(np.mean(co2_series(scen, years)))


if __name__ == "__main__":
    print(f"TAG={TAG}  基准 {HIS_LABEL}  未来 {FUT_LABEL}  支撑区门槛 ≥{SUPPORT_MIN_YEARS}/{len(HIS_YEARS)} 年")
    print(f"预测根目录：{PRED_ROOT}")
    print(f"下游读取：{pred_dir()}  （{PRED_MODE}, {CO2_VARIANT}）")
    print(f"示例下游目录：{out_dir('quality_zones')}")
    print(f"统一格网域：{domain_file('ri1').parent}  （{'已生成' if domain_file('ri1').exists() else '未生成，先运行 03b'}）"
          f"；汇总权重 SUMMARY_WEIGHT={SUMMARY_WEIGHT}，诊断量 DIAG_WEIGHT={DIAG_WEIGHT}")
    print(f"固定 CO2 反事实参照基准：{CO2_CF_BASELINE}；价格 {P_ORD}/{P_HQ} 元/kg；汇率 {CNY_PER_USD}")
    print(f"分区阈值：{Q_THRESH_MODE}  " + (f"Q_LOW={Q_LOW} Q_HIGH={Q_HIGH}" if Q_THRESH_MODE == "fixed"
          else f"基准分位数 {Q_PCT[0]:.3f}/{Q_PCT[1]:.3f}  两季合并={Q_POOL_SEASONS}") + f"  目录后缀='{thresh_tag()}'")
