# ============================================================
# config_1004.py —— 全流程统一配置（所有 *_1004.py 脚本都从这里读取）
#   改时段 / 改输出后缀 / 切换 CO2 方案 / 改分区阈值，只改这一个文件
# ============================================================
import os
from pathlib import Path

# ---------------- 版本与时段 ----------------
TAG = "1004b"                               # 所有输出目录后缀（1004 = 1991–2010/2041–2060；1004b = 1981–2010/2041–2070）
HIS = (1981, 2010)                          # 基准期
FUT = (2041, 2070)                          # 未来期
SSPS = ["ssp126", "ssp585"]
RI_TYPES = ["ri1", "ri2"]
CROP_MODELS = ["pDSSAT", "CYGMA1p74", "ISAM", "LDNDC", "LPJ-GUESS"]
GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]

HIS_P, FUT_P = f"{HIS[0]}_{HIS[1]}", f"{FUT[0]}_{FUT[1]}"          # 文件名用
HIS_LABEL, FUT_LABEL = f"{HIS[0]}–{HIS[1]}", f"{FUT[0]}–{FUT[1]}"  # 图上用
HIS_YEARS = list(range(HIS[0], HIS[1] + 1))
FUT_YEARS = list(range(FUT[0], FUT[1] + 1))

# 基准支撑区：有效物候年数 ≥ 85%（向下取整：20 年 → 17 年；30 年 → 25 年）
SUPPORT_MIN_YEARS = int(0.85 * len(HIS_YEARS))

# 品质分区阈值（Q 合意度；普通 < Q_LOW ≤ 中等 < Q_HIGH ≤ 优质）
#   Q_THRESH_MODE = "fixed"        → 用下面的固定值 Q_LOW / Q_HIGH（原方案，输出目录不加后缀）
#   Q_THRESH_MODE = "baseline_pct" → 用基准期 Q 的面积加权分位数 Q_PCT 定阈值，再原样套到未来
#                                    Q_POOL_SEASONS=True：两季合并算一套阈值（"一把尺子"，保留早晚稻品质差异）
#                                    输出目录自动加后缀，如 _q33（三分位）、_q25（25/75 分位）
Q_THRESH_MODE  = "fixed"             # ★ 1004b 最终采用固定阈值
Q_LOW, Q_HIGH  = 0.60, 0.66          # ★ [1031] 国标三级限值换算（35 号）：恰在限值上的籽粒（HRR、CD 取三级限值，PC = 9.0）的 Q，
                                     #   籼（中粒）粳平均：GB/T 17891-2017 → 0.600，NY/T 593-2021 → 0.655（取两位 0.66）；与锚点同一把尺子
Q_PCT          = (1 / 3, 2 / 3)      # baseline_pct 模式使用：(普通/中等分界, 中等/优质分界)
Q_POOL_SEASONS = True


def thresh_tag():
    if Q_THRESH_MODE == "fixed":
        return "" if (Q_LOW, Q_HIGH) == (0.63, 0.68) else f"_fix{int(round(Q_LOW * 100))}{int(round(Q_HIGH * 100))}"
    tag = f"_q{int(round(Q_PCT[0] * 100))}" if abs(Q_PCT[0] + Q_PCT[1] - 1) < 1e-6 else \
          f"_q{int(round(Q_PCT[0] * 100))}-{int(round(Q_PCT[1] * 100))}"
    return tag + ("" if Q_POOL_SEASONS else "s")

# ---------------- 预测方案 ----------------
PRED_MODE   = "annual"      # 主结果：逐年预测再平均；对照 "meanclim"
CO2_VARIANT = "withco2"     # 下游读取的预测："withco2"（CO2 随情景）| "fixedco2"（CO2 固定为基准均值）

# ============================================================
# ★ 全流程唯一统计口径（1013 定稿；所有脚本遵守，改这里即全局生效）
#   1) 格网域 D(ri) = 基准支撑区（01）∩ 分季面积 > 0（面积来自 1 km 种植像元聚合到 0.5°）
#      由 03b 写出 analysis_domain 文件；support_file(ri) 返回它，所有脚本都经由它取域
#      （01 生成的原始支撑区用 raw_support_file，只有 01、03b 使用）；成员缺值格网不另处理
#   2) 中心值 = 集合平均（每个成员逐年预测 → 时段均值 → 25 个成员平均）；
#      成员范围 = 25 个成员各自同法计算 → 中位数 [最小, 最大]（只作离散度，放补充材料）
#   3) 全国/全区汇总（气候、物候、品质性状、Q）= D 内按分季面积加权（SUMMARY_WEIGHT）
#      变化 = 未来均值 − 基准均值；相对变化 = 变化 / 基准均值 × 100；各情景同一组格网，基准唯一
#   4) 模型诊断量（SHAP、方差分解、外推比例 S10、输入分布 S18）按格网（或格网×年×成员）等权（DIAG_WEIGHT）
#   5) 面积量（经济价值与损失、整精米总产、优质与各等级面积占比、迁移面积）按 1 km 聚合面积求和
#   6) CO2 反事实（CO2_CF_BASELINE = "withco2"）：基准只有一个 = 历史逐年预测（当年实际 CO2）的均值；
#        固定 CO2 只用于未来（CO2 恒为 1981–2010 均值）；气候部分 = F(固定 CO2) − B；CO2 部分 = F(随情景 CO2) − F(固定 CO2)
#   7) 灌浆天数沿用原定义（07：平均 MA − 平均 HE；05/06/26：窗口日数 gf_days），两者实测相差 ≤ 0.07 d
# ============================================================
SUMMARY_WEIGHT  = "area"      # 描述性汇总的权重："area" = 面积加权（定稿）| "cell" = 逐格等权
DIAG_WEIGHT     = "cell"      # 模型诊断量（SHAP、方差分解、S10、S18）的权重："cell"（定稿）| "area"
CO2_CF_BASELINE = "withco2"   # 固定 CO2 反事实的基准："withco2"（定稿：唯一基准 = 历史逐年实际 CO2）| "own"（固定 CO2 另算历史基准）

# 品质指数锚点（训练集 p1, p99）与价格、汇率 —— 11/14/16/24/25/27/28 与 q_common 均从这里读取
ANCHORS     = {"HRR": (23.38, 73.77), "CD": (0.31, 44.59), "PC": (4.50, 12.18)}   # 训练集 p1/p99（定稿保留）
#   [1031] 原 Table S6 的 ROC 阈值（0.62/0.72）是在 p5/p95 锚点、实测值上算的，与分区所用尺子不一致；
#          现阈值改由国标三级限值在本锚点下直接换算（35 号），ROC 只用于说明 Q 的判别力（33/34/35 号）
ANCHOR_SIGN = {"HRR": +1, "CD": -1, "PC": -1}          # +1 越高越好；−1 越低越好
P_ORD, P_HQ = 5.23, 5.96                              # 元/kg：中等+普通 / 优质
CNY_PER_USD = 7.0

# ---------------- 路径 ----------------
# [1108f] 路径可用环境变量覆盖（GitHub 版复现用）；不设环境变量时与服务器原路径完全一致
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

MODEL_DIR = Path(os.environ.get("RICEQ_MODEL_DIR", "/home/debian/pythonfile/results/Models"))                 # 品质模型（含 TSR）
F_TIF     = Path(os.environ.get("RICEQ_F_TIF", "/home/debian/pythonfile/china/f_indica_fraction_0p5deg.tif"))
FACE_XLSX = Path(os.environ.get("RICEQ_TRAIN_XLSX", "/home/debian/pythonfile/FACE_V1002.xlsx"))                # 训练数据（确认路径）
CHINA_SHP = Path(os.environ.get("RICEQ_CHINA_SHP", "/home/debian/pythonfile/china/china.shp"))
POLY_SHP  = Path(os.environ.get("RICEQ_POLY_SHP", "/home/debian/pythonfile/china/Polygon.shp"))

CO2_FILES = {"historical": CO2_DIR / "co2_historical_annual_1850_2014.txt",
             "ssp126":     CO2_DIR / "co2_ssp126_annual_2015_2100.txt",
             "ssp585":     CO2_DIR / "co2_ssp585_annual_2015_2100.txt"}

# 气候提取输出（按时段命名，不同时段互不覆盖）
CLIM_HIS = OUTPUT / f"historical_{HIS_P}_v4"
CLIM_FUT = OUTPUT / f"future_{FUT_P}"

# 品质预测输出
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
