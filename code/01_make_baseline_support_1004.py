# ============================================================
# 01_make_baseline_support_1004.py —— 基准期物候有效年数与共同支撑区
#   基准期与门槛来自 config_1004（HIS、SUPPORT_MIN_YEARS）
#   输入：phenology_corrected/historical_annual/{ri}_{HE|MA}_{yr}_{ens25|observed}.tif
#         masks/{ri}_mask_0.5deg.tif；output/area_weights/area_weight_{ri}_2000_2010.tif
#   输出：{ri}_baseline_nyears_{HIS_P}.tif、{ri}_baseline_support_{HIS_P}.tif（≥门槛）
#   [1013] 这里写的是原始支撑区（raw_support_file）；全流程统一格网域由 03b 在此基础上生成
# ============================================================
import numpy as np
from config_1004 import (RI_TYPES, HIS_YEARS, HIS_LABEL, SUPPORT_MIN_YEARS, HIS_ANN,
                         raw_support_file, nyears_file, area_file, hist_pheno_tag)
from pheno_common_1004 import read_tif, save_tif, load_mask

N = len(HIS_YEARS)
for ri in RI_TYPES:
    mask = load_mask(ri)
    n = np.zeros(mask.shape, int)
    for yr in HIS_YEARS:
        tag = hist_pheno_tag(yr)
        he = read_tif(HIS_ANN / f"{ri}_HE_{yr}_{tag}.tif")
        ma = read_tif(HIS_ANN / f"{ri}_MA_{yr}_{tag}.tif")
        n[np.isfinite(he) & np.isfinite(ma) & (he < ma) & mask] += 1
    sup = mask & (n >= SUPPORT_MIN_YEARS)
    save_tif(np.where(mask, n, np.nan).astype(np.float32), nyears_file(ri))
    save_tif(np.where(sup, 1.0, np.nan), raw_support_file(ri))

    area = read_tif(area_file(ri))
    tot = np.nansum(area[mask])
    print(f"\n[{ri}] 基准 {HIS_LABEL}（{N} 年） mask 格网 {mask.sum()}  面积 {tot / 1e4:.2f} Mha")
    for thr in sorted({N, SUPPORT_MIN_YEARS, int(0.67 * N), 1}, reverse=True):
        sel = mask & (n >= thr)
        print(f"  ≥{thr:>2}/{N} 年: 格网 {sel.sum():>5} ({sel.sum() / mask.sum():.1%})  "
              f"面积 {np.nansum(area[sel]) / 1e4:.2f} Mha ({np.nansum(area[sel]) / tot:.1%})")
    print(f"  [saved] {raw_support_file(ri).name}（≥{SUPPORT_MIN_YEARS}/{N}：{sup.sum()} 格网）")
