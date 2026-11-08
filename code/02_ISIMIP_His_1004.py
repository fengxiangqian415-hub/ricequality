import numpy as np
import xarray as xr

from config_1004 import RI_TYPES, GCMS, HIS, HIS_YEARS, HIS_P, CLIM_HIS, hist_pheno_tag
from pheno_common_1004 import OUT_HIS, read_tif, save_tif, load_mask
from clim_common_1004 import VAR_AGG, VAR_LABELS, EXTRA_LABELS, load_daily, extract_years, \
    save_annual_nc, period_mean, baseline_window

Y0, Y1   = HIS
YEARS    = HIS_YEARS
PERIOD   = HIS_P
OUT_DIR  = CLIM_HIS
ANN_DIR  = OUT_DIR / "annual"
FIX_DIR  = OUT_DIR / "fixed_window"
FORCE    = False


def load_hist_pheno(ri, mask):
    """唯一历史物候：{yr: (he, ma)}，套 mask，剔除 HE>=MA"""
    ph = {}
    for yr in YEARS:
        tag = hist_pheno_tag(yr)
        fh, fm = OUT_HIS / f"{ri}_HE_{yr}_{tag}.tif", OUT_HIS / f"{ri}_MA_{yr}_{tag}.tif"
        if not (fh.exists() and fm.exists()):
            print(f"  ⚠️ 缺物候 {ri} {yr} ({tag})")
            continue
        he, ma = read_tif(fh), read_tif(fm)
        bad = ~mask | ~np.isfinite(he) | ~np.isfinite(ma) | (he >= ma)
        he[bad] = np.nan
        ma[bad] = np.nan
        ph[yr] = (he, ma)
    return ph


for ri in RI_TYPES:
    print(f"\n{'=' * 60}\n[{ri}] 历史 {PERIOD}")
    mask = load_mask(ri)
    pheno = load_hist_pheno(ri, mask)
    print(f"  物候年份 {len(pheno)}/{len(YEARS)}")
    he_b, ma_b = baseline_window(ri, mask, YEARS)

    ens, ens_fix = {}, {}
    for gcm in GCMS:
        fp_ann = ANN_DIR / f"{gcm}_historical_{ri}_{PERIOD}_annual.nc"
        fp_fix = FIX_DIR / "annual" / f"{gcm}_historical_{ri}_{PERIOD}_fixedwindow_annual.nc"
        need_A, need_F = FORCE or not fp_ann.exists(), FORCE or not fp_fix.exists()
        daily = None
        if need_A or need_F:
            daily = {}
            for var in VAR_AGG:
                vals, times = load_daily(gcm, "historical", var, Y0, Y1)
                daily[var] = vals
        if need_A:
            res = extract_years(daily, times, YEARS, lambda yr: pheno.get(yr))
            save_annual_nc(res, YEARS, fp_ann,
                           attrs={"gcm": gcm, "ri": ri, "period": PERIOD, "scenario": "historical",
                                  "phenology": "pre-2000 ens25 bias-corrected mean; 2000 onward observed"})
            print(f"  [saved] {fp_ann.name}")
        else:
            print(f"  [skip] {fp_ann.name}")
            with xr.open_dataset(fp_ann) as ds:
                res = {k: ds[k].values for k in ds.data_vars}
        if need_F:
            res_fix = extract_years(daily, times, YEARS, lambda yr: (he_b, ma_b))
            save_annual_nc(res_fix, YEARS, fp_fix,
                           attrs={"gcm": gcm, "ri": ri, "period": PERIOD, "scenario": "historical",
                                  "window": f"baseline {Y0}-{Y1} mean HE-MA"})
        else:
            with xr.open_dataset(fp_fix) as ds:
                res_fix = {k: ds[k].values for k in ds.data_vars}
        del daily
        for lab, arr in res_fix.items():
            m, _ = period_mean(arr)
            save_tif(m, FIX_DIR / gcm / f"{gcm}_historical_{ri}_fixedwindow_{lab}_{PERIOD}_mean.tif")
            ens_fix.setdefault(lab, []).append(m)

        out_dir = OUT_DIR / gcm
        for lab, arr in res.items():
            m, n = period_mean(arr)
            save_tif(m, out_dir / f"{gcm}_historical_{ri}_{lab}_{PERIOD}_mean.tif")
            ens.setdefault(lab, []).append(m)
            if lab == VAR_LABELS["tas"]:
                save_tif(n, out_dir / f"{gcm}_historical_{ri}_nyears_{PERIOD}.tif")
        t = np.nanmean(period_mean(res[VAR_LABELS["tas"]])[0])
        d = np.nanmean(period_mean(res[EXTRA_LABELS["days"]])[0])
        print(f"  [{gcm}] 灌浆期均温 {t:.2f} °C  灌浆天数 {d:.1f}  "
              f"有效格网 {int(np.isfinite(period_mean(res[VAR_LABELS['tas']])[0]).sum())}")

    for lab, lst in ens.items():
        save_tif(np.nanmean(np.stack(lst), axis=0),
                 OUT_DIR / "ensemble" / f"ensemble_historical_{ri}_{lab}_{PERIOD}_mean.tif")
    for lab, lst in ens_fix.items():
        save_tif(np.nanmean(np.stack(lst), axis=0),
                 FIX_DIR / "ensemble" / f"ensemble_historical_{ri}_fixedwindow_{lab}_{PERIOD}_mean.tif")
    print(f"  [saved] ensemble（{len(GCMS)} GCM 平均）")

print(f"\n完成。输出: {OUT_DIR}")
