import numpy as np
import xarray as xr

from config_1004 import RI_TYPES, GCMS, CROP_MODELS, SSPS, FUT, FUT_YEARS, FUT_P, HIS_YEARS, CLIM_FUT
from pheno_common_1004 import OUT_HIS, OUT_FUT, read_tif, save_tif, load_mask
from clim_common_1004 import VAR_AGG, VAR_LABELS, EXTRA_LABELS, load_daily, extract_years, \
    save_annual_nc, period_mean, clean_window as clean, baseline_window

Y0, Y1   = FUT
YEARS    = FUT_YEARS
PERIOD   = FUT_P
BASE_YEARS = HIS_YEARS
OUT_DIR  = CLIM_FUT
ANN_DIR  = OUT_DIR / "annual"
FIX_DIR  = OUT_DIR / "fixed_window"
FORCE    = False


def member_pheno(model, gcm, ssp, ri, mask):
    ph = {}
    d = OUT_FUT / model / gcm / ssp
    for yr in YEARS:
        fh = d / f"{model}_{gcm}_{ssp}_{ri}_HE_{yr}_corrected.tif"
        fm = d / f"{model}_{gcm}_{ssp}_{ri}_MA_{yr}_corrected.tif"
        if fh.exists() and fm.exists():
            ph[yr] = clean(read_tif(fh), read_tif(fm), mask)
    return ph


def save_means(res, out_dir, prefix, ens=None):
    for lab, arr in res.items():
        m, n = period_mean(arr)
        save_tif(m, out_dir / f"{prefix}_{lab}_{PERIOD}_mean.tif")
        if ens is not None:
            ens.setdefault(lab, []).append(m)
        if lab == VAR_LABELS["tas"]:
            save_tif(n, out_dir / f"{prefix}_nyears_{PERIOD}.tif")


def load_nc(fp):
    with xr.open_dataset(fp) as ds:
        return {k: ds[k].values for k in ds.data_vars}


for ssp in SSPS:
    for ri in RI_TYPES:
        print(f"\n{'=' * 60}\n[{ssp} / {ri}] 未来 {PERIOD}")
        mask = load_mask(ri)
        he_b, ma_b = baseline_window(ri, mask, BASE_YEARS)
        print(f"  基准窗口：有效格网 {int(np.isfinite(he_b).sum())}  "
              f"HE 均值 {np.nanmean(he_b):.1f}  MA 均值 {np.nanmean(ma_b):.1f}")
        ens_B, ens_C = {}, {}

        for gcm in GCMS:
            fp_C = FIX_DIR / "annual" / f"{gcm}_{ssp}_{ri}_{PERIOD}_fixedwindow_annual.nc"
            todo = [m for m in CROP_MODELS if FORCE or not
                    (ANN_DIR / ssp / m / gcm / f"{m}_{gcm}_{ssp}_{ri}_{PERIOD}_annual.nc").exists()]
            need_C = FORCE or not fp_C.exists()

            daily = None
            if todo or need_C:
                daily = {}
                for var in VAR_AGG:
                    vals, times = load_daily(gcm, ssp, var, Y0, Y1)
                    daily[var] = vals

            if need_C:
                resC = extract_years(daily, times, YEARS, lambda yr: (he_b, ma_b))
                save_annual_nc(resC, YEARS, fp_C, attrs={"gcm": gcm, "ssp": ssp, "ri": ri,
                               "period": PERIOD, "window": f"baseline {BASE_YEARS[0]}-{BASE_YEARS[-1]} mean HE-MA"})
            else:
                resC = load_nc(fp_C)
            save_means(resC, FIX_DIR / gcm, f"{gcm}_{ssp}_{ri}_fixedwindow", ens_C)

            for model in CROP_MODELS:
                fp_B = ANN_DIR / ssp / model / gcm / f"{model}_{gcm}_{ssp}_{ri}_{PERIOD}_annual.nc"
                if model in todo:
                    ph = member_pheno(model, gcm, ssp, ri, mask)
                    if len(ph) < len(YEARS):
                        print(f"  ⚠️ {model}/{gcm}: 物候年份 {len(ph)}/{len(YEARS)}")
                    resB = extract_years(daily, times, YEARS, lambda yr: ph.get(yr))
                    save_annual_nc(resB, YEARS, fp_B, attrs={"model": model, "gcm": gcm, "ssp": ssp,
                                   "ri": ri, "period": PERIOD, "window": "member bias-corrected HE-MA"})
                else:
                    resB = load_nc(fp_B)
                save_means(resB, OUT_DIR / ssp / model / gcm, f"{model}_{gcm}_{ssp}_{ri}", ens_B)
                tB = np.nanmean(period_mean(resB[VAR_LABELS["tas"]])[0])
                dB = np.nanmean(period_mean(resB[EXTRA_LABELS["days"]])[0])
                print(f"  [{model}/{gcm}] 均温 {tB:.2f} °C  天数 {dB:.1f}"
                      + ("" if model in todo else "  (skip, 已有)"))
            tC = np.nanmean(period_mean(resC[VAR_LABELS["tas"]])[0])
            print(f"  [fixed window/{gcm}] 均温 {tC:.2f} °C")
            del daily

        for lab, lst in ens_B.items():
            save_tif(np.nanmean(np.stack(lst), axis=0),
                     OUT_DIR / "ensemble" / f"ensemble_{ssp}_{ri}_{lab}_{PERIOD}_mean.tif")
        for lab, lst in ens_C.items():
            save_tif(np.nanmean(np.stack(lst), axis=0),
                     FIX_DIR / "ensemble" / f"ensemble_{ssp}_{ri}_fixedwindow_{lab}_{PERIOD}_mean.tif")
        print(f"  [saved] ensemble：B {len(ens_B[VAR_LABELS['tas']])} 成员，C {len(ens_C[VAR_LABELS['tas']])} GCM")

print(f"\n完成。输出: {OUT_DIR}")
