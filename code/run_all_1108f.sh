#!/usr/bin/env bash
# ============================================================
# run_all_1108f.sh —— full pipeline in dependency order (final version used for the paper)
#   bash run_all_1108f.sh            # everything
#   bash run_all_1108f.sh 9          # from step 9 on (steps are numbered as listed below)
#   bash run_all_1108f.sh 31 36      # only steps 31–36
#   Logs: logs_<TAG>_all/<script>.log ; stops at the first failure and prints the step to resume from.
#   Paths: edit config_1004.py or set RICEQ_BASE / RICEQ_MODEL_DIR / RICEQ_TRAIN_XLSX / RICEQ_CHINA_SHP / RICEQ_POLY_SHP / RICEQ_F_TIF.
#   Model training (results/Models, results/Tables) is a separate workflow (see training/README.md).
# ============================================================
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
START=${1:-0}
END=${2:-999}
TAG=$($PY -c "from config_1004 import TAG; print(TAG)")
LOG=logs_${TAG}_all; mkdir -p "$LOG"

# step-number | script (steps 02 and 03 run in parallel)
STEPS=(
  "1|01_make_baseline_support_1004.py"
  "2|02_ISIMIP_His_1004.py,03_ISIMIP_Fut_1004.py"
  "3|03b_make_analysis_domain_1004.py"
  "4|04_predict_rice_quality_1004.py"
  "5|05_decompose_373_1004.py"
  "6|06_plot_climate_change_1004.py"
  "7|07_plot_phenology_1004.py"
  "8|08_plot_feature_distribution_1004.py"
  "9|09_plot_rice_quality_change_1004.py"
  "10|10_calc_head_rice_yield_1004.py"
  "11|11_build_quality_zones_1004.py"
  "12|12_diagnose_zones_1004.py"
  "13|13_build_migration_stats_1004.py"
  "14|14_compute_production_loss_1004.py"
  "15|15_plot_fig4_grid_1004.py"
  "16|16_plot_bivariate_loss_1004.py"   # first pass: writes province_frac_1km.npz
  "17|24_econ_members_1004.py"
  "18|24b_province_members_1004.py"   # needs province_frac_1km.npz from the first pass of 16
  "19|16_plot_bivariate_loss_1004.py"   # second pass: Fig. 4 with member box plots
  "20|17_plot_co2_response_diagnosis_1004.py"
  "21|18_export_shap_features_1004.py"
  "22|19_shap_analysis_1004.py"
  "23|19b_shap_summary_both_1004.py"
  "24|20_uncertainty_1004.py"
  "25|23_q_sensitivity_1004.py"
  "26|25_econ_qhigh_sensitivity_1004.py"
  "27|26_province_drivers_1004.py"
  "28|27_province_attribution_1004.py"
  "29|28_econ_accounting_table_1004.py"
  "30|29_extrapolation_check_1004.py"
  "31|31_annual_classification_1031.py"
  "32|32_quality_error_mc_1031.py"
  "33|33_roc_at_threshold_1031.py"
  "34|34_calibrate_thresholds_cv_1031.py"
  "35|35_thresholds_from_limits_1031.py"
  "36|36_yield_only_valuation_1106.py"
  "37|figs_1020/check_co2_response.py"
  "38|figs_1020/check_hrr_regional.py"
  "39|figs_1020/figS_hrr_regional.py"
  "40|figs_1020/figS_co2_interaction.py"
  "41|figs_1020/figS_hrr_response.py"
  "42|figs_1020/fig1a_sampling_map.py"
  "43|figs_1020/fig1_framework.py"
  "44|figs_1020/fig2_main.py"
  "45|figs_1020/figS01_model_benchmark.py"
  "46|figS22_price_trends.py"
  "47|30_consistency_audit_1004.py"
  "48|21_export_paper_data_1004.py"
  "49|collect_figs_1108.py"
  "50|collect_data_1108.py"
  "51|export_tif_1108.py"
)

run_one () {
  local f=$1 t0=$(date +%s)
  echo "[$(date '+%F %T')] > $f"
  if $PY "$f" > "$LOG/$(basename "${f%.py}").log" 2>&1; then
    echo "[$(date '+%F %T')] ok $f ($(( $(date +%s) - t0 )) s)"
  else
    echo "[$(date '+%F %T')] FAILED $f — last lines:"; tail -n 20 "$LOG/$(basename "${f%.py}").log" | sed 's/^/    /'
    return 1
  fi
}

$PY config_1004.py || exit 1
for item in "${STEPS[@]}"; do
  n=${item%%|*}; files=${item#*|}
  (( n < START || n > END )) && continue
  if [[ $files == *","* ]]; then
    IFS=',' read -ra par <<< "$files"; pids=()
    for f in "${par[@]}"; do run_one "$f" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || { echo ">>> resume with: bash run_all_1108f.sh $n"; exit 1; }; done
  else
    run_one "$files" || { echo ">>> resume with: bash run_all_1108f.sh $n"; exit 1; }
  fi
done
echo "All steps finished. Logs in $LOG/"
