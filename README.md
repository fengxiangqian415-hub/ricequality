# ricequality

Code and gridded data for

> Feng, X., Li, G., Deng, J., Jiang, S., Lu, J., Yang, Q., Shi, S., Zhang, Y., Chen, S., Wang, D., Zhao, C., Harrison, M. T., Zhou, M. & Liu, K.
> **Climate change downgrades China's rice harvest even where yields rise.** *Nature Food* (submitted).

The workflow trains ensemble models of seven rice grain-quality traits on 1,136 Chinese field and enrichment records,
projects the traits with 25 ISIMIP3b crop-model × climate-model members (baseline 1981–2010; 2041–2070 under SSP1-2.6 and
SSP5-8.5), combines head-rice rate, chalkiness degree and protein content into a composite index Q, classifies the existing
rice area into premium, moderate and common classes, and values head-rice production with grade-specific prices.

## Repository layout

| Path | Content |
|---|---|
| `code/` | Analysis pipeline (`01_…` to `36_…`), shared modules (`config_1004.py`, `q_common_1004.py`, …) and helper tools |
| `code/figs_1020/` | Figure scripts for Fig. 1, Fig. 2 and several Supplementary Figures, plus `assets/` (grain scans and plant outlines used in Fig. 1) |
| `code/static_figs/` | Supplementary Figs. 24 (analysis workflow, drawn separately) and 25 (predictor screening, from the training workflow), exported from the final Supplementary Information |
| `data/gridded/` | GeoTIFF deposit described in Supplementary Table 13 (written by `code/export_tif_1108.py`) |
| `data/README.md` | Data description, naming convention, format and licence |

## Figures and the scripts that make them

Supplementary items are numbered by first citation in the paper. `code/collect_figs_1108.py` gathers every figure under these names.

| Figure | Script (`code/`) | Exported as |
|---|---|---|
| Fig. 1 | `figs_1020/fig1_framework.py` (with `fig1_concept.py`, `fig1a_sampling_map.py`, `fig1_workflow.py`) | `Fig1_data_framework` |
| Fig. 2 | `figs_1020/fig2_main.py` (inputs from 05, 09, 19/19b and 20) | `Fig2_main` |
| Fig. 3 | `15_plot_fig4_grid_1004.py` (inputs from 11 and 13) | `Fig3_zones` |
| Fig. 4 | `16_plot_bivariate_loss_1004.py` (inputs from 10, 14 and 24b) | `Fig4_bivariate_loss` |
| Supplementary Fig. 1 | `figs_1020/figS01_model_benchmark.py` (training outputs; see `sync_model_skill_1108.py`) | `FigS01_model_benchmark` |
| Supplementary Fig. 2 | `08_plot_feature_distribution_1004.py` | `FigS02_feature_distribution` |
| Supplementary Fig. 3 | `07_plot_phenology_1004.py` | `FigS03_phenology` |
| Supplementary Figs. 4–9 | `06_plot_climate_change_1004.py` | `FigS04_Tavg` … `FigS09_RHavg` |
| Supplementary Figs. 10–12, 14–16, 18 | `09_plot_rice_quality_change_1004.py` | `FigS10_HRR`, `FigS11_BRR`, `FigS12_MRR`, `FigS14_CGR`, `FigS15_CD`, `FigS16_PC`, `FigS18_AC` |
| Supplementary Fig. 13 | `figs_1020/figS_hrr_regional.py` | `FigS13_hrr_regional` |
| Supplementary Fig. 17 | `figs_1020/figS_co2_interaction.py` | `FigS17_co2_interaction` |
| Supplementary Fig. 19 | `20_uncertainty_1004.py` | `FigS19_uncertainty` |
| Supplementary Fig. 20 | `19_shap_analysis_1004.py`, `19b_shap_summary_both_1004.py` | `FigS20_shap_summary` |
| Supplementary Fig. 21 | `10_calc_head_rice_yield_1004.py` | `FigS21_head_rice_yield` |
| Supplementary Fig. 22 | `figS22_price_trends.py` (with the two price CSV files) | `FigS22_price_trend` |
| Supplementary Fig. 23 | `36_yield_only_valuation_1106.py` | `FigS23_valuation` |
| Supplementary Fig. 24 | `static_figs/` (schematic) | `FigS24_workflow` |
| Supplementary Fig. 25 | `static_figs/` (training workflow) | `FigS25_correlation` |

| Supplementary Table | Source |
|---|---|
| 1–2 | Training outputs (`results/Tables`); checked and formatted by `sync_model_skill_1108.py` |
| 3–4 | `35_thresholds_from_limits_1031.py` (with `33_roc_at_threshold_1031.py`, `34_calibrate_thresholds_cv_1031.py`) |
| 5 | `31_annual_classification_1031.py` |
| 6 | `11_build_quality_zones_1004.py`, `12_diagnose_zones_1004.py` |
| 7 | `23_q_sensitivity_1004.py` |
| 8a–e | `14`, `24`, `24b`, `26`, `27`, `28` |
| 9 | Price series used by `figS22_price_trends.py`; prices in `config_1004.py` |
| 10 | `36_yield_only_valuation_1106.py` |
| 11 | `25_econ_qhigh_sensitivity_1004.py`, `28_econ_accounting_table_1004.py` |
| 12 | `32_quality_error_mc_1031.py` |
| 13 | `export_tif_1108.py` (this deposit) |

`21_export_paper_data_1004.py` writes every number quoted in the text and the source-data workbook.

## Licence

Code: MIT (see `LICENSE`). Data in `data/`: CC BY 4.0. Please cite the paper and the archived release.
