# training (to be added)

The model-training workflow runs separately from this pipeline and writes the files that the pipeline reads:

| Output | Used by |
|---|---|
| `results/Models/{trait}_Blending.joblib` (inverse-RMSE-weighted VotingRegressor of the five best learners per trait) | `04_predict_rice_quality_1004.py`, `17_…`, `19_shap_analysis_1004.py` |
| `results/Tables/r2_matrix.csv`, `rmse_matrix.csv`, `mae_matrix.csv` (five-fold CV of 21 algorithms) | Supplementary Fig. 1a, Supplementary Table 1 |
| `results/Tables/Blending_predictions.csv` (observed and predicted values, training and test sets) | Supplementary Fig. 1b, Supplementary Table 2 |
| `results/Tables/Ensemble_CV_folds_training.csv`, `Test_bootstrap_CI95.csv` | Supplementary Table 2, Fig. 1 |

Place here: the predictor-screening script (pairwise Pearson correlation, |r| ≥ 0.7; Supplementary Fig. 25) and the
training script (80/20 random split with seed 42, five-fold CV on the training set, 21 candidate algorithms with fixed
hyperparameters, top-5 ensembles weighted by inverse CV RMSE). Software versions: Python 3.10.20, scikit-learn 1.7.2,
XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10, SHAP 0.49.1.
