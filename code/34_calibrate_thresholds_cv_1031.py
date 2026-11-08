# ============================================================
# 34_calibrate_thresholds_cv_1031.py —— 在"模型预测值"的尺子上校准 Q 分级阈值（证据链补全）
#   为什么：Table S6 的 ROC 用的是 1,136 条实测值的 Q；格网上分级的却是模型预测（再经 25 成员 × 30 年平均），
#          预测值的离散比实测小得多，同一阈值对应的品质水平不同。阈值应在与格网相同的量（预测值）上校准。
#   做法：对 HRR、CD、PC 的最终集成模型（MODEL_DIR/{t}_Blending.joblib，VotingRegressor）做 clone，
#         5 折交叉验证（KFold, shuffle, seed 42）得到全部记录的样本外预测 → Q_pred（config.ANCHORS 锚点）；
#         标签仍按实测值是否满足两套国标三级限值（Table S5 规则，PC ≤ pc_cap）；
#         ROC：AUC、Youden 最优阈值及 1,000 次 bootstrap 区间、J 平台范围（J ≥ Jmax − 0.01）；
#         同时给出实测 Q 的 ROC（应复现 Table S6 的口径）和两种 Q 的分位数，用于说明"压缩"。
#   建议阈值：Q_LOW = GB/T 17891 在预测尺上的最优阈值；Q_HIGH = 两标准最优阈值的中点（与正文逻辑链相同）。
#   输出：out_dir("sensitivity")/threshold_calibration_cv.csv、cv_predictions.csv、threshold_calibration_cv.txt
#   用法：python 34_calibrate_thresholds_cv_1031.py [--pc_cap 9.0] [--type_col TYPE] [--seed 42] [--nboot 1000]
# ============================================================
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

import config_1004 as C
import q_common_1004 as Q

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--xlsx", default=None)
p.add_argument("--type_col", default="TYPE")
p.add_argument("--pc_cap", type=float, default=9.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--nboot", type=int, default=1000)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
BASE = ["CO2", "Tavg", "Tdif", "CTSR", "TSR", "Rain", "RHavg"]
TR = ["HRR", "CD", "PC"]

df = pd.read_excel(a.xlsx or C.FACE_XLSX)
models = {t: joblib.load(C.MODEL_DIR / f"{t}_Blending.joblib") for t in TR}
feats = {t: list(getattr(models[t], "feature_names_in_", BASE)) for t in TR}
need = sorted(set(sum(feats.values(), [])) | set(TR) | {a.type_col})
df = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)
print(f"记录 {len(df)} 条；特征 {feats}")

# ---- 5 折样本外预测（与训练同一 KFold 设置）
kf = KFold(n_splits=5, shuffle=True, random_state=a.seed)
pred = pd.DataFrame(index=df.index)
for t in TR:
    est = clone(models[t])
    pred[f"{t}_cv"] = cross_val_predict(est, df[feats[t]], df[t].values, cv=kf)
    pred[f"{t}_fit"] = models[t].predict(df[feats[t]])          # 全量拟合的样本内预测（参考）
    rmse = float(np.sqrt(np.mean((pred[f"{t}_cv"] - df[t]) ** 2)))
    print(f"  {t}: CV RMSE {rmse:.2f}")
q_obs = Q.q_index(df.HRR.values, df.CD.values, df.PC.values)
q_cv = Q.q_index(pred.HRR_cv.values, pred.CD_cv.values, pred.PC_cv.values)
q_fit = Q.q_index(pred.HRR_fit.values, pred.CD_fit.values, pred.PC_fit.values)
pred["Q_obs"], pred["Q_cv"], pred["Q_fit"] = q_obs, q_cv, q_fit

# ---- 标签（实测值 vs 国标三级限值）
ind = df[a.type_col].astype(float) == 1
LIM = {"GB/T 17891-2017": dict(hrr=np.where(ind, 46, 55), cd=np.where(ind, 8.0, 6.0)),
       "NY/T 593-2021":   dict(hrr=np.where(ind, 50, 63), cd=np.where(ind, 5.0, 5.0))}
labels = {s: ((df.HRR.values >= L["hrr"]) & (df.CD.values <= L["cd"]) & (df.PC.values <= a.pc_cap)).astype(int)
          for s, L in LIM.items()}


def youden(y, q):
    fpr, tpr, thr = roc_curve(y, q)
    J = tpr - fpr
    j = int(np.argmax(J))
    plateau = thr[J >= J[j] - 0.01]
    plateau = plateau[np.isfinite(plateau)]
    return float(thr[j]), float(tpr[j]), float(1 - fpr[j]), float(J[j]), float(plateau.min()), float(plateau.max())


rows, lines = [], []
lines.append(f"Q 锚点（config.ANCHORS）: {Q.ANCHORS}\n")
lines.append("Q 分位数 (p5, p25, p50, p75, p95)：")
for nm, q in (("实测 Q_obs", q_obs), ("CV 预测 Q_cv", q_cv), ("全量拟合 Q_fit", q_fit)):
    lines.append(f"  {nm:14s} " + "  ".join(f"{v:.3f}" for v in np.percentile(q, [5, 25, 50, 75, 95])))
lines.append("")
opt = {}
for scale, q in (("observed", q_obs), ("cv_pred", q_cv), ("fit_pred", q_fit)):
    for s, y in labels.items():
        auc = roc_auc_score(y, q)
        thr, se, sp, J, plo, phi = youden(y, q)
        boots = []
        for _ in range(a.nboot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            boots.append(youden(y[i], q[i])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(scale=scale, standard=s, positive_pct=100 * y.mean(), auc=auc, youden_threshold=thr,
                         sensitivity=se, specificity=sp, J=J, plateau_lo=plo, plateau_hi=phi, boot_lo=lo, boot_hi=hi))
        opt[(scale, s)] = thr
        lines.append(f"[{scale:9s}] {s}: 阳性 {100*y.mean():.1f}%  AUC {auc:.3f}  最优阈值 {thr:.3f} "
                     f"(95% CI {lo:.3f}–{hi:.3f}; J 平台 {plo:.3f}–{phi:.3f})  sens {se:.2f} spec {sp:.2f}")
    lines.append("")
tab = pd.DataFrame(rows)

# ---- 建议阈值（预测尺）
qlo = opt[("cv_pred", "GB/T 17891-2017")]
qny = opt[("cv_pred", "NY/T 593-2021")]
qhi = (qlo + qny) / 2
lines.append(f"建议（CV 预测尺）：Q_LOW = {qlo:.3f}（GB/T 最优）；NY/T 最优 {qny:.3f}；Q_HIGH = 中点 {qhi:.3f}")
lines.append(f"  → config_1004.py: Q_LOW, Q_HIGH = {round(qlo, 2):.2f}, {round(qhi, 2):.2f}")
for s, y in labels.items():
    for t in (qlo, qhi):
        pr = (q_cv >= t).astype(int)
        lines.append(f"  {s} @ Q_cv ≥ {t:.3f}: sens {(pr[y==1]==1).mean():.2f} spec {(pr[y==0]==0).mean():.2f}")
# 同一阈值在实测尺上的敏感度/特异度（说明两把尺子的差别）
lines.append("对照：同一阈值用于实测 Q_obs 时")
for s, y in labels.items():
    for t in (qlo, qhi):
        pr = (q_obs >= t).astype(int)
        lines.append(f"  {s} @ Q_obs ≥ {t:.3f}: sens {(pr[y==1]==1).mean():.2f} spec {(pr[y==0]==0).mean():.2f}")

OUT = C.out_dir("sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
tab.to_csv(OUT / "threshold_calibration_cv.csv", index=False, float_format="%.4f")
pd.concat([df[[a.type_col] + TR], pred], axis=1).to_csv(OUT / "cv_predictions.csv", index=False, float_format="%.4f")
(OUT / "threshold_calibration_cv.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\n[saved] {OUT / 'threshold_calibration_cv.csv'}\n        {OUT / 'cv_predictions.csv'}\n        {OUT / 'threshold_calibration_cv.txt'}")
