import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config_1004 as C
import q_common_1004 as Q

p = argparse.ArgumentParser()
p.add_argument("--caps", default="9.0,11.0", help="PC 上限（第一个为主方案）")
p.add_argument("--type_col", default="TYPE")
p.add_argument("--xlsx", default=None)
p.add_argument("--cv", default=None, help="34 号 cv_predictions.csv；默认先找本阈值目录，再找 output/sensitivity_*/ 下最新的")
a = p.parse_args()
CAPS = [float(x) for x in a.caps.split(",")]
LIMITS = {"GB/T 17891-2017": {"indica": (46, 8.0), "japonica": (55, 6.0)},
          "NY/T 593-2021":   {"indica": (50, 5.0), "japonica": (63, 5.0)}}

df = pd.read_excel(a.xlsx or C.FACE_XLSX).dropna(subset=["HRR", "CD", "PC", a.type_col]).reset_index(drop=True)
print("Q 锚点（config.ANCHORS）:", Q.ANCHORS)
print("训练集分位数 (p1, p5, p95, p99)：")
for t in ("HRR", "CD", "PC"):
    print(f"  {t}: " + "  ".join(f"{v:.2f}" for v in np.percentile(df[t], [1, 5, 95, 99])))

cvp = Path(a.cv) if a.cv else C.out_dir("sensitivity") / "cv_predictions.csv"
if not cvp.exists():
    cands = sorted(C.OUTPUT.glob("sensitivity_*/cv_predictions.csv"), key=lambda x: x.stat().st_mtime)
    cvp = cands[-1] if cands else None
q_cv = None
if cvp is not None:
    cv = pd.read_csv(cvp)
    if len(cv) == len(df):
        q_cv = Q.q_index(cv.HRR_cv.values, cv.CD_cv.values, cv.PC_cv.values)
        print(f"CV 预测：{cvp}")
    else:
        print(f"[提示] {cvp} 行数 {len(cv)} ≠ 训练表 {len(df)}，跳过预测尺")
else:
    print("[提示] 未找到 34 号 cv_predictions.csv，只给实测尺")
q_obs = Q.q_index(df.HRR.values, df.CD.values, df.PC.values)
ind = df[a.type_col].astype(float) == 1

rows = []
for cap in CAPS:
    print(f"\n===== PC 上限 {cap}" + ("（主方案）" if cap == CAPS[0] else "（敏感性）"))
    for std, L in LIMITS.items():
        qi, qj = Q.q_from_limits(*L["indica"], cap), Q.q_from_limits(*L["japonica"], cap)
        thr = (qi + qj) / 2
        y = ((df.HRR.values >= np.where(ind, L["indica"][0], L["japonica"][0])) &
             (df.CD.values <= np.where(ind, L["indica"][1], L["japonica"][1])) & (df.PC.values <= cap)).astype(int)
        r = dict(pc_cap=cap, standard=std, Q_indica=qi, Q_japonica=qj, threshold=thr, threshold_2dp=round(thr, 2),
                 positive_pct=100 * y.mean())
        for scale, q in (("obs", q_obs), ("cv", q_cv)):
            if q is None:
                continue
            pr = (q >= round(thr, 2)).astype(int)
            r.update({f"auc_{scale}": roc_auc_score(y, q),
                      f"sens_{scale}": (pr[y == 1] == 1).mean(), f"spec_{scale}": (pr[y == 0] == 0).mean(),
                      f"npv_{scale}": (y[pr == 0] == 0).mean() if (pr == 0).any() else np.nan})
        rows.append(r)
        msg = (f"{std}: Q(籼,中粒) {qi:.3f}  Q(粳) {qj:.3f}  平均 {thr:.3f} → {round(thr, 2):.2f}  阳性 {r['positive_pct']:.1f}%"
               f"\n    实测 Q: AUC {r['auc_obs']:.3f}  sens {r['sens_obs']:.2f}  spec {r['spec_obs']:.2f}  "
               f"Q<阈值者不达标的比例 {r['npv_obs']:.2f}")
        if q_cv is not None:
            msg += (f"\n    CV 预测 Q: AUC {r['auc_cv']:.3f}  sens {r['sens_cv']:.2f}  spec {r['spec_cv']:.2f}  "
                    f"Q<阈值者不达标的比例 {r['npv_cv']:.2f}")
        print(msg)
OUT = C.out_dir("sensitivity"); OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "table_S6_thresholds_from_limits.csv", index=False, float_format="%.4f")
print(f"\n[saved] {OUT / 'table_S6_thresholds_from_limits.csv'}")
