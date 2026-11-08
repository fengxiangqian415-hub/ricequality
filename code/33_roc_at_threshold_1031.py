# ============================================================
# 33_roc_at_threshold_1031.py —— Table S6 补充：Q 在给定阈值处对两套国标三级标签的敏感度/特异度（主方案阈值 0.67）
#   标签规则同 Supplementary Table S5：HRR ≥ 限值（籼稻按中粒：GB/T 17891 46、NY/T 593 50；粳稻 55、63），
#   CD ≤ 限值（GB/T 籼 8.0、粳 6.0；NY/T 5.0/5.0），且 PC ≤ 9.0（DB21/T 4211）。
#   同时重算 AUC 与 Youden 最优阈值，用于核对 Table S6（应得 0.90 与 0.62 / 0.72）。
#   用法：python 33_roc_at_threshold_1031.py [--xlsx 训练表] [--type_col TYPE] [--thresholds 0.62,0.66,0.67,0.72] [--pc_cap 9.0]
# ============================================================
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config_1004 as C
import q_common_1004 as Q

p = argparse.ArgumentParser()
p.add_argument("--xlsx", default=None)
p.add_argument("--type_col", default="TYPE", help="籼/粳列：1 = indica, 2 = japonica")
p.add_argument("--thresholds", default="0.62,0.66,0.67,0.72")
p.add_argument("--pc_cap", type=float, default=9.0)
p.add_argument("--anchors", default=None,
               help="覆盖 Q 锚点，格式 HRRlo,HRRhi,CDlo,CDhi,PClo,PChi；如正文 p5/p95：35.6,71.8,1.0,27.0,5.6,11.0（默认用 config.ANCHORS）")
a = p.parse_args()
df = pd.read_excel(a.xlsx or C.FACE_XLSX)
need = ["HRR", "CD", "PC"]
miss = [c for c in need + [a.type_col] if c not in df.columns]
if miss:
    raise SystemExit(f"训练表缺列 {miss}；现有列：{list(df.columns)}")
df = df.dropna(subset=need + [a.type_col]).copy()
ind = df[a.type_col].astype(float) == 1
LIM = {"GB/T 17891-2017": dict(hrr=np.where(ind, 46, 55), cd=np.where(ind, 8.0, 6.0)),
       "NY/T 593-2021":   dict(hrr=np.where(ind, 50, 63), cd=np.where(ind, 5.0, 5.0))}
if a.anchors:
    v = [float(x) for x in a.anchors.split(",")]
    Q.ANCHORS = {"HRR": (v[0], v[1]), "CD": (v[2], v[3]), "PC": (v[4], v[5])}
print("Q 锚点:", Q.ANCHORS)
q = Q.q_index(df.HRR.values.astype(float), df.CD.values.astype(float), df.PC.values.astype(float))
TH = [float(t) for t in a.thresholds.split(",")]
rows = []
for std, L in LIM.items():
    y = ((df.HRR.values >= L["hrr"]) & (df.CD.values <= L["cd"]) & (df.PC.values <= a.pc_cap)).astype(int)
    auc = roc_auc_score(y, q)
    fpr, tpr, thr = roc_curve(y, q)
    j = np.argmax(tpr - fpr)
    print(f"{std}: 阳性 {100*y.mean():.1f}%  AUC {auc:.3f}  Youden 最优阈值 {thr[j]:.3f}（sens {tpr[j]:.2f}, spec {1-fpr[j]:.2f}）")
    for t in TH:
        pred = (q >= t).astype(int)
        sens = (pred[y == 1] == 1).mean(); spec = (pred[y == 0] == 0).mean()
        rows.append(dict(standard=std, threshold=t, sensitivity=round(sens, 3), specificity=round(spec, 3), auc=round(auc, 3)))
        print(f"   Q ≥ {t:.2f}: sensitivity {sens:.2f}  specificity {spec:.2f}")
out = C.out_dir("sensitivity"); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "roc_at_thresholds.csv", index=False)
print(f"[saved] {out / 'roc_at_thresholds.csv'}")
