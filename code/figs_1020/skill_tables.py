import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("/home/debian/pythonfile/results/Tables")
TRAIT_KEYS = {"BRR": ["brr", "brown"], "MRR": ["mrr", "milled"], "HRR": ["hrr", "head"],
              "CGR": ["cgr", "chalky grain", "chalky_grain", "chalkygrain"],
              "CD": ["cd", "chalkiness", "chalk degree"], "Acontent": ["acontent", "ac", "amylose"],
              "PC": ["pc", "protein"]}


def _norm_trait(v):
    s = str(v).strip().lower()
    for t, keys in TRAIT_KEYS.items():
        if s in keys:
            return t
    for t, keys in TRAIT_KEYS.items():
        if any(k in s for k in keys if len(k) > 3):
            return t
    return None


def _trait_col(df):
    for c in df.columns:
        if "trait" in c.lower() or "target" in c.lower() or "variable" in c.lower():
            return c
    for c in df.columns:
        if df[c].map(_norm_trait).notna().sum() >= 5:
            return c
    raise KeyError(f"找不到性状列：{list(df.columns)}")


def _find(cols, must, avoid=()):
    for c in cols:
        lc = c.lower().replace("²", "2")
        if all(re.search(m, lc) for m in must) and not any(re.search(a, lc) for a in avoid):
            return c
    return None


LO = (r"lo(w|wer)?\b|lower|2\.5|_lo|ci_l|low")
HI = (r"hi(gh|gher)?\b|upper|97\.5|_hi|ci_u|high")


def read_test_ci(path):
    df = pd.read_csv(path)
    tc = _trait_col(df)
    mcol = _find(df.columns, [r"metric|stat|measure"])
    if mcol is not None:
        sub = df[df[mcol].astype(str).str.lower().str.replace("²", "2").str.contains(r"^r2|r\^?2|r_2")]
        est = _find(sub.columns, [r"est|value|point|mean|score|r2"], [LO, HI, mcol.lower()])
        lo = _find(sub.columns, [LO]); hi = _find(sub.columns, [HI])
    else:
        sub = df
        est = _find(df.columns, [r"r2|r_2|r\^2"], [LO, HI, r"cv|train"])
        lo = _find(df.columns, [r"r2|r_2|r\^2", LO]); hi = _find(df.columns, [r"r2|r_2|r\^2", HI])
    ci = _find(sub.columns, [r"r2|r_2|r\^2", r"ci"]) if (lo is None or hi is None) else None
    if est is None or (ci is None and None in (lo, hi)):
        raise KeyError(f"{path.name}: 识别不到 R² / CI 列；列名 = {list(df.columns)}")
    out = pd.DataFrame({"trait": sub[tc].map(_norm_trait), "test_r2": sub[est].astype(float)})
    if ci is not None:
        parts = sub[ci].astype(str).str.split(r"\s*(?:–|—|-|~|to|,)\s*", n=1, regex=True)
        out["test_r2_lo"] = parts.str[0].astype(float); out["test_r2_hi"] = parts.str[1].astype(float)
    else:
        out["test_r2_lo"] = sub[lo].astype(float); out["test_r2_hi"] = sub[hi].astype(float)
    for m in ("rmse", "mae"):
        c = _find(sub.columns, [m], [r"ci", LO, HI])
        if c is not None:
            out[f"test_{m}"] = pd.to_numeric(sub[c], errors="coerce")
    return out.dropna(subset=["trait"]).groupby("trait", as_index=False).first()


def read_cv_folds(path):
    """逐折 R²（长表：trait, r2），Fig. 1b 箱线图用"""
    path = Path(path)
    df = pd.read_csv(path)
    tc = _trait_col(df)
    mcol = _find(df.columns, [r"metric|stat|measure"])
    if mcol is not None:
        sub = df[df[mcol].astype(str).str.lower().str.replace("²", "2").str.contains(r"^r2|r\^?2|r_2")]
        val = _find(sub.columns, [r"value|score|r2|est"], [mcol.lower()])
    else:
        sub = df
        val = _find(df.columns, [r"r2|r_2|r\^2"], [r"rmse|mae|sd|std|lo|hi"])
    if val is None:
        raise KeyError(f"{path.name}: 识别不到逐折 R² 列；列名 = {list(df.columns)}")
    return pd.DataFrame({"trait": sub[tc].map(_norm_trait), "r2": sub[val].astype(float)}).dropna(subset=["trait"])


def read_cv(path):
    g = read_cv_folds(path)
    return g.groupby("trait").r2.agg(cv_r2_mean="mean", cv_r2_sd=lambda s: s.std(ddof=1)).reset_index()


def load_skill(tables=TABLES, reference=None):
    tables = Path(tables)
    t = read_test_ci(tables / "Test_bootstrap_CI95.csv")
    c = read_cv(tables / "Ensemble_CV_folds_training.csv")
    sk = t.merge(c, on="trait", how="left")
    print("[Fig. 1b] 读自训练输出：\n" + sk.round(3).to_string(index=False))
    if reference is not None and Path(reference).exists():
        ref = pd.read_csv(reference).set_index("trait")
        bad = []
        for _, r in sk.iterrows():
            if r.trait not in ref.index:
                continue
            for col in ("test_r2", "test_r2_lo", "test_r2_hi", "cv_r2_mean", "cv_r2_sd", "test_rmse", "test_mae"):
                if col in ref.columns and col in r.index and np.isfinite(r[col]) \
                        and abs(r[col] - ref.loc[r.trait, col]) > 0.0051:
                    bad.append(f"{r.trait} {col}: 训练输出 {r[col]:.3f} vs 表 S3 {ref.loc[r.trait, col]:.2f}")
        print("[Fig. 1b] 与 SI 表 S3 一致 ✔" if not bad else "[Fig. 1b] ⚠️ 与 SI 表 S3 不一致：\n  " + "\n  ".join(bad))
    return sk
