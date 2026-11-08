import numpy as np
import rasterio
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

from config_1004 import (TAG, HIS_P, FUT_P, Q_LOW as _QL, Q_HIGH as _QH, CROP_MODELS, GCMS, SSPS, RI_TYPES,
                         Q_THRESH_MODE, Q_PCT, Q_POOL_SEASONS, thresh_tag,
                         pred_dir, out_dir, support_file, area_file, ANCHORS, ANCHOR_SIGN)
import json
OUT_ROOT = out_dir("quality_zones")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "annual":   {"mode": "annual"},
    "meanclim": {"mode": "meanclim"},
}

ANCHOR = {t: (*ANCHORS[t], ANCHOR_SIGN[t]) for t in ("HRR", "CD", "PC")}
COMPUTE_AGREEMENT = True

Q_LOW, Q_HIGH = _QL, _QH

def read_tif(p):
    if not Path(p).exists(): return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64"); nd = ds.nodata
        if nd is not None: a[a == nd] = np.nan
    return a

def his_path(v, ri, t):        return pred_dir(v["mode"])/"ensemble"/f"ensemble_historical_{ri}_{t}_{HIS_P}.tif"
def fut_ens_path(v, ri, s, t): return pred_dir(v["mode"])/"ensemble"/f"ensemble_{s}_{ri}_{t}_{FUT_P}.tif"
def fut_member_path(v, ri, s, m, g, t):
    return pred_dir(v["mode"])/s/f"{m}_{g}_{s}_{ri}_{t}_{FUT_P}.tif"
def his_member_path(v, ri, g, t):
    return pred_dir(v["mode"])/"historical"/f"{g}_historical_{ri}_{t}_{HIS_P}.tif"

def get_meta(v, ri):
    for t in ANCHOR:
        p = his_path(v, ri, t)
        if Path(p).exists():
            with rasterio.open(p) as ds: m = ds.meta.copy()
            m.update(count=1, dtype="float32", nodata=np.nan, compress="lzw"); return m
    raise FileNotFoundError(f"no historical tif for {ri}")

def save(arr, path, meta, dtype="float32", nodata=np.nan):
    m = meta.copy(); m.update(dtype=dtype, nodata=nodata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **m) as d: d.write(arr.astype(dtype), 1)

def desir(x, trait):
    lo, hi, sign = ANCHOR[trait]
    z = np.clip((x - lo)/(hi - lo), 0.0, 1.0)
    return z if sign > 0 else (1.0 - z)

def compute_Q(arrays, idx, shape):
    """绝对归一合意度 + 等权 3 指标 → 全栅格 Q(域外NaN) 及逐性状 d(域内)。"""
    d = {t: desir(arrays[t].ravel()[idx], t) for t in ANCHOR}
    Q_flat = (d["HRR"] + d["CD"] + d["PC"]) / 3.0
    Q = np.full(shape[0]*shape[1], np.nan); Q[idx] = Q_flat
    return Q.reshape(shape), d

def assign_zone(Q, lo, hi):
    z = np.zeros_like(Q, dtype="uint8"); ok = ~np.isnan(Q)
    z[ok & (Q < lo)] = 1
    z[ok & (Q >= lo) & (Q < hi)] = 2
    z[ok & (Q >= hi)] = 3
    return z

def wpct(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])


def baseline_Q_all(v):
    """各季基准期 Q 与面积权重（与下文分区相同的有效域）"""
    out = {}
    for ri in RI_TYPES:
        meta = get_meta(v, ri); shape = (meta["height"], meta["width"])
        w2d = read_tif(area_file(ri))
        base = {t: read_tif(his_path(v, ri, t)) for t in ANCHOR}
        if w2d is None or any(x is None for x in base.values()):
            continue
        dom = np.ones(shape, bool)
        for t in ANCHOR: dom &= np.isfinite(base[t])
        dom &= np.isfinite(w2d) & (w2d > 0)
        sup = read_tif(support_file(ri))
        if sup is not None: dom &= np.isfinite(sup)
        idx = np.where(dom.ravel())[0]
        Qb, _ = compute_Q(base, idx, shape)
        out[ri] = (Qb.ravel()[idx], w2d.ravel()[idx])
    return out


for vname, v in VARIANTS.items():
    print(f"\n{'#'*58}\n# {vname}")
    out_dir = OUT_ROOT / vname; out_dir.mkdir(parents=True, exist_ok=True)

    TH = {}
    if Q_THRESH_MODE == "baseline_pct":
        qb_all = baseline_Q_all(v)
        if Q_POOL_SEASONS:
            q = np.concatenate([a for a, _ in qb_all.values()]); w = np.concatenate([b for _, b in qb_all.values()])
            lo, hi = wpct(q, w, Q_PCT[0]), wpct(q, w, Q_PCT[1])
            TH = {ri: (lo, hi) for ri in RI_TYPES}
        else:
            TH = {ri: (wpct(a, b, Q_PCT[0]), wpct(a, b, Q_PCT[1])) for ri, (a, b) in qb_all.items()}
    else:
        TH = {ri: (_QL, _QH) for ri in RI_TYPES}
    for ri, (lo, hi) in TH.items():
        print(f"  [{ri}] 阈值 普通<{lo:.4f}≤中等<{hi:.4f}≤优质  （{Q_THRESH_MODE}"
              + (f"，分位数 {Q_PCT[0]:.3f}/{Q_PCT[1]:.3f}，两季合并={Q_POOL_SEASONS}" if Q_THRESH_MODE != "fixed" else "") + "）")
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(dict(mode=Q_THRESH_MODE, pct=Q_PCT, pool_seasons=Q_POOL_SEASONS, tag=thresh_tag(),
                       thresholds={ri: dict(Q_LOW=lo, Q_HIGH=hi) for ri, (lo, hi) in TH.items()}), f, indent=2)

    for ri in RI_TYPES:
        Q_LOW, Q_HIGH = TH.get(ri, (_QL, _QH))
        print(f"\n[{vname}/{ri}]")
        meta = get_meta(v, ri); H, W = meta["height"], meta["width"]; shape = (H, W)
        w2d = read_tif(area_file(ri))
        if w2d is None: print("  no area weight, skip"); continue
        base = {t: read_tif(his_path(v, ri, t)) for t in ANCHOR}
        if any(x is None for x in base.values()): print("  baseline traits missing, skip"); continue

        dom = np.ones(shape, bool)
        for t in ANCHOR: dom &= np.isfinite(base[t])
        dom &= np.isfinite(w2d) & (w2d > 0)
        sup = read_tif(support_file(ri))
        if sup is not None: dom &= np.isfinite(sup)
        else: print("  [warn] 无基准支撑区文件，未筛选")
        idx = np.where(dom.ravel())[0]
        if idx.size == 0: print("  empty domain, skip"); continue
        w_dom = w2d.ravel()[idx]

        Q_b, d_b = compute_Q(base, idx, shape)
        zone_b = assign_zone(Q_b, Q_LOW, Q_HIGH)
        def prop(z):
            """[1013] 分母 = 该层有分级的格网面积（与 23/24 的 shares 同口径）"""
            zz=z.ravel()[idx]; tw=w_dom[zz > 0].sum()
            return {g:100*w_dom[zz==g].sum()/tw for g in (3,2,1)}
        pb=prop(zone_b)
        print(f"  阈值 优质>={Q_HIGH:.4f} 中等>={Q_LOW:.4f}  基线Q均值={np.nanmean(Q_b):.3f}")
        print(f"  基线占比 优/中/普 = {pb[3]:.0f}/{pb[2]:.0f}/{pb[1]:.0f} %")

        member_rows = []
        def member_share(arrs):
            """[1013] 分母 = 该成员有值格网的面积（缺值格网不再算作"非优质"）；另记覆盖率"""
            Qm, _ = compute_Q(arrs, idx, shape)
            zm = assign_zone(Qm, Q_LOW, Q_HIGH).ravel()[idx]
            tw = w_dom[zm > 0].sum()
            sh = {g: 100*w_dom[zm == g].sum()/tw for g in (3, 2, 1)} if tw > 0 else {g: np.nan for g in (3, 2, 1)}
            sh["coverage"] = 100 * tw / w_dom.sum()
            return zm, sh
        for g in GCMS:
            mt = {t: read_tif(his_member_path(v, ri, g, t)) for t in ANCHOR}
            if any(x is None for x in mt.values()): continue
            _, sh = member_share(mt)
            member_rows.append(["baseline", g, round(sh[3],2), round(sh[2],2), round(sh[1],2), round(sh["coverage"],2)])

        save(Q_b, out_dir/f"Q_{ri}_baseline.tif", meta)
        save(zone_b, out_dir/f"zone_{ri}_baseline.tif", meta, "uint8", 0)
        for t in ANCHOR:
            full = np.full(H*W, np.nan); full[idx] = d_b[t]
            save(full.reshape(shape), out_dir/f"d_{t}_{ri}_baseline.tif", meta)

        for ssp in SSPS:
            fut = {t: read_tif(fut_ens_path(v, ri, ssp, t)) for t in ANCHOR}
            if any(x is None for x in fut.values()): print(f"  future {ssp} missing, skip"); continue
            Q_f, d_f = compute_Q(fut, idx, shape)
            zone_f = assign_zone(Q_f, Q_LOW, Q_HIGH)
            save(Q_f, out_dir/f"Q_{ri}_{ssp}_future.tif", meta)
            save(zone_f, out_dir/f"zone_{ri}_{ssp}_future.tif", meta, "uint8", 0)
            for t in ANCHOR:
                full = np.full(H*W, np.nan); full[idx] = d_f[t]
                save(full.reshape(shape), out_dir/f"d_{t}_{ri}_{ssp}_future.tif", meta)

            if COMPUTE_AGREEMENT:
                agree = np.zeros(idx.size); n_ok = 0
                for m in CROP_MODELS:
                    for g in GCMS:
                        mt = {t: read_tif(fut_member_path(v, ri, ssp, m, g, t)) for t in ANCHOR}
                        if any(x is None for x in mt.values()): continue
                        zm, sh = member_share(mt)
                        member_rows.append([ssp, f"{m}_{g}", round(sh[3],2), round(sh[2],2), round(sh[1],2),
                                            round(sh["coverage"],2)])
                        zf_ = zone_f.ravel()[idx]
                        agree += (zm == zf_) & (zm > 0); n_ok += 1
                if n_ok:
                    fr = np.full(H*W, np.nan); fr[idx] = agree/n_ok
                    save(fr.reshape(shape), out_dir/f"agreement_{ri}_{ssp}.tif", meta)

            pf=prop(zone_f)
            _cov = 100 * w_dom[zone_f.ravel()[idx] > 0].sum() / w_dom.sum()
            if _cov < 99.99:
                print(f"  ⚠️ [{ssp}] 集合未来 Q 只覆盖基准分区面积的 {_cov:.2f}%（缺值格网不计入占比分母）")
            print(f"  [{ssp}] 未来Q均值={np.nanmean(Q_f):.3f}  未来占比 优/中/普 = "
                  f"{pf[3]:.0f}/{pf[2]:.0f}/{pf[1]:.0f} %")
            prem = [r[2] for r in member_rows if r[0] == ssp]
            if prem:
                print(f"         成员级优质占比 中位数 {np.median(prem):.1f}% [{min(prem):.1f}–{max(prem):.1f}]  n={len(prem)}")

        import csv
        with open(out_dir/f"member_zone_share_{ri}.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["period", "member", "premium_pct", "moderate_pct", "common_pct", "coverage_pct"])
            wr.writerows(member_rows)
        pb_m = [r[2] for r in member_rows if r[0] == "baseline"]
        if pb_m:
            print(f"  基线成员级优质占比 中位数 {np.median(pb_m):.1f}% [{min(pb_m):.1f}–{max(pb_m):.1f}]  n={len(pb_m)}")

print(f"\n完成 -> {OUT_ROOT}")
