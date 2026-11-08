import argparse, csv, datetime, hashlib, shutil, sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_1004 as C

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None, help="输出目录（默认 output/ricequality_data_1108/gridded）")
ap.add_argument("--no-rewrite", action="store_true", help="只复制不重写格式/元数据")
ap.add_argument("--zip", action="store_true", help="另按子目录打 zip（Zenodo 网页上传用）")
A = ap.parse_args()
DEST = Path(A.out).expanduser() if A.out else C.OUTPUT / "ricequality_data_1108" / "gridded"
if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)

PRED = C.pred_dir()
HRY = C.out_dir("head_rice_yield", thresh=False)
TRAITS = ["BRR", "MRR", "HRR", "CGR", "CD", "Acontent", "PC"]
TNAME = {"BRR": "brown-rice rate", "MRR": "milled-rice rate", "HRR": "head-rice rate", "CGR": "chalky-grain rate",
         "CD": "chalkiness degree", "Acontent": "amylose content (AC)", "PC": "protein content"}
RINAME = {"ri1": "single/early rice", "ri2": "late rice"}
H, F = C.HIS_P, C.FUT_P
EXP_SHAPE = (72, 126)
EXP_BOUNDS = (73.0, 18.0, 136.0, 54.0)
CITE = ("Feng, X. et al. Climate change downgrades China's rice harvest even where yields rise (Nature Food, submitted); "
        "https://github.com/fengxiangqian415-hub/ricequality")
LICENCE = "CC BY 4.0"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


checks, rows = [], []


def put(src, rel, meta, tags):
    """复制（默认重写为 float32 / NaN / LZW 并写元数据）并核对网格。meta = [product, ri, trait, period_or_ssp, GGCM, GCM]"""
    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if not src.exists():
        checks.append([rel, "MISSING", str(src)]); return
    with rasterio.open(src) as ds:
        a = ds.read(1).astype("float32"); prof = ds.profile.copy()
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a[a == ds.nodata] = np.nan
        shp, b, crs = (ds.height, ds.width), ds.bounds, ds.crs
    ok = shp == EXP_SHAPE and all(abs(x - y) < 1e-6 for x, y in zip((b.left, b.bottom, b.right, b.top), EXP_BOUNDS)) \
        and crs is not None and crs.to_epsg() == 4326
    if not ok:
        checks.append([rel, "GRID/CRS", f"shape={shp} bounds={tuple(round(v, 3) for v in b)} crs={crs}"])
    if A.no_rewrite:
        shutil.copy2(src, dst)
    else:
        prof.update(driver="GTiff", dtype="float32", nodata=np.nan, count=1, compress="lzw", predictor=3, tiled=False)
        for k in ("blockxsize", "blockysize"):
            prof.pop(k, None)
        with rasterio.open(dst, "w", **prof) as out:
            out.write(a, 1)
            out.update_tags(**{k: v for k, v in tags.items() if v}, citation=CITE, licence=LICENCE,
                            created=datetime.date.today().isoformat())
            out.set_band_description(1, tags.get("variable", ""))
            out.update_tags(1, units=tags.get("units", ""))
    rows.append([rel, *meta, int(np.isfinite(a).sum()), dst.stat().st_size, sha256(dst)])


for ri in C.RI_TYPES:
    for t in TRAITS:
        put(PRED / "ensemble" / f"ensemble_historical_{ri}_{t}_{H}.tif",
            f"01_quality_historical_mean/{t}_{ri}_historical_{H}_mean.tif",
            ["quality_historical_mean", ri, t, "historical", "", ""],
            dict(product="ensemble-mean grain-quality trait", variable=TNAME[t], trait=t, units="%",
                 period=H.replace("_", "-"), scenario="historical", rice_season=RINAME[ri]))
for s in C.SSPS:
    for m in C.CROP_MODELS:
        for g in C.GCMS:
            for ri in C.RI_TYPES:
                for t in TRAITS:
                    nm = f"{m}_{g}_{s}_{ri}_{t}_{F}.tif"
                    put(PRED / s / nm, f"02_quality_future_members/{s}/{nm}",
                        ["quality_future_member", ri, t, s, m, g],
                        dict(product="member-level grain-quality trait (scenario CO2)", variable=TNAME[t], trait=t, units="%",
                             period=F.replace("_", "-"), scenario=s, GGCM=m, GCM=g, rice_season=RINAME[ri]))
for ri in C.RI_TYPES:
    for per in ["historical", *C.SSPS]:
        nm = f"head_rice_yield_{ri}_{per}.tif"
        put(HRY / nm, f"03_head_rice_yield/{nm}", ["head_rice_yield_mean", ri, "HRY", per, "", ""],
            dict(product="ensemble-mean head-rice yield (GGCM paddy yield x predicted HRR / 100)", variable="head-rice yield",
                 units="t ha-1", period=(H if per == "historical" else F).replace("_", "-"), scenario=per, rice_season=RINAME[ri]))

with open(DEST / "file_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "product", "rice_season", "trait", "period_or_scenario", "GGCM", "GCM", "n_valid_cells", "bytes", "sha256"])
    w.writerows(rows)
with open(DEST / "CHECK_grid.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["file", "issue", "detail"]); w.writerows(checks)

n_by = {}
for r in rows:
    n_by[r[1]] = n_by.get(r[1], 0) + 1
EXPECT = {"quality_historical_mean": 14, "quality_future_member": 700, "head_rice_yield_mean": 6}
(DEST / "README.md").write_text(f"""# Gridded rice grain-quality and head-rice yield projections for China

Data deposit for Feng, X. et al. *Climate change downgrades China's rice harvest even where yields rise* (Nature Food, submitted).
File inventory and naming follow Supplementary Table 13 of the paper. Generated {datetime.date.today()} by `code/export_tif_1108.py`.

| Folder | Product | Files |
|---|---|---|
| `01_quality_historical_mean/` | Ensemble-mean grain-quality traits, baseline 1981–2010 | {n_by.get('quality_historical_mean', 0)} |
| `02_quality_future_members/ssp126`, `ssp585` | Member-level (GGCM × GCM) traits, 2041–2070, scenario CO₂ | {n_by.get('quality_future_member', 0)} |
| `03_head_rice_yield/` | Ensemble-mean head-rice yield, baseline and both SSPs | {n_by.get('head_rice_yield_mean', 0)} |

## File naming
- `{{trait}}_{{ri}}_historical_{H}_mean.tif`
- `{{GGCM}}_{{GCM}}_{{ssp}}_{{ri}}_{{trait}}_{F}.tif`
- `head_rice_yield_{{ri}}_{{historical|ssp126|ssp585}}.tif`

ri1 = single/early rice; ri2 = late rice. trait ∈ {{BRR, MRR, HRR, CGR, CD, Acontent, PC}} (Acontent = amylose content, AC).
GGCM ∈ {{{', '.join(C.CROP_MODELS)}}}. GCM ∈ {{{', '.join(C.GCMS)}}}. ssp ∈ {{{', '.join(C.SSPS)}}}.

## Format
Single-band GeoTIFF, float32, NoData = NaN, LZW-compressed, EPSG:4326, 0.5° × 0.5° grid spanning 73–136° E and 18–54° N
(126 × 72 cells). Values are defined only within the rice model-support mask.
Units: quality traits in per cent; head-rice yield in t ha⁻¹.
Each file carries metadata tags (product, trait, units, period, scenario, GGCM, GCM, rice season, citation, licence).

Traits: BRR brown-rice rate; MRR milled-rice rate; HRR head-rice rate; CGR chalky-grain rate; CD chalkiness degree;
AC amylose content; PC protein content. Future traits are predicted from each member's grain-filling climate and the
scenario CO₂ concentration (Methods). Head-rice yield = GGCM paddy yield × predicted HRR / 100, averaged over members.

## Integrity
`file_inventory.csv` lists every file with its SHA-256 checksum and number of valid cells; `CHECK_grid.csv` is empty
(header only) when every raster is present and on the common grid.

## Licence and citation
Data: {LICENCE}. Please cite the paper and the archived release (Zenodo DOI to be added on publication).
Contact: Xiangqian Feng (fengxiangqian415@gmail.com).
""", encoding="utf-8")

if A.zip:
    zdir = DEST.parent / f"{DEST.name}_zip"; zdir.mkdir(exist_ok=True)
    for sub in ["01_quality_historical_mean", "03_head_rice_yield"]:
        shutil.make_archive(str(zdir / sub), "zip", DEST, sub)
    for s in C.SSPS:
        shutil.make_archive(str(zdir / f"02_quality_future_members_{s}"), "zip", DEST / "02_quality_future_members", s)
    print("zip →", zdir)

size = sum(f.stat().st_size for f in DEST.rglob("*.tif")) / 1024 ** 2
print(f"\n[export_tif_1108] {len(rows)} 个 GeoTIFF（{size:.1f} MB）→ {DEST}")
bad = {k: (n_by.get(k, 0), v) for k, v in EXPECT.items() if n_by.get(k, 0) != v}
for k, v in EXPECT.items():
    print(f"  {k:26s} {n_by.get(k, 0):4d} / {v}")
if checks or bad:
    print(f"  ⚠ 网格/缺失问题 {len(checks)} 条（CHECK_grid.csv）；数量不符：{bad}")
    sys.exit(1)
print("  全部齐全、同一网格 ✔")
