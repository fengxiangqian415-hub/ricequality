from pathlib import Path
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle, Wedge, Patch, Ellipse
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import sys, os

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import paper_palette as PP

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,  "ytick.major.size": 2.0,
})

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--fig1", action="store_true", help="输出 Fig. 1 a 面板用的紧凑版")
_ap.add_argument("--datadir", default=None, help="map.xlsx 等输入表所在目录（默认自动查找）")
_ap.add_argument("--out", default=None, help="输出目录（默认见文件头）")
EMBED = globals().get("EXT_AX") is not None
_A = EXT_ARGS if EMBED else _ap.parse_args()
FULL_OUT = Path("/home/debian/pythonfile/output/figures/sampling_map_1013")


def _fig1_out():
    try:
        sys.path.insert(0, os.path.dirname(_HERE))
        import config_1004 as C
        return C.FIGURES / "main_figs_1020"
    except Exception:
        return Path(_HERE) / "main_figs_1020"


def find_datadir():
    """map.xlsx 所在目录：先查常用位置，再在 /home/debian/pythonfile 下浅层搜索（跳过大数据目录）"""
    home = Path("/home/debian/pythonfile")
    cands = [Path.cwd(), home, home / "data", home / "map", Path(_HERE), Path(_HERE).parent,
             Path(_HERE).parent.parent, FULL_OUT]
    for d in cands:
        if (d / "map.xlsx").exists():
            return d
    skip = {"ISIMIP3b", "ISMIP3b", "output", "results", "china", "area", "masks", "__pycache__", ".git"}
    for root, dirs, files in os.walk(home):
        if "map.xlsx" in files:
            return Path(root)
        depth = Path(root).relative_to(home).parts
        dirs[:] = [] if len(depth) >= 3 else [d for d in dirs if d not in skip]
    raise FileNotFoundError("找不到 map.xlsx，请用 --datadir 指定它所在的目录")


DATA = Path(_A.datadir) if _A.datadir else find_datadir()
print(f"[map] 输入表目录：{DATA}")
XLSX      = DATA / "map.xlsx"
PHENO_XLSX= DATA / "CTSR&RAIN.xlsx"
FIG_W = 118 if _A.fig1 else 180
FSZ = (dict(tick=6.5, north=7.5, leg=6.5, scale=6.5, key=6.5, wh_label=6, wh_tick=5.5, wh_leg=5, wh_title=5.5,
            dot=4, wh_dot=7, wh_mk=3.2, leg_mk=2.5) if _A.fig1 else
       dict(tick=15, north=15, leg=9, scale=9, key=9, wh_label=10, wh_tick=9, wh_leg=7, wh_title=7.5,
            dot=7, wh_dot=14, wh_mk=4.5, leg_mk=3))
RICE_DIR       = Path("/home/debian/IF10")
RICE_PATTERNS  = ["CHN_Rice(SR&ER)_HE_{year}.tif",
                  "CHN_Rice(LR)_HE_{year}.tif"]
RICE_YEARS     = range(2000, 2011)
RICE_MIN_YEARS = 1
CHINA_SHP = Path("/home/debian/pythonfile/china/china.shp")
POLY_SHP  = Path("/home/debian/pythonfile/china/Polygon.shp")
OUT_DIR   = Path(_A.out) if _A.out else (_fig1_out() if _A.fig1 else FULL_OUT)
if not EMBED:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM     = "Fig1a_sampling_map" if _A.fig1 else "fig_sampling_sites"

MAP_EXTENT   = [73, 136, 17, 54]
MAP_ASPECT   = 1 / np.cos(np.deg2rad(np.mean([MAP_EXTENT[2], MAP_EXTENT[3]])))
INSET_EXTENT = [105, 2, 125, 26]
AREA_COLOR   = PP.RICE_AREA
DOT_FC       = PP.SITE
PIE_SINGLE   = PP.PIE_SINGLE
PIE_LATE     = PP.PIE_LATE
SPLIT_DOY    = 244
SCALE_KM     = 1000
LAT_REF      = 35.0

RICE_REPROJ_RES = 0.02
RICE_GRID       = (73.0, 136.0, 15.0, 54.0)

PIE_RMIN = 0.3
PIE_RMAX = 1.0

def load_basemap():
    for p in (CHINA_SHP, POLY_SHP):
        if not p.exists():
            raise FileNotFoundError(f"底图 shapefile 不存在: {p}（检查 CHINA_SHP / POLY_SHP 路径）")
    china = gpd.read_file(CHINA_SHP).to_crs("EPSG:4326")
    poly  = gpd.read_file(POLY_SHP).to_crs("EPSG:4326")
    nanhai = poly[poly.geometry.apply(lambda g: g.bounds[1] < 25)]
    jd     = poly[poly["adchar"] == "JD"] if "adchar" in poly.columns else nanhai
    return china, nanhai, jd


china, nanhai, jd_line = load_basemap()
PROV_COL = "name" if "name" in china.columns else china.columns[0]


def load_rice_area():
    """[1013] Rice area from the 1-km phenology product -> (mask, extent_lonlat).

    Same source and period as the analysis area weights (area_weight_{ri}_2000_2010):
    for each season pattern and each year in RICE_YEARS, a 1-km pixel is rice if
    its HE value is valid (> 0 and not nodata). Each yearly 0/1 mask is
    reprojected to a FIXED EPSG:4326 grid (RICE_GRID, RICE_REPROJ_RES) with
    Resampling.max (a target cell is rice if ANY source rice pixel falls in it).
    A cell is shown if it is rice in >= RICE_MIN_YEARS years for either season.

    Fallback (no files found): a synthetic mask so the layout can be QA'd locally.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    x0, x1, y0, y1 = RICE_GRID
    dst_w = int(round((x1 - x0) / RICE_REPROJ_RES))
    dst_h = int(round((y1 - y0) / RICE_REPROJ_RES))
    dst_transform = from_origin(x0, y1, RICE_REPROJ_RES, RICE_REPROJ_RES)

    shown = np.zeros((dst_h, dst_w), dtype=bool)
    n_found = 0
    for pat in RICE_PATTERNS:
        n_years = np.zeros((dst_h, dst_w), dtype=np.int16)
        found = []
        for yr in RICE_YEARS:
            fp = RICE_DIR / pat.format(year=yr)
            if not fp.exists():
                continue
            with rasterio.open(fp) as ds:
                src = ds.read(1)
                ok = src > 0
                if ds.nodata is not None:
                    ok &= src != ds.nodata
                if np.issubdtype(src.dtype, np.floating):
                    ok &= np.isfinite(src)
                dst = np.zeros((dst_h, dst_w), dtype="uint8")
                reproject(source=ok.astype("uint8"), destination=dst,
                          src_transform=ds.transform, src_crs=ds.crs,
                          dst_transform=dst_transform, dst_crs="EPSG:4326",
                          resampling=Resampling.max)
            n_years += (dst == 1)
            found.append(yr)
        if found:
            shown |= n_years >= RICE_MIN_YEARS
            n_found += len(found)
            lat_c = y1 - (np.arange(dst_h) + 0.5) * RICE_REPROJ_RES
            north = (n_years >= RICE_MIN_YEARS) & (lat_c[:, None] > 50)
            print(f"[rice area] {pat}: {found[0]}–{found[-1]} ({len(found)} yrs), "
                  f"cells shown {int((n_years >= RICE_MIN_YEARS).sum())}, north of 50°N {int(north.sum())}")
        else:
            print(f"[rice area] ⚠️ no files for {pat} in {RICE_DIR} ({RICE_YEARS.start}–{RICE_YEARS.stop - 1})")
    if n_found:
        return shown, (x0, x1, y0, y1)

    print("[rice area] ⚠️ no phenology tiles found -> SYNTHETIC rice area (layout QA only)")
    ny, nx = 700, 1000
    lon = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], nx)
    lat = np.linspace(MAP_EXTENT[3], MAP_EXTENT[2], ny)
    LON, LAT = np.meshgrid(lon, lat)
    rng = np.random.default_rng(0)
    noise = rng.random((ny, nx))
    core = (((LON > 108) & (LON < 123) & (LAT > 22) & (LAT < 34)) |
            ((LON > 112) & (LON < 118) & (LAT > 24) & (LAT < 30)) |
            ((LON > 122) & (LON < 132) & (LAT > 43) & (LAT < 48)) |
            ((LON > 98)  & (LON < 106) & (LAT > 22) & (LAT < 28)))
    prob = np.where(core, 0.55, 0.02)
    prob[(LON < 100) & (LAT > 33)] = 0.0
    return (noise < prob), (MAP_EXTENT[0], MAP_EXTENT[1], MAP_EXTENT[2], MAP_EXTENT[3])


def load_rice_area_cached():
    """1 km 瓦片重投影较慢：真实数据算一次后缓存到 main_figs_1020/_cache/（改 RICE_* 参数会自动重算）"""
    key = f"rice_area_{RICE_YEARS.start}_{RICE_YEARS.stop - 1}_min{RICE_MIN_YEARS}_{RICE_REPROJ_RES}.npz"
    cache = _fig1_out() / "_cache" / key
    if cache.exists():
        z = np.load(cache)
        print(f"[rice area] 读缓存 {cache}")
        return z["mask"], tuple(z["ext"])
    m, ext = load_rice_area()
    if any((RICE_DIR / p.format(year=y)).exists() for p in RICE_PATTERNS for y in RICE_YEARS):
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, mask=m, ext=np.array(ext, float))
    return m, ext


rice_mask, rice_ext = load_rice_area_cached()


df = pd.read_excel(XLSX).dropna(subset=["LONGITUDE", "LATITUDE"])

pheno = pd.read_excel(PHENO_XLSX)[["ID", "HE"]].copy()
pheno["HE"] = pd.to_datetime(pheno["HE"], errors="coerce")
pheno["he_doy"] = pheno["HE"].dt.dayofyear
df = df.merge(pheno[["ID", "he_doy"]], on="ID", how="left")
df["season"] = np.where(df["he_doy"] >= SPLIT_DOY, "Late", "Single")

pts = gpd.GeoDataFrame(
    df, geometry=[Point(x, y) for x, y in zip(df.LONGITUDE, df.LATITUDE)],
    crs="EPSG:4326")
prov = china[[PROV_COL, "geometry"]].rename(columns={PROV_COL: "prov"})
j = gpd.sjoin(pts, prov, how="left", predicate="within")
miss = j["prov"].isna()
if miss.any():
    nn = gpd.sjoin_nearest(pts[miss.values].to_crs(3857),
                           prov.to_crs(3857), how="left")
    j.loc[miss.values, "prov"] = nn["prov"].values
df["prov"] = j["prov"].values

prov_stats = (df.groupby("prov")
                .agg(n=("ID", "size"),
                     late=("season", lambda s: int((s == "Late").sum())))
                .reset_index())
prov_stats["single"] = prov_stats["n"] - prov_stats["late"]
cent = prov.set_index("prov").geometry.representative_point()
prov_stats["lon"] = prov_stats["prov"].map(lambda p: cent[p].x)
prov_stats["lat"] = prov_stats["prov"].map(lambda p: cent[p].y)

n_tot = int(prov_stats["n"].sum())
n_sites = int(df[["LONGITUDE", "LATITUDE"]].round(3).drop_duplicates().shape[0])
n_late = int(prov_stats["late"].sum())
print(prov_stats.sort_values("n", ascending=False).to_string(index=False))
print(f"total = {n_tot} ; single/early = {n_tot-n_late} ; late = {n_late}")


def add_scale_bar(ax, x_left, cy, length_km=SCALE_KM, lat_ref=LAT_REF,
                  n_seg=4, h=None):
    """Scale bar in figure coords, anchored by its LEFT edge (x_left) so it can
    be left-aligned with the legend. The distance label sits to the RIGHT of
    the bar, vertically centred (parallel), not above it.

    Bar WIDTH is still derived from the map scale (km -> degrees -> figure
    fraction), so it stays geographically correct wherever it is placed.
    """
    fig = ax.figure
    fw_mm, fh_mm = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    if h is None:
        h = 0.011 * FIG_W * 130 / 180 / fh_mm
    gap = 0.006 * FIG_W / fw_mm
    deg = length_km / (111.320 * np.cos(np.deg2rad(lat_ref)))
    x_lo, x_hi = ax.get_xlim(); pos = ax.get_position()
    w = deg / (x_hi - x_lo) * pos.width
    seg = w / n_seg
    for i in range(n_seg):
        fig.patches.append(Rectangle((x_left + i*seg, cy), seg, h,
                           facecolor="black" if i % 2 == 0 else "white",
                           edgecolor="black", linewidth=0.5,
                           transform=fig.transFigure, zorder=10, figure=fig))
    return fig.text(x_left + w + gap, cy + h/2, f"{length_km:,} km",
                    ha="left", va="center", fontsize=FSZ["scale"], zorder=10)


def add_north_arrow(ax, x=0.955, y=0.955, size=0.055):
    x_lo, x_hi = ax.get_xlim(); y_lo, y_hi = ax.get_ylim()
    cx = x_lo + x*(x_hi-x_lo); cy = y_lo + y*(y_hi-y_lo)
    h = size*(y_hi-y_lo); w = h*0.36
    ax.add_patch(plt.Polygon([[cx, cy], [cx-w, cy-h], [cx, cy-h*0.62]], closed=True,
                 facecolor="black", edgecolor="black", linewidth=0.4, zorder=10))
    ax.add_patch(plt.Polygon([[cx, cy], [cx+w, cy-h], [cx, cy-h*0.62]], closed=True,
                 facecolor="white", edgecolor="black", linewidth=0.4, zorder=10))
    ax.text(cx, cy + h*0.10, "N", ha="center", va="bottom",
            fontsize=FSZ["north"], fontweight="bold", zorder=10)


def add_province_pies(ax, stats, rmin=PIE_RMIN, rmax=PIE_RMAX):
    """One pie per province: Single/early vs Late. Radius ~ sqrt(n).

    Size is controlled by PIE_RMIN / PIE_RMAX in the Config block above,
    not here — edit those two numbers to resize every pie at once.
    """
    nmax = stats["n"].max()
    for _, r in stats.iterrows():
        if r["n"] == 0:
            continue
        rad = rmin + (rmax - rmin) * np.sqrt(r["n"] / nmax)
        start = 90.0
        tr = (mtransforms.Affine2D()
              .translate(-r["lon"], -r["lat"])
              .scale(1.0, 1.0 / MAP_ASPECT)
              .translate(r["lon"], r["lat"]) + ax.transData)
        for v, col in zip([r["single"], r["late"]], [PIE_SINGLE, PIE_LATE]):
            if v <= 0:
                continue
            ang = 360.0 * v / r["n"]
            w = Wedge((r["lon"], r["lat"]), rad, start - ang, start,
                      facecolor=col, edgecolor="white", linewidth=0.3,
                      alpha=0.85, zorder=4)
            w.set_transform(tr)
            ax.add_patch(w)
            start -= ang
        ax.add_patch(Ellipse((r["lon"], r["lat"]), width=2*rad,
                             height=2*rad/MAP_ASPECT, fill=False,
                             edgecolor="0.25", linewidth=0.3, zorder=5))


def add_pie_size_key(ax, stats, cx, y_base, ref=(50, 200, 350),
                     rmin=PIE_RMIN, rmax=PIE_RMAX, lead=None, label_gap=None):
    """Nested-circle size key, bottom-aligned (as in a standard bubble legend):
    all reference circles share the same BASE line, so smaller circles nest
    inside larger ones. Value labels are stacked at an EVEN vertical spacing on
    the right, and a leader line runs from the top of each circle across to its
    label — this avoids the labels piling up when the circles are similar in
    size (which happens with small PIE_RMAX).

    cx      : x of the shared vertical centre line (data coords, lon)
    y_base  : y of the shared bottom line (data coords, lat)

    ---- 微调（怎么调）----------------------------------------------------
    位置      : add_pie_size_key(..., cx=<经度>, y_base=<纬度>)
    圆大小    : 跟随 PIE_RMIN / PIE_RMAX，自动与地图上的饼一致
    标注档位  : ref=(50, 200, 350) 改成你想标的样本数
    引线长度  : lead（单位=经纬度），None 时自动=rmax+1.5
    标签行距  : label_gap（单位=纬度），None 时自动，值越大数值分得越开
                >>> 数值挤到一起时，把 label_gap 调大即可 <<<
    """
    nmax = stats["n"].max()
    if lead is None:
        lead = rmax + 1.5
    order = sorted(ref, reverse=True)
    tops = []
    for nref in order:
        rad = rmin + (rmax - rmin) * np.sqrt(nref / nmax)
        ry = rad / MAP_ASPECT
        cy = y_base + ry
        ax.add_patch(Ellipse((cx, cy), width=2*rad, height=2*ry, fill=False,
                             edgecolor="0.35", linewidth=0.5, zorder=9))
        tops.append(cy + ry)
    top_max = max(tops)
    if label_gap is None:
        label_gap = (rmax / MAP_ASPECT) * 0.9
    label_x = cx + lead
    for i, (nref, y_top) in enumerate(zip(order, tops)):
        y_lab = top_max - i * label_gap
        ax.plot([cx, label_x], [y_top, y_lab], color="0.55",
                linewidth=0.3, zorder=9)
        ax.text(label_x + 0.4, y_lab, f"{nref}", va="center", ha="left",
                fontsize=FSZ["key"], zorder=9)


def add_nanhai_inset(ax):
    pos = ax.get_position()
    iw = pos.width * 0.155; ih = pos.height * 0.275
    ix = pos.x1 - iw * 1.06; iy = pos.y0 + pos.height * 0.012
    iax = ax.figure.add_axes([ix, iy, iw, ih])
    iax.imshow(np.ma.masked_where(~rice_mask, rice_mask.astype(float)),
               extent=(rice_ext[0], rice_ext[1], rice_ext[2], rice_ext[3]),
               origin="upper", cmap=ListedColormap([AREA_COLOR]), vmin=0, vmax=1,
               interpolation="nearest", zorder=1)
    china.boundary.plot(ax=iax, linewidth=0.25, color="black", zorder=4)
    nanhai.boundary.plot(ax=iax, linewidth=0.30, color="black", zorder=4)
    jd_line.boundary.plot(ax=iax, linewidth=0.50, color="black", zorder=5)
    iax.set_xlim(INSET_EXTENT[0], INSET_EXTENT[2])
    iax.set_ylim(INSET_EXTENT[1], INSET_EXTENT[3])
    iax.set_xticks([]); iax.set_yticks([]); iax.set_xlabel(""); iax.set_ylabel("")
    for s in iax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    return iax


BIOME_COL={'Tundra':'#C1E1DD','Boreal forest':'#A5C790',
 'Temperate seasonal forest':'#97B669','Temperate rain forest':'#75A95E',
 'Tropical rain forest':'#317A22','Tropical seasonal forest/savanna':'#A09700',
 'Subtropical desert':'#DCBB50','Temperate grassland/desert':'#FCD57A',
 'Woodland/shrubland':'#D16E3F'}
BIOME_ORDER=['Tundra','Boreal forest','Temperate seasonal forest','Temperate rain forest',
 'Tropical rain forest','Tropical seasonal forest/savanna','Subtropical desert',
 'Temperate grassland/desert','Woodland/shrubland']
REGIONS=['华南双季稻区','华中双单季稻区','西南单双季稻区','华北单季稻区','东北早熟单季稻区','西北干燥区单季稻区']
EN={'华南双季稻区':'SCD',
    '华中双单季稻区':'CCDS',
    '西南单双季稻区':'SWP',
    '华北单季稻区':'NCS',
    '东北早熟单季稻区':'NECS',
    '西北干燥区单季稻区':'NWCS'}
RICE_MARKER={'华南双季稻区':'o','华中双单季稻区':'^','西南单双季稻区':'s',
             '华北单季稻区':'D','东北早熟单季稻区':'v','西北干燥区单季稻区':'P'}
RICE_FC='#9E9E9E'; RICE_EC='#3A3A3A'
BIOME_ABBR={'Tundra':'Tu','Boreal forest':'BF','Temperate seasonal forest':'TSF',
            'Temperate rain forest':'TRF','Tropical rain forest':'TrRF',
            'Tropical seasonal forest/savanna':'TSS','Subtropical desert':'SD',
            'Temperate grassland/desert':'TGD','Woodland/shrubland':'WS'}


def add_whittaker_inset(ax, points_csv=DATA / "whittaker_points.csv",
                        biomes_csv=DATA / "whittaker_biomes.csv",
                        rect=(0.060, 0.41, 0.43, 0.58)):
    """Whittaker biome background + sample points (gray, shape=rice region),
    drawn as an inset in the TOP-LEFT blank area (NW China, no samples).
    Style identical to the standalone Whittaker figure."""
    import pandas as _pd
    wb=_pd.read_csv(biomes_csv)

    fig=ax.figure; pos=ax.get_position()
    ix=pos.x0+rect[0]*pos.width; iy=pos.y0+rect[1]*pos.height
    iw=rect[2]*pos.width; ih=rect[3]*pos.height
    pad_l=iw*0.135; pad_b=ih*0.12
    pad_r=iw*0.01;  pad_t=ih*0.01
    bax=fig.add_axes([ix-pad_l, iy-pad_b, iw+pad_l+pad_r, ih+pad_b+pad_t])
    bax.set_facecolor("#DDDDDD"); bax.patch.set_alpha(0.20)
    bax.set_xticks([]); bax.set_yticks([])
    for s in bax.spines.values(): s.set_visible(False)
    iax=fig.add_axes([ix,iy,iw,ih])
    iax.set_facecolor("none")
    for b in BIOME_ORDER:
        sub=wb[wb.biome==b]
        if len(sub)<3: continue
        _poly=sub[['temp_c','precp_cm']].values.astype(float); _poly[:,1]/=100.0
        iax.add_patch(MplPolygon(_poly, closed=True,
                      facecolor=BIOME_COL[b], edgecolor='white', linewidth=0.4, zorder=1))
    pts=_pd.read_csv(points_csv).dropna(subset=['MAT_C','MAP_cm','region'])
    for reg in REGIONS:
        s=pts[pts.region==reg]
        if len(s)==0: continue
        iax.scatter(s.MAT_C, s.MAP_cm/100.0, s=FSZ["wh_dot"], marker=RICE_MARKER[reg],
                    facecolor=RICE_FC, edgecolor=RICE_EC, linewidth=0.3,
                    alpha=0.9, zorder=4)
    iax.set_xlim(-20,31); iax.set_ylim(-0.05,4.6)
    iax.set_xlabel('MAT (°C)', fontsize=FSZ["wh_label"], labelpad=0.5)
    iax.set_ylabel('MAP (\u00d71000 mm)', fontsize=FSZ["wh_label"], labelpad=0.5)
    iax.tick_params(labelsize=FSZ["wh_tick"], length=2.0, width=0.5)
    for s in iax.spines.values(): s.set_linewidth(0.5)
    rice=[Line2D([],[],marker=RICE_MARKER[r],linestyle='none',markersize=FSZ["wh_mk"],
                 markerfacecolor=RICE_FC,markeredgecolor=RICE_EC,markeredgewidth=0.4,
                 label=EN[r]) for r in REGIONS]
    l1=iax.legend(handles=rice, title='Rice region', loc='upper left',
                  fontsize=FSZ["wh_leg"], title_fontsize=FSZ["wh_title"], handletextpad=0.3,
                  labelspacing=0.25, columnspacing=0.7, ncol=2,
                  bbox_to_anchor=(0.005,0.995), alignment='left')
    iax.add_artist(l1)
    fig.canvas.draw()
    bb=l1.get_window_extent().transformed(iax.transAxes.inverted())
    bio=[Patch(facecolor=BIOME_COL[b],edgecolor='white',label=BIOME_ABBR[b]) for b in BIOME_ORDER]
    iax.legend(handles=bio, title='Whittaker biomes', loc='upper left',
               fontsize=FSZ["wh_leg"], title_fontsize=FSZ["wh_title"], handlelength=0.9, handletextpad=0.3,
               labelspacing=0.25, columnspacing=0.7, ncol=2,
               bbox_to_anchor=(0.005, bb.y0-0.01), alignment='left')
    return iax


def draw_whittaker_panel(iax, fs=5.5, dot=7.0, mk=3.0, xmin=-20, ltop=0.955, bfs=None, rfs=None):
    """[1021] Whittaker 图单独成面板（fig1_framework.py 调用），画法同原插图：群系色块（白边）+ 采样点
    （灰色，形状 = 水稻生态区），两组图例放在图内左上（上 = 群系全称，下 = 稻区）。
    MAT 轴左端 = xmin（默认 −20，同原图）；图例顶边位置 = ltop（轴坐标）。"""
    wb = pd.read_csv(DATA / "whittaker_biomes.csv")
    pts = pd.read_csv(DATA / "whittaker_points.csv").dropna(subset=['MAT_C', 'MAP_cm', 'region'])
    print(f"[c] Whittaker 散点 {len(pts)} 条（应与 b 的记录数一致；不一致说明 whittaker_points.csv 缺站点的 MAT/MAP）")
    for b in BIOME_ORDER:
        sub = wb[wb.biome == b]
        if len(sub) < 3:
            continue
        _poly = sub[['temp_c', 'precp_cm']].values.astype(float); _poly[:, 1] /= 100.0
        iax.add_patch(MplPolygon(_poly, closed=True, facecolor=BIOME_COL[b], edgecolor='white',
                                 linewidth=0.4, zorder=1))
    for reg in REGIONS:
        s_ = pts[pts.region == reg]
        if len(s_):
            iax.scatter(s_.MAT_C, s_.MAP_cm / 100.0, s=dot, marker=RICE_MARKER[reg], facecolor=RICE_FC,
                        edgecolor=RICE_EC, linewidth=0.3, alpha=0.9, zorder=4)
    iax.set_xlim(xmin, 31); iax.set_ylim(-0.05, 4.6)
    iax.set_xticks([t for t in range(-40, 31, 10) if t >= xmin])
    iax.set_xlabel('MAT (°C)', fontsize=fs, labelpad=1)
    iax.set_ylabel('MAP (\u00d71000 mm)', fontsize=fs, labelpad=1)
    iax.tick_params(labelsize=fs - 0.3, length=1.5, width=0.5, pad=1)
    for sp in iax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
    bio = [Patch(facecolor=BIOME_COL[b], edgecolor='white', label=b) for b in BIOME_ORDER]
    bfs = fs - 0.5 if bfs is None else bfs
    rfs = fs - 0.5 if rfs is None else rfs
    l1 = iax.legend(handles=bio, title='Whittaker biomes', loc='upper left', fontsize=bfs,
                    title_fontsize=max(bfs, rfs) + 0.2, handlelength=0.9, handletextpad=0.3, labelspacing=0.22, ncol=1,
                    bbox_to_anchor=(0.01, ltop), alignment='left', frameon=False)
    iax.add_artist(l1)
    iax.figure.canvas.draw()
    bb = l1.get_window_extent().transformed(iax.transAxes.inverted())
    rice = [Line2D([], [], marker=RICE_MARKER[r], linestyle='none', markersize=mk, markerfacecolor=RICE_FC,
                   markeredgecolor=RICE_EC, markeredgewidth=0.4, label=EN[r]) for r in REGIONS]
    iax.legend(handles=rice, title='Rice region', loc='upper left', fontsize=rfs,
               title_fontsize=max(bfs, rfs) + 0.2, handlelength=0.9, handletextpad=0.4, labelspacing=0.22,
               columnspacing=0.9, ncol=2,
               bbox_to_anchor=(0.01, bb.y0 - 0.01), alignment='left', frameon=False)
    return iax

def save_pub(fig, stem, dpi=600):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png",  dpi=400, bbox_inches="tight")


if EMBED:
    fig, ax = EXT_FIG, EXT_AX
else:
    fig, ax = plt.subplots(figsize=(FIG_W / 25.4, FIG_W * 130 / 180 / 25.4))

area_cmap = ListedColormap([AREA_COLOR])
ax.imshow(np.ma.masked_where(~rice_mask, rice_mask.astype(float)),
          extent=(rice_ext[0], rice_ext[1], rice_ext[2], rice_ext[3]),
          origin="upper", cmap=area_cmap, vmin=0, vmax=1,
          interpolation="nearest", zorder=1)

china.plot(ax=ax, facecolor="none", edgecolor="0.45", linewidth=0.35, zorder=2)
china.boundary.plot(ax=ax, linewidth=0.35, color="0.25", zorder=3)
ax.set_xlabel(""); ax.set_ylabel("")

ax.scatter(df.LONGITUDE, df.LATITUDE, s=FSZ["dot"], facecolor=DOT_FC, edgecolor="white",
           linewidth=0.25, alpha=1.0, zorder=6)

add_province_pies(ax, prov_stats)

ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
ax.set_aspect(MAP_ASPECT)

ax.set_xticks(np.arange(80, 131, 10))
ax.set_yticks(np.arange(20, 51, 10))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}\u00B0E"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}\u00B0N"))
ax.tick_params(labelsize=FSZ["tick"], direction="out")
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(0.7)



_legend_items = {
    "rice":   Patch(facecolor=AREA_COLOR, edgecolor="0.45", linewidth=0.3,
                    label=f"Rice area ({RICE_YEARS.start}–{RICE_YEARS.stop - 1})"),
    "site":   plt.Line2D([], [], marker="o", linestyle="none", markersize=FSZ["leg_mk"],
                    markerfacecolor=DOT_FC, markeredgecolor="white",
                    markeredgewidth=0.2, label=f"Sampling site (n = {n_sites:,})"),
    "single": Patch(facecolor=PIE_SINGLE, edgecolor="white", linewidth=0.3,
                    label="Single / early"),
    "late":   Patch(facecolor=PIE_LATE, edgecolor="white", linewidth=0.3,
                    label="Late"),
}
LEGEND_ORDER = ["site", "rice", "single", "late"]
handles = [_legend_items[k] for k in LEGEND_ORDER]
ax.apply_aspect()
MM_DEG = ax.get_position().height * fig.get_figheight() * 25.4 / (MAP_EXTENT[3] - MAP_EXTENT[2])
LEGEND_X, LEGEND_Y = 0.015, max(0.2, 16.0 / (MM_DEG * (MAP_EXTENT[3] - MAP_EXTENT[2])))

SCALEBAR_DX, SCALEBAR_DY = 0.0, 0.0
SIZEKEY_DX,  SIZEKEY_DY  = 0.0, 0.0
NARROW_DX,   NARROW_DY   = 0.0, 0.0

TOPLEFT = not getattr(_A, "whittaker", True)
if TOPLEFT:
    mm = lambda v: v / MM_DEG
    _r_ref = PIE_RMIN + (PIE_RMAX - PIE_RMIN) * np.sqrt(350 / prov_stats["n"].max())
    _r_top = max(_r_ref, PIE_RMAX)
    SIZEKEY_LABEL_GAP = max(1.0, mm(2.4))
    TITLE_Y = MAP_EXTENT[3] - mm(4.0)
    KEY_TOP = TITLE_Y - mm(2.5 + 2.4)
    SIZEKEY_YBASE = KEY_TOP - 2 * _r_top / MAP_ASPECT + SIZEKEY_DY
    _lab_low = KEY_TOP - 2 * SIZEKEY_LABEL_GAP
    LEG_TOP = min(SIZEKEY_YBASE, _lab_low - mm(1.2)) - mm(1.8)
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(LEGEND_X, (LEG_TOP - MAP_EXTENT[2]) / (MAP_EXTENT[3] - MAP_EXTENT[2])),
                    fontsize=FSZ["leg"], handlelength=1.0, handletextpad=0.5, labelspacing=0.5,
                    borderaxespad=0.0)
    leg.set_zorder(9)
    fig.canvas.draw()
    _bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    LEGEND_X0, LEGEND_Y0 = _bb.x0, _bb.y0
    inv = ax.transData.inverted()
    lx_data = inv.transform(fig.transFigure.transform((LEGEND_X0, LEGEND_Y0)))[0]
    SIZEKEY_CX = lx_data + 1.5 + SIZEKEY_DX
    add_pie_size_key(ax, prov_stats, cx=SIZEKEY_CX, y_base=SIZEKEY_YBASE, label_gap=SIZEKEY_LABEL_GAP)
    ax.text(SIZEKEY_CX - 1.5, TITLE_Y, "Samples per province", fontsize=FSZ["key"], ha="left", va="top",
            zorder=9)
    _fh = fig.get_figheight() * 25.4
    _sb_h = 0.011 * FIG_W * 130 / 180 / _fh
    _sb_text = add_scale_bar(ax, x_left=LEGEND_X0 + 0.004 + SCALEBAR_DX,
                             cy=LEGEND_Y0 - 2.6 / _fh - _sb_h + SCALEBAR_DY)
    add_north_arrow(ax, x=0.93 + NARROW_DX, y=0.90 + NARROW_DY)
    fig.canvas.draw()
    _tb = _sb_text.get_window_extent().transformed(fig.transFigure.inverted())
    _ap = ax.get_position()
    _x1 = max(_bb.x1, _tb.x1) + 1.2 / (fig.get_figwidth() * 25.4)
    _y0 = LEGEND_Y0 - 2.6 / _fh - _sb_h - 1.2 / _fh
    _ix, _iy = 0.35 / (fig.get_figwidth() * 25.4), 0.35 / _fh
    ax.add_patch(Rectangle((_ap.x0 + _ix, _y0), _x1 - _ap.x0 - _ix, _ap.y1 - _iy - _y0, transform=fig.transFigure,
                           facecolor="white", edgecolor="none", alpha=0.85, zorder=4.5))
else:
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(LEGEND_X, LEGEND_Y),
                    fontsize=FSZ["leg"], handlelength=1.0, handletextpad=0.5,
                    labelspacing=0.5, borderaxespad=0.0)
    fig.canvas.draw()
    _bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    LEGEND_X0 = _bb.x0
    LEGEND_Y0 = _bb.y0
    LEGEND_Y1 = _bb.y1
    LEGEND_ROW = (_bb.y1 - _bb.y0) / len(handles)

    _sb_x = fig.transFigure.inverted().transform(ax.transData.transform((90, 17)))[0]
    _late_y = LEGEND_Y0 + LEGEND_ROW*0.5
    add_scale_bar(ax, x_left=_sb_x + SCALEBAR_DX,
                  cy=_late_y + SCALEBAR_DY)
    add_north_arrow(ax, x=0.93 + NARROW_DX, y=0.90 + NARROW_DY)

    inv = ax.transData.inverted()
    lx_data, ly_top = inv.transform(fig.transFigure.transform((LEGEND_X0, LEGEND_Y1)))
    SIZEKEY_CX    = lx_data + 1.5 + SIZEKEY_DX
    SIZEKEY_YBASE = ly_top + 1.2 + SIZEKEY_DY
    SIZEKEY_LABEL_GAP = max(1.0, 2.4 / MM_DEG)
    add_pie_size_key(ax, prov_stats, cx=SIZEKEY_CX, y_base=SIZEKEY_YBASE,
                     label_gap=SIZEKEY_LABEL_GAP)
    _r_ref = PIE_RMIN + (PIE_RMAX - PIE_RMIN) * np.sqrt(350 / prov_stats["n"].max())
    ax.text(SIZEKEY_CX - 1.5, SIZEKEY_YBASE + 2*max(_r_ref, PIE_RMAX)/MAP_ASPECT + max(1.0, 3.2 / MM_DEG),
            "Samples per province", fontsize=FSZ["key"], ha="left", va="bottom")

add_nanhai_inset(ax)
if not getattr(_A, "whittaker", True):
    pass
elif (DATA / "whittaker_points.csv").exists() and (DATA / "whittaker_biomes.csv").exists():
    add_whittaker_inset(ax)
else:
    print(f"[map] ⚠️ {DATA} 下没有 whittaker_points.csv / whittaker_biomes.csv，本次不画 Whittaker 插图")

if not EMBED:
    save_pub(fig, STEM)
    plt.close(fig)
    print(f"[saved] {STEM}.svg/pdf/tiff/png -> {OUT_DIR}")
