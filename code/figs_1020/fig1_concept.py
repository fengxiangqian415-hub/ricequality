from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

from fig1_workflow import Canvas, FRAME, GREY, INK, ZONE

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ROW_COL = {"high": ZONE["premium"], "low": "#6E6E6E"}
ROW_LAB = {"high": "High\nquality", "low": "Low\nquality"}
TRAY_BG = "#1C232B"
MILLED = dict(fc="#ECE9DE", ec="#BDB7A6")
TRANSL = dict(fc="#A3ABA9", ec="#7F8886")
CHALK = "#F7F6F0"
BROWN = dict(fc="#C9AE78", ec="#8F7748")


def _photo(cv, stem):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        f = ASSETS / f"{stem}{ext}"
        if f.exists():
            return f
    return None


def _place_photo(cv, f, x, y, w, h, clip):
    """照片按 w:h 居中裁切后贴进 (x, y, w, h)，裁成圆角"""
    from PIL import Image
    im = Image.open(f).convert("RGB")
    W_, H_ = im.size
    r = w / h
    if W_ / H_ > r:
        nw = int(H_ * r); x0 = (W_ - nw) // 2; im = im.crop((x0, 0, x0 + nw, H_))
    else:
        nh = int(W_ / r); y0 = (H_ - nh) // 2; im = im.crop((0, y0, W_, y0 + nh))
    tw = int(round(w / 25.4 * 600))
    if im.width > tw:
        im = im.resize((tw, int(round(tw * im.height / im.width))), Image.LANCZOS)
    art = cv.ax.imshow(np.asarray(im), extent=(x, x + w, y, y + h), origin="upper", zorder=4,
                       interpolation="lanczos")
    art.set_clip_path(clip)


def _fragment(cx, cy, L, Wd, ang, rng):
    """碎米：半个椭圆 + 一条略斜的断口"""
    t = np.linspace(-np.pi / 2, np.pi / 2, 16)
    pts = np.c_[L * np.cos(t), Wd / 2 * np.sin(t)]
    pts = np.vstack([pts, [-0.15 * L * rng.uniform(0, 1), -Wd / 2], [0.1 * L * rng.uniform(-1, 1), 0],
                     [-0.15 * L * rng.uniform(0, 1), Wd / 2]])[::-1]
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ R.T + [cx, cy]


def _grains(cv, x, y, w, h, kind, seed, clip):
    """矢量示意的扫描图：kind ∈ whole / broken / translucent / chalky"""
    rng = np.random.default_rng(seed)
    cw, ch = 4.5, 2.75
    nx, ny = int(w // cw), int(h // ch)
    ox, oy = x + (w - nx * cw) / 2 + cw / 2, y + (h - ny * ch) / 2 + ch / 2
    L, Wd = 3.3, 1.42
    for i in range(nx):
        for j in range(ny):
            cx = ox + i * cw + rng.uniform(-0.45, 0.45)
            cy = oy + j * ch + rng.uniform(-0.25, 0.25)
            ang = rng.uniform(-28, 28) + (180 if rng.random() < 0.5 else 0)
            if kind in ("broken", "brown_broken") and rng.random() < 0.55:
                for k in range(1 if rng.random() < 0.6 else 2):
                    Lf = L * rng.uniform(0.28, 0.42)
                    dx = (k - 0.5) * 1.6 * np.cos(np.deg2rad(ang)) if k else 0
                    p = Polygon(_fragment(cx + dx, cy, Lf, Wd, ang + (180 if k else 0), rng), closed=True,
                                lw=0.2, zorder=5, **(BROWN if kind.startswith("brown") else MILLED))
                    cv.ax.add_patch(p); p.set_clip_path(clip)
                continue
            sty = TRANSL if kind in ("translucent", "chalky") else (BROWN if kind.startswith("brown") else MILLED)
            e = Ellipse((cx, cy), L, Wd, angle=ang, lw=0.2, zorder=5, **sty)
            cv.ax.add_patch(e); e.set_clip_path(clip)
            if kind == "chalky" and rng.random() < 0.85:
                f = rng.uniform(0.45, 0.75)
                off = rng.uniform(-0.12, 0.12) * Wd
                a = np.deg2rad(ang)
                c = Ellipse((cx - off * np.sin(a), cy + off * np.cos(a)), L * f, Wd * rng.uniform(0.55, 0.8),
                            angle=ang, fc=CHALK, ec="none", zorder=6)
                cv.ax.add_patch(c); c.set_clip_path(clip)
    cv.text(x + w - 0.6, y + h - 0.4, "placeholder", size=4.2, color="#8B949C", ha="right", italic=True)


COLS = [("brown", ("Whole brown rice", "Broken brown rice")),
        ("milling", ("Whole milled rice", "Broken milled rice")),
        ("appearance", ("Low-chalkiness rice", "High-chalkiness rice"))]
GROUPS = [("Milling quality", "BRR · MRR · HRR", (0, 1)), ("Appearance quality", "CGR · CD", (2,))]
LX, GAP = 12.0, 3.0
TAG, RGAP, HEAD = 2.7, 1.0, 7.0
TOP_PAD, BOT_PAD = 6.4, 1.8
DEFAULT_ASPECT = 92.2 / 29.7


def _aspect():
    """扫描图的宽高比（读 assets 里第一张；照片不裁切，格子比例跟照片一致）"""
    from PIL import Image
    for key, _ in COLS:
        for r in ("high", "low"):
            f = _photo(None, f"scanner_{key}_{r}")
            if f is not None:
                with Image.open(f) as im:
                    return im.width / im.height
    return DEFAULT_ASPECT


def _geom(W):
    img_w = (W - 0.6 - 1.4 - 1.8 - LX - 2 * GAP) / 3
    return img_w, img_w / _aspect()


def concept_height(W=180.0):
    """a 面板（含虚线框和面板字母）的高度，fig1_framework 用来排版"""
    _, img_h = _geom(W)
    return TOP_PAD + BOT_PAD + HEAD + 2 * TAG + RGAP + 2 * img_h


def draw_concept(fig, x0, y0, fig_w, fig_h, W=180.0, H=None, label="a", frame=True):
    H = H or concept_height(W)
    cv = Canvas(fig, x0, y0, W, H, fig_w, fig_h)
    cv.ax.text(0.0, H, label, fontsize=8, fontweight="bold", ha="left", va="top", color=INK)
    FX0, FX1, FY0, FY1 = 0.6, W - 0.6, 0.6, H - 4.2
    if frame:
        cv.rbox(FX0, FY0, FX1 - FX0, FY1 - FY0, fc="none", ec=FRAME, lw=0.9, r=1.6, ls=(0, (4.5, 2.8)), z=1)
    CX0, CY0 = FX0 + 1.2, BOT_PAD
    img_w, img_h = _geom(W)
    y_low = CY0 + HEAD + TAG
    y_high = y_low + img_h + RGAP + TAG
    rows = {"high": y_high, "low": y_low}
    xs = [CX0 + LX + k * (img_w + GAP) for k in range(3)]
    for r, ry in rows.items():
        cv.text(CX0 + LX / 2 - 0.3, ry + img_h / 2, ROW_LAB[r], size=6.3, bold=True, color=ROW_COL[r],
                va="center", within=(CX0 - 0.2, ry, CX0 + LX, ry + img_h))
    kinds = {"brown": ("brown_whole", "brown_broken"), "milling": ("whole", "broken"),
             "appearance": ("translucent", "chalky")}
    for k, ((key, tags), x) in enumerate(zip(COLS, xs)):
        for (r, ry), tag in zip(rows.items(), tags):
            tray = FancyBboxPatch((x, ry), img_w, img_h, boxstyle="round,pad=0,rounding_size=0.8", fc=TRAY_BG,
                                  ec="none", zorder=3)
            cv.ax.add_patch(tray)
            f = _photo(cv, f"scanner_{key}_{r}")
            if f is not None:
                _place_photo(cv, f, x, ry, img_w, img_h, tray)
            else:
                _grains(cv, x, ry, img_w, img_h, kinds[key][0 if r == "high" else 1], 3 + k + (r == "low"), tray)
            cv.text(x + img_w / 2, ry - 1.95, tag, size=5.8, bold=True, color=ROW_COL[r], va="baseline",
                    within=(x - GAP / 2, ry - TAG, x + img_w + GAP / 2, ry))
    yb = CY0 + HEAD - 0.7
    for title, traits, ks in GROUPS:
        gx0, gx1 = xs[ks[0]], xs[ks[-1]] + img_w
        cv.line([gx0, gx0, gx1, gx1], [yb + 0.8, yb, yb, yb + 0.8], color=GREY, lw=0.5)
        mid = (gx0 + gx1) / 2
        box = (gx0 - 1, CY0 - 0.3, gx1 + 1, yb)
        cv.text(mid, yb - 0.4, title, size=6.5, bold=True, within=box)
        cv.text(mid, yb - 3.3, traits, size=5.3, color=GREY, within=box)
    return cv
