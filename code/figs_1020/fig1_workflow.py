import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

PT = 25.4 / 72
INK, FRAME, GREY = "#1A1A1A", "#1E3448", "#666666"
NEUTRAL, NEUTRAL_DARK = "#EEF1F4", "#D9DFE6"
L1 = ("#0D4F8B", "#CFE2F1", "#E8F1F9")
L2 = ("#5B3F8E", "#E3DAF0", "#F1ECF8")
OLIVE = ("#4E6B1A", "#E2EAD1", "#F1F5E8")
RUST = ("#A3461E", "#F5DCCF", "#FBEFE8")
TEAL = ("#1B6A6F", "#D2E7E8", "#E9F3F4")
GOLD = ("#7A5C10", "#EFE4C4", "#F8F2E2")
CV = {"train": "#9CC7E6", "validation": "#F29D52", "test": "#E6E6E6", "test_edge": "#4C9A47",
      "test_fill": "#DDEFD8", "selected": "#5B3F8E", "other": "#C9CFD8", "null": "#B0B0B0"}
ZONE = {"common": "#8C8C8C", "moderate": "#E4A73C", "premium": "#2C9E77"}
SCEN = {"baseline": "#7D7D7D", "ssp126": "#2C6FB0", "ssp585": "#E8833A"}
RMSE_AC = [2.90, 2.96, 2.97, 2.98, 3.11, 3.20, 3.13, 3.25, 3.35, 3.58, 3.52, 3.52, 3.52, 3.52, 3.52, 3.52,
           3.52, 3.55, 3.57, 3.79, 4.97]
NULL_RMSE = 3.79

T_TITLE, T_BODY, T_SMALL = 5.8, 5.5, 5.3


def eff(size):
    """[1108d] 最终字号：保存前 fig_common.scale_fonts 统一 ×FONT_SCALE、且不低于 MIN_FONT_PT。
    原来按放大前的字号排版，放大后（5.1→6.0 pt 等）文字顶出框、互相贴住；现在行高、间距都按最终字号算"""
    try:
        import fig_common as F
        return max(size * F.FONT_SCALE, F.MIN_FONT_PT)
    except Exception:
        return size


def lh(size, n=1):
    """n 行文字的高度（mm，按最终字号）"""
    return n * eff(size) * 1.22 * PT


class Canvas:
    """毫米坐标的画布 + 文字溢出检查（相当于 S17 的 qa_drawio.py）"""

    def __init__(self, fig, x0, y0, w, h, fig_w, fig_h):
        self.ax = fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.fig = fig
        self.qa = []

    def text(self, x, y, s, size=T_BODY, color=INK, ha="center", va="top", bold=False, italic=False,
             within=None, name="", **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, linespacing=1.15,
                         fontweight="bold" if bold else "normal", fontstyle="italic" if italic else "normal",
                         zorder=6, **kw)
        if within is not None:
            self.qa.append((t, within, name or s.split("\n")[0]))
        return t

    def rbox(self, x, y, w, h, fc="white", ec=FRAME, lw=0.6, r=1.0, ls="-", z=2):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                           ls=ls, zorder=z)
        self.ax.add_patch(p)
        return p

    def rect(self, x, y, w, h, fc, ec="none", lw=0.5, z=4):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))

    def line(self, xs, ys, color=INK, lw=0.6, z=5, **kw):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt", **kw)

    def harrow(self, x0, x1, yc, hw=1.0, hh=2.0, hl=1.6):
        """空心横向箭头（S17 的 singleArrow）"""
        xs = x1 - hl
        pts = [(x0, yc - hw), (xs, yc - hw), (xs, yc - hh), (x1, yc), (xs, yc + hh), (xs, yc + hw), (x0, yc + hw)]
        self.ax.add_patch(Polygon(pts, closed=True, fc="white", ec=FRAME, lw=0.7, zorder=7, joinstyle="miter"))

    def varrow(self, xc, y0, y1, hw=0.8, hh=1.6, hl=1.2):
        """空心向下箭头（y0 在上）"""
        ys = y1 + hl
        pts = [(xc - hw, y0), (xc - hw, ys), (xc - hh, ys), (xc, y1), (xc + hh, ys), (xc + hw, ys), (xc + hw, y0)]
        self.ax.add_patch(Polygon(pts, closed=True, fc="white", ec=FRAME, lw=0.6, zorder=7, joinstyle="miter"))

    def darrow2(self, xc, y0, y1, color=FRAME, lw=0.7):
        """竖向双箭头（y0 在上，y1 在下）"""
        self.ax.annotate("", xy=(xc, y1), xytext=(xc, y0),
                         arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw, mutation_scale=5,
                                         shrinkA=0, shrinkB=0), zorder=7)

    def section(self, x, y, w, h, label, col):
        """虚线大框 + 顶边居中的色块标题"""
        self.rbox(x, y, w, h, fc="none", ec=FRAME, lw=0.9, r=1.6, ls=(0, (4.5, 2.8)), z=1)
        self.ax.text(x + w / 2, y + h, label, fontsize=6.7, fontweight="bold", color=INK, ha="center",
                     va="center", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.75", fc=col[1], ec="none"))

    def check(self, tol=0.15, min_size=4.2):
        """每段文字是否落在它所属的框内；超出的自动缩小字号（字体比 Arial 宽时，如服务器只有 DejaVu Sans）"""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()

        def over(t, box):
            x0, y0, x1, y1 = box
            bb = t.get_window_extent(r).transformed(inv)
            return max(x0 - bb.x0, bb.x1 - x1, y0 - bb.y0, bb.y1 - y1)

        shrunk, bad = [], 0
        for t, box, name in self.qa:
            size0 = t.get_fontsize()
            while over(t, box) > tol and t.get_fontsize() - 0.2 >= min_size:
                t.set_fontsize(t.get_fontsize() - 0.2)
            if over(t, box) > tol and t.get_fontweight() in ("bold", 700):
                t.set_fontweight("normal")
                shrunk.append(f"{name}（改常规字重）")
            if t.get_fontsize() < size0:
                shrunk.append(f"{name}（{size0:.1f}→{t.get_fontsize():.1f} pt）")
            o = over(t, box)
            if o > tol:
                bad += 1
                print(f"[Fig. 1 流程 QA] ⚠️ 缩到 {min_size} pt 仍超出框 {o:.2f} mm：{name}")
        if shrunk:
            print(f"[Fig. 1 流程 QA] {len(shrunk)} 处自动缩小字号或改常规字重：" + "；".join(shrunk))
        print(f"[Fig. 1 流程 QA] 文字溢出检查：{bad} 处" + (" ✔" if bad == 0 else ""))
        return bad


class Box:
    """白色圆角框，内容自上而下排（光标 cy）；所有文字自动登记溢出检查"""

    def __init__(self, cv, x, y, w, h, title=None, tcol=INK, pad=1.1, ec=FRAME, lw=0.6):
        self.cv, self.x, self.y, self.w, self.h = cv, x, y, w, h
        cv.rbox(x, y, w, h, ec=ec, lw=lw)
        self.rect_ = (x, y, x + w, y + h)
        self.cx = x + w / 2
        self.cy = y + h - pad
        if title:
            cv.text(self.cx, self.cy, title, size=T_TITLE, color=tcol, bold=True, within=self.rect_)
            self.cy -= lh(T_TITLE, title.count("\n") + 1) + 0.5

    def space(self, d):
        self.cy -= d

    def text(self, s, size=T_BODY, color=INK, italic=False, bold=False, gap=0.5):
        self.cv.text(self.cx, self.cy, s, size=size, color=color, italic=italic, bold=bold, within=self.rect_)
        self.cy -= lh(size, s.count("\n") + 1) + gap
        self.last_gap = gap

    def pill(self, lines, fc=NEUTRAL, size=T_BODY, ec="none", lw=0.6, inset=1.0, gap=0.8, color=INK):
        """lines: 字符串，或 [(文字, 是否粗体), ...]"""
        if isinstance(lines, str):
            lines = [(s, False) for s in lines.split("\n")]
        h = lh(size, len(lines)) + 1.1
        x0, w = self.x + inset, self.w - 2 * inset
        self.cv.rbox(x0, self.cy - h, w, h, fc=fc, ec=ec, lw=lw, r=1.0, z=3)
        y = self.cy - 0.55
        for s, b in lines:
            self.cv.text(self.cx, y, s, size=size, bold=b, color=color, within=(x0, self.cy - h, x0 + w, self.cy))
            y -= lh(size)
        self.cy -= h + gap
        self.last_gap = gap

    def glyph(self, h, draw, gap=0.8):
        """在 (x, cy-h, w, h) 区域画小示意图：draw(cv, x, y, w, h)"""
        draw(self.cv, self.x, self.cy - h, self.w, h)
        self.cy -= h + gap
        self.last_gap = gap

    def done(self, name):
        left = self.cy + getattr(self, "last_gap", 0.0) - self.y
        if left < 0.4:
            print(f"[Fig. 1 流程 QA] ⚠️ 「{name}」内容超出框底 {0.4 - left:.1f} mm")


def g_split(cv, x, y, w, h):
    """训练/测试划分：两个胶囊"""
    bw = (w - 3.2) / 2
    for k, (lab, n, fc, ec) in enumerate([("Train", "908", CV["train"], "none"),
                                          ("Test", "228", CV["test"], CV["test_edge"])]):
        x0 = x + 1.1 + k * (bw + 1.0)
        cv.rbox(x0, y, bw, h, fc=fc, ec=ec, lw=0.7, r=0.9, z=3)
        cv.text(x0 + bw / 2, y + h - 0.45, f"{lab}\nn = {n}", size=T_SMALL, bold=False,
                within=(x0, y, x0 + bw, y + h), name=lab)


def g_cvgrid(cv, x, y, w, h):
    """5 折交叉验证：每行一折，验证折沿反对角线（同 S17a）"""
    n, g = 5, 0.35
    top = lh(4.9) + 0.25
    cw = min((w - 4.5 - g * (n - 1)) / n, 3.0); ch = (h - top - g * (n - 1)) / n
    gx = x + (w - (n * cw + (n - 1) * g)) / 2
    for r in range(n):
        for c in range(n):
            cv.rect(gx + c * (cw + g), y + h - top - (r + 1) * ch - r * g, cw, ch,
                    CV["validation"] if r == n - 1 - c else CV["train"])
    for c in range(n):
        cv.text(gx + c * (cw + g) + cw / 2, y + h, f"{c + 1}", size=4.9, color=GREY)


def g_legend_cv(cv, x, y, w, h):
    items = [(CV["train"], "train"), (CV["validation"], "validation")]
    tw = [len(lab) * eff(5.1) * 0.45 * PT for _, lab in items]
    xx = x + (w - (sum(tw) + 2 * 2.2 + 1.4)) / 2
    for (fc, lab), wl in zip(items, tw):
        cv.rect(xx, y + h / 2 - 0.75, 1.8, 1.5, fc)
        t = cv.text(xx + 2.2, y + h, lab, size=5.1, ha="left", within=(x, y - 0.2, x + w, y + h + 0.3))
        xx += 2.2 + wl + 1.4


def g_rank(cv, x, y, w, h):
    """21 个候选按 CV RMSE 排序：前 5（紫）进入集成，灰 = 其余，深灰 = 零模型（S17a 的 AC 示例）"""
    v = sorted(RMSE_AC)
    pitch = h / len(v); bh = pitch * 0.72
    x0 = x + 3.0; S = (w - 6.0) / 5.0
    top5 = set(range(5))
    null_i = v.index(NULL_RMSE)
    for i, r in enumerate(v):
        col = CV["selected"] if i in top5 else (CV["null"] if i == null_i else CV["other"])
        cv.rect(x0, y + h - (i + 1) * pitch + (pitch - bh) / 2, r * S, bh, col, z=4)
    yb = y + h - 5 * pitch
    xe = x0 + v[4] * S + 0.6
    cv.line([xe, xe + 0.6, xe + 0.6, xe], [y + h, y + h, yb, yb], color=L2[0], lw=0.5)


def g_members(cv, x, y, w, h):
    """5 GCM × 5 GGCM = 25 个成员"""
    s, g = 1.45, 0.35
    gw = 5 * s + 4 * g
    lab_l = 8.0
    gx = x + (w - (lab_l + gw)) / 2 + lab_l
    gy = y + h - lh(T_SMALL) - 0.3 - gw
    for r in range(5):
        for c in range(5):
            cv.rect(gx + c * (s + g), gy + (4 - r) * (s + g), s, s, OLIVE[0] if (r, c) == (2, 3) else OLIVE[1])
    box = (x, y, x + w, y + h)
    cv.text(gx + gw / 2, y + h, "5 GCMs", size=T_SMALL, within=box)
    cv.text(gx - 0.6, gy + gw / 2, "5\nGGCMs", size=T_SMALL, ha="right", va="center", within=box)
    cv.text(x + w / 2, gy - 0.4, "= 25 members", size=T_SMALL, bold=True, within=box)


def g_window(cv, x, y, w, h):
    """灌浆窗口示意：日气象曲线，抽穗（HE）到成熟（MA）阴影"""
    gx, gw = x + 2.0, w - 4.0
    gy = y + lh(5.1) + 0.45
    gh = h - (gy - y) - 0.4
    he, ma = gx + 0.52 * gw, gx + 0.80 * gw
    cv.rect(he, gy, ma - he, gh, OLIVE[1], z=2)
    t = np.linspace(0, 1, 60)
    k = np.array([0, .15, .32, .48, .62, .78, .92, 1.0]); vv = np.array([.18, .42, .74, .92, .88, .62, .30, .14])
    cv.line(gx + gw * t, gy + 0.6 + (gh - 1.4) * np.interp(t, k, vv), color=OLIVE[0], lw=0.9)
    cv.line([gx, gx + gw], [gy, gy], color=INK, lw=0.5)
    box = (x, y, x + w, y + h)
    cv.text(he, gy - 0.25, "HE", size=5.1, bold=True, within=box)
    cv.text(ma, gy - 0.25, "MA", size=5.1, bold=True, within=box)


def g_co2(cv, x, y, w, h):
    """CO₂ 时段均值（ppm），颜色 = 情景（与 Fig. 2–4 一致）"""
    rows = [("1981–2010", 362.43, SCEN["baseline"]), ("SSP1-2.6", 470.30, SCEN["ssp126"]),
            ("SSP5-8.5", 611.74, SCEN["ssp585"])]
    box = (x, y, x + w, y + h)
    lab_w = 10.4
    S = (w - lab_w - 5.0) / 620.0
    pitch = h / 3
    for k, (lab, v, col) in enumerate(rows):
        yc = y + h - (k + 0.5) * pitch
        cv.text(x + lab_w, yc, lab, size=5.1, ha="right", va="center", within=box)
        cv.rect(x + lab_w + 0.5, yc - 0.6, v * S, 1.2, col)
        cv.text(x + lab_w + 0.9 + v * S, yc, f"{v:.0f}", size=5.1, ha="left", va="center", within=box)


def g_ramps(cv, x, y, w, h):
    """可取度函数 d：p5–p95 之间线性，HRR 越高越好，CD、PC 越低越好"""
    rows = [("HRR", True, "23.4–73.8%", "Milling"), ("CD", False, "0.3–44.6%", "Appearance"),
            ("PC", False, "4.5–12.2%", "Cooking")]
    cw = w / 3
    for k, (tr, up, anc, dim) in enumerate(rows):
        cx = x + k * cw + cw / 2
        box = (x + k * cw, y, x + (k + 1) * cw, y + h)
        cv.text(cx, y + h, dim, size=5.5, bold=True, within=box)
        ax_, aw = cx - 5.0, 10.0
        ay = y + lh(4.8, 2) + 0.6
        ah = min(3.6, y + h - lh(5.5) - 0.35 - ay)
        cv.line([ax_, ax_, ax_ + aw], [ay + ah, ay, ay], color="#9A9A9A", lw=0.5)
        lo, hi = ay + 0.3, ay + ah - 0.3
        xs = [ax_, ax_ + 2.2, ax_ + aw - 2.2, ax_ + aw]
        cv.line(xs, [lo, lo, hi, hi] if up else [hi, hi, lo, lo], color=TEAL[0], lw=1.0)
        cv.text(cx, ay - 0.35, f"{tr}\n{anc}", size=4.8, color=GREY, within=box)


def g_zones(cv, x, y, w, h):
    """分级：common | moderate | premium，阈值 0.60、0.66（示意比例 = 两季合并基准面积占比 20/42/38）"""
    bx, bw = x + 0.8, w - 1.6
    segs = [("common", 0.20), ("moderate", 0.42), ("premium", 0.38)]
    xx = bx
    bh = max(2.6, lh(5.1) + 0.2)
    by = y + h - bh
    for name, f in segs:
        cv.rect(xx, by, bw * f, bh, ZONE[name])
        cv.text(xx + bw * f / 2, by + bh / 2, name, size=5.1, color="white", bold=True, va="center",
                within=(xx + 0.25, by - 0.3, xx + bw * f - 0.25, by + bh + 0.3))
        xx += bw * f
    for v, f in (("0.60", 0.20), ("0.66", 0.62)):
        cv.line([bx + bw * f] * 2, [by - 0.6, by], color=INK, lw=0.5)
        cv.text(bx + bw * f, by - 0.6, v, size=5.1)
    cv.text(bx + bw + 0.2, by - 0.6, "Q", size=5.1, ha="right", italic=True)


ICON = {"temp": "#C0504D", "rad": "#E0A12B", "water": "#3C7FB8", "co2": "#5E5E5E"}


def i_thermo(cv, cx, cy, col):
    """温度计：竖管 + 底部圆球"""
    cv.rbox(cx - 0.5, cy - 1.3, 1.0, 3.3, fc="white", ec=col, lw=0.5, r=0.5, z=6)
    cv.rect(cx - 0.22, cy - 1.3, 0.44, 2.1, col, z=7)
    cv.ax.add_patch(Circle((cx, cy - 1.45), 0.8, fc=col, ec=col, lw=0.5, zorder=7))


def i_sun(cv, cx, cy, col):
    """太阳：圆 + 8 条光线"""
    cv.ax.add_patch(Circle((cx, cy), 0.9, fc=col, ec="none", zorder=7))
    for a in np.deg2rad(np.arange(0, 360, 45)):
        cv.line([cx + 1.3 * np.cos(a), cx + 1.85 * np.cos(a)], [cy + 1.3 * np.sin(a), cy + 1.85 * np.sin(a)],
                color=col, lw=0.55, z=7)


def i_drop(cv, cx, cy, col, r=1.0):
    """水滴：下半圆弧 + 顶点"""
    t = np.deg2rad(np.linspace(150, 390, 40))
    pts = np.c_[cx + r * np.cos(t), cy - 0.45 * r + r * np.sin(t)]
    pts = np.vstack([pts, [cx, cy - 0.45 * r + 2.0 * r]])
    cv.ax.add_patch(Polygon(pts, closed=True, fc=col, ec="none", zorder=7))


def i_co2(cv, cx, cy, col):
    """CO₂ 分子：O=C=O"""
    for dx, rr, fc in ((-1.35, 0.55, "white"), (0.0, 0.62, col), (1.35, 0.55, "white")):
        cv.ax.add_patch(Circle((cx + dx, cy), rr, fc=fc, ec=col, lw=0.5, zorder=8))
    for dx in (-0.8, 0.8):
        for dy in (-0.18, 0.18):
            cv.line([cx + dx - 0.25, cx + dx + 0.25], [cy + dy, cy + dy], color=col, lw=0.4, z=7)


def g_predictors(cv, x, y, w, h):
    """7 个预测因子竖排：温度、辐射、水分三组（图标 + 变量名）+ CO₂"""
    groups = [(i_thermo, ICON["temp"], ("Tavg", "Tdif")), (i_sun, ICON["rad"], ("CTSR", "TSR")),
              (i_drop, ICON["water"], ("Rain", "RHavg"))]
    box = (x, y, x + w, y + h)
    x0, pw = x + 1.0, w - 2.0
    PS = 5.4
    ch = lh(PS) + 0.5
    gh, gg = (h - ch - 0.9 - 0.5) / 3, 0.5
    ix, tx = x0 + 3.9, x0 + 7.3
    yy = y + h
    for icon, col, names in groups:
        cv.rbox(x0, yy - gh, pw, gh, fc=OLIVE[2], ec="none", r=0.8, z=3)
        icon(cv, ix, yy - gh / 2 + 0.05, col)
        for k, nm in enumerate(names):
            cv.text(tx, yy - gh / 2 + (0.5 - k) * gh * 0.46, nm, size=PS, ha="left", va="center", within=box)
        yy -= gh + gg
    yy -= 0.4
    cv.rbox(x0, yy - ch, pw, ch, fc=NEUTRAL_DARK, ec="none", r=0.8, z=3)
    i_co2(cv, ix, yy - ch / 2, ICON["co2"])
    cv.text(tx, yy - ch / 2, "CO₂", size=PS, bold=True, ha="left", va="center", within=box)


MILLED = dict(fc="#ECE7D8", ec="#8E8A80")
PADDY = dict(fc="#D9A93F", ec="#8F6A1E")


def grain(cv, cx, cy, w=1.45, h=3.1, fc="#ECE7D8", ec="#8E8A80", lw=0.4, angle=0.0, z=6):
    cv.ax.add_patch(Ellipse((cx, cy), w, h, angle=angle, fc=fc, ec=ec, lw=lw, zorder=z))


def small_arrow(cv, x0, x1, y):
    cv.ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=7,
                   arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.45, mutation_scale=3.2, shrinkA=0, shrinkB=0))



def _bez(p0, p1, p2, n=30):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2, t[:, 0]


def _blade(p0, p1, p2, w0):
    """叶片：沿二次贝塞尔中线的细长尖叶"""
    c, t = _bez(np.array(p0, float), np.array(p1, float), np.array(p2, float), 40)
    d = np.gradient(c, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[-d[:, 1], d[:, 0]]
    wid = w0 * (0.55 + 0.45 * np.sin(np.pi * np.clip(t * 1.6, 0, 1))) * (1 - t) ** 0.7
    return np.vstack([c + nrm * wid[:, None] / 2, (c - nrm * wid[:, None] / 2)[::-1]])


def rice_plant(ax, px, py, H, seed=0, greens=("#2E7D4F", "#3E9558", "#5BAE68", "#7CC27A"), spread=1.0,
               n_leaf=14, n_pan=6, z=5):
    """水稻植株模式图（矢量）：叶丛 + 下垂稻穗 + 须根；(px, py) = 茎基，H = 茎基以上高度（mm）"""
    rng = np.random.default_rng(seed)
    W = 0.45 * H * spread
    for k in range(11):
        a = np.deg2rad(-90 + (k - 5) * 11 + rng.uniform(-4, 4))
        L = 0.2 * H * rng.uniform(0.65, 1.0)
        bend = rng.uniform(-0.04, 0.04) * H
        c, _ = _bez(np.array([px, py]), np.array([px + 0.5 * L * np.cos(a) + bend, py + 0.5 * L * np.sin(a)]),
                    np.array([px + L * np.cos(a), py + L * np.sin(a)]), 10)
        ax.plot(c[:, 0], c[:, 1], color="#7A5A43", lw=0.22, zorder=z, solid_capstyle="round")
        if k % 3 == 1:
            m = c[5]
            ax.plot([m[0], m[0] + 0.06 * H * np.sign(np.cos(a) + 1e-9)], [m[1], m[1] - 0.05 * H],
                    color="#7A5A43", lw=0.16, zorder=z)

    def leaf(f, zz, L):
        tipx = px + f * W * 1.05
        tipy = py + L * (1.0 - 0.5 * abs(f) ** 1.4)
        ctrl = (px + f * W * 0.18, py + L * (0.78 + 0.25 * abs(f)))
        base = (px + rng.uniform(-0.035, 0.035) * H, py + 0.03 * H)
        col = greens[rng.integers(len(greens))]
        ax.add_patch(Polygon(_blade(base, ctrl, (tipx, tipy), 0.05 * H), closed=True, fc=col, ec="none", zorder=zz))

    fs = np.linspace(-1, 1, n_leaf) + rng.uniform(-0.07, 0.07, n_leaf)
    for f in fs:
        leaf(float(np.clip(f, -1.05, 1.05)), z, H * rng.uniform(0.75, 1.05) * (1 - 0.2 * abs(f)))
    pf = np.linspace(-0.85, 0.85, n_pan) + rng.uniform(-0.06, 0.06, n_pan)
    for k, f in enumerate(pf):
        top = np.array([px + f * W * 0.45, py + H * (0.92 - 0.1 * abs(f)) + rng.uniform(-0.04, 0.04) * H])
        c, _ = _bez(np.array([px, py]), np.array([px + f * W * 0.08, py + 0.55 * H]), top, 20)
        ax.plot(c[:, 0], c[:, 1], color="#9DBA5A", lw=0.3, zorder=z + 1)
        side = np.sign(f) if abs(f) > 0.12 else (1 if k % 2 else -1)
        P = 0.36 * H * rng.uniform(0.85, 1.05)
        p0 = top
        p1 = top + np.array([side * 0.32 * P, 0.22 * P])
        p2 = top + np.array([side * 0.55 * P, -0.75 * P])
        pc, tt = _bez(p0, p1, p2, 40)
        ax.plot(pc[:, 0], pc[:, 1], color="#B08A2E", lw=0.25, zorder=z + 2)
        d = np.gradient(pc, axis=0)
        for j in range(6, 40, 2):
            ang = np.degrees(np.arctan2(d[j, 1], d[j, 0]))
            sgn = 1 if (j // 2) % 2 else -1
            off = 0.022 * H
            na = np.deg2rad(ang + 90)
            ax.add_patch(Ellipse((pc[j, 0] + sgn * off * np.cos(na), pc[j, 1] + sgn * off * np.sin(na)),
                                 0.044 * H, 0.02 * H, angle=ang + sgn * 30,
                                 fc="#D9A930", ec="#A67C1A", lw=0.12, zorder=z + 3))
    for f in (-0.35, 0.15, 0.55):
        leaf(f + rng.uniform(-0.05, 0.05), z + 4, H * rng.uniform(0.55, 0.75))


from pathlib import Path as _P
_PLANTS = None


def draw_plant_asset(cv, key, cx, ybot, height, z=6):
    """把矢量化的植株画到 (cx 居中, ybot 底边, 高 height mm)；返回 (左, 右) 边界。读不到素材时退回程序绘制"""
    global _PLANTS
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    if _PLANTS is None:
        f = _P(__file__).resolve().parent / "assets" / "rice_plants.npz"
        _PLANTS = dict(np.load(f)) if f.exists() else {}
        if not _PLANTS:
            print(f"[Fig. 1 流程] ⚠️ 找不到 {f}，水稻植株改用程序绘制")
    if f"{key}_v" not in _PLANTS:
        rice_plant(cv.ax, cx, ybot + 0.2 * height / 1.2, height / 1.2)
        return cx - 0.5 * height * 0.6, cx + 0.5 * height * 0.6
    w, h = _PLANTS[f"{key}_wh"]
    sc = height / h
    tr = Affine2D().scale(sc, -sc).translate(cx - w * sc / 2, ybot + height) + cv.ax.transData
    V, C, off, col = (_PLANTS[f"{key}_{k}"] for k in ("v", "c", "off", "col"))
    for i in range(len(off) - 1):
        cv.ax.add_patch(PathPatch(MPath(V[off[i]:off[i + 1]], C[off[i]:off[i + 1]]), fc=str(col[i]), ec="none",
                                  lw=0, transform=tr, zorder=z))
    return cx - w * sc / 2, cx + w * sc / 2


def g_traits(cv, x, y, w, h):
    """7 个性状一排：加工（稻谷 BRR、精米含碎米 MRR、整精米 HRR）│ 垩白（CGR：几粒中有垩白粒；
    CD：单粒腹部白斑）│ 成分（PC：氨基酸串；AC：葡萄糖单元直链）"""
    box = (x, y - 0.2, x + w, y + h)
    xs = [x + w * (k + 0.5) / 7 for k in range(7)]
    LAB = lh(4.8) + 0.2
    yc = y + LAB + 0.15 + (h - LAB - 0.15) / 2
    gw, gh_ = 1.4, 3.2
    grain(cv, xs[0], yc, w=gw, h=gh_, **PADDY)
    cv.line([xs[0], xs[0]], [yc - 1.1, yc + 1.1], color=PADDY["ec"], lw=0.3, z=7)
    cv.line([xs[0], xs[0] + 0.12], [yc + 1.4, yc + 1.8], color=PADDY["ec"], lw=0.5, z=7)
    grain(cv, xs[1] - 0.35, yc, w=gw, h=gh_, **MILLED)
    grain(cv, xs[1] + 0.75, yc + 0.6, w=0.7, h=1.05, angle=-28, **MILLED)
    grain(cv, xs[1] + 0.7, yc - 0.75, w=0.65, h=0.9, angle=22, **MILLED)
    grain(cv, xs[2], yc, w=gw, h=gh_, **MILLED)
    for k, dx in enumerate((-1.05, 0.0, 1.05)):
        grain(cv, xs[3] + dx, yc, w=0.8, h=2.2, **MILLED)
        if k == 1:
            cv.ax.add_patch(Ellipse((xs[3] + dx + 0.1, yc), 0.45, 1.1, fc="white", ec="#B9B4A8", lw=0.25, zorder=7))
    grain(cv, xs[4], yc, w=gw, h=gh_, **MILLED)
    cv.ax.add_patch(Ellipse((xs[4] + 0.18, yc - 0.05), 0.72, 1.5, fc="white", ec="#B9B4A8", lw=0.25, zorder=7))
    t = np.linspace(-1.25, 1.25, 5)
    px, py = xs[5] + t, yc + 0.5 * np.sin(t * 2.2)
    cv.line(px, py, color="#6B5BA8", lw=0.5, z=6)
    for k, (a_, b_) in enumerate(zip(px, py)):
        cv.ax.add_patch(Circle((a_, b_), 0.36, fc=("#8C7FC7", "#C9C1EA")[k % 2], ec="#6B5BA8", lw=0.3, zorder=7))
    r6 = 0.5
    hx = [xs[6] - 1.25, xs[6], xs[6] + 1.25]
    ang = np.deg2rad(np.arange(30, 390, 60))
    for k, hc in enumerate(hx):
        cv.ax.add_patch(Polygon(np.c_[hc + r6 * np.cos(ang), yc + r6 * np.sin(ang)], closed=True,
                                fc="#E1EBF4", ec="#4F6D8A", lw=0.4, zorder=7))
        if k:
            cv.line([hx[k - 1] + r6 * 0.87, hc - r6 * 0.87], [yc - 0.1, yc - 0.1], color="#4F6D8A", lw=0.4, z=6)
    for xx, lab in zip(xs, ("BRR", "MRR", "HRR", "CGR", "CD", "PC", "AC")):
        cv.text(xx, y + LAB, lab, size=4.8, color=GREY, within=box)


def g_outputs(cv, x, y, w, h):
    """投影输出：上排 = 7 个性状的图示；下排三格（图 + 短标签）= 30 年均值 × 25 成员 │ 2 个稻作季 │ 相对变化 %"""
    r1h = 7.2
    g_traits(cv, x + 0.2, y + h - r1h, w - 0.4, r1h)
    cv.line([x + 1.0, x + w - 1.0], [y + h - r1h - 0.55] * 2, color="#D5D9DE", lw=0.5, z=3)
    h2 = h - r1h - 1.1
    cw = w / 3
    xe = x + cw * np.arange(4)
    cap_h = lh(T_SMALL, 2)
    gy0, gh = y + cap_h + 1.0, h2 - cap_h - 1.0
    caps = ["30-yr mean\n25 members", "2 rice\nseasons", "Change (%)\nvs baseline"]
    for k, cap in enumerate(caps):
        cv.text((xe[k] + xe[k + 1]) / 2, y + cap_h + 0.2, cap, size=T_SMALL, within=(xe[k], y - 0.3, xe[k + 1], y + h2))
        if k:
            cv.line([xe[k]] * 2, [y + 0.6, y + h2 - 0.3], color="#D5D9DE", lw=0.5, z=3)
    cx = (xe[0] + xe[1]) / 2
    fw_, fh_ = 8.0, max(3.6, gh - 1.5 - lh(4.9))
    fx0, fy0 = cx - fw_ / 2 - 0.6, gy0 + 0.3
    for o in (1.0, 0.5):
        cv.rbox(fx0 + o, fy0 + o, fw_, fh_, fc="white", ec="#B8BEC6", lw=0.4, r=0.5, z=4)
    cv.rbox(fx0, fy0, fw_, fh_, fc="white", ec="#8A929C", lw=0.5, r=0.5, z=5)
    rng = np.random.default_rng(3)
    v = 0.45 + 0.25 * rng.standard_normal(30).clip(-1.6, 1.6) / 1.6
    bw = (fw_ - 1.4) / 30
    for i, vi in enumerate(v):
        cv.rect(fx0 + 0.7 + i * bw, fy0 + 0.5, bw * 0.72, vi * (fh_ - 1.1), "#E7B49B", z=6)
    my = fy0 + 0.5 + v.mean() * (fh_ - 1.1)
    cv.line([fx0 + 0.5, fx0 + fw_ - 0.5], [my, my], color=RUST[0], lw=0.9, z=7)
    cv.text(fx0 + fw_ + 1.0, fy0 + fh_ + 1.1, "×25", size=4.9, bold=True, color=GREY, ha="right", va="bottom",
            within=(xe[0], y, xe[1], y + h2 + 0.3))
    cx = (xe[1] + xe[2]) / 2
    PH = gh - 0.1
    for dx, key, lab in ((-3.2, "SE", "S/E"), (3.0, "L", "L")):
        x0_, x1_ = draw_plant_asset(cv, key, cx + dx, gy0 + 0.15, PH)
        cv.text(x0_ + 0.8 * (x1_ - x0_), gy0 + 0.15 + 0.1 * PH, lab, size=4.8, color=GREY, ha="left",
                va="center", within=(xe[1], y, xe[2], y + h2))
    cx = (xe[2] + xe[3]) / 2 - 1.2
    base_y = gy0 + 0.3
    f_h = gh - 0.6; b_h, bw2 = 0.64 * f_h, 2.4
    cv.line([cx - 4.2, cx + 4.0], [base_y, base_y], color=INK, lw=0.5, z=6)
    cv.rect(cx - 3.4, base_y, bw2, b_h, SCEN["baseline"], z=5)
    cv.rect(cx - 0.4, base_y, bw2, f_h, RUST[0], z=5)
    cv.line([cx - 3.4, cx + 2.7], [base_y + b_h] * 2, color=GREY, lw=0.4, z=6, ls=(0, (1.5, 1)))
    cv.darrow2(cx + 2.9, base_y + f_h, base_y + b_h, color=RUST[0], lw=0.6)
    cv.text(cx + 3.4, base_y + (b_h + f_h) / 2, "Δ%", size=5.6, bold=True, color=RUST[0], ha="left", va="center",
            within=(xe[2], y, xe[3] + 0.3, y + h2))


def g_scen(cv, x, y, w, h, keys, filled, labels):
    """情景色块：实心 = 情景 CO₂，空心 = CO₂ 固定（与 Fig. 2、3 相同编码）"""
    n = len(keys)
    widths = [2.0 + 0.8 + len(l) * eff(5.1) * 0.53 * PT for l in labels]
    tot = sum(widths) + 1.8 * (n - 1)
    xx = x + (w - tot) / 2
    for k, l, wd in zip(keys, labels, widths):
        col = SCEN[k]
        cv.rect(xx, y + 0.3, 2.0, 1.6, col if filled else "white", ec=col, lw=0.9)
        cv.text(xx + 2.8, y + h, l, size=5.1, ha="left", within=(x, y - 0.3, x + w, y + h + 0.3))
        xx += wd + 1.8


def draw_workflow(fig, x0, y0, fig_w, fig_h, W=180.0, H=67.6, label="d", skill_ref="b", skill_txt=None):
    """[1021] 精简版：保留各段框线、小标题、示意图（图例式元素）和关键数字，删去说明性文字"""
    cv = Canvas(fig, x0, y0, W, H, fig_w, fig_h)
    TOP = H - 4.2
    cv.ax.text(0.0, H, label, fontsize=8, fontweight="bold", ha="left", va="top", color=INK)

    X1, W1 = 0.6, 76.0
    Y2 = 0.6
    H2 = (TOP - Y2 - 4.4) * 0.524
    Y1, H1 = Y2 + H2 + 4.4, TOP - (Y2 + H2 + 4.4)
    cv.section(X1, Y1, W1, H1, "Model training", L1)
    by, bh = Y1 + 1.6, H1 - 4.6
    b1 = Box(cv, X1 + 2.0, by, 21.4, bh, "Observations", L1[0])
    b1.pill([("1,136 records", True), ("92 sites", False)], fc=NEUTRAL_DARK)
    b1.text("FACE/OTC +\nfield trials", size=5.1)
    b1.glyph(lh(T_SMALL, 2) + 0.7, g_split)
    b1.done("Observations")
    cv.harrow(X1 + 23.7, X1 + 26.5, by + bh / 2)
    b2 = Box(cv, X1 + 26.8, by, 22.2, bh, "Layer 1: screening", L1[0])
    b2.pill("21 algorithms", fc=NEUTRAL, size=T_SMALL)
    b2.glyph(10.4, g_cvgrid, gap=0.5)
    b2.glyph(lh(5.1) + 0.1, g_legend_cv, gap=0.5)
    b2.done("Layer 1")
    cv.harrow(X1 + 49.3, X1 + 52.1, by + bh / 2)
    b3 = Box(cv, X1 + 52.4, by, 21.6, bh, "Layer 2: ensemble", L2[0])
    b3.glyph(7.6, g_rank, gap=0.7)
    b3.text("top 5,\n1/RMSE weights", size=5.1, color=L2[0])
    b3.pill(skill_txt or (f"test R² ({skill_ref})" if skill_ref else "held-out test R²"), fc=CV["test_fill"],
            ec=CV["test_edge"], size=T_SMALL, inset=0.9)
    b3.done("Layer 2")

    cv.section(X1, Y2, W1, H2, "Climate inputs — 25 members", OLIVE)
    by, bh = Y2 + 1.6, H2 - 4.6
    c1 = Box(cv, X1 + 2.0, by, 26.0, bh, "Climate × crop models", OLIVE[0])
    c1.text("ISIMIP3b · 0.5° · daily", size=T_SMALL)
    c1.glyph(2 * lh(T_SMALL) + 9.6, g_members)
    c1.pill("SSP1-2.6 · SSP5-8.5", fc=NEUTRAL, size=T_SMALL, inset=0.9)
    c1.done("Climate × crop models")
    cv.harrow(X1 + 28.3, X1 + 31.1, by + bh / 2)
    c2 = Box(cv, X1 + 31.4, by, 19.6, bh, "Grain-filling\nwindow", OLIVE[0])
    c2.glyph(9.6, g_window)
    c2.pill("corrected to 1-km\nphenology", fc=OLIVE[2], size=T_SMALL, inset=0.9)
    c2.done("Grain-filling window")
    cv.harrow(X1 + 51.3, X1 + 54.1, by + bh / 2)
    c3 = Box(cv, X1 + 54.4, by, 19.6, bh, "Predictors", OLIVE[0])
    c3.glyph(c3.cy - by - 0.9, g_predictors, gap=0.0)
    c3.done("Predictors")

    X3, W3 = 82.5, 42.0
    cv.harrow(X1 + W1 + 0.5, X3 - 0.6, Y1 + H1 / 2)
    cv.harrow(X1 + W1 + 0.5, X3 - 0.6, Y2 + H2 / 2)

    Y3, H3 = 0.6, TOP - 0.6
    cv.section(X3, Y3, W3, H3, "Projection — paired CO₂ runs", RUST)
    ix, iw = X3 + 2.0, W3 - 4.0
    runs = [("Baseline · 1981–2010", ["baseline"], True, ["historical"]),
            ("Future · 2041–2070", ["ssp126", "ssp585"], True, ["SSP1-2.6", "SSP5-8.5"]),
            ("Fixed-CO₂ run", ["ssp126", "ssp585"], False, ["SSP1-2.6", "SSP5-8.5"])]
    sh = lh(5.1) + 0.1
    hb, gap = 1.1 + lh(T_TITLE) + 0.5 + sh + 0.7, 4.3
    yy = Y3 + H3 - 3.6
    edges = []
    for ttl, keys, filled, labs in runs:
        b = Box(cv, ix, yy - hb, iw, hb, ttl, RUST[0])
        b.glyph(sh, lambda c, x, y, w, hh, k=keys, f=filled, l=labs: g_scen(c, x, y, w, hh, k, f, l), gap=0.6)
        b.done(ttl)
        edges.append((yy, yy - hb))
        yy -= hb + gap
    xa = ix + iw * 0.30
    for (top_a, bot_a), (top_b, _), lab in zip(edges[:-1], edges[1:], ["Change", "CO₂ effect"]):
        cv.darrow2(xa, bot_a - 0.3, top_b + 0.3)
        cv.text(xa + 1.6, (bot_a + top_b) / 2, lab, size=T_SMALL, ha="left", va="center", bold=True,
                color=RUST[0], within=(ix, top_b, ix + iw, bot_a))
    yy += gap - 2.2
    pb = Box(cv, ix, Y3 + 2.0, iw, yy - Y3 - 2.0, None)
    pb.space(0.6)
    pb.glyph(pb.cy - pb.y - 1.4, g_outputs, gap=0.0)
    pb.done("Projection outputs")

    X4 = X3 + W3 + 6.0
    W4 = W - X4 - 0.6
    cv.harrow(X3 + W3 + 0.5, X4 - 0.6, Y3 + H3 / 2)

    cv.section(X4, Y3, W4, H3, "Zoning, value and attribution", TEAL)
    ix, iw = X4 + 2.0, W4 - 4.0
    yy = Y3 + H3 - 3.6
    ttl = 1.1 + lh(T_TITLE) + 0.5
    ramp_h = lh(5.5) + 0.35 + 3.2 + lh(4.8, 2) + 0.6
    zone_h = max(2.6, lh(5.1) + 0.2) + 0.6 + lh(5.1) + 0.1
    spec = [ttl + lh(T_BODY) + 0.5 + ramp_h + 0.9, ttl + zone_h + 1.0, ttl + 2 * (lh(T_BODY) + 0.5) + 0.4]
    d1 = Box(cv, ix, yy - spec[0], iw, spec[0], "Composite quality index $Q$", TEAL[0])
    d1.text("$Q$ = (d$_{\\mathrm{HRR}}$ + d$_{\\mathrm{CD}}$ + d$_{\\mathrm{PC}}$) / 3", size=T_BODY)
    d1.glyph(ramp_h, g_ramps, gap=0.5)
    d1.done("Q")
    yy -= spec[0]
    cv.varrow(ix + iw / 2, yy - 0.2, yy - 2.2)
    yy -= 2.4
    d2 = Box(cv, ix, yy - spec[1], iw, spec[1], "Quality zones", TEAL[0])
    d2.glyph(zone_h, g_zones, gap=0.6)
    d2.done("zones")
    yy -= spec[1]
    cv.varrow(ix + iw / 2, yy - 0.2, yy - 2.2)
    yy -= 2.4
    d3 = Box(cv, ix, yy - spec[2], iw, spec[2], "Head-rice yield and value", TEAL[0])
    d3.text("HRY = $Y$ × HRR / 100", size=T_BODY)
    d3.text("$V$ = $a$ · HRY · $p_Z$", size=T_BODY)
    d3.done("value")
    yy -= spec[2] + 2.0
    d4 = Box(cv, ix, Y3 + 2.0, iw, yy - Y3 - 2.0, "Attribution", GOLD[0], ec=GOLD[0])
    ph, pg = lh(T_SMALL) + 1.1, 1.0
    pw = (iw - 2 * 1.0 - pg) / 2
    for k, s in enumerate(("four-factor ANOVA", "SHAP (ΔSHAP)")):
        px0 = ix + 1.0 + k * (pw + pg)
        cv.rbox(px0, d4.cy - ph, pw, ph, fc=GOLD[2], ec="none", r=1.0, z=3)
        cv.text(px0 + pw / 2, d4.cy - 0.55, s, size=T_SMALL, within=(px0, d4.cy - ph, px0 + pw, d4.cy))
    d4.cy -= ph + 0.8
    d4.last_gap = 0.8
    d4.done("attribution")
    return cv
