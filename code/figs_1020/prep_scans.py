# ============================================================
# prep_scans.py —— Fig. 1a 扫描图预处理：定位、纠偏、统一尺寸裁剪、统一调亮（所有图同一组参数）
#   1. 在 1/8 缩略图上找到 50 粒米，只把米粒区域（四周 +25 mm）读成全分辨率，省内存
#      （14479 × 9600 的 TIFF 整张转浮点会占几 GB）
#   2. 竖着扫的（米粒区高 > 宽）转 90°；按米粒质心把每行摆平（纠偏，|角度| ≤ --max_skew）
#   3. 所有图用同一个裁剪尺寸（毫米）：取各图米粒外接框的最大宽、高，四周加 --margin，右侧再留 --bar_space
#      放比例尺（--aspect > 0 时再补成该宽高比），米粒区居中 —— 输出尺寸、比例尺完全一致，
#      不同面板里的米粒大小可以直接比较
#   4. 统一色阶（作用于整幅图）：黑场 = 背景高分位数，白场 = 同组各图米粒像素合并后的 --white_pct 分位数，
#      中间调 gamma（非线性，须在图注说明）。--groups 可把要直接比较的图分成组（如糙米一组、精米一组），
#      组内参数完全相同；不分组时所有图共用一组参数
#   5. --tint 指定的样品（糙米）做整幅图的伪彩色（灰度 → 黄褐色调），须在图注说明；其余保持灰度
#   6. 右下角加比例尺（--bar_mm，默认 5 mm，白色横线；[1027] 默认不标数字、不留白，--bar_label / --bar_space 可恢复）：按 Fig. 1a 格子宽度（--fig_slot_mm）换算字号和线粗，
#      保证放进 Fig. 1 后标注约 --bar_pt 磅；各分辨率分别绘制（不是缩放出来的）
#   7. 输出：assets/scanner_<样品名>.png（--save_dpi，默认 600 dpi，Fig. 1 用）；--master 目录另存全分辨率
#      处理参数写入 assets/scanner_prep_log.txt；--sheet 另出一张带标签的总览图（检查用）
#   用法（扫描仪写的 dpi 不对时用 --dpi 指定）：
#     python figs_1020/prep_scans.py 糙米.tif=brown_high 糙米碎.tif=brown_low 精米.tif=milling_high \
#            碎米.tif=milling_low 低垩白.tif=appearance_high 高垩白.tif=appearance_low \
#            --dpi 1200 --tint brown_high,brown_low --groups "brown_high,brown_low" --sheet 总览.png
#   原图不改
# ============================================================
import argparse
import gc
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("pairs", nargs="+", help="扫描文件=样品名，如 x.tif=milling_high")
p.add_argument("--aspect", type=float, default=0, help="裁剪宽高比；0 = 按米粒区实际大小（不补边）")
p.add_argument("--margin", type=float, default=2.5, help="最大米粒外接框四周留白（mm）")
p.add_argument("--gamma", type=float, default=0.6, help="中间调（<1 提亮暗部，如透明米）")
p.add_argument("--white_pct", type=float, default=99.7, help="白场 = 所有图米粒像素合并后的该分位数")
p.add_argument("--dpi", type=float, default=None, help="扫描分辨率（文件里没记或记错时指定，如 1200）")
p.add_argument("--max_skew", type=float, default=12.0, help="纠偏搜索范围（度）")
p.add_argument("--rows", type=int, default=5, help="每张图米粒的行数（纠偏用）")
p.add_argument("--tint", default="", help="做黄褐色伪彩色的样品名，逗号分隔（如 brown_high,brown_low）")
p.add_argument("--tint_strength", type=float, default=0.85, help="伪彩色强度 0–1（0 = 灰度，1 = 全色调）")
p.add_argument("--groups", default="", help="色阶分组：组内逗号、组间分号，如 \"brown_high,brown_low\"；未列出的归为另一组")
p.add_argument("--bar_mm", type=float, default=5.0, help="比例尺长度（mm）；0 = 不加")
p.add_argument("--bar_space", type=float, default=0.0, help="右侧留给比例尺的宽度（mm，实物尺寸）；[1027] 默认 0 = 不留白，线压在右下角的留白边上")
p.add_argument("--bar_label", action="store_true", help="在线上方标注长度（[1027] 默认不标，长度写在图注里）")
p.add_argument("--fig_slot_mm", type=float, default=52.7, help="Fig. 1a 每个格子的宽度（mm），用来换算比例尺字号")
p.add_argument("--bar_pt", type=float, default=5.5, help="比例尺标注在 Fig. 1 里的字号（磅）")
p.add_argument("--bar_line_mm", type=float, default=0.45, help="比例尺线在 Fig. 1 里的粗细（mm）")
p.add_argument("--save_dpi", type=float, default=600, help="assets 里的输出分辨率")
p.add_argument("--master", default=None, help="另存全分辨率结果的目录（可选）")
p.add_argument("--sheet", default=None, help="带标签的总览图路径（可选）")
p.add_argument("--out", default=str(HERE / "assets"))
a = p.parse_args()

Image.MAX_IMAGE_PIXELS = None
TINT = dict(black=(0, 0, 0), mid=(170, 138, 78), white=(255, 244, 214))   # 糙米：灰度 → 黄褐色调（再按强度与灰度混合）


def font_path():
    """Arial → Liberation Sans → DejaVu Sans（与 Fig. 1 的字体一致）"""
    from matplotlib import font_manager as fm
    return fm.findfont(fm.FontProperties(family=["Arial", "Liberation Sans", "DejaVu Sans"]))


def add_scalebar(img, dpi_img, W_mm, H_mm, Wg_mm):
    """右下角白色比例尺：线长 --bar_mm；标注在线的正上方；尺寸按放进 Fig. 1 后的大小换算"""
    from PIL import ImageFont
    k = a.fig_slot_mm / W_mm                                       # 实物 → Fig. 1 的缩放
    pm = dpi_img / 25.4                                            # 每毫米像素
    em = a.bar_pt * 25.4 / 72 / k                                  # 标注字高（实物 mm）
    thick = a.bar_line_mm / k
    if a.bar_space > 0:                                            # 右侧留白的中线；线的下沿 = 米粒区下沿
        cx, yb = (Wg_mm + W_mm) / 2, H_mm - a.margin
    else:                                                          # [1027] 不留白：线放在底部留白边的中间，右端距边 1.5 mm
        cx, yb = W_mm - 1.5 - a.bar_mm / 2, H_mm - (a.margin - thick) / 2
    d = ImageDraw.Draw(img)
    white = 255 if img.mode == "L" else (255, 255, 255)
    d.rectangle([round((cx - a.bar_mm / 2) * pm), round((yb - thick) * pm),
                 round((cx + a.bar_mm / 2) * pm) - 1, round(yb * pm) - 1], fill=white)
    if not a.bar_label:
        return img
    font = ImageFont.truetype(font_path(), size=max(8, round(em * pm)))
    d.text((round(cx * pm), round((yb - thick - 0.25 * em) * pm)), f"{a.bar_mm:g} mm", font=font, fill=white,
           anchor="ms")
    return img


def comps(g, min_px, edge=None):
    """米粒连通域：去灰尘（< min_px）和贴边的杂物；返回掩膜、质心、外接框"""
    thr = max(8.0, float(np.percentile(g, 50)) + 8)
    m = ndi.binary_opening(g > thr, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return m, np.zeros((0, 2)), None
    sizes = ndi.sum(m, lab, range(1, n + 1))
    sl = ndi.find_objects(lab)
    H, W = g.shape
    ok = []
    for i in range(n):
        if sizes[i] < min_px:
            continue
        if edge is not None:
            ex, ey = edge
            if sl[i][1].start < ex or sl[i][1].stop > W - ex or sl[i][0].start < ey or sl[i][0].stop > H - ey:
                continue
        ok.append(i)
    if not ok:
        return np.zeros_like(m), np.zeros((0, 2)), None
    med = np.median(sizes[ok])
    keep = [i + 1 for i in ok if 0.12 * med <= sizes[i] <= 4 * med]
    mask = np.isin(lab, keep)
    cm = np.array(ndi.center_of_mass(m, lab, keep))
    ys, xs = np.where(mask)
    return mask, cm, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def skew_angle(cm, n_rows, lim):
    """让每行米粒质心的 y 最集中的旋转角（度）"""
    best, ang_best = np.inf, 0.0
    for ang in np.arange(-lim, lim + 1e-9, 0.1):
        t = np.deg2rad(ang)
        y = np.sort(cm[:, 0] * np.cos(t) - cm[:, 1] * np.sin(t))
        cost = sum(((gr - gr.mean()) ** 2).sum() for gr in np.array_split(y, n_rows))
        if cost < best:
            best, ang_best = cost, ang
    return float(ang_best)


items = []
for pair in a.pairs:
    fn, name = pair.split("=")
    im = Image.open(fn)
    dpi = float(a.dpi or im.info.get("dpi", (300, 300))[0] or 300)
    L = im.convert("L"); del im; gc.collect()                       # 转灰度（扫描为灰度时无损）
    pxmm = dpi / 25.4
    # ---- 1. 1/8 缩略图上定位米粒区 ----
    f = 8
    sm = np.asarray(L.reduce(f)).astype(np.float32)
    _, _, bb = comps(sm, min_px=0.4 * (pxmm / f) ** 2, edge=(int(0.04 * sm.shape[1]), int(0.02 * sm.shape[0])))
    if bb is None:
        raise SystemExit(f"{fn}：没找到米粒")
    pad = int(25 * pxmm)
    x0, y0, x1, y1 = [v * f for v in bb]
    region = np.asarray(L.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))).astype(np.float32)
    del L, sm; gc.collect()
    if (y1 - y0) > (x1 - x0):                                       # 竖着扫的转成横向
        region = np.ascontiguousarray(np.rot90(region, k=1))
    # ---- 2. 纠偏 ----
    min_px = 0.4 * pxmm ** 2                                        # 0.4 mm² 以下当灰尘
    _, cm, _ = comps(region, min_px)
    ang = skew_angle(cm, a.rows, a.max_skew) if len(cm) >= 10 else 0.0
    bg = float(np.percentile(region, 20))
    if abs(ang) > 0.05:
        region = np.asarray(Image.fromarray(region).rotate(ang, resample=Image.BICUBIC, fillcolor=bg)).astype(np.float32)
    mask, cm, bb = comps(region, min_px)
    items.append(dict(fn=fn, name=name, dpi=dpi, ang=ang, g=region, mask=mask, bb=bb, n=len(cm), bg=bg))
    print(f"[读入] {Path(fn).name} → {name}：{len(cm)} 粒，纠偏 {ang:+.1f}°")

# ---- 3. 统一裁剪尺寸（毫米相同） ----
dpis = {round(it["dpi"]) for it in items}
if len(dpis) > 1:
    raise SystemExit(f"扫描分辨率不一致 {dpis}，用 --dpi 统一后再跑")
dpi = items[0]["dpi"]; pxmm = dpi / 25.4
mpx = a.margin * pxmm
Wg = max(it["bb"][2] - it["bb"][0] for it in items) + 2 * mpx       # 米粒区（含四周留白）
Hc = max(it["bb"][3] - it["bb"][1] for it in items) + 2 * mpx
Wc = Wg + (a.bar_space * pxmm if a.bar_mm > 0 else 0)              # 右侧加比例尺留白
if a.aspect > 0:
    if Wc / Hc < a.aspect:
        Wc = Hc * a.aspect
    else:
        Hc = Wc / a.aspect
Wc, Hc = int(round(Wc)), int(round(Hc))
for it in items:
    cx, cy = (it["bb"][0] + it["bb"][2]) / 2, (it["bb"][1] + it["bb"][3]) / 2
    X0, Y0 = int(round(cx - Wg / 2)), int(round(cy - Hc / 2))      # 米粒区居中于左侧 Wg 范围
    H_, W_ = it["g"].shape
    crop = np.full((Hc, Wc), it["bg"], np.float32); mcrop = np.zeros((Hc, Wc), bool)
    sx0, sy0, sx1, sy1 = max(0, X0), max(0, Y0), min(W_, X0 + Wc), min(H_, Y0 + Hc)
    crop[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = it["g"][sy0:sy1, sx0:sx1]
    mcrop[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = it["mask"][sy0:sy1, sx0:sx1]
    it.update(crop=crop, mcrop=mcrop); del it["g"], it["mask"]
    lost = it["n"] - comps(crop, 0.4 * pxmm ** 2)[1].shape[0]
    if lost > 0:
        print(f"[⚠️] {it['name']}：裁剪后少了 {lost} 粒，加大 --margin 或改 --aspect")
gc.collect()

# ---- 4. 统一色阶（组内共用；不分组时所有图共用） ----
bgmask = [~ndi.binary_dilation(it["mcrop"], iterations=int(0.5 * pxmm)) for it in items]
black = float(np.median([np.percentile(it["crop"][bm], 99.5) for it, bm in zip(items, bgmask)]))
groups = [[n.strip() for n in g_.split(",") if n.strip()] for g_ in a.groups.split(";") if g_.strip()]
listed = {n for g_ in groups for n in g_}
rest = [it["name"] for it in items if it["name"] not in listed]
if rest:
    groups.append(rest)
WHITE = {}
for g_ in groups:
    pool = np.concatenate([it["crop"][it["mcrop"]][::7] for it in items if it["name"] in g_])
    w_ = float(np.percentile(pool, a.white_pct))
    for n in g_:
        WHITE[n] = w_
    del pool
tint = {s.strip() for s in a.tint.split(",") if s.strip()}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
master = Path(a.master) if a.master else None
if master:
    master.mkdir(parents=True, exist_ok=True)
log = [f"统一色阶（作用于整幅图）：黑场 {black:.1f}，gamma {a.gamma}；白场 = 同组米粒像素合并 {a.white_pct} 分位："
       + "；".join(f"{'、'.join(g_)} → {WHITE[g_[0]]:.1f}" for g_ in groups)
       + f"。灰度扫描；纠偏后按同一尺寸 {Wc / pxmm:.1f} × {Hc / pxmm:.1f} mm 裁剪"
       f"（米粒区四周至少留 {a.margin} mm，右侧 {a.bar_space if a.bar_mm > 0 else 0} mm 放比例尺）；"
       f"扫描 {dpi:.0f} dpi（1 mm = {pxmm:.2f} 像素），Fig. 1 用图 {a.save_dpi:.0f} dpi"]
if a.bar_mm > 0:
    log.append(f"比例尺：{a.bar_mm:g} mm 白色横线，右下角，{'标注约 ' + str(a.bar_pt) + ' 磅，' if a.bar_label else '不标注（长度写在图注），'}"
               f"按 Fig. 1a 格子宽 {a.fig_slot_mm} mm 换算，放进 Fig. 1 后线粗 {a.bar_line_mm} mm")
if tint:
    log.append(f"伪彩色：{', '.join(sorted(tint))} 整幅图按灰度映射到黄褐色调（黑 → {TINT['mid']} → {TINT['white']}，"
               f"强度 {a.tint_strength}），表示糙米的颜色；扫描本身为灰度")
thumbs = []
for it in items:
    white = WHITE[it["name"]]
    v = np.clip((it["crop"] - black) / max(white - black, 1.0), 0, 1)
    clip_pct = 100 * float((it["crop"][it["mcrop"]] >= white).mean())
    v = v ** a.gamma
    img = Image.fromarray((v * 255 + 0.5).astype(np.uint8), "L")
    if it["name"] in tint:
        img = Image.blend(img.convert("RGB"), ImageOps.colorize(img, **TINT), a.tint_strength)
    W_mm, H_mm, Wg_mm = Wc / pxmm, Hc / pxmm, Wg / pxmm
    k = a.save_dpi / dpi
    small = img.resize((int(round(img.width * k)), int(round(img.height * k))), Image.LANCZOS) if k < 1 else img.copy()
    if a.bar_mm > 0:                                               # 比例尺在各分辨率上分别绘制
        img = add_scalebar(img, dpi, W_mm, H_mm, Wg_mm)
        small = add_scalebar(small, min(a.save_dpi, dpi), W_mm, H_mm, Wg_mm)
    if master:
        img.save(master / f"scanner_{it['name']}_{dpi:.0f}dpi.png", dpi=(dpi, dpi))
    fn_out = out / f"scanner_{it['name']}.png"
    small.save(fn_out, dpi=(min(a.save_dpi, dpi),) * 2)
    log.append(f"{Path(it['fn']).name} → {fn_out.name}：{it['n']} 粒，纠偏 {it['ang']:+.1f}°，"
               f"高光截断 {clip_pct:.2f}% 的米粒像素，输出 {small.width} × {small.height} px")
    thumbs.append((it["name"], small.convert("RGB").resize((small.width // 4, small.height // 4), Image.LANCZOS)))
print("\n".join(log))
(out / "scanner_prep_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

if a.sheet:                                                         # 总览：2 列，同一比例尺
    tw, th = thumbs[0][1].size
    rows = (len(thumbs) + 1) // 2
    sheet = Image.new("RGB", (2 * tw + 30, rows * (th + 26) + 10), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    for i, (nm, t) in enumerate(thumbs):
        x, y = 10 + (i % 2) * (tw + 10), 10 + (i // 2) * (th + 26)
        sheet.paste(t, (x, y + 16)); d.text((x, y), nm, fill=(255, 255, 255))
    sheet.save(a.sheet)
    print(f"[总览] {a.sheet}")
