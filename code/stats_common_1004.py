# ============================================================
# stats_common_1004.py —— 统一口径的读写与汇总统计（1013；权重见 config.SUMMARY_WEIGHT）
#   口径定义见 config_1004.py 的"全流程唯一统计口径"；05/06/07/21/30 共用这里的函数，
#   保证同一个量在不同脚本里由同一段代码算出。
# ============================================================
from pathlib import Path
import numpy as np
import rasterio

import config_1004 as C

_CACHE = {}


def read(path):
    """读栅格为 float64；nodata → NaN；文件不存在返回 None"""
    p = Path(path)
    if not p.exists():
        return None
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a[a == ds.nodata] = np.nan
    return a


def domain(ri):
    """统一格网域 D(ri)（bool）"""
    k = ("dom", ri)
    if k not in _CACHE:
        _CACHE[k] = np.isfinite(read(C.support_file(ri)))
    return _CACHE[k]


def area(ri):
    """D 内分季面积（km²），域外 NaN —— 只用于面积量（经济、优质面积等）的求和"""
    k = ("area", ri)
    if k not in _CACHE:
        a = read(C.area_file(ri))
        _CACHE[k] = np.where(domain(ri) & np.isfinite(a) & (a > 0), a, np.nan)
    return _CACHE[k]


def weights(ri):
    """汇总平均的权重（config.SUMMARY_WEIGHT）："cell" → D 内每个格网 1；"area" → D 内面积；域外 NaN"""
    a = area(ri)
    return np.where(np.isfinite(a), 1.0, np.nan) if C.SUMMARY_WEIGHT == "cell" else a


def cells(ri, *arrays):
    """D（支撑区 ∩ 面积>0）∩ 所有给定数组都有值（None 视为全缺）"""
    m = np.isfinite(area(ri))
    for a in arrays:
        if a is None:
            return np.zeros_like(m)
        m &= np.isfinite(a)
    return m


def wmean(x, w, m):
    m = m & np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


def wquant(x, w, m, q):
    m = m & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    v, ww = x[m], w[m]
    o = np.argsort(v)
    c = np.cumsum(ww[o]) / ww.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def rel(f, b):
    return 100.0 * (f - b) / abs(b) if (b is not None and np.isfinite(b) and b != 0) else np.nan


def mrange(v, fmt="{:+.2f}"):
    """成员离散度：中位数 [最小, 最大]"""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return ""
    f = lambda x: fmt.format(x)
    return f"{f(np.median(v))} [{f(v.min())}, {f(v.max())}]"
