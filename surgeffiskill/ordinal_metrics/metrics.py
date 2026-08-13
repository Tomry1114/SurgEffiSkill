# -*- coding: utf-8 -*-
"""ordinal_metrics.metrics — 序数分级指标：Accuracy / Macro-F1 / MAE / 加权 Cohen's κ / Spearman ρ。

纯 numpy 实现，不依赖 sklearn / scipy —— 推理用的 qwen_cpu 环境没有 sklearn。
test_metrics.py 会在装了 sklearn/scipy 的环境（easyr1）里逐个与官方实现对数，
确保这里的数字和 sklearn.metrics / scipy.stats 一致。

口径（写论文时要说清楚的几点）：
* **Macro-F1** 默认对 `y_true ∪ y_pred` 里出现过的类求平均（sklearn 的默认行为）。
  传 `labels=[1,2,3,4,5]` 可以强制按整套 5 级平均 —— 数据里没出现的档会得 F1=0
  并把宏平均拉低，两种口径差别很大，报数时必须注明用的哪种。
* **加权 κ** 默认用**整条 1..5 序数标尺**（不是只用出现过的类）建权重矩阵，
  因为 κ 的权重取决于两个等级在标尺上的距离，只用观测到的类会改变距离定义。
  默认二次权重（ordinal 场景惯例），`weights='linear'` 可切线性。
* **Spearman ρ** 用平均秩处理并列；p 值只在装了 scipy 时给出，否则为 None
  （不自己拿近似式糊一个 p 出来）。
"""

import math
from typing import Optional, Sequence

import numpy as np

NAN = float('nan')


def _as_arrays(y_true: Sequence, y_pred: Sequence):
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    if yt.shape != yp.shape:
        raise ValueError('y_true/y_pred 长度不一致: %s vs %s' % (yt.shape, yp.shape))
    return yt, yp


def accuracy(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    return float(np.mean(yt == yp))


def macro_f1(y_true, y_pred, labels: Optional[Sequence] = None) -> float:
    """未指定 labels 时按 y_true ∪ y_pred 出现过的类求宏平均（同 sklearn 默认）。"""
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    labs = np.unique(np.concatenate([yt, yp])) if labels is None else np.asarray(labels, dtype=float)
    f1s = []
    for c in labs:
        tp = float(np.sum((yt == c) & (yp == c)))
        fp = float(np.sum((yt != c) & (yp == c)))
        fn = float(np.sum((yt == c) & (yp != c)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else NAN


def acc_within(y_true, y_pred, tol: int = 1) -> float:
    """±tol 级容差准确率：|pred - gt| <= tol 即算命中（序数任务的临床可接受判据）。"""
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    return float(np.mean(np.abs(yt - yp) <= tol))


def error_hist(y_true, y_pred, num_levels: int = 5) -> dict:
    """|Δ| 的分布：0 = 精确命中，>=2 = 严重错误。"""
    yt, yp = _as_arrays(y_true, y_pred)
    out = {}
    n = max(yt.size, 1)
    for k in range(num_levels):
        c = int(np.sum(np.abs(yt - yp) == k)) if yt.size else 0
        out[str(k)] = {'n': c, 'frac': (c / n) if yt.size else NAN}
    sev = int(np.sum(np.abs(yt - yp) >= 2)) if yt.size else 0
    out['severe_ge2'] = {'n': sev, 'frac': (sev / n) if yt.size else NAN}
    return out


def mae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    return float(np.mean(np.abs(yt - yp)))


def weighted_kappa(y_true, y_pred, weights: str = 'quadratic',
                   labels: Optional[Sequence] = None, num_levels: int = 5) -> float:
    """加权 Cohen's κ。labels 默认取整条 1..num_levels 标尺。"""
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    labs = (np.arange(1, num_levels + 1, dtype=float) if labels is None
            else np.asarray(labels, dtype=float))
    k = len(labs)
    if k < 2:
        return NAN
    idx = {v: i for i, v in enumerate(labs)}
    O = np.zeros((k, k), dtype=float)
    n = 0
    for a, b in zip(yt, yp):
        if a in idx and b in idx:
            O[idx[a], idx[b]] += 1.0
            n += 1
    if n == 0:
        return NAN
    O /= n
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col)

    i = np.arange(k).reshape(-1, 1)
    j = np.arange(k).reshape(1, -1)
    if weights == 'quadratic':
        W = ((i - j) ** 2) / float((k - 1) ** 2)
    elif weights == 'linear':
        W = np.abs(i - j) / float(k - 1)
    else:
        raise ValueError("weights 只支持 'quadratic' / 'linear'，收到 %r" % weights)

    denom = float(np.sum(W * E))
    if denom == 0:
        return NAN                      # 边缘分布退化，κ 无定义
    return float(1.0 - np.sum(W * O) / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """并列取平均秩（同 scipy.stats.rankdata 的 'average'）。"""
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(y_true, y_pred):
    """-> (rho, p)。p 只在装了 scipy 时给出，否则 None。"""
    yt, yp = _as_arrays(y_true, y_pred)
    n = yt.size
    if n < 3:
        return NAN, None
    rt, rp = _average_ranks(yt), _average_ranks(yp)
    st, sp = rt.std(), rp.std()
    if st == 0 or sp == 0:
        return NAN, None                # 有一侧是常数，ρ 无定义
    rho = float(np.mean((rt - rt.mean()) * (rp - rp.mean())) / (st * sp))
    rho = max(-1.0, min(1.0, rho))
    p = None
    try:
        from scipy import stats
        if abs(rho) < 1.0:
            t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
            p = float(2 * stats.t.sf(abs(t), n - 2))
        else:
            p = 0.0
    except Exception:
        p = None
    return rho, p


def all_metrics(y_true, y_pred, num_levels: int = 5,
                kappa_weights: str = 'quadratic',
                macro_f1_labels: Optional[Sequence] = None) -> dict:
    """一次算齐 5 个指标。"""
    rho, p = spearman(y_true, y_pred)
    return {
        'n': int(len(list(y_true))),
        'accuracy': accuracy(y_true, y_pred),
        'acc_within_1': acc_within(y_true, y_pred, 1),
        'error_hist': error_hist(y_true, y_pred, num_levels),
        'macro_f1': macro_f1(y_true, y_pred, labels=macro_f1_labels),
        'mae': mae(y_true, y_pred),
        'kappa_%s' % kappa_weights: weighted_kappa(
            y_true, y_pred, weights=kappa_weights, num_levels=num_levels),
        'spearman_rho': rho,
        'spearman_p': p,
    }


METRIC_KEYS = ['accuracy', 'acc_within_1', 'macro_f1', 'mae', 'kappa_quadratic', 'spearman_rho']
