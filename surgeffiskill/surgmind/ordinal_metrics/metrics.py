import math
from typing import Optional, Sequence

import numpy as np

NAN = float('nan')


def _as_arrays(y_true: Sequence, y_pred: Sequence):
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    if yt.shape != yp.shape:
        raise ValueError('y_true/y_pred length mismatch: %s vs %s' % (yt.shape, yp.shape))
    return yt, yp


def accuracy(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    return float(np.mean(yt == yp))


def macro_f1(y_true, y_pred, labels: Optional[Sequence] = None) -> float:
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
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return NAN
    return float(np.mean(np.abs(yt - yp) <= tol))


def error_hist(y_true, y_pred, num_levels: int = 5) -> dict:
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
        raise ValueError("weights supports only 'quadratic' / 'linear', got %r" % weights)

    denom = float(np.sum(W * E))
    if denom == 0:
        return NAN
    return float(1.0 - np.sum(W * O) / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
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
    yt, yp = _as_arrays(y_true, y_pred)
    n = yt.size
    if n < 3:
        return NAN, None
    rt, rp = _average_ranks(yt), _average_ranks(yp)
    st, sp = rt.std(), rp.std()
    if st == 0 or sp == 0:
        return NAN, None
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
