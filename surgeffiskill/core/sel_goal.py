# -*- coding: utf-8 -*-
"""与验收标准同构的 train-CV 选择准则，以及在该准则下的逐维序数标定 λ。

为什么换：原 `train_mmoe_lite.sel_score` 是
`0.30·QWK + 0.15·rho + 0.15·F1 + 0.20·QWK(Tissue) + 0.10·Exact + 0.10·±1`
—— **不含 MAE**，且 ±1 只占 0.10。而验收标准要求 MAE < 0.460、±1 ≥ 0.98。
实测后果（`skillmoe_v2_final`）：CV 选出 λ=(0.5,0.75,0.5,0.75)，
Bimanual 的 ±1 掉到 0.88（3 个 |Δ|≥2），整体 ±1 0.960 / MAE 0.530，两项都不达标。
即我在优化一个与验收口径不一致的目标。

新准则 = 六个验收指标的等权平均 + **±1 的铰链惩罚**（低于 0.98 才罚，且罚得重，
使其接近硬约束）：

    S = (QWK + Exact + MacroF1 + rho + QWK_Tissue + (1 − MAE)) / 6
        + 2 · min(0, ±1 − 0.98)

常数偏移不影响排序，所以这等价于对验收指标做等权组合；铰链项是唯一的非线性部分。
λ 的选择用同一形状的逐维版本，全部在 train 折外计算，不看 val。
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import train_mmoe_lite as T
from skill_moe_v8 import DIMS, TISSUE, BASE_THR

PM1_TARGET = 0.98
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
SHIFTS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


def sel_goal(pred, gt):
    y, p = gt.ravel(), pred.ravel()
    rho = spearmanr(y, p)[0]
    rho = 0.0 if rho != rho else rho
    mae = float(np.abs(y - p).mean())
    pm1 = float((np.abs(y - p) <= 1).mean())
    core = (T.qwk(y, p) + float((y == p).mean()) + T._f1_macro(y, p) + rho
            + T.qwk(gt[:, TISSUE], pred[:, TISSUE]) + (1.0 - mae)) / 6.0
    return core + 2.0 * min(0.0, pm1 - PM1_TARGET)


def _dim_goal(y, p):
    mae = float(np.abs(y - p).mean())
    pm1 = float((np.abs(y - p) <= 1).mean())
    return ((T.qwk(y, p) + float((y == p).mean()) + T._f1_macro(y, p) + (1.0 - mae)) / 4.0
            + 2.0 * min(0.0, pm1 - PM1_TARGET))


def pick_lambda_goal(oof, Ytr, seed=7):
    """逐维选 λ：内层再切 5 折（阈值在 4/5 上拟合、指标在留出的 1/5 上算），
    目标函数与验收标准同构；不优于 λ=0 就退回 λ=0（无标定）。"""
    inner = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(Ytr))))
    lam_l, qthr_l, trace = [], [], []
    for i in range(len(DIMS)):
        sc = []
        for lam in LAMS:
            v = []
            for a, b in inner:
                cnt = np.bincount(Ytr[a, i], minlength=6)[1:].astype(float)
                cum = np.cumsum(cnt) / cnt.sum()
                qt = np.array([np.quantile(oof[a, i], q) for q in cum[:-1]])
                pl = np.clip(np.searchsorted((1 - lam) * BASE_THR + lam * qt, oof[b, i]) + 1, 1, 5)
                v.append(_dim_goal(Ytr[b, i], pl))
            sc.append(float(np.mean(v)))
        j = int(np.argmax(sc))
        if sc[j] <= sc[0]:
            j = 0
        lam_l.append(LAMS[j])
        cnt = np.bincount(Ytr[:, i], minlength=6)[1:].astype(float)
        cum = np.cumsum(cnt) / cnt.sum()
        qthr_l.append(np.array([np.quantile(oof[:, i], q) for q in cum[:-1]]))
        trace.append([round(x, 4) for x in sc])
    return lam_l, qthr_l, trace


def pick_calib_goal(oof, Ytr, seed=7):
    """逐维联合选 (λ, δ)：thresholds = (1−λ)·BASE + λ·分位阈值 **+ δ**。

    只有 λ 是不够的：λ 只能在"基准阈值 ↔ 分位阈值"这一条一维线上取值，
    它改变的是阈值的**相对间距**；而实测的失配是**整体偏移**——
    例如 Depth 的预测分布 {3:7, 4:8, 5:10} 对真值 {3:7, 4:12, 5:6}，明显偏高。
    偏移直接决定 Exact 与 MAE，这正是 MoE 残差改善了排序（QWK/rho）却没改善落点的原因。
    δ 与 λ 一样，全部在 train 折外用与验收标准同构的准则选，不看 val。
    """
    inner = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(Ytr))))
    lam_l, qthr_l, sh_l, trace = [], [], [], []
    for i in range(len(DIMS)):
        best, base = None, None
        for lam in LAMS:
            for dl in SHIFTS:
                v = []
                for a, b in inner:
                    cnt = np.bincount(Ytr[a, i], minlength=6)[1:].astype(float)
                    cum = np.cumsum(cnt) / cnt.sum()
                    qt = np.array([np.quantile(oof[a, i], q) for q in cum[:-1]])
                    thr = (1 - lam) * BASE_THR + lam * qt + dl
                    pl = np.clip(np.searchsorted(thr, oof[b, i]) + 1, 1, 5)
                    v.append(_dim_goal(Ytr[b, i], pl))
                sc = float(np.mean(v))
                if lam == 0.0 and dl == 0.0:
                    base = sc
                if best is None or sc > best[0]:
                    best = (sc, lam, dl)
        sc, lam, dl = best
        if sc <= base:                       # 不优于"无标定"就退回无标定
            lam, dl, sc = 0.0, 0.0, base
        lam_l.append(lam)
        sh_l.append(dl)
        cnt = np.bincount(Ytr[:, i], minlength=6)[1:].astype(float)
        cum = np.cumsum(cnt) / cnt.sum()
        qthr_l.append(np.array([np.quantile(oof[:, i], q) for q in cum[:-1]]))
        trace.append('λ=%.2f δ=%+.1f 分数 %.4f（无标定 %.4f）' % (lam, dl, sc, base))
    return lam_l, qthr_l, sh_l, trace
