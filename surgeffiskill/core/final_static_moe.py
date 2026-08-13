# -*- coding: utf-8 -*-
"""SurgMind-Static：异质证据专家 + **逐技能维度静态门控**，α 由内层 CV 选定。

这是被 oracle 诊断逼出来的最终形态：

  * Oracle 逐病例选专家 → QWK 0.6988 / Exact 0.6400，**远高于 ck250**，
    说明专家池信息充足，瓶颈在门控而非证据；
  * Oracle 静态混合（逐维一组固定权重）已在全部六项超过 ck250，
    说明**不需要逐样本动态路由**，一组稳定的逐维证据偏好就够；
  * 直接在训练集折外预测上网格搜 α（非 oracle）也已六项全过，
    但只用原始 train QWK 选、1001 点/维，有过拟合训练集的风险。

因此这里把 α 的选择改成**内层 CV**：在 4/5 折上搜 α、在留出的 1/5 折上打分，
重复多轮取平均，最后用全训练集重搜一次部署。评测口径与 ck250 严格一致
（固定阈值 {1.5,2.5,3.5,4.5} 解码、无任何后处理标定、同一个 OrdinalEvaluator）。

专家池（异质证据）：
  f28 / fps4  两种时间采样的冻结 ViT（视觉、纹理、空间线索）
  base        基座 LLM 语义
  ck250       手术微调后 LLM 语义
  joint       全部来源拼接后的早融合 ridge（跨来源交互）
  sft         SFT 模型（ck250）自身的生成式预测
前六路是 ridge 探针（逐维选 PCA 维数与 alpha，train 折外 / val 全拟合）；
`sft` 的训练集预测是**样本内**的（train Exact 0.6148 vs val 0.5200），
会让门控高估它——这一偏差在结果里必须声明。
"""
import argparse
import io
import itertools
import json
import os
import sys

import numpy as np

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'revision/scripts/sft_grpo'))
import train_mmoe_lite as T                                        # noqa: E402
import ridge_moe as RM                                            # noqa: E402

DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']
K = len(DIMS)
THR = np.array([1.5, 2.5, 3.5, 4.5])


def lvl(y):
    return np.clip(np.searchsorted(THR, y) + 1, 1, 5).astype(int)


def dim_obj(y, p, mode):
    """逐维选择目标。mode=qwk 只看排序一致性；mode=mix 兼顾 Exact/MAE/±1。"""
    q = T.qwk(y, p)
    if mode == 'qwk':
        return q
    ex = float((y == p).mean())
    mae = float(np.abs(y - p).mean())
    pm1 = float((np.abs(y - p) <= 1).mean())
    return 0.4 * q + 0.25 * ex + 0.25 * (1 - mae / 4.0) + 0.10 * pm1


def _mix(P, base, A, Kp, idx, i):
    mix = P[idx, i, :] @ A[i]
    return mix if base is None else (1 - Kp[i]) * base[idx, i] + Kp[i] * mix


def simplex(m, step):
    n = int(round(1.0 / step))
    for c in itertools.product(range(n + 1), repeat=m):
        if sum(c) == n:
            yield np.array(c, dtype=float) / n


def pick_alpha(P, Y, idx, grids, mode, base=None, kappas=(1.0,)):
    """逐维搜最优 (α, κ)。base 给出固定基线路径时：
       s_d = (1−κ_d)·base_d + κ_d·Σ_m α_{d,m}·ŷ_{d,m}
    """
    A, Kp = [], []
    for i in range(K):
        best, bw, bk = -9e9, None, 1.0
        for w in grids:
            mix = P[idx, i, :] @ w
            for kp in kappas:
                sc = mix if base is None else (1 - kp) * base[idx, i] + kp * mix
                v = dim_obj(Y[idx, i], lvl(sc), mode)
                if v > best:
                    best, bw, bk = v, w, kp
        A.append(bw)
        Kp.append(bk)
    return np.stack(A), np.array(Kp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--modes', default='qwk,mix')
    ap.add_argument('--drop', default='')
    ap.add_argument('--no-sft', dest='nosft', action='store_true')
    ap.add_argument('--no-joint', dest='nojoint', action='store_true')
    ap.add_argument('--sft-as-base', dest='sftbase', action='store_true',
                    help='把 SFT 当**固定基线路径**而非参与竞争的专家。\n'
                         '原因：ck250 是在这 98 个训练视频上微调的，它在训练集上的预测是'
                         '**样本内**的（train Exact 0.6148 vs val 0.5200），'
                         '而 ridge 专家是折外的。让二者竞争 α 会系统性高估 SFT。\n'
                         '此模式下 α 只在**全为折外**的 ridge 专家间分配（竞争对称），\n'
                         '    s_d = (1−κ_d)·ŷ_sft,d + κ_d·Σ_m α_{d,m}·ŷ_{d,m}\n'
                         'κ_d 由内层 CV 选。样本内偏差会让 κ_d 偏**小**，'
                         '即偏向退回基线——是保守方向，不会制造虚假增益。')
    ap.add_argument('--kappas', default='0,0.2,0.4,0.6,0.8,1.0')
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--tag', default='surgmind_static')
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()

    src, Y, istr, names = RM.load_sources(a.feats, a.pca)
    for k in [x for x in a.drop.split(',') if x]:
        src.pop(k, None)
    snames = list(src)
    Ytr, Yva = Y[istr], Y[~istr]
    ntr = [n for n, t in zip(names, istr) if t]
    nva = [n for n, t in zip(names, istr) if not t]
    fo = np.arange(len(Ytr)) % 5
    Otr, Ova = [], []
    for k in snames:
        o1, o2, _ = RM.source_expert_preds(src[k][istr], Ytr, src[k][~istr], fo)
        Otr.append(o1)
        Ova.append(o2)
    if not a.nojoint:
        Xtr = np.hstack([src[k][istr] for k in snames])
        Xva = np.hstack([src[k][~istr] for k in snames])
        o1, o2, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
        Otr.append(o1)
        Ova.append(o2)
        snames = snames + ['joint']
    if not a.nosft:
        R = os.path.dirname(a.feats)

        def lp(fn):
            rr = [json.loads(l) for l in io.open(os.path.join(R, fn), encoding='utf-8')]
            return {r['video']: r['pred_levels'] for r in rr}
        pt, pv = lp('pred_sftbalpd3_ck250_train_pd.jsonl'), lp('pred_sftbalpd3_ck250_val_pd.jsonl')
        Otr.append(np.array([pt[v] for v in ntr], dtype=float))
        Ova.append(np.array([pv[v] for v in nva], dtype=float))
        snames = snames + ['sft']
    Ptr, Pva = np.stack(Otr, -1), np.stack(Ova, -1)
    base_tr = base_va = None
    if a.sftbase:
        assert snames[-1] == 'sft', 'sft-as-base 需要 sft 在专家池末位'
        base_tr, base_va = Ptr[:, :, -1], Pva[:, :, -1]
        Ptr, Pva = Ptr[:, :, :-1], Pva[:, :, :-1]
        snames = snames[:-1]
        print('  SFT 已移出竞争池，改作固定基线路径；α 仅在折外 ridge 专家间分配', flush=True)
    M = len(snames)
    print('%s | 专家 %s | train %d / val %d' % (a.tag, snames, len(Ytr), len(Yva)), flush=True)
    for j, k in enumerate(snames):
        L = lvl(Pva[:, :, j])
        print('  专家[%-6s] 单独 val: Exact %.3f QWK %.3f'
              % (k, (L == Yva).mean(), T.qwk(Yva.ravel(), L.ravel())), flush=True)

    KAP = [float(x) for x in a.kappas.split(',')] if a.sftbase else [1.0]
    grids = list(simplex(M, a.step))
    print('单纯形网格 %d 点/维' % len(grids), flush=True)

    # ---------- 内层 CV 选 mode（α 的选择目标），只用训练视频
    best_mode, best_sc = None, -9e9
    for mode in a.modes.split(','):
        sc = []
        for r in range(a.repeats):
            sh = np.random.RandomState(9000 + r).permutation(len(Ytr))
            f5 = np.zeros(len(Ytr), dtype=int)
            for i, v in enumerate(sh):
                f5[v] = i % 5
            for f in range(5):
                tr = np.where(f5 != f)[0]
                te = np.where(f5 == f)[0]
                A, Kp = pick_alpha(Ptr, Ytr, tr, grids, mode, base_tr, KAP)
                pred = np.stack([lvl(_mix(Ptr, base_tr, A, Kp, te, i)) for i in range(K)], 1)
                sc.append(np.mean([dim_obj(Ytr[te, i], pred[:, i], 'mix') for i in range(K)]))
        m = float(np.mean(sc))
        print('  内层 CV mode=%-4s %.4f ± %.4f'
              % (mode, m, float(np.std(sc, ddof=1) / np.sqrt(len(sc)))), flush=True)
        if m > best_sc:
            best_mode, best_sc = mode, m
    print('选中 α 的搜索目标 mode = %s' % best_mode)

    # ---------- 用全部训练视频重搜 α → 25 视频评一次
    A, Kp = pick_alpha(Ptr, Ytr, np.arange(len(Ytr)), grids, best_mode, base_tr, KAP)
    idx_va = np.arange(len(Yva))
    pred = np.stack([lvl(_mix(Pva, base_va, A, Kp, idx_va, i)) for i in range(K)], 1)
    extra = {'method': 'SurgMind-Static: heterogeneous evidence experts (per-source ridge '
                       'probes + early-fusion joint ridge + generative SFT prediction) '
                       'combined by a per-competency static gate alpha_d; fixed ordinal '
                       'thresholds {1.5,2.5,3.5,4.5}; NO post-hoc calibration '
                       '(protocol-matched to ck250)',
             'experts': snames, 'alpha_mode': best_mode, 'inner_cv': round(best_sc, 4),
             'simplex_step': a.step, 'grid_points_per_dim': len(grids),
             'alpha': {DIMS[i]: dict(zip(snames, [round(float(x), 2) for x in A[i]]))
                       for i in range(K)},
             'kappa': {DIMS[i]: float(Kp[i]) for i in range(K)} if a.sftbase else None,
             'sft_as_base': bool(a.sftbase),
             'note_sft_in_sample': 'sft 专家的训练集预测为样本内（ck250 在这 98 个视频上'
                                   '微调过，train Exact 0.6148 vs val 0.5200），'
                                   '门控会高估它；这一偏差在解读 alpha 时必须声明。'}
    res = T.score(pred, Yva, nva, a.tag, a.outdir, extra=extra)
    print('\n===== %s（25 视频，评一次；与 ck250 同评测脚本 / 同解码口径）=====' % a.tag)
    print(T.format_report(res))
    print(T.format_confusion(res))
    print('逐技能维度的证据偏好 α_d%s:' % ('（κ = 从 SFT 基线移向证据混合的幅度）'
                                            if a.sftbase else ''))
    for i, d in enumerate(DIMS):
        kk = '  κ=%.1f' % Kp[i] if a.sftbase else ''
        print('  %-24s %s%s' % (d, dict(zip(snames, [round(float(x), 2) for x in A[i]])), kk))


if __name__ == '__main__':
    main()
