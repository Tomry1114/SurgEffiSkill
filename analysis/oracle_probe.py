# -*- coding: utf-8 -*-
"""专家池的**可融合上限**诊断 —— 区分瓶颈是"证据不够"还是"gate 没学会选"。

背景与自我更正：此前我根据"各专家与 ck250 的排序误差相关性 0.81–0.88"下过一句
"QWK 的上限是结构性的，约 0.589"。**这句话是过度声明并已撤回**——
相关性不构成严格上界，0.85 的相关也不等于预测相同：专家仍可能在
误差幅度、局部样本、不同技能维度上互补。

因此这里不猜上限，而是**直接测**三件事（全部在 val 上，属于 oracle 诊断，
**不是可报告的模型结果**，只用于决定下一步方向）：

  A. Oracle 专家选择   每个 (病例, 维度) 直接选离真值最近的那个专家
                       → 若 oracle 也只到 ~0.59，说明专家池确实接近天花板；
                         若能到 0.65–0.70，说明证据够、是 gate 没学会选。
  B. Oracle 静态混合   逐维在单纯形上网格搜最优 α（再加一个无约束最小二乘的更松上界）
                       → 回答"不考虑泛化，这几路专家能不能组合出更好的结果"。
  C. 误差互补计数      ck250 错而某专家对的病例数 / 至少一个专家对的病例数 /
                       各专家独有的纠错病例数，逐维统计
                       → 比相关系数更直接：其他专家到底有没有能力救 ck250 的错。
"""
import io
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/hpc2hdd/home/ce483/hkust_surgery_video_learning/revision/scripts/sft_grpo')
import train_mmoe_lite as T                                        # noqa: E402
import ridge_moe as RM                                            # noqa: E402

DIMS = ['Depth', 'Bimanual', 'Efficiency', 'Tissue']
THR = np.array([1.5, 2.5, 3.5, 4.5])
R = '/hpc2hdd/home/ce483/hkust_surgery_video_learning/revision/evidence/results/sft_grpo'
CK = dict(Exact=.52, pm1=.99, F1=.4856, MAE=.49, QWK=.5894, rho=.5831)


def lvl(y):
    return np.clip(np.searchsorted(THR, y) + 1, 1, 5).astype(int)


def met(p, y):
    from scipy.stats import spearmanr
    a, b = y.ravel(), p.ravel()
    r = spearmanr(a, b)[0]
    return dict(Exact=float((a == b).mean()), pm1=float((np.abs(a - b) <= 1).mean()),
                F1=float(T._f1_macro(a, b)), MAE=float(np.abs(a - b).mean()),
                QWK=float(T.qwk(a, b)), rho=0.0 if r != r else float(r))


def show(tag, m):
    print('  %-30s Exact %.4f  ±1 %.4f  F1 %.4f  MAE %.4f  QWK %.4f  rho %.4f'
          % (tag, m['Exact'], m['pm1'], m['F1'], m['MAE'], m['QWK'], m['rho']))


def main():
    src, Y, istr, names = RM.load_sources(os.path.join(R, 'feats_4src.npz'), 64)
    Ytr, Yva = Y[istr], Y[~istr]
    nva = [n for n, t in zip(names, istr) if not t]
    fo = np.arange(len(Ytr)) % 5
    O = {}
    for k, v in src.items():
        _, o2, _ = RM.source_expert_preds(v[istr], Ytr, v[~istr], fo)
        O[k] = o2
    rr = [json.loads(l) for l in io.open(
        os.path.join(R, 'pred_sftbalpd3_ck250_val_pd.jsonl'), encoding='utf-8')]
    pm = {r['video']: r['pred_levels'] for r in rr}
    O['sft'] = np.array([pm[v] for v in nva], dtype=float)
    ks = list(O)
    P = np.stack([O[k] for k in ks], -1)                       # (n, K, M) 连续/等级分

    print('=== 参照 ===')
    show('ck250 (SFT baseline)', CK)
    print('\n=== 各专家单独 ===')
    for j, k in enumerate(ks):
        show(k, met(lvl(P[:, :, j]), Yva))

    print('\n=== A. Oracle 专家选择（每病例每维选离真值最近的专家）===')
    err = np.abs(P - Yva[:, :, None])
    best = err.argmin(-1)
    orc = np.take_along_axis(P, best[:, :, None], -1)[:, :, 0]
    show('oracle-select (连续分)', met(lvl(orc), Yva))
    orcl = np.take_along_axis(lvl(P), best[:, :, None], -1)[:, :, 0]
    show('oracle-select (先离散后选)', met(orcl, Yva))
    print('  各专家被 oracle 选中的次数（逐维）:')
    for i, d in enumerate(DIMS):
        c = np.bincount(best[:, i], minlength=len(ks))
        print('    %-11s %s' % (d, dict(zip(ks, c.tolist()))))

    print('\n=== B. Oracle 静态混合（逐维在单纯形上网格搜，step=0.1）===')
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(ks))
            if abs(sum(w) - 1.0) < 1e-6]
    print('  单纯形网格点数 %d' % len(grid))
    bestmix = np.zeros_like(Yva, dtype=float)
    alphas = []
    for i in range(len(DIMS)):
        bq, bw = -9, None
        for w in grid:
            q = T.qwk(Yva[:, i], lvl(P[:, i, :] @ np.array(w)))
            if q > bq:
                bq, bw = q, w
        alphas.append(bw)
        bestmix[:, i] = P[:, i, :] @ np.array(bw)
        print('    %-11s 最优 α = %s → 该维 QWK %.4f'
              % (DIMS[i], dict(zip(ks, [round(x, 1) for x in bw])), bq))
    show('oracle-convex-mixture', met(lvl(bestmix), Yva))

    ls = np.zeros_like(Yva, dtype=float)
    for i in range(len(DIMS)):
        A = np.hstack([P[:, i, :], np.ones((len(Yva), 1))])
        coef, *_ = np.linalg.lstsq(A, Yva[:, i].astype(float), rcond=None)
        ls[:, i] = A @ coef
    show('oracle-unconstrained-LS（更松上界）', met(lvl(ls), Yva))

    print('\n=== C. 误差互补计数（以 ck250 为参照）===')
    Lp = lvl(P)
    sj = ks.index('sft')
    for i, d in enumerate(DIMS):
        sw = Lp[:, i, sj] != Yva[:, i]                          # ck250 错的病例
        print('  [%s] ck250 错 %d / %d 例' % (d, sw.sum(), len(Yva)))
        anyfix = np.zeros(len(Yva), dtype=bool)
        for j, k in enumerate(ks):
            if k == 'sft':
                continue
            fix = sw & (Lp[:, i, j] == Yva[:, i])
            anyfix |= fix
            print('      %-6s 能救 %2d 例' % (k, fix.sum()))
        print('      至少一个专家能救: %d 例（占 ck250 错误的 %.0f%%）'
              % (anyfix.sum(), 100 * anyfix.sum() / max(sw.sum(), 1)))
        uniq = {}
        for j, k in enumerate(ks):
            if k == 'sft':
                continue
            fix = sw & (Lp[:, i, j] == Yva[:, i])
            others = np.zeros(len(Yva), dtype=bool)
            for j2, k2 in enumerate(ks):
                if k2 in ('sft', k):
                    continue
                others |= sw & (Lp[:, i, j2] == Yva[:, i])
            uniq[k] = int((fix & ~others).sum())
        print('      各专家**独有**纠错例数: %s' % uniq)

    # ---- 关键补测：去掉 sft 专家后的 oracle。
    # sft 的训练集预测是**样本内**的（ck250 在这 98 个视频上微调过，train Exact 0.6148
    # vs val 0.5200），而 ridge 专家的训练集预测是**折外**的。于是 gate 在训练时
    # 系统性高估 sft、低估 ridge —— 正好与 oracle 想要的相反（oracle 给 ck250-ridge 0.6–0.8）。
    # 若纯 ridge 池的 oracle 仍能超过 ck250，这个偏差问题就可以直接绕开。
    print('\n=== D. 去掉 sft 专家后的 oracle（绕开样本内偏差）===')
    keep = [j for j, k in enumerate(ks) if k != 'sft']
    kk = [ks[j] for j in keep]
    Pr = P[:, :, keep]
    er = np.abs(Pr - Yva[:, :, None])
    bs = er.argmin(-1)
    show('oracle-select (仅 ridge)', met(lvl(np.take_along_axis(Pr, bs[:, :, None], -1)[:, :, 0]), Yva))
    gr = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(kk))
          if abs(sum(w) - 1.0) < 1e-6]
    mx = np.zeros_like(Yva, dtype=float)
    for i in range(len(DIMS)):
        bq, bw = -9, None
        for w in gr:
            q = T.qwk(Yva[:, i], lvl(Pr[:, i, :] @ np.array(w)))
            if q > bq:
                bq, bw = q, w
        mx[:, i] = Pr[:, i, :] @ np.array(bw)
        print('    %-11s 最优 α = %s → QWK %.4f'
              % (DIMS[i], dict(zip(kk, [round(x, 1) for x in bw])), bq))
    show('oracle-convex-mixture (仅 ridge)', met(lvl(mx), Yva))
    lsr = np.zeros_like(Yva, dtype=float)
    for i in range(len(DIMS)):
        A = np.hstack([Pr[:, i, :], np.ones((len(Yva), 1))])
        c, *_ = np.linalg.lstsq(A, Yva[:, i].astype(float), rcond=None)
        lsr[:, i] = A @ c
    show('oracle-LS (仅 ridge)', met(lvl(lsr), Yva))

    print('\n=== E. 训练集上的最优静态 α（可实际使用，不是 oracle）===')
    # 在**训练集折外预测**上搜同样的单纯形，得到一个真正可部署的 α，然后在 val 上评。
    Otr = {}
    for k, v in src.items():
        o1, _, _ = RM.source_expert_preds(v[istr], Ytr, v[~istr], fo)
        Otr[k] = o1
    rr2 = [json.loads(l) for l in io.open(
        os.path.join(R, 'pred_sftbalpd3_ck250_train_pd.jsonl'), encoding='utf-8')]
    pm2 = {r['video']: r['pred_levels'] for r in rr2}
    ntr = [n for n, t in zip(names, istr) if t]
    Otr['sft'] = np.array([pm2[v] for v in ntr], dtype=float)
    Ptr = np.stack([Otr[k] for k in ks], -1)
    for tagname, idxs in [('全部专家', list(range(len(ks)))), ('仅 ridge', keep)]:
        nm2 = [ks[j] for j in idxs]
        g2 = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(idxs))
              if abs(sum(w) - 1.0) < 1e-6]
        out = np.zeros_like(Yva, dtype=float)
        aa = []
        for i in range(len(DIMS)):
            bq, bw = -9, None
            for w in g2:
                q = T.qwk(Ytr[:, i], lvl(Ptr[:, i, idxs] @ np.array(w)))
                if q > bq:
                    bq, bw = q, w
            aa.append(dict(zip(nm2, [round(x, 1) for x in bw])))
            out[:, i] = P[:, i, idxs] @ np.array(bw)
        print('  [%s] train 上选出的 α: %s' % (tagname, aa))
        show('train-fitted static mixture (%s)' % tagname, met(lvl(out), Yva))

    print('\n=== 结论判据 ===')
    print('  若 oracle-select / oracle-mixture 明显高于 ck250（如 QWK ≥ 0.65），')
    print('  则瓶颈是 gate 没学会选，应继续改路由；')
    print('  若 oracle 本身也接近 0.59，则需引入新的 evidence source')
    print('  （optical flow / motion / temporal event 等）。')


if __name__ == '__main__':
    main()
