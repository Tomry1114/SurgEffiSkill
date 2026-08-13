"""Pooled cross-fitted OOF selection of competency-specific z_d —— 本线**最后一次**尝试。

诊断定位（R34）：
  · 用训练 loss 学 z：低方差，但在 Efficiency 上**过拟合**（学出 0.28–0.51，
    而零训练 OOF 扫描给出最优 0）；
  · 用 3 个各 ~26 条 inner fold 的 QWK 选 z：方向可信但**方差过大**（G1/G2 收回 23–24%）。

本方法取两者之间：**不让梯度决定该结构变量，直接让折外泛化决定**，
且不在小 fold 上分别估计，而是在 outer-train 内做完整 cross-fitting、
把 ~78 条 OOF 预测**汇总后算一次 QWK**。
（QWK 是 dataset-level、不可分解的指标——汇总算一次比在 3 个小 fold 上分别估更符合其定义。）

流程（每个 outer fold）：
  outer-train ≈78 → 内层 5 折 cross-fitting → 78 条真正 OOF 的 s_global / s_hot
  → 每个 competency 在**汇总 78 条**上扫 z → z*_d = argmax QWK
  → 在完整 outer-train 上重训 → 施加 s(z*)=(1−z*)s_global+z*·s_hot 于 outer-test

**运行前定死，不看结果再改**：z 网格 = {0,0.1,…,1.0}；无 λ_z、无 selector、
无 sparsity、无 proxy、无人工规则。

**预先声明的停止条件**：
  a) QWK vs hot 配对 t > 2
  b) Efficiency 多数外层折选 z ≤ 0.3
  c) Tissue   多数外层折选 z ≥ 0.7
三条全过 → 才评 25 条；任一不过 → **停掉 competency-adaptive selection 这条线**。

train-only，**val 从不读取**。
"""
import os, io, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, fit_dtr, DIMS, K, PROJECT
from surgmind_sp import qwk, HOT

ZGRID = np.round(np.linspace(0, 1, 11), 2)          # 运行前定死
ARMS = ['global_only', 'hot', 'oof_z', 'oracle_z']


def fit_pair(P, Y, tr, te, grids, seeds):
    """在 tr 上拟合，返回 te 上的连续分数 (s_global, s_hot)。"""
    A1, _ = FS.pick_alpha(P, Y, tr, grids, 'qwk')
    U = routing_signals(P, A1, None)
    ms = fit_dtr(U[tr], P[tr], Y[tr], A1, HOT, seeds)
    ue = torch.from_numpy(U[te]).float(); oe = torch.from_numpy(P[te]).float()
    ut = torch.from_numpy(U[tr]).float(); ot = torch.from_numpy(P[tr]).float()
    Se, At = [], []
    for g in ms:
        g.eval()
        with torch.no_grad():
            s_, _, _ = g(ue, oe)
            _, al_, _ = g(ut, ot)
        Se.append(s_.numpy()); At.append(al_.numpy())
    s_hot = np.mean(Se, 0)
    Ps = np.mean(At, 0).mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
    s_glob = np.stack([P[te, i, :] @ Ps[i] for i in range(K)], 1)
    return s_glob, s_hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--inner', type=int, default=5)
    ap.add_argument('--tag', default='oof_z')
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()

    src, Y, istr, names = RM.load_sources(a.feats, a.pca)
    sn = list(src); Ytr = Y[istr]
    ntr = [n for n, t in zip(names, istr) if t]
    fo = np.arange(len(Ytr)) % 5
    O = []
    for k in sn:
        o1, _, _ = RM.source_expert_preds(src[k][istr], Ytr, src[k][~istr], fo)
        O.append(o1)
    Xtr = np.hstack([src[k][istr] for k in sn]); Xva = np.hstack([src[k][~istr] for k in sn])
    o1, _, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
    O.append(o1); sn = sn + ['joint']
    Rd = os.path.dirname(a.feats)
    rr = [json.loads(l) for l in io.open(
        os.path.join(Rd, 'pred_sftbalpd3_ck250_train_pd.jsonl'), encoding='utf-8')]
    pt = {r['video']: r['pred_levels'] for r in rr}
    O.append(np.array([pt[v] for v in ntr], float)); sn = sn + ['sft']
    P = np.stack(O, -1); n = len(Ytr)
    grids = list(FS.simplex(P.shape[-1], a.step))
    print('%s | train %d | z 网格 %s（运行前定死）| 内层 %d 折 cross-fitting | '
          '**val 从不读取**' % (a.tag, n, list(ZGRID), a.inner), flush=True)

    per = {x: [] for x in ARMS}
    zsel = [[] for _ in range(K)]

    for rep in range(a.repeats):
        rs = np.random.RandomState(41000 + rep)
        fold = rs.permutation(n) % 5
        oof = {x: np.zeros((n, K)) for x in ARMS}
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]

            # ---- 内层 5 折 cross-fitting：得到 outer-train 上真正的 OOF 预测
            rs2 = np.random.RandomState(43000 + rep * 5 + f)
            ifold = rs2.permutation(len(tr)) % a.inner
            SG = np.zeros((len(tr), K)); SH = np.zeros((len(tr), K))
            for gi in range(a.inner):
                m = ifold == gi
                ite = tr[m]; itr = tr[~m]
                sg, sh = fit_pair(P, Ytr, itr, ite, grids, a.seeds)
                SG[m] = sg; SH[m] = sh

            # ---- 在**汇总的 ~78 条 OOF** 上逐维扫 z（QWK 汇总算一次）
            zstar = np.zeros(K)
            for i in range(K):
                sc = [qwk(FS.lvl((1 - z) * SG[:, i] + z * SH[:, i]), Ytr[tr, i])
                      for z in ZGRID]
                zstar[i] = ZGRID[int(np.argmax(sc))]
                zsel[i].append(float(zstar[i]))

            # ---- 完整 outer-train 重训 → 施加于 outer-test
            s_glob, s_hot = fit_pair(P, Ytr, tr, te, grids, a.seeds)
            oof['global_only'][te] = FS.lvl(s_glob)
            oof['hot'][te] = FS.lvl(s_hot)
            for i in range(K):
                oof['oof_z'][te, i] = FS.lvl(
                    (1 - zstar[i]) * s_glob[:, i] + zstar[i] * s_hot[:, i])
                bo = [qwk(FS.lvl((1 - z) * s_glob[:, i] + z * s_hot[:, i]), Ytr[te, i])
                      for z in ZGRID]
                zb = ZGRID[int(np.argmax(bo))]
                oof['oracle_z'][te, i] = FS.lvl(
                    (1 - zb) * s_glob[:, i] + zb * s_hot[:, i])
        for x in ARMS:
            p_ = oof[x]
            per[x].append(dict(exact=float((p_ == Ytr).mean()),
                               mae=float(np.abs(p_ - Ytr).mean()),
                               qwk=qwk(p_.reshape(-1), Ytr.reshape(-1)),
                               perdim_qwk=[qwk(p_[:, i], Ytr[:, i]) for i in range(K)]))
        print('  repeat %d/%d' % (rep + 1, a.repeats), flush=True)

    nf = a.repeats * 5
    out = {'note': 'pooled cross-fitted OOF selection of z_d; train-only',
           'zgrid': [float(z) for z in ZGRID], 'z_selected': {}, 'arms': {}}
    print('\n===== 1. 各 competency 选中的 z*（%d 个外层折）=====' % nf)
    print('%-22s %8s %8s %10s %10s   分布' % ('维度', 'z*均值', 'z*中位', 'z*≤0.3', 'z*≥0.7'))
    from collections import Counter
    for i in range(K):
        v = np.array(zsel[i])
        c = dict(sorted(Counter(np.round(v, 1)).items()))
        out['z_selected'][DIMS[i]] = dict(
            mean=round(float(v.mean()), 3), median=round(float(np.median(v)), 3),
            frac_le_03=round(float((v <= 0.3).mean()), 3),
            frac_ge_07=round(float((v >= 0.7).mean()), 3),
            hist={str(k): int(x) for k, x in c.items()})
        print('%-22s %8.3f %8.3f %9.1f%% %9.1f%%   %s'
              % (DIMS[i], v.mean(), np.median(v), 100 * (v <= 0.3).mean(),
                 100 * (v >= 0.7).mean(), c))

    print('\n===== 2. 折外表现 =====')
    print('%-14s %14s %14s %14s' % ('arm', 'Exact', 'MAE', 'QWK(pooled)'))
    for x in ARMS:
        r = per[x]
        m = {k: (float(np.mean([q[k] for q in r])),
                 float(np.std([q[k] for q in r], ddof=1) / np.sqrt(len(r))))
             for k in ('exact', 'mae', 'qwk')}
        out['arms'][x] = {'cv': m}
        print('%-14s %7.4f±%.4f %7.4f±%.4f %7.4f±%.4f%s'
              % (x, m['exact'][0], m['exact'][1], m['mae'][0], m['mae'][1],
                 m['qwk'][0], m['qwk'][1], '  ← 上界(非方法)' if 'oracle' in x else ''))
    print('\n  配对差:')
    tstats = {}
    for x in ('oof_z', 'oracle_z'):
        for ref in ('global_only', 'hot'):
            cells = []
            for k in ('exact', 'qwk'):
                d = np.array([per[x][i][k] - per[ref][i][k] for i in range(a.repeats)])
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = d.mean() / se if se > 1e-12 else np.nan
                cells.append('%s %+.4f(t=%s)' % (k, d.mean(), '%+.2f' % t))
                out['arms'][x].setdefault('vs_' + ref, {})[k] = [
                    round(float(d.mean()), 5), round(float(t), 2)]
                if x == 'oof_z' and ref == 'hot' and k == 'qwk':
                    tstats['qwk_t'] = float(t); tstats['qwk_d'] = float(d.mean())
            print('    %-10s vs %-12s %s' % (x, ref, '  '.join(cells)))
    print('\n===== 3. 逐维 QWK =====')
    print('  %-22s %s' % ('维度', ' '.join('%14s' % x for x in ARMS)))
    for i in range(K):
        cells = []
        for x in ARMS:
            v = [q['perdim_qwk'][i] for q in per[x]]
            cells.append('%7.4f±%.4f' % (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        print('  %-22s %s' % (DIMS[i], ' '.join(cells)))

    print('\n===== 预先声明的停止条件裁决 =====')
    ca = tstats.get('qwk_t', 0) > 2
    cb = out['z_selected'][DIMS[2]]['frac_le_03'] > 0.5
    cc = out['z_selected'][DIMS[3]]['frac_ge_07'] > 0.5
    rows = [('a) QWK vs hot 配对 t > 2', ca, 't=%+.2f (Δ=%+.4f)'
             % (tstats.get('qwk_t', float('nan')), tstats.get('qwk_d', float('nan')))),
            ('b) Efficiency 多数折 z ≤ 0.3', cb,
             '%.1f%%' % (100 * out['z_selected'][DIMS[2]]['frac_le_03'])),
            ('c) Tissue 多数折 z ≥ 0.7', cc,
             '%.1f%%' % (100 * out['z_selected'][DIMS[3]]['frac_ge_07']))]
    allp = True
    for nm, ok, det in rows:
        allp = allp and bool(ok)
        print('  [%s] %-30s %s' % ('通过' if ok else '不通过', nm, det))
    out['verdict'] = dict(criteria={r[0]: bool(r[1]) for r in rows}, pass_all=bool(allp))
    print('\n  ==> %s' % ('**三条全过**：可评 25 条'
                          if allp else '**未全过**：按预先声明，'
                                       '**停掉 competency-adaptive selection 这条线**'))
    p = os.path.join(a.outdir, 'oof_z.json')
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
