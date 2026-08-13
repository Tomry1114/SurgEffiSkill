"""把 (hot − global_only) 的增益按四个 competency 分开。

为什么这个差就是「逐样本分量」：`global_only` 用的 α = hot 门控的跨样本平均 π*，
`hot` 用的是逐样本的 α_d(x)。两者共用同一次拟合、同一批参数，
**唯一差别就是 α 是否随样本变化** → 差值即纯逐样本效应，不含全局先验的贡献。

输出：
 1. 逐维 (hot − global_only) 的配对 CV 差（Exact / QWK / MAE），mean ± SE 与 t
 2. 逐维干预分析：hot 相对 global_only 改了哪些、修正/打坏、分歧百分位
 3. 同样给出 (hot − static) 作对照

train-only，10 repeats × 5 折，per-fold π；**val 从不读取**。
"""
import os, io, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, fit_dtr, predict, DIMS, K, PROJECT

HOT = dict(rank=2, lr=1e-2, epochs=300, lam_pair=0.5, lam_beta=0.0, wd=0.0, batch=32)
ARMS = ['static', 'global_only', 'hot']


def qwk(p, y, kmin=1, kmax=5):
    n = kmax - kmin + 1
    O = np.zeros((n, n))
    for aa, bb in zip(y.astype(int), p.astype(int)):
        O[aa - kmin, bb - kmin] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], float) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    de = float((W * E).sum())
    return 1 - float((W * O).sum()) / de if de > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--tag', default='dyn_by_skill')
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
    DIS = P.std(-1)
    grids = list(FS.simplex(P.shape[-1], a.step))
    print('%s | train %d | **val 从不读取**' % (a.tag, n), flush=True)

    # per_rep[arm][dim] = list over repeats of dict(exact,qwk,mae)
    per = {arm: [[] for _ in range(K)] for arm in ARMS}
    chg = {'hot_vs_global': [], 'hot_vs_static': []}

    for rep in range(a.repeats):
        rs = np.random.RandomState(3000 + rep)
        fold = rs.permutation(n) % 5
        oof = {arm: np.zeros((n, K)) for arm in ARMS}
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            A1, _ = FS.pick_alpha(P, Ytr, tr, grids, 'qwk')
            U = routing_signals(P, A1, None)
            oof['static'][te] = np.stack([FS.lvl(P[te, i, :] @ A1[i]) for i in range(K)], 1)
            gs = fit_dtr(U[tr], P[tr], Ytr[tr], A1, HOT, a.seeds)
            pl, _, _ = predict(gs, U[te], P[te])
            oof['hot'][te] = pl
            _, al_tr, _ = predict(gs, U[tr], P[tr])
            Ps = al_tr.mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
            oof['global_only'][te] = np.stack(
                [FS.lvl(P[te, i, :] @ Ps[i]) for i in range(K)], 1)
            for nm, bn in (('hot_vs_global', 'global_only'), ('hot_vs_static', 'static')):
                for i in range(K):
                    for j in np.where(oof['hot'][te, i] != oof[bn][te, i])[0]:
                        chg[nm].append(dict(dim=i, dis=float(DIS[te[j], i]),
                                            base=int(oof[bn][te[j], i]),
                                            new=int(oof['hot'][te[j], i]),
                                            y=int(Ytr[te[j], i])))
        for arm in ARMS:
            for i in range(K):
                p = oof[arm][:, i]
                per[arm][i].append(dict(exact=float((p == Ytr[:, i]).mean()),
                                        mae=float(np.abs(p - Ytr[:, i]).mean()),
                                        qwk=qwk(p, Ytr[:, i])))
        print('  repeat %d/%d' % (rep + 1, a.repeats), flush=True)

    out = {'note': 'train-only; hot−global_only = 纯逐样本分量', 'per_skill': {}}
    print('\n===== 1. 逐维折外表现（%d repeats, mean ± SE）=====' % a.repeats)
    print('%-22s %-12s %14s %14s %14s' % ('维度', 'arm', 'Exact', 'QWK', 'MAE'))
    for i in range(K):
        for arm in ARMS:
            r = per[arm][i]
            m = {k: (np.mean([x[k] for x in r]),
                     np.std([x[k] for x in r], ddof=1) / np.sqrt(len(r)))
                 for k in ('exact', 'qwk', 'mae')}
            print('%-22s %-12s %7.4f±%.4f %7.4f±%.4f %7.4f±%.4f'
                  % (DIMS[i] if arm == ARMS[0] else '', arm,
                     m['exact'][0], m['exact'][1], m['qwk'][0], m['qwk'][1],
                     m['mae'][0], m['mae'][1]))
        print()

    print('===== 2. **逐维**配对差 (hot − global_only) = 纯逐样本分量 =====')
    print('%-22s %18s %18s %18s' % ('维度', 'ΔExact', 'ΔQWK', 'ΔMAE(负=好)'))
    for i in range(K):
        row = {}
        cells = []
        for k in ('exact', 'qwk', 'mae'):
            d = np.array([per['hot'][i][j][k] - per['global_only'][i][j][k]
                          for j in range(a.repeats)])
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 1e-12 else np.nan
            row[k] = [round(float(d.mean()), 5), round(float(se), 5),
                      round(float(t), 2) if se > 1e-12 else None]
            cells.append('%+.4f(t=%s)' % (d.mean(), '%+.2f' % t if se > 1e-12 else 'n/a'))
        out['per_skill'].setdefault(DIMS[i], {})['hot_minus_global'] = row
        print('%-22s %18s %18s %18s' % (DIMS[i], *cells))

    print('\n  对照：逐维配对差 (hot − static)')
    for i in range(K):
        cells = []
        for k in ('exact', 'qwk', 'mae'):
            d = np.array([per['hot'][i][j][k] - per['static'][i][j][k]
                          for j in range(a.repeats)])
            se = d.std(ddof=1) / np.sqrt(len(d))
            cells.append('%+.4f(t=%s)' % (d.mean(),
                         '%+.2f' % (d.mean() / se) if se > 1e-12 else 'n/a'))
        print('  %-22s %18s %18s %18s' % (DIMS[i], *cells))

    print('\n===== 3. 逐维干预分析 =====')
    dis_all = DIS.reshape(-1)
    for nm, rec in chg.items():
        print('\n  --- hot 相对 %s ---' % nm.split('_vs_')[1])
        if not rec:
            print('    未改变任何预测'); continue
        for i in range(K):
            ri = [r for r in rec if r['dim'] == i]
            if not ri:
                print('    %-22s 0 次' % DIMS[i]); continue
            fi = sum(1 for r in ri if r['base'] != r['y'] and r['new'] == r['y'])
            bi = sum(1 for r in ri if r['base'] == r['y'] and r['new'] != r['y'])
            cd = np.array([r['dis'] for r in ri])
            pct = float(np.mean([(dis_all < d).mean() for d in cd]))
            if nm == 'hot_vs_global':
                out['per_skill'].setdefault(DIMS[i], {})['changed_vs_global'] = dict(
                    n=len(ri), fix=fi, brk=bi, net=fi - bi, dis_pct=round(pct, 4))
            print('    %-22s %4d 次 | 修正 %3d / 打坏 %3d → **净 %+d** | 分歧百分位 %.1f%%'
                  % (DIMS[i], len(ri), fi, bi, fi - bi, 100 * pct))

    print('\n===== 4. 结论指向 =====')
    for i in range(K):
        q = out['per_skill'][DIMS[i]]['hot_minus_global']['qwk']
        c = out['per_skill'][DIMS[i]].get('changed_vs_global', {})
        v = ('有益' if (q[2] or 0) > 2 else ('有害' if (q[2] or 0) < -2 else '无显著效应'))
        print('  %-22s ΔQWK %+.4f (t=%s) | 干预净 %+d → **动态在该维%s**'
              % (DIMS[i], q[0], q[2], c.get('net', 0), v))

    p = os.path.join(a.outdir, 'dyn_by_skill.json')
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
