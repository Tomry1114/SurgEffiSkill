"""零训练检查：凸 competency routing 的 z–QWK 曲线是否有结构。

背景：当前 SQ 用 α=softmax(θ+zΔ)。因 softmax 非线性，
      softmax(θ+0.4Δ) ≠ 0.6·softmax(θ)+0.4·softmax(θ+Δ)，
      故 z **不是** global→dynamic 的插值系数，中间值可能比两端都差
      （FINAL_sq 的 Efficiency 正是如此：z=0 好、z=1 好、z=0.404 坏）。

改用凸路由：α_d(x)=(1−z_d)·π_d + z_d·softmax(θ_d+Δ_d(x))
        ⇒ **分数层严格线性** s_d(x)=(1−z_d)·s_global(x)+z_d·s_hot(x)
        ⇒ 若 global 与 hot 同落一个等级区间，任何 z 亦然（对序数任务尤其重要）。

本脚本**不训练任何 z**：直接用现成的 s_global 与 s_hot 做凸插值，
扫 z∈{0,0.1,…,1}，画四个 competency 的折外 QWK 曲线。
  · 若 Depth/Bimanual/Efficiency 最优靠近 0 而 Tissue 靠近 1
    → competency-specific dynamic allocation 假设直接成立，值得训练可学 z_d。
  · 若曲线无结构 → 不必再投入。
train-only（10 repeats × 5 折，per-fold π*），**val 从不读取**。
"""
import os, io, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, fit_dtr, DIMS, K, PROJECT
from surgmind_sp import qwk, HOT

ZS = np.round(np.linspace(0, 1, 11), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--tag', default='z_sweep')
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
    print('%s | train %d | 零训练凸插值扫 z=%s | **val 从不读取**'
          % (a.tag, n, list(ZS)), flush=True)

    # curves[dim][z] = list over repeats of QWK
    curves = {i: {z: [] for z in ZS} for i in range(K)}
    exact_c = {i: {z: [] for z in ZS} for i in range(K)}
    ident = {i: [] for i in range(K)}          # global 与 hot 的等级一致率

    for rep in range(a.repeats):
        rs = np.random.RandomState(23000 + rep)
        fold = rs.permutation(n) % 5
        SG = np.zeros((n, K)); SH = np.zeros((n, K))
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            A1, _ = FS.pick_alpha(P, Ytr, tr, grids, 'qwk')
            U = routing_signals(P, A1, None)
            ms = fit_dtr(U[tr], P[tr], Ytr[tr], A1, HOT, a.seeds)
            ue = torch.from_numpy(U[te]).float(); oe = torch.from_numpy(P[te]).float()
            ut = torch.from_numpy(U[tr]).float(); ot = torch.from_numpy(P[tr]).float()
            Se, At = [], []
            for g in ms:
                g.eval()
                with torch.no_grad():
                    s_, _, _ = g(ue, oe)
                    _, al_, _ = g(ut, ot)
                Se.append(s_.numpy()); At.append(al_.numpy())
            SH[te] = np.mean(Se, 0)
            Ps = np.mean(At, 0).mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
            SG[te] = np.stack([P[te, i, :] @ Ps[i] for i in range(K)], 1)
        for i in range(K):
            ident[i].append(float((FS.lvl(SG[:, i]) == FS.lvl(SH[:, i])).mean()))
            for z in ZS:
                s = (1 - z) * SG[:, i] + z * SH[:, i]
                curves[i][z].append(qwk(FS.lvl(s), Ytr[:, i]))
                exact_c[i][z].append(float((FS.lvl(s) == Ytr[:, i]).mean()))
        print('  repeat %d/%d' % (rep + 1, a.repeats), flush=True)

    out = {'note': 'zero-training convex interpolation sweep; train-only', 'curves': {}}
    print('\n===== 逐 competency 的 z–QWK 曲线（%d repeats, mean±SE）=====' % a.repeats)
    hdr = '%-22s ' % '维度' + ' '.join('%7.1f' % z for z in ZS)
    print(hdr)
    for i in range(K):
        mu = [np.mean(curves[i][z]) for z in ZS]
        se = [np.std(curves[i][z], ddof=1) / np.sqrt(a.repeats) for z in ZS]
        best = int(np.argmax(mu))
        out['curves'][DIMS[i]] = dict(
            z=[float(z) for z in ZS], qwk_mean=[round(float(x), 4) for x in mu],
            qwk_se=[round(float(x), 4) for x in se], argmax_z=float(ZS[best]),
            qwk_at_0=round(float(mu[0]), 4), qwk_at_1=round(float(mu[-1]), 4),
            qwk_best=round(float(mu[best]), 4),
            global_hot_level_agreement=round(float(np.mean(ident[i])), 4))
        line = '%-22s ' % DIMS[i] + ' '.join('%7.4f' % x for x in mu)
        print(line)
        print('%-22s ' % '' + ' '.join(('  ←最优' if j == best else '       ')
                                       for j in range(len(ZS))))
    print('\n%-22s %10s %10s %10s %10s %14s'
          % ('维度', 'z*=argmax', 'QWK(z=0)', 'QWK(z=1)', 'QWK(z*)', 'global≡hot 比例'))
    for i in range(K):
        c = out['curves'][DIMS[i]]
        print('%-22s %10.1f %10.4f %10.4f %10.4f %13.1f%%'
              % (DIMS[i], c['argmax_z'], c['qwk_at_0'], c['qwk_at_1'], c['qwk_best'],
                 100 * c['global_hot_level_agreement']))

    print('\n===== 逐 competency 的 z–Exact 曲线 =====')
    print(hdr)
    for i in range(K):
        mu = [np.mean(exact_c[i][z]) for z in ZS]
        out['curves'][DIMS[i]]['exact_mean'] = [round(float(x), 4) for x in mu]
        print('%-22s ' % DIMS[i] + ' '.join('%7.4f' % x for x in mu))

    print('\n判定：若 Depth/Bimanual/Efficiency 的 z* 靠近 0 且 Tissue 的 z* 靠近 1，'
          '\n      则 competency-specific dynamic allocation 假设成立，值得训练可学 z_d。')
    zst = [out['curves'][DIMS[i]]['argmax_z'] for i in range(K)]
    ok = (zst[3] >= 0.7) and all(zst[i] <= 0.3 for i in (0, 1, 2))
    print('  z* = %s → %s' % (zst, '**结构成立**' if ok else '**结构不成立/不清晰**'))
    out['verdict_structure'] = bool(ok)
    p = os.path.join(a.outdir, 'z_sweep.json')
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
