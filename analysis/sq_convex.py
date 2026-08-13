"""Convex Competency Routing（SQ_convex）：z_d 真正是 global→dynamic 的分配系数。

    α_d(x) = (1 − z_d)·π_d + z_d·softmax(θ_d + Δ_d(x))
  ⇒ s_d(x) = (1 − z_d)·s_global(x) + z_d·s_dyn(x)      **分数层严格线性**

对比原 SQ 的 α=softmax(θ+z·Δ)：因 softmax 非线性，z 并非插值系数，
预测可跑到 [s_global, s_hot] 线段**之外**——TD2 实测 Efficiency 上
|s_SQ−s_hot|=0.0318、|s_SQ−s_g|=0.0288 双双大于 |s_hot−s_g|=0.0162，即为明证；
这正是 z_Eff=0.404 比 z=0 与 z=1 都差的机制原因。

凸路由的序数性质：若 s_global 与 s_hot 同落一个等级区间，则任意 z∈[0,1] 亦然
→ 结构上排除「两端都对、中间翻错」。

前置（ZS 零训练扫描，train-only）：z* = [0, 0, 0, 1]，四条曲线**单调**
→ competency-specific dynamic allocation 假设成立，故值得训练可学 z_d。

其余一切保持不变：θ 冻结、soft-QWK 代理、hard-concrete z + L0、同 λ 网格、
同 train-only CV 协议。**val 从不读取。**
"""
import os, io, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, fit_dtr, pair_loss, DIMS, K, PROJECT
from surgmind_sp import qwk, HOT, soft_lvl, soft_qwk_loss, HC_B, HC_G, HC_Z

LZS = [0.002, 0.005, 0.01, 0.02, 0.05]


class ConvexGate(nn.Module):
    """s_d(x) = (1−z_d)·s_global(x) + z_d·s_dyn(x)；z_d 为 hard-concrete 近似二值。"""

    def __init__(self, nf, M, pi, num_dims=K, rank=2):
        super().__init__()
        self.nd = num_dims
        self.register_buffer('pi', torch.as_tensor(pi, dtype=torch.float32))     # π*（冻结）
        self.register_buffer('theta', torch.log(
            torch.as_tensor(pi, dtype=torch.float32) + 1e-8))                    # θ（冻结）
        self.V = nn.Linear(nf, rank, bias=False)
        self.U = nn.Parameter(torch.zeros(num_dims, rank, M))
        self.log_alpha = nn.Parameter(torch.zeros(num_dims))
        self.force_z = None

    def z(self):
        if self.force_z is not None:
            return torch.full((self.nd,), float(self.force_z))
        if self.training:
            u = torch.rand(self.nd).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / HC_B)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (HC_Z - HC_G) + HC_G).clamp(0, 1)

    def l0(self):
        return torch.sigmoid(self.log_alpha - HC_B * np.log(-HC_G / HC_Z)).sum()

    def forward(self, U_feat, O):
        r = torch.tanh(self.V(U_feat))
        delta = torch.einsum('nkr,kro->nko', r, self.U)
        a_dyn = F.softmax(self.theta.unsqueeze(0) + delta, -1)
        s_dyn = (a_dyn * O).sum(-1)
        s_glob = (self.pi.unsqueeze(0) * O).sum(-1)
        z = self.z()
        s = (1 - z).view(1, -1) * s_glob + z.view(1, -1) * s_dyn
        a = ((1 - z).view(1, -1, 1) * self.pi.unsqueeze(0)
             + z.view(1, -1, 1) * a_dyn)
        return s, a, z


def fit_convex(Ufe, O, Y, pi, V0, U0, lam_z, cfg, seeds=2):
    ms = []
    for sd in range(seeds):
        torch.manual_seed(200 + sd); np.random.seed(200 + sd)
        g = ConvexGate(Ufe.shape[-1], O.shape[-1], pi, rank=cfg['rank'])
        with torch.no_grad():
            g.V.weight.copy_(V0); g.U.copy_(U0)
        opt = torch.optim.AdamW([                       # 门控参数一律 wd=0
            {'params': [g.U] + list(g.V.parameters()), 'lr': cfg['lr'], 'weight_decay': 0.0},
            {'params': [g.log_alpha], 'lr': cfg['lr'] * 3, 'weight_decay': 0.0}])
        u = torch.from_numpy(Ufe).float(); o = torch.from_numpy(O).float()
        yt = torch.from_numpy(Y).long()
        for _ in range(cfg['epochs']):
            g.train()
            perm = torch.randperm(len(u))
            for i in range(0, len(perm), cfg['batch']):
                b = perm[i:i + cfg['batch']]
                s, _, _ = g(u[b], o[b])
                loss = torch.stack([
                    soft_qwk_loss(soft_lvl(s[:, d], cfg['tau']), yt[b][:, d], tau=cfg['tau'])
                    for d in range(s.shape[1])]).mean()
                if cfg['lam_pair'] > 0:
                    loss = loss + cfg['lam_pair'] * pair_loss(s, yt[b])
                loss = loss + lam_z * g.l0()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(g.parameters(), 1.0)
                opt.step()
        ms.append(g)
    return ms


def cvx_predict(ms, Ufe, O):
    u = torch.from_numpy(Ufe).float(); o = torch.from_numpy(O).float()
    S, Z = [], []
    for g in ms:
        g.eval()
        with torch.no_grad():
            s, _, z = g(u, o)
        S.append(s.numpy()); Z.append(z.numpy())
    return FS.lvl(np.mean(S, 0)), np.mean(Z, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--tag', default='sq_convex')
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
    ARMS = (['global_only', 'hot'] + ['cvx_lz%g' % l for l in LZS]
            + ['cvx_nested', 'oracle_z'])
    print('%s | train %d | 凸路由 | soft-QWK τ=%.2f | λ_z 扫 %s | **val 从不读取**'
          % (a.tag, n, a.tau, LZS), flush=True)

    per = {x: [] for x in ARMS}
    zrec = {l: [[] for _ in range(K)] for l in LZS}
    lzsel = []
    sanity = []

    def prep(tr):
        A1, _ = FS.pick_alpha(P, Ytr, tr, grids, 'qwk')
        U = routing_signals(P, A1, None)
        ms = fit_dtr(U[tr], P[tr], Ytr[tr], A1, HOT, a.seeds)
        ut = torch.from_numpy(U[tr]).float(); ot = torch.from_numpy(P[tr]).float()
        Al, Bl = [], []
        for g in ms:
            g.eval()
            with torch.no_grad():
                _, al, be = g(ut, ot)
            Al.append(al.numpy()); Bl.append(be.numpy())
        Ps = np.mean(Al, 0).mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
        bm = np.mean(Bl, 0).mean(0)
        V0 = ms[0].V.weight.data.clone()
        U0 = ms[0].U.data.clone() * torch.as_tensor(bm, dtype=torch.float32).view(-1, 1, 1)
        return A1, U, ms, Ps, V0, U0

    for rep in range(a.repeats):
        rs = np.random.RandomState(25000 + rep)
        fold = rs.permutation(n) % 5
        oof = {x: np.zeros((n, K)) for x in ARMS}
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            A1, U, hm, Ps, V0, U0 = prep(tr)
            ue = torch.from_numpy(U[te]).float(); oe = torch.from_numpy(P[te]).float()
            Se = []
            for g in hm:
                g.eval()
                with torch.no_grad():
                    s_, _, _ = g(ue, oe)
                Se.append(s_.numpy())
            s_hot = np.mean(Se, 0)
            s_glob = np.stack([P[te, i, :] @ Ps[i] for i in range(K)], 1)
            oof['hot'][te] = FS.lvl(s_hot)
            oof['global_only'][te] = FS.lvl(s_glob)
            # 零训练 oracle：逐维在 {0,1} 上取更优（与 ZS 的 z*=[0,0,0,1] 呼应）
            for i in range(K):
                oof['oracle_z'][te, i] = (FS.lvl(s_hot[:, i])
                                          if qwk(FS.lvl(s_hot[:, i]), Ytr[te, i])
                                          > qwk(FS.lvl(s_glob[:, i]), Ytr[te, i])
                                          else FS.lvl(s_glob[:, i]))
            cfg = dict(rank=HOT['rank'], lr=3e-3, epochs=200, lam_pair=0.5,
                       batch=32, tau=a.tau)
            Uc = routing_signals(P, Ps, None)          # 路由信号按 π* 重算
            for lz in LZS:
                ms = fit_convex(Uc[tr], P[tr], Ytr[tr], Ps, V0, U0, lz, cfg, a.seeds)
                pl, z = cvx_predict(ms, Uc[te], P[te])
                oof['cvx_lz%g' % lz][te] = pl
                for i in range(K):
                    zrec[lz][i].append(float(z[i]))
                if lz == LZS[0] and rep == 0 and f == 0:
                    # 自检：强制 z=0 应严格等于 global_only；z=1 应接近 hot 的动态支路
                    for g in ms:
                        g.force_z = 0.0
                    p0, _ = cvx_predict(ms, Uc[te], P[te])
                    for g in ms:
                        g.force_z = None
                    sanity.append(float((p0 == oof['global_only'][te]).mean()))
            # 嵌套选 λ_z（只用外层训练折）
            iw = {lz: 0.0 for lz in LZS}
            rs2 = np.random.RandomState(31000 + rep * 5 + f)
            ifold = rs2.permutation(len(tr)) % 3
            for gi in range(3):
                ite = tr[ifold == gi]; itr = tr[ifold != gi]
                _, _, _, Ps2, V2, U2 = prep(itr)
                Uc2 = routing_signals(P, Ps2, None)
                for lz in LZS:
                    m2 = fit_convex(Uc2[itr], P[itr], Ytr[itr], Ps2, V2, U2, lz, cfg, a.seeds)
                    p2, _ = cvx_predict(m2, Uc2[ite], P[ite])
                    iw[lz] += np.mean([qwk(p2[:, i], Ytr[ite, i]) for i in range(K)])
            best = max(LZS, key=lambda l: iw[l]); lzsel.append(best)
            ms = fit_convex(Uc[tr], P[tr], Ytr[tr], Ps, V0, U0, best, cfg, a.seeds)
            pl, _ = cvx_predict(ms, Uc[te], P[te])
            oof['cvx_nested'][te] = pl
        for x in ARMS:
            p_ = oof[x]
            per[x].append(dict(exact=float((p_ == Ytr).mean()),
                               mae=float(np.abs(p_ - Ytr).mean()),
                               qwk=qwk(p_.reshape(-1), Ytr.reshape(-1)),
                               perdim_qwk=[qwk(p_[:, i], Ytr[:, i]) for i in range(K)]))
        print('  repeat %d/%d' % (rep + 1, a.repeats), flush=True)

    from collections import Counter
    out = {'note': 'convex competency routing; train-only', 'tau': a.tau,
           'sanity_z0_equals_global': [round(float(np.mean(sanity)), 4)] if sanity else None,
           'lz_selected': {str(k): v for k, v in Counter(lzsel).items()}, 'z': {}}
    print('\n===== 0. 自检：强制 z=0 是否严格等于 global_only =====')
    print('  一致率 %.4f（凸参数化下应为 1.0000）' % (np.mean(sanity) if sanity else -1))

    print('\n===== 1. 学到的 z_d（%d 折）=====' % (a.repeats * 5))
    for lz in LZS:
        print('  --- λ_z=%g ---' % lz)
        out['z'][str(lz)] = {}
        for i in range(K):
            v = np.array(zrec[lz][i])
            out['z'][str(lz)][DIMS[i]] = [round(float(v.mean()), 3),
                                          round(float((v > 0.5).mean()), 3)]
            print('    %-22s z 均值 %.3f  SD %.3f | z>0.5 %5.1f%%'
                  % (DIMS[i], v.mean(), v.std(), 100 * (v > 0.5).mean()))
    print('\n  嵌套选中的 λ_z: %s' % dict(Counter(lzsel)))

    print('\n===== 2. 折外表现 =====')
    print('%-14s %14s %14s %14s' % ('arm', 'Exact', 'MAE', 'QWK(pooled)'))
    out['arms'] = {}
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
    for x in ['cvx_lz%g' % l for l in LZS] + ['cvx_nested', 'oracle_z']:
        for ref in ('global_only', 'hot'):
            cells = []
            for k in ('exact', 'qwk'):
                d = np.array([per[x][i][k] - per[ref][i][k] for i in range(a.repeats)])
                se = d.std(ddof=1) / np.sqrt(len(d))
                cells.append('%s %+.4f(t=%s)' % (
                    k, d.mean(), '%+.2f' % (d.mean() / se) if se > 1e-12 else 'n/a'))
                out['arms'][x].setdefault('vs_' + ref, {})[k] = [
                    round(float(d.mean()), 5),
                    round(float(d.mean() / se), 2) if se > 1e-12 else None]
            print('    %-14s vs %-12s %s' % (x, ref, '  '.join(cells)))
    print('\n===== 3. 逐维 QWK =====')
    print('  %-22s %s' % ('维度', ' '.join('%12s' % x for x in ARMS)))
    for i in range(K):
        cells = []
        for x in ARMS:
            v = [q['perdim_qwk'][i] for q in per[x]]
            cells.append('%6.4f±%.4f' % (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        print('  %-22s %s' % (DIMS[i], ' '.join(cells)))
    p = os.path.join(a.outdir, 'sq_convex.json')
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
