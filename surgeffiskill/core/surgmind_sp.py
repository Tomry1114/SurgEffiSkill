"""SurgMind-SP：Self-Pruning Competency MMoE。

    α_d(x) = softmax[ θ_d + z_d · Δ_d(x) ]

  θ_d       全局 competency 先验，**强制保留**（warm-start 自 π*）
  Δ_d(x)    低秩逐样本动态残差分支（warm-start 自 hot）
  z_d       **近似二值**的 competency 开关（hard-concrete / L0，Louizos et al. 2018）

与 G1/G2/G3/E1 的本质区别：那四个都是在**固定 expert 预测上事后做离散模型选择**
（26 条/折 → 必然高方差）；本方法把结构选择放进**训练目标**，用全部训练数据的梯度决定。

为何用 hard-concrete 而非线性 γ：G3 的收缩公式不动点在中间
（Tissue 需要 ≈1 却被压在 0.6）；hard-concrete 的不动点在两端。

损失：Huber + λ_pair·pairwise + **λ_z·L0(z)** [+ λ_g·group-lasso(Δ_d)]
默认倾向 z_d=0（即 global-only）；只有该 competency 的动态分支持续降低
训练损失，才有足够梯度把 z_d 打开。

**λ_z 不做选择**：扫 {0.01,0.05,0.2} 全部报告——要看的是 z 的**模式**
（z_Tissue→1 而其余→0）在不同 λ 下是否稳定，挑最好的 λ 即是变相 cherry-picking。

**warm-start 自检**：z 强制为 1 时 SP 必须复现 hot，脚本内打印一致率。
train-only，val 从不读取。
"""
import os, io, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
import surgmind_dtr as SD
from surgmind_dtr import routing_signals, fit_dtr, pair_loss, DIMS, K, PROJECT

HOT = dict(rank=2, lr=1e-2, epochs=300, lam_pair=0.5, lam_beta=0.0, wd=0.0, batch=32)
# hard-concrete 常数（Louizos et al. 2018 的标准取值）
HC_B, HC_G, HC_Z = 2.0 / 3.0, -0.1, 1.1
LZS = [0.01, 0.05, 0.2]
THR_T = torch.tensor([1.5, 2.5, 3.5, 4.5])


def soft_lvl(s, tau=0.1):
    return 1.0 + torch.sigmoid((s.unsqueeze(-1) - THR_T) / tau).sum(-1)


def soft_qwk_loss(sl, y, kmin=1, kmax=5, tau=0.1):
    """可导 QWK 代理（与 soft_surrogate2.py 同一实现）。
    SS2 门槛检验：Tissue 上 Δ=−0.035 (t≈−3.1, 三个 τ 一致) → 该代理**看得见**
    hot 的收益，而 Huber/L1 均盲。"""
    ks = torch.arange(kmin, kmax + 1, dtype=torch.float32)
    Pm = torch.softmax(-(sl.unsqueeze(-1) - ks) ** 2 / (2 * tau ** 2), -1)
    Ym = torch.zeros_like(Pm)
    Ym.scatter_(1, (y.long() - kmin).clamp(0, len(ks) - 1).unsqueeze(-1), 1.0)
    C = len(ks)
    W = ((ks.view(-1, 1) - ks.view(1, -1)) ** 2) / (C - 1) ** 2
    Om = Ym.t() @ Pm
    Em = torch.outer(Ym.sum(0), Pm.sum(0)) / max(len(y), 1)
    return (W * Om).sum() / (W * Em).sum().clamp_min(1e-8)


def qwk(p, y, kmin=1, kmax=5):
    n = kmax - kmin + 1
    O = np.zeros((n, n))
    for aa, bb in zip(y.astype(int), p.astype(int)):
        O[aa - kmin, bb - kmin] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], float) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    de = float((W * E).sum())
    return 1 - float((W * O).sum()) / de if de > 0 else 0.0


class SPGate(nn.Module):
    """global 分支强制保留；dynamic 分支由近似二值的 z_d 开关控制。"""

    def __init__(self, nf, M, num_dims=K, rank=2):
        super().__init__()
        self.M, self.nd = M, num_dims
        self.theta = nn.Parameter(torch.zeros(num_dims, M))     # θ_d（可学，warm-start π*）
        self.V = nn.Linear(nf, rank, bias=False)
        self.U = nn.Parameter(torch.zeros(num_dims, rank, M))
        # log α 初值 → z 初值约 0.5，不预设开或关
        self.log_alpha = nn.Parameter(torch.zeros(num_dims))
        self.force_z = None                                      # 自检用：强制 z

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
        """L0 期望惩罚（可导）。"""
        return torch.sigmoid(self.log_alpha - HC_B * np.log(-HC_G / HC_Z)).sum()

    def forward(self, U_feat, O):
        r = torch.tanh(self.V(U_feat))
        delta = torch.einsum('nkr,kro->nko', r, self.U)
        z = self.z()
        a = F.softmax(self.theta.unsqueeze(0) + z.view(1, -1, 1) * delta, -1)
        return (a * O).sum(-1), a, z


def fit_sp(Ufe, O, Y, theta0, V0, U0, lam_z, lam_g, cfg, seeds=2, freeze_theta=False):
    ms = []
    for sd in range(seeds):
        torch.manual_seed(100 + sd); np.random.seed(100 + sd)
        g = SPGate(Ufe.shape[-1], O.shape[-1], rank=cfg['rank'])
        with torch.no_grad():
            g.theta.copy_(torch.as_tensor(theta0, dtype=torch.float32))
            g.V.weight.copy_(V0)
            g.U.copy_(U0)
        # 门控参数一律 weight_decay=0（本项目已多次因 wd 压死门控踩坑）
        if freeze_theta:
            g.theta.requires_grad_(False)          # θ 固定 → z=0 严格等于 global_only
        main_p = ([g.U] + list(g.V.parameters())
                  if freeze_theta else [g.U, g.theta] + list(g.V.parameters()))
        opt = torch.optim.AdamW([
            {'params': main_p, 'lr': cfg['lr'], 'weight_decay': 0.0},
            {'params': [g.log_alpha], 'lr': cfg['lr'] * 3, 'weight_decay': 0.0}])
        u = torch.from_numpy(Ufe).float(); o = torch.from_numpy(O).float()
        yt = torch.from_numpy(Y).long()
        for _ in range(cfg['epochs']):
            g.train()
            perm = torch.randperm(len(u))
            for i in range(0, len(perm), cfg['batch']):
                b = perm[i:i + cfg['batch']]
                s, _, _ = g(u[b], o[b])
                if cfg.get('surrogate') == 'sqwk':
                    # 逐维求 soft-QWK 再取平均（κ 需要成批样本，故按 batch 计算）
                    loss = torch.stack([
                        soft_qwk_loss(soft_lvl(s[:, d], cfg.get('tau', 0.1)),
                                      yt[b][:, d], tau=cfg.get('tau', 0.1))
                        for d in range(s.shape[1])]).mean()
                else:
                    loss = F.huber_loss(s, yt[b].float())
                if cfg['lam_pair'] > 0:
                    loss = loss + cfg['lam_pair'] * pair_loss(s, yt[b])
                loss = loss + lam_z * g.l0()
                if lam_g > 0:                       # 逐维 group-lasso：整条动态分支一起压
                    loss = loss + lam_g * g.U.pow(2).sum((1, 2)).add(1e-8).sqrt().sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(g.parameters(), 1.0)
                opt.step()
        ms.append(g)
    return ms


def sp_predict(ms, Ufe, O, force_z=None):
    u = torch.from_numpy(Ufe).float(); o = torch.from_numpy(O).float()
    S, Z = [], []
    for g in ms:
        g.eval(); g.force_z = force_z
        with torch.no_grad():
            s, _, z = g(u, o)
        g.force_z = None
        S.append(s.numpy()); Z.append(z.numpy())
    return FS.lvl(np.mean(S, 0)), np.mean(Z, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--lam-g', dest='lam_g', type=float, default=0.0)
    ap.add_argument('--tag', default='surgmind_sp')
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
    ARMS = ['static', 'global_only', 'hot'] + ['sp_lz%g' % l for l in LZS] + ['oracle']
    print('%s | train %d | λ_z 扫 %s (全部报告) | λ_g=%g | **val 从不读取**'
          % (a.tag, n, LZS, a.lam_g), flush=True)

    per = {arm: [] for arm in ARMS}
    zrec = {l: [[] for _ in range(K)] for l in LZS}
    warm_agree = []

    for rep in range(a.repeats):
        rs = np.random.RandomState(11000 + rep)
        fold = rs.permutation(n) % 5
        oof = {arm: np.zeros((n, K)) for arm in ARMS}
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            A1, _ = FS.pick_alpha(P, Ytr, tr, grids, 'qwk')
            U = routing_signals(P, A1, None)
            hm = fit_dtr(U[tr], P[tr], Ytr[tr], A1, HOT, a.seeds)

            # hot 的折外预测 + π*（θ_d 的 warm-start）
            ue = torch.from_numpy(U[te]).float(); oe = torch.from_numpy(P[te]).float()
            ut = torch.from_numpy(U[tr]).float(); ot = torch.from_numpy(P[tr]).float()
            Se, At, Bt = [], [], []
            for g in hm:
                g.eval()
                with torch.no_grad():
                    s_, _, _ = g(ue, oe)
                    _, al_, be_ = g(ut, ot)
                Se.append(s_.numpy()); At.append(al_.numpy()); Bt.append(be_.numpy())
            oof['hot'][te] = FS.lvl(np.mean(Se, 0))
            Ps = np.mean(At, 0).mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
            beta_mean = np.mean(Bt, 0).mean(0)                    # (K,) hot 的平均 β
            oof['static'][te] = np.stack([FS.lvl(P[te, i, :] @ A1[i]) for i in range(K)], 1)
            oof['global_only'][te] = np.stack(
                [FS.lvl(P[te, i, :] @ Ps[i]) for i in range(K)], 1)
            for i in range(K):
                bo = qwk(oof['hot'][te, i], Ytr[te, i]) > qwk(oof['global_only'][te, i],
                                                              Ytr[te, i])
                oof['oracle'][te, i] = (oof['hot'][te, i] if bo
                                        else oof['global_only'][te, i])

            # warm-start：θ←log π*；V←hot.V；U←hot.U × 平均 β（吸收 β 使 z=1 ≈ hot）
            theta0 = np.log(Ps + 1e-8)
            V0 = hm[0].V.weight.data.clone()
            U0 = hm[0].U.data.clone() * torch.as_tensor(
                beta_mean, dtype=torch.float32).view(-1, 1, 1)

            for lz in LZS:
                cfg = dict(rank=HOT['rank'], lr=3e-3, epochs=200,
                           lam_pair=0.5, batch=32)
                ms = fit_sp(U[tr], P[tr], Ytr[tr], theta0, V0, U0,
                            lz, a.lam_g, cfg, a.seeds)
                pl, z = sp_predict(ms, U[te], P[te])
                oof['sp_lz%g' % lz][te] = pl
                for i in range(K):
                    zrec[lz][i].append(float(z[i]))
                if lz == LZS[1] and rep == 0:      # warm-start 自检（未训练前）
                    m0 = fit_sp(U[tr], P[tr], Ytr[tr], theta0, V0, U0, lz, a.lam_g,
                                dict(cfg, epochs=0), 1)
                    p1, _ = sp_predict(m0, U[te], P[te], force_z=1.0)
                    warm_agree.append(float((p1 == oof['hot'][te]).mean()))
        for arm in ARMS:
            p_ = oof[arm]
            per[arm].append(dict(
                exact=float((p_ == Ytr).mean()), mae=float(np.abs(p_ - Ytr).mean()),
                qwk=qwk(p_.reshape(-1), Ytr.reshape(-1)),
                mix=float(np.mean([FS.dim_obj(Ytr[:, i], p_[:, i], 'mix') for i in range(K)])),
                perdim_qwk=[qwk(p_[:, i], Ytr[:, i]) for i in range(K)]))
        print('  repeat %d/%d' % (rep + 1, a.repeats), flush=True)

    print('\n===== 0. warm-start 自检（z 强制=1，未训练）=====')
    print('  与 hot 的逐元素一致率: %.3f ± %.3f（越接近 1 说明 warm-start 确实复现了 hot）'
          % (np.mean(warm_agree), np.std(warm_agree)) if warm_agree else '  (未采集)')

    out = {'note': 'train-only; self-pruning competency switch', 'lam_g': a.lam_g,
           'warm_start_agreement_with_hot': [round(float(np.mean(warm_agree)), 4),
                                             round(float(np.std(warm_agree)), 4)]
           if warm_agree else None, 'arms': {}}

    print('\n===== 1. 学到的 z_d（%d 折；关键结果）=====' % (a.repeats * 5))
    out['z'] = {}
    for lz in LZS:
        print('  --- λ_z=%g ---' % lz)
        out['z'][str(lz)] = {}
        for i in range(K):
            v = np.array(zrec[lz][i])
            out['z'][str(lz)][DIMS[i]] = [round(float(v.mean()), 3),
                                          round(float(v.std()), 3),
                                          round(float((v > 0.5).mean()), 3)]
            print('    %-22s z 均值 %.3f  SD %.3f  | z>0.5 的折 %5.1f%%  | z>0.9 %5.1f%%'
                  % (DIMS[i], v.mean(), v.std(), 100 * (v > 0.5).mean(),
                     100 * (v > 0.9).mean()))

    print('\n===== 2. 折外表现 =====')
    print('%-16s %14s %14s %14s %14s' % ('arm', 'Exact', 'MAE', 'QWK(pooled)', 'mix'))
    for arm in ARMS:
        r = per[arm]
        m = {k: (float(np.mean([x[k] for x in r])),
                 float(np.std([x[k] for x in r], ddof=1) / np.sqrt(len(r))))
             for k in ('exact', 'mae', 'qwk', 'mix')}
        out['arms'][arm] = {'cv': m}
        print('%-16s %7.4f±%.4f %7.4f±%.4f %7.4f±%.4f %7.4f±%.4f%s'
              % (arm, m['exact'][0], m['exact'][1], m['mae'][0], m['mae'][1],
                 m['qwk'][0], m['qwk'][1], m['mix'][0], m['mix'][1],
                 '  ← 上界(非方法)' if arm == 'oracle' else ''))

    print('\n  配对差:')
    for arm in ['sp_lz%g' % l for l in LZS] + ['oracle']:
        for ref in ('global_only', 'hot'):
            cells = []
            for k in ('exact', 'qwk', 'mix'):
                d = np.array([per[arm][i][k] - per[ref][i][k] for i in range(a.repeats)])
                se = d.std(ddof=1) / np.sqrt(len(d))
                cells.append('%s %+.4f(t=%s)' % (
                    k, d.mean(), '%+.2f' % (d.mean() / se) if se > 1e-12 else 'n/a'))
                out['arms'][arm].setdefault('vs_' + ref, {})[k] = [
                    round(float(d.mean()), 5), round(float(se), 5),
                    round(float(d.mean() / se), 2) if se > 1e-12 else None]
            print('    %-16s vs %-12s %s' % (arm, ref, '  '.join(cells)))

    print('\n===== 3. 逐维 QWK =====')
    print('  %-22s %s' % ('维度', ' '.join('%13s' % x for x in ARMS)))
    for i in range(K):
        cells = []
        for arm in ARMS:
            v = [x['perdim_qwk'][i] for x in per[arm]]
            cells.append('%6.4f±%.4f' % (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
        print('  %-22s %s' % (DIMS[i], ' '.join(cells)))

    p = os.path.join(a.outdir, 'surgmind_sp.json')
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
