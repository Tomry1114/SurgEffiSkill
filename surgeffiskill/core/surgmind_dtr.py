# -*- coding: utf-8 -*-
"""SurgMind-DTR：Static Prior + **Disagreement-Triggered** Dynamic Routing。

设计动机（全部由本项目的实测逼出来）：
  * 逐技能维度的**静态**证据偏好 π_d 已验证有效（25 视频上六项达到/超过 SFT baseline）；
  * 而自由的逐样本 dynamic routing 反复被内层 CV 拒绝（总选 rank=0），
    因为它拿 4 万维特征去为 98 个病例各猜一组权重，自由度过高；
  * 但 oracle 显示空间确实存在：静态混合 QWK 0.6309 vs **逐病例选专家 0.6988**。
    中间 0.068 就是"某些病例应该改变 mixture"。

所以不是"要不要动态路由"，而是"**如何让它只在有依据时才动**"：

    α_d(x) = softmax( log π_d + β_d(x) · Δ_d(u_d(x)) )
    β_d(x) = σ( a_dᵀ u_d(x) + b_d )        ← 这个病例**是否需要**动态修正

  π_d      冻结自网格搜索得到的已验证静态解
  u_d(x)   **低维**路由信号（约 10–20 个标量），不是原始 embedding：
             各专家预测值 / 专家间 std / max−min / SFT 与 ridge 均值之差 /
             静态混合分到四个序数阈值的距离 / clip 级时间方差
  Δ_d      极小的修正网络（共享低秩投影 + 逐维读出）
  β_d(x)   触发门：只有当专家分歧大、分数贴近阈值、时间不一致时才接近 1

**零初始化保证**：Δ 的读出层零初始化 ⇒ epoch 0 时 α_d ≡ π_d，
即模型起点**逐位等于**已验证的静态最优解；内层 CV 若判定修正无益就停在 epoch 0。
因此这个结构在 CV 口径下**不可能比静态版更差**。

与近期 uncertainty-aware routing 的关系：MoFME / MoGU 把 expert uncertainty 作为
routing 信号而非仅用线性 router；MLoRE 保留稳定共享路径、只让低秩部分做受约束适配。
本设计把两者结合到小样本临床评分场景：稳定的 task prior 承担主要预测，
分歧度触发的低维修正只在必要时介入。
"""
import argparse
import io
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'revision/scripts/sft_grpo'))
import train_mmoe_lite as T                                        # noqa: E402
import ridge_moe as RM                                            # noqa: E402
import final_static_moe as FS                                     # noqa: E402

DIMS = FS.DIMS
K = FS.K
THR = FS.THR
torch.set_num_threads(int(os.environ.setdefault('OMP_NUM_THREADS', '4')))


def routing_signals(O, pi, tvar=None):
    """由**专家预测本身**构造低维路由信号 u_d(x)。

    O:   (n, K, M) 各专家在等级尺度上的预测
    pi:  (K, M) 静态先验（用来算静态混合分，再算它到序数阈值的距离）
    tvar:(n, K) 可选的 clip 级时间方差（时间不一致性）
    返回 (n, K, F)。F ≈ M + 3 + 4 (+1)，约 12–13 维——不是原始 embedding。
    """
    n, Kd, M = O.shape
    mix = np.einsum('nkm,km->nk', O, pi)
    feats = [O,                                             # 各专家预测值 (M)
             O.std(-1, keepdims=True),                      # 专家间分歧（std）
             (O.max(-1) - O.min(-1))[..., None],            # 分歧幅度 max−min
             (O[:, :, -1:] - O[:, :, :-1].mean(-1, keepdims=True))]   # SFT 与 ridge 均值之差
    feats.append(np.stack([np.abs(mix - t) for t in THR], -1))       # 到四个阈值的距离
    if tvar is not None:
        feats.append(tvar[:, :, None])                      # clip 级时间方差
    return np.concatenate(feats, -1).astype(np.float32)


class DTRGate(nn.Module):
    def __init__(self, nf, M, num_dims=K, rank=2):
        super().__init__()
        self.M, self.num_dims = M, num_dims
        self.register_buffer('logpi', torch.zeros(num_dims, M))     # 冻结的静态先验
        self.V = nn.Linear(nf, rank, bias=False)                    # 共享低秩投影
        self.U = nn.Parameter(torch.zeros(num_dims, rank, M))       # 零初始化 → 起点 = π
        self.a = nn.Parameter(torch.zeros(num_dims, nf))            # 触发门 β_d(x)
        self.b = nn.Parameter(torch.full((num_dims,), -2.0))        # 初始 β≈0.12，默认不触发

    def set_prior(self, pi):
        with torch.no_grad():
            self.logpi.copy_(torch.log(torch.as_tensor(pi, dtype=torch.float32) + 1e-8))

    def forward(self, U_feat, O):
        """U_feat: (n,K,F)  O: (n,K,M)"""
        r = torch.tanh(self.V(U_feat))                              # (n,K,rank)
        delta = torch.einsum('nkr,kro->nko', r, self.U)             # (n,K,M)
        beta = torch.sigmoid((self.a.unsqueeze(0) * U_feat).sum(-1) + self.b)   # (n,K)
        a = F.softmax(self.logpi.unsqueeze(0) + beta.unsqueeze(-1) * delta, -1)
        return (a * O).sum(-1), a, beta


def pair_loss(s, y, margin=1.0):
    tot, c = 0.0, 0
    for d in range(s.shape[1]):
        diff = y[:, d].float().unsqueeze(0) - y[:, d].float().unsqueeze(1)
        m = diff > 0
        if not m.any():
            continue
        sd = s[:, d].unsqueeze(0) - s[:, d].unsqueeze(1)
        tot = tot + F.relu(margin - sd)[m].mean()
        c += 1
    return tot / max(c, 1)


def fit_dtr(Ufe, O, Y, pi, cfg, seeds=4):
    ms = []
    for sd in range(seeds):
        torch.manual_seed(sd)
        np.random.seed(sd)
        g = DTRGate(Ufe.shape[-1], O.shape[-1], rank=cfg['rank'])
        g.set_prior(pi)
        opt = torch.optim.AdamW([
            {'params': [g.U] + list(g.V.parameters()), 'lr': cfg['lr'],
             'weight_decay': cfg['wd']},
            {'params': [g.a, g.b], 'lr': cfg['lr'], 'weight_decay': cfg['wd']}])
        u = torch.from_numpy(Ufe).float()
        o = torch.from_numpy(O).float()
        yt = torch.from_numpy(Y).long()
        for _ in range(cfg['epochs']):
            g.train()
            perm = torch.randperm(len(u))
            for i in range(0, len(perm), cfg['batch']):
                b = perm[i:i + cfg['batch']]
                s, _, beta = g(u[b], o[b])
                loss = F.huber_loss(s, yt[b].float())
                if cfg['lam_pair'] > 0:
                    loss = loss + cfg['lam_pair'] * pair_loss(s, yt[b])
                # 触发稀疏惩罚：默认不动，只有有依据时才 β→1
                loss = loss + cfg['lam_beta'] * beta.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
                opt.step()
        ms.append(g)
    return ms


def predict(ms, Ufe, O):
    u = torch.from_numpy(Ufe).float()
    o = torch.from_numpy(O).float()
    S, A, Bt = [], [], []
    for g in ms:
        g.eval()
        with torch.no_grad():
            s, a, beta = g(u, o)
        S.append(s.numpy())
        A.append(a.numpy())
        Bt.append(beta.numpy())
    s = np.mean(S, 0)
    return FS.lvl(s), np.mean(A, 0), np.mean(Bt, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--clip-feats', dest='cf', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/clip_feats.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--ranks', default='1,2')
    ap.add_argument('--lrs', default='3e-3,1e-2')
    ap.add_argument('--epochs-grid', dest='eg', default='0,100,300')
    ap.add_argument('--lam-pairs', dest='lp', default='0,0.5')
    ap.add_argument('--lam-betas', dest='lb', default='0.01,0.1')
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--no-tvar', dest='notvar', action='store_true')
    ap.add_argument('--strict-fold-feats', dest='sff', action='store_true',
                    help='严格版：内层每折用**该折的** π 重算路由信号。'
                         '默认版用全训练集拟合的 π 构造"到阈值距离"这一项，'
                         '不泄漏 val 但会让内层 CV 对路由特征略微乐观。')
    ap.add_argument('--force-cfg', dest='force', default='',
                    help='强制使用 rank,lr,epochs,lam_pair,lam_beta（跳过 1-SE 选择）。'
                         '仅用于**机制消融**——回答「强制动态时门控如何动作」，'
                         '绝不可作为模型主结果，因为它绕开了模型选择协议。')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--final-seeds', dest='fseeds', type=int, default=8)
    ap.add_argument('--tag', default='surgmind_dtr')
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()

    # ---- 专家池（与静态版完全一致）
    src, Y, istr, names = RM.load_sources(a.feats, a.pca)
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
    Xtr = np.hstack([src[k][istr] for k in snames])
    Xva = np.hstack([src[k][~istr] for k in snames])
    o1, o2, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
    Otr.append(o1)
    Ova.append(o2)
    snames = snames + ['joint']
    R = os.path.dirname(a.feats)

    def lp_(fn):
        rr = [json.loads(l) for l in io.open(os.path.join(R, fn), encoding='utf-8')]
        return {r['video']: r['pred_levels'] for r in rr}
    pt, pv = lp_('pred_sftbalpd3_ck250_train_pd.jsonl'), lp_('pred_sftbalpd3_ck250_val_pd.jsonl')
    Otr.append(np.array([pt[v] for v in ntr], dtype=float))
    Ova.append(np.array([pv[v] for v in nva], dtype=float))
    snames = snames + ['sft']                       # sft 放末位，路由信号里要用
    Ptr, Pva = np.stack(Otr, -1), np.stack(Ova, -1)
    M = len(snames)
    print('%s | 专家 %s | train %d / val %d' % (a.tag, snames, len(Ytr), len(Yva)), flush=True)

    # ---- 时间不一致性信号（clip 级预测的逐视频方差）
    tv_tr = tv_va = None
    if not a.notvar and os.path.exists(a.cf):
        d = np.load(a.cf, allow_pickle=True)
        Xc, mc = d['X'], json.loads(str(d['meta']))
        vid = np.array([m['video'] for m in mc])
        cstr = np.array([m['split'] == 'train' for m in mc])
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit(Xc[cstr]).transform(Xc)
        Zc = PCA(n_components=32, random_state=0).fit(Xs[cstr]).transform(Xs).astype(np.float32)
        from sklearn.linear_model import Ridge
        yv = {m['video']: m['label'] for m in mc}
        vt = np.array(sorted(set(vid[cstr])))
        Yc = np.array([yv[v] for v in vid])
        pred = np.zeros((len(Zc), K), dtype=np.float32)
        for i in range(K):
            pred[:, i] = Ridge(alpha=1e3).fit(Zc[cstr], Yc[cstr, i]).predict(Zc)

        def per_video_std(mask, order):
            out = np.zeros((len(order), K), dtype=np.float32)
            for j, v in enumerate(order):
                s = pred[(vid == v)]
                out[j] = s.std(0) if len(s) > 1 else 0.0
            return out
        tv_tr, tv_va = per_video_std(cstr, ntr), per_video_std(~cstr, nva)
        print('  已加入 clip 级时间方差信号（逐视频、逐维）', flush=True)

    # ---- 静态先验 π：与静态版同一套网格搜索（只用 train 折外预测）
    grids = list(FS.simplex(M, a.step))
    A0, _ = FS.pick_alpha(Ptr, Ytr, np.arange(len(Ytr)), grids, 'qwk')
    print('  冻结的静态先验 π_d:')
    for i, dd in enumerate(DIMS):
        print('    %-24s %s' % (dd, {k: round(float(x), 2)
                                     for k, x in zip(snames, A0[i]) if x > 0}))
    base_lvl = np.stack([FS.lvl(Pva[:, i, :] @ A0[i]) for i in range(K)], 1)
    print('  静态版在 val 上：Exact %.4f  QWK %.4f'
          % ((base_lvl == Yva).mean(), T.qwk(Yva.ravel(), base_lvl.ravel())), flush=True)

    Utr = routing_signals(Ptr, A0, tv_tr)
    Uva = routing_signals(Pva, A0, tv_va)
    mu, sg = Utr.reshape(-1, Utr.shape[-1]).mean(0), Utr.reshape(-1, Utr.shape[-1]).std(0) + 1e-6
    Utr, Uva = (Utr - mu) / sg, (Uva - mu) / sg
    print('  路由信号维度 F = %d（不是原始 embedding）' % Utr.shape[-1], flush=True)

    GRID = [(r, lr, ep, lpv, lbv)
            for r in [int(x) for x in a.ranks.split(',')]
            for lr in [float(x) for x in a.lrs.split(',')]
            for ep in [int(x) for x in a.eg.split(',')]
            for lpv in [float(x) for x in a.lp.split(',')]
            for lbv in [float(x) for x in a.lb.split(',')]]
    GRID = [g for g in GRID if g[2] > 0] + [(1, 3e-3, 0, 0.0, 0.0)]   # ep=0 → 纯静态
    print('  网格 %d（含 ep=0 即纯静态，保证不可能更差）' % len(GRID), flush=True)

    scores = {}
    for rep in range(a.repeats):
        sh = np.random.RandomState(7000 + rep).permutation(len(Ytr))
        f5 = np.zeros(len(Ytr), dtype=int)
        for i, v in enumerate(sh):
            f5[v] = i % 5
        for gk in GRID:
            r, lr, ep, lpv, lbv = gk
            cfg = dict(rank=r, lr=lr, epochs=ep, lam_pair=lpv, lam_beta=lbv,
                       wd=a.wd, batch=a.batch)
            acc = []
            for f in range(5):
                tr, te = np.where(f5 != f)[0], np.where(f5 == f)[0]
                A1, _ = FS.pick_alpha(Ptr, Ytr, tr, grids, 'qwk')
                if ep == 0:
                    pl = np.stack([FS.lvl(Ptr[te, i, :] @ A1[i]) for i in range(K)], 1)
                else:
                    if a.sff:            # 该折自己的 π 重算路由信号，并用折内统计标准化
                        Uf = routing_signals(Ptr, A1, tv_tr)
                        m2 = Uf[tr].reshape(-1, Uf.shape[-1]).mean(0)
                        s2 = Uf[tr].reshape(-1, Uf.shape[-1]).std(0) + 1e-6
                        Uf = (Uf - m2) / s2
                    else:
                        Uf = Utr
                    gs = fit_dtr(Uf[tr], Ptr[tr], Ytr[tr], A1, cfg, a.seeds)
                    pl, _, _ = predict(gs, Uf[te], Ptr[te])
                acc.append(np.mean([FS.dim_obj(Ytr[te, i], pl[:, i], 'mix') for i in range(K)]))
            scores.setdefault(gk, []).extend(acc)
        print('  内层 repeat %d 完成' % rep, flush=True)

    st = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
          for k, v in scores.items()}
    best = max(st, key=lambda k: st[k][0])
    thr = st[best][0] - st[best][1]
    sel = min([k for k in st if st[k][0] >= thr], key=lambda k: (k[2], k[0]))
    print('\n内层 CV Top-6 (rank, lr, epochs, λ_pair, λ_beta):')
    for k in sorted(st, key=lambda k: -st[k][0])[:6]:
        print('  %s  %.4f ± %.4f%s' % (k, st[k][0], st[k][1], '  ← 最优' if k == best else ''))
    print('1-SE 阈 %.4f → 选中 %s%s' % (thr, sel, '（= 纯静态）' if sel[2] == 0 else ''))

    if a.force:
        f = [x.strip() for x in a.force.split(',')]
        sel = (int(f[0]), float(f[1]), int(f[2]), float(f[3]), float(f[4]))
        print('!! 强制配置 %s —— 机制消融，非 CV 选出的模型，不得用于指标声明' % (sel,),
              flush=True)
    r, lr, ep, lpv, lbv = sel
    if ep == 0:
        pred, alpha, beta = base_lvl, np.repeat(A0[None], len(Yva), 0), np.zeros((len(Yva), K))
    else:
        cfg = dict(rank=r, lr=lr, epochs=ep, lam_pair=lpv, lam_beta=lbv,
                   wd=a.wd, batch=a.batch)
        gs = fit_dtr(Utr, Ptr, Ytr, A0, cfg, a.fseeds)
        pred, alpha, beta = predict(gs, Uva, Pva)

    extra = {'method': 'SurgMind-DTR: static per-competency evidence prior + '
                       'disagreement-triggered low-dimensional dynamic correction; '
                       'alpha_d(x)=softmax(log pi_d + beta_d(x)*Delta_d(u_d(x))); '
                       'zero-init => epoch0 identical to validated static solution; '
                       'fixed ordinal thresholds, NO post-hoc calibration',
             'experts': snames, 'routing_signal_dim': int(Utr.shape[-1]),
             'selected': dict(zip(['rank', 'lr', 'epochs', 'lam_pair', 'lam_beta'], sel)),
             'strict_fold_feats': bool(a.sff),
             'FORCED_not_cv_selected': bool(a.force),
             # 强制配置可能不在网格里 → 此时没有对应的内层 CV 分数
             'inner_cv': (round(st[sel][0], 4) if sel in st else None),
             'static_pi': {DIMS[i]: dict(zip(snames, [round(float(x), 2) for x in A0[i]]))
                           for i in range(K)},
             'beta_mean': {DIMS[i]: round(float(beta[:, i].mean()), 4) for i in range(K)},
             'beta_max': {DIMS[i]: round(float(beta[:, i].max()), 4) for i in range(K)},
             'alpha_mean': np.mean(alpha, 0).round(4).tolist(),
             # 逐病例 β 与逐病例专家分歧度——判据 3 必须用它们做相关，
             # 才能判断"按分歧触发"的机制解释是否成立（而不只是"一个恰好有效的门控"）。
             'beta_per_case': {v: [round(float(x), 4) for x in beta[j]]
                               for j, v in enumerate(nva)},
             'expert_disagreement_per_case': {
                 v: [round(float(x), 4) for x in Pva[j].std(-1)]
                 for j, v in enumerate(nva)}}
    res = T.score(pred, Yva, nva, a.tag, a.outdir, extra=extra)
    print('\n===== %s（25 视频，评一次；与 ck250 同评测脚本 / 同解码口径）=====' % a.tag)
    print(T.format_report(res))
    print(T.format_confusion(res))
    print('触发门 β_d 统计（0=完全用静态先验，1=完全启用修正）:')
    dis = Pva.std(-1)
    for i, dd in enumerate(DIMS):
        b, g = beta[:, i], dis[:, i]
        c = np.corrcoef(b, g)[0, 1] if b.std() > 1e-9 else float('nan')
        print('  %-24s 均值 %.3f  最大 %.3f  >0.5 的病例 %d/%d  corr(β, 专家分歧)=%+.3f'
              % (dd, b.mean(), b.max(), int((b > 0.5).sum()), len(Yva), c))
    print('  ↑ corr>0 才支持"按分歧触发"的机制解释；否则 β 学到的是别的东西。')


if __name__ == '__main__':
    main()
