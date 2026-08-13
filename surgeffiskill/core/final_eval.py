"""最终 val 评测 —— 三个候选走**同一入口、同一打分路径**（只跑一次）。

统一口径保证：同 25 视频 · 同专家池构造 · 同序数阈值 · **无后验校准** ·
同 `train_mmoe_lite.score` 打分 → 与 ck250 的 `ordinal_*.json` 完全可比。

  --arm hot          π(网格) + 无正则动态门控 (rank2/lr1e-2/ep300/λpair.5/λbeta0)
  --arm global_only  π* = hot 门控在 98 train 上的跨样本平均 α，**静态**施用
  --arm weak         π(网格) + 强正则动态门控 (rank1/lr3e-3/ep100/λpair.5/λbeta.01)
  --arm sq_nested    **competency-adaptive SurgMind**：α_d(x)=softmax[θ_d + z_d·Δ_d(x)]
                     θ_d 冻结为 log π*；z_d 为 hard-concrete 近似二值开关；
                     soft-QWK 代理 + L0 稀疏；**λ_z 由 98 train 内的 3 折内层 CV 选出**。
                     val 不参与任何选择。

三者的**超参全部来自 train-only 重复 CV**（R23–R24），本脚本不做任何选择，
只执行已选定的臂。val 只在最后一步被读取一次。
"""
import os, io, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mmoe_lite as T
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, fit_dtr, predict, DIMS, K, PROJECT
from surgmind_sp import fit_sp, sp_predict, qwk as _qwk

LZS_SEL = [0.002, 0.005, 0.01, 0.02, 0.05]

CFG = {'hot': dict(rank=2, lr=1e-2, epochs=300, lam_pair=0.5, lam_beta=0.0,
                   wd=0.0, batch=32),
       'weak': dict(rank=1, lr=3e-3, epochs=100, lam_pair=0.5, lam_beta=0.01,
                    wd=0.0, batch=32)}
CFG['global_only'] = CFG['hot']            # π* 由 hot 门控导出
CFG['sq_nested'] = CFG['hot']              # warm-start 源


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['hot', 'global_only', 'weak', 'sq_nested'])
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()
    tag = a.tag or ('FINAL_' + a.arm)

    # ---- 专家池：与所有 train-only 实验完全同一条构造路径
    src, Y, istr, names = RM.load_sources(a.feats, a.pca)
    sn = list(src)
    Ytr, Yva = Y[istr], Y[~istr]
    ntr = [n for n, t in zip(names, istr) if t]
    nva = [n for n, t in zip(names, istr) if not t]
    fo = np.arange(len(Ytr)) % 5
    Otr, Ova = [], []
    for k in sn:
        o1, o2, _ = RM.source_expert_preds(src[k][istr], Ytr, src[k][~istr], fo)
        Otr.append(o1); Ova.append(o2)
    Xtr = np.hstack([src[k][istr] for k in sn])
    Xva = np.hstack([src[k][~istr] for k in sn])
    o1, o2, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
    Otr.append(o1); Ova.append(o2); sn = sn + ['joint']
    R = os.path.dirname(a.feats)

    def lp(fn):
        return {r['video']: r['pred_levels'] for r in
                [json.loads(l) for l in io.open(os.path.join(R, fn), encoding='utf-8')]}

    pt, pv = lp('pred_sftbalpd3_ck250_train_pd.jsonl'), lp('pred_sftbalpd3_ck250_val_pd.jsonl')
    Otr.append(np.array([pt[v] for v in ntr], float))
    Ova.append(np.array([pv[v] for v in nva], float))
    sn = sn + ['sft']
    Ptr, Pva = np.stack(Otr, -1), np.stack(Ova, -1)
    print('%s | arm=%s | 专家 %s | train %d / val %d' % (tag, a.arm, sn, len(Ytr), len(Yva)),
          flush=True)

    # ---- 第一层：网格先验（只用 98 train）
    grids = list(FS.simplex(Ptr.shape[-1], a.step))
    A0, _ = FS.pick_alpha(Ptr, Ytr, np.arange(len(Ytr)), grids, 'qwk')
    U = routing_signals(Ptr, A0, None)
    print('  网格 π: %s' % json.dumps(
        {DIMS[i]: {sn[j]: round(float(A0[i, j]), 2)
                   for j in range(len(sn)) if A0[i, j] > 0} for i in range(K)},
        ensure_ascii=False), flush=True)

    beta = np.zeros((len(Yva), K))
    if a.arm == 'sq_nested':
        # ---- warm-start：全 98 train 上拟合 hot，取 π*、V、U（× 平均 β）
        gs = fit_dtr(U, Ptr, Ytr, A0, CFG['hot'], a.seeds)
        ut = torch.from_numpy(U).float(); ot = torch.from_numpy(Ptr).float()
        Al, Bl = [], []
        for g in gs:
            g.eval()
            with torch.no_grad():
                _, al, be = g(ut, ot)
            Al.append(al.numpy()); Bl.append(be.numpy())
        Ps = np.mean(Al, 0).mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
        bm = np.mean(Bl, 0).mean(0)
        theta0 = np.log(Ps + 1e-8)
        V0 = gs[0].V.weight.data.clone()
        U0 = gs[0].U.data.clone() * torch.as_tensor(bm, dtype=torch.float32).view(-1, 1, 1)
        cfg = dict(rank=CFG['hot']['rank'], lr=3e-3, epochs=200, lam_pair=0.5,
                   batch=32, surrogate='sqwk', tau=0.1)
        print('  π*: %s' % json.dumps(
            {DIMS[i]: {sn[j]: round(float(Ps[i, j]), 3)
                       for j in range(len(sn)) if Ps[i, j] > 0.01} for i in range(K)},
            ensure_ascii=False), flush=True)

        # ---- 内层 3 折 CV 选 λ_z（**只用 98 train**；val 不参与）
        rs = np.random.RandomState(0)
        ifold = rs.permutation(len(Ytr)) % 3
        score = {lz: 0.0 for lz in LZS_SEL}
        for gidx in range(3):
            ite = np.where(ifold == gidx)[0]; itr = np.where(ifold != gidx)[0]
            A2, _ = FS.pick_alpha(Ptr, Ytr, itr, grids, 'qwk')
            U2 = routing_signals(Ptr, A2, None)
            h2 = fit_dtr(U2[itr], Ptr[itr], Ytr[itr], A2, CFG['hot'], 2)
            u2 = torch.from_numpy(U2[itr]).float(); o2 = torch.from_numpy(Ptr[itr]).float()
            A_, B_ = [], []
            for g in h2:
                with torch.no_grad():
                    _, a2, b2 = g(u2, o2)
                A_.append(a2.numpy()); B_.append(b2.numpy())
            Ps2 = np.mean(A_, 0).mean(0); Ps2 = Ps2 / Ps2.sum(-1, keepdims=True)
            bm2 = np.mean(B_, 0).mean(0)
            th2 = np.log(Ps2 + 1e-8)
            V2 = h2[0].V.weight.data.clone()
            U2p = h2[0].U.data.clone() * torch.as_tensor(
                bm2, dtype=torch.float32).view(-1, 1, 1)
            for lz in LZS_SEL:
                m2 = fit_sp(U2[itr], Ptr[itr], Ytr[itr], th2, V2, U2p, lz, 0.0,
                            cfg, 2, freeze_theta=True)
                p2, _ = sp_predict(m2, U2[ite], Ptr[ite])
                score[lz] += np.mean([_qwk(p2[:, i], Ytr[ite, i]) for i in range(K)])
        best_lz = max(LZS_SEL, key=lambda l: score[l])
        print('  内层 3 折选 λ_z: %s → **选中 %g**'
              % ({k: round(v / 3, 4) for k, v in score.items()}, best_lz), flush=True)

        # ---- 用选中的 λ_z 在全 98 上重训，施加于 val
        ms = fit_sp(U, Ptr, Ytr, theta0, V0, U0, best_lz, 0.0, cfg, a.seeds,
                    freeze_theta=True)
        Uva = routing_signals(Pva, Ps, None)
        pred, zlearn = sp_predict(ms, Uva, Pva)
        print('  学到的 z_d: %s' % {DIMS[i]: round(float(zlearn[i]), 3) for i in range(K)},
              flush=True)
        alpha = np.repeat(Ps[None], len(Yva), 0)
        prior = Ps
        _extra_sp = dict(lam_z_selected=best_lz, inner_scores={str(k): round(v / 3, 4)
                                                               for k, v in score.items()},
                         z_learned={DIMS[i]: round(float(zlearn[i]), 4) for i in range(K)})
    elif a.arm == 'global_only':
        gs = fit_dtr(U, Ptr, Ytr, A0, CFG['hot'], a.seeds)
        _, al, _ = predict(gs, U, Ptr)                 # 只在 **train** 上取平均 α
        Ps = al.mean(0); Ps = Ps / Ps.sum(-1, keepdims=True)
        print('  π*: %s' % json.dumps(
            {DIMS[i]: {sn[j]: round(float(Ps[i, j]), 3)
                       for j in range(len(sn)) if Ps[i, j] > 0.01} for i in range(K)},
            ensure_ascii=False), flush=True)
        print('  |π*−π|₁ = %.4f' % float(np.abs(Ps - A0).sum()), flush=True)
        pred = np.stack([FS.lvl(Pva[:, i, :] @ Ps[i]) for i in range(K)], 1)
        alpha = np.repeat(Ps[None], len(Yva), 0)
        prior = Ps
    else:
        gs = fit_dtr(U, Ptr, Ytr, A0, CFG[a.arm], a.seeds)
        Uva = routing_signals(Pva, A0, None)
        pred, alpha, beta = predict(gs, Uva, Pva)
        prior = A0

    extra = {'arm': a.arm, 'experts': sn,
             'config': (cfg if a.arm == 'sq_nested' else CFG[a.arm]),
             'selection': 'train-only repeated CV (R23-R24); 本脚本不做任何选择',
             'prior': {DIMS[i]: {sn[j]: round(float(prior[i, j]), 3)
                                 for j in range(len(sn)) if prior[i, j] > 1e-3}
                       for i in range(K)},
             'alpha_mean': np.mean(alpha, 0).round(4).tolist(),
             'beta_mean': {DIMS[i]: round(float(beta[:, i].mean()), 4) for i in range(K)},
             'beta_max': {DIMS[i]: round(float(beta[:, i].max()), 4) for i in range(K)},
             'beta_per_case': {v: [round(float(x), 4) for x in beta[j]]
                               for j, v in enumerate(nva)},
             'expert_disagreement_per_case': {
                 v: [round(float(x), 4) for x in Pva[j].std(-1)] for j, v in enumerate(nva)}}
    if a.arm == 'sq_nested':
        extra.update(_extra_sp)
    res = T.score(pred, Yva, nva, tag, a.outdir, extra=extra)
    print('\n===== %s（25 视频，只评一次；与 ck250 同评测脚本 / 同解码口径）=====' % tag)
    print(T.format_report(res))
    print(T.format_confusion(res))
    if beta.max() > 1e-9:
        dis = Pva.std(-1)
        print('触发门 β_d 统计:')
        for i, dd in enumerate(DIMS):
            b, g = beta[:, i], dis[:, i]
            c = np.corrcoef(b, g)[0, 1] if b.std() > 1e-9 else float('nan')
            print('  %-24s 均值 %.3f  最大 %.3f  >0.5 病例 %d/%d  corr(β,分歧)=%+.3f'
                  % (dd, b.mean(), b.max(), int((b > 0.5).sum()), len(Yva), c))


if __name__ == '__main__':
    main()
