# -*- coding: utf-8 -*-
"""训练 / 扫描 SkillMMoE-Lite，并用未改动的 OrdinalEvaluator 评分。

**协议（关键）**：所有配置搜索、epoch、解码阈值 λ 都只用 **train(98) 内部 5 折 CV** 打分；
只有 CV 选出的唯一配置才拿去 val(25) 评一次。几十个配置对着 val 挑最好的属于
cherry-picking，本脚本在结构上排除这种做法：sweep 阶段根本不读 val。

选择准则 = 0.5·QWK(overall) + 0.5·QWK(Tissue)，对应"Tissue 修复 + 序数一致性优先"。
"""
import argparse
import collections
import io
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'revision/scripts/sft_grpo'))
from ordinal_metrics import OrdinalEvaluator, format_confusion, format_report   # noqa: E402
from skill_mmoe_lite import DIMS, TISSUE, SkillMMoELite, pcgrad_step            # noqa: E402

RUBRIC = json.load(io.open(os.path.join(PROJECT, 'surg_rubric.json'), encoding='utf-8'))
LEVELS = RUBRIC['levels']
BASE_THR = np.array([1.5, 2.5, 3.5, 4.5])
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def render(l):
    return '\n'.join('%s (%d): %s' % (d, int(x), LEVELS[d][str(int(x))])
                     for d, x in zip(DIMS, l))


def qwk(y, p, L=5):
    y, p = np.asarray(y, int), np.asarray(p, int)
    O = np.zeros((L, L))
    for a, b in zip(y, p):
        O[a - 1, b - 1] += 1
    w = (np.arange(L)[:, None] - np.arange(L)[None, :]) ** 2 / (L - 1.0) ** 2
    E = np.outer(np.bincount(y - 1, minlength=L), np.bincount(p - 1, minlength=L)) / max(len(y), 1)
    de = (w * E).sum()
    return 1.0 - (w * O).sum() / de if de > 0 else 0.0


def _f1_macro(y, p):
    lab = sorted(set(np.asarray(y).tolist()) | set(np.asarray(p).tolist()))
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=lab, average='macro', zero_division=0)


def sel_score(pred, gt):
    """train 内 CV 的选择准则，权重照抄验收标准的优先级：
    Tissue 修复 + QWK/Spearman/Macro-F1 > Exact > ±1。
    ±1 与 MAE 必须**进目标**，否则强序数标定会为了 QWK 把 ±1 从 0.98 砸到 0.93
    （quick 扫描实测），那不是 Pareto improvement。"""
    from scipy.stats import spearmanr
    y, p = gt.ravel(), pred.ravel()
    rho = spearmanr(y, p)[0]
    rho = 0.0 if rho != rho else rho
    return (0.30 * qwk(y, p) + 0.15 * rho + 0.15 * _f1_macro(y, p)
            + 0.20 * qwk(gt[:, TISSUE], pred[:, TISSUE])
            + 0.10 * float((y == p).mean())
            + 0.10 * float((np.abs(y - p) <= 1).mean()))


class Cfg(dict):
    __getattr__ = dict.get


def build(in_dim, c):
    return SkillMMoELite(in_dim, num_experts=c['num_experts'],
                         expert_hidden=c['expert_hidden'], dropout=c['dropout'],
                         task_experts=c['task_experts'], use_probe=c['use_probe'],
                         arch=c['arch'], layers=c['layers'], gate=c['gate'],
                         topk=c['topk'], task_weight=c['task_weight'])


def fit(Ztr, Ytr, Zev, n_epochs, c, seed=0, eval_every=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = build(Ztr.shape[1], c)
    m.set_class_weights(Ytr)
    opt = torch.optim.AdamW(m.parameters(), lr=c['lr'], weight_decay=c['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
    zt, yt = torch.from_numpy(Ztr).float(), torch.from_numpy(Ytr).long()
    ze = torch.from_numpy(Zev).float()
    dim_w = [1.0, 1.0, 1.0, c['tissue_weight']]
    traj = []
    for ep in range(1, n_epochs + 1):
        m.train()
        perm = torch.randperm(len(zt))
        for i in range(0, len(perm), c['batch']):
            b = perm[i:i + c['batch']]
            out = m(zt[b])
            L = m.per_dim_loss(out, yt[b], dim_w)
            aux = m.aux(out, yt[b], c['w_bal'], c['w_div'], c['w_dec'])
            if c['pcgrad']:
                pcgrad_step(m, list(L) + [aux], opt)
            else:
                opt.zero_grad()
                (L.sum() + aux).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
        if eval_every and ep % eval_every == 0:
            m.eval()
            with torch.no_grad():
                traj.append((ep, m.expected_level(m(ze)['cum_logits']).numpy()))
    m.eval()
    with torch.no_grad():
        o = m(ze)
        return m, m.expected_level(o['cum_logits']).numpy(), o['gate'].mean(0).numpy(), traj


def cv_eval(Ztr, Ytr, c, grid, folds=5):
    """只用 train：返回 (各 epoch 的 CV 选择分, 折外连续分)。val 全程未出现。"""
    from sklearn.model_selection import KFold
    cv = np.zeros(len(grid))
    oof = np.zeros((len(grid), len(Ztr), len(DIMS)), dtype=np.float32)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(Ztr):
        _, _, _, traj = fit(Ztr[tr], Ytr[tr], Ztr[te], max(grid), c, seed=0,
                            eval_every=grid[1] - grid[0] if len(grid) > 1 else max(grid))
        for j, (ep, pc) in enumerate(traj[:len(grid)]):
            oof[j, te] = pc
            cv[j] += sel_score(np.clip(np.rint(pc), 1, 5).astype(int), Ytr[te])
    return cv / folds, oof


def pick_lambda(oof_j, Ytr, no_calib=False):
    from sklearn.model_selection import KFold
    inner = list(KFold(5, shuffle=True, random_state=7).split(np.arange(len(Ytr))))
    lam_l, qthr_l, sc_l = [], [], []
    for i in range(len(DIMS)):
        sc = []
        for lam in LAMS:
            ks = []
            for a, b in inner:
                cnt = np.bincount(Ytr[a, i], minlength=6)[1:].astype(float)
                cum = np.cumsum(cnt) / cnt.sum()
                qt = np.array([np.quantile(oof_j[a, i], q) for q in cum[:-1]])
                pl = np.clip(np.searchsorted((1 - lam) * BASE_THR + lam * qt, oof_j[b, i]) + 1, 1, 5)
                yb = Ytr[b, i]
                ks.append(0.45 * qwk(yb, pl) + 0.15 * _f1_macro(yb, pl)
                          + 0.15 * float((yb == pl).mean())
                          + 0.25 * float((np.abs(yb - pl) <= 1).mean()))
            sc.append(float(np.mean(ks)))
        lam_l.append(0.0 if no_calib else LAMS[int(np.argmax(sc))])
        cnt = np.bincount(Ytr[:, i], minlength=6)[1:].astype(float)
        cum = np.cumsum(cnt) / cnt.sum()
        qthr_l.append(np.array([np.quantile(oof_j[:, i], q) for q in cum[:-1]]))
        sc_l.append([round(x, 3) for x in sc])
    return lam_l, qthr_l, sc_l


def score(pred, gt, names, tag, outdir, extra=None):
    os.makedirs(outdir, exist_ok=True)
    recs = [{'video': v, 'label_score': [int(x) for x in g], 'gt_levels': [int(x) for x in g],
             'pred_levels': [int(x) for x in p], 'pred': render(p), 'gt': render(g)}
            for p, g, v in zip(pred, gt, names)]
    with io.open(os.path.join(outdir, 'pred_%s.jsonl' % tag), 'w', encoding='utf-8') as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + u'\n')
    res = OrdinalEvaluator().score_records(recs)
    uq = collections.Counter(tuple(r['pred_levels']) for r in recs)
    res['run'] = dict(tag=tag, mode='SkillMMoE-Lite (MMoE/CGC/PLE) on pooled frozen features',
                      **(extra or {}))
    res['diversity'] = {'unique_pred_level_vectors': len(uq),
                        'top_level_vectors': [[list(k), v] for k, v in uq.most_common(5)]}
    with io.open(os.path.join(outdir, 'ordinal_%s.json' % tag), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    return res


def load_feats(path, pca):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = np.load(path, allow_pickle=True)
    X, meta = d['X'], json.loads(str(d['meta']))
    Y = np.array([m['label'] for m in meta], dtype=np.int64)
    istr = np.array([m['split'] == 'train' for m in meta])
    names = [m['video'] for m in meta]
    D3 = X.shape[1] // 3
    Xs = np.hstack([StandardScaler().fit(X[istr][:, s * D3:(s + 1) * D3])
                    .transform(X[:, s * D3:(s + 1) * D3]) for s in range(3)])
    p = PCA(n_components=pca, random_state=0).fit(Xs[istr])
    Z = p.transform(Xs).astype(np.float32)
    Z = (Z - Z[istr].mean(0)) / (Z[istr].std(0) + 1e-6)
    return Z, Y, istr, names, float(p.explained_variance_ratio_.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(PROJECT, 'revision/evidence/results/sft_grpo/vis_feats_f28.npz'))
    ap.add_argument('--outdir', default=os.path.join(PROJECT, 'revision/evidence/results/sft_grpo'))
    ap.add_argument('--tag', default='mmoelite')
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--max-epochs', dest='max_epochs', type=int, default=200)
    ap.add_argument('--eval-every', dest='eval_every', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--sweep', default='full', choices=['full', 'quick', 'none'])
    ap.add_argument('--arch-filter', dest='arch_filter', default='',
                    help='只扫指定拓扑（逗号分隔），用于把扫描拆成并行作业')
    args = ap.parse_args()

    Z, Y, istr, names, evr = load_feats(args.feats, args.pca)
    Ztr, Ytr, Zva, Yva = Z[istr], Y[istr], Z[~istr], Y[~istr]
    nva = [n for n, t in zip(names, istr) if not t]
    print('%s | train %s val %s | PCA %d 解释方差 %.3f'
          % (args.tag, Ztr.shape, Zva.shape, args.pca, evr), flush=True)

    base = dict(num_experts=4, expert_hidden=16, task_experts=1, dropout=0.1,
                lr=3e-3, wd=1e-2, batch=16, arch='cgc', layers=1, gate='softmax',
                topk=2, task_weight='fixed', tissue_weight=1.5, use_probe=True,
                pcgrad=False, w_bal=0.01, w_div=0.05, w_dec=0.2)
    if args.sweep == 'quick':
        space = dict(arch=['mmoe', 'cgc'], num_experts=[2, 4], task_experts=[1])
    elif args.sweep == 'full':
        space = dict(arch=['mmoe', 'cgc', 'ple'], num_experts=[2, 4, 6, 8],
                     task_experts=[1, 2], gate=['softmax', 'topk'],
                     task_weight=['fixed', 'uncertainty'], layers=[1])
        if args.arch_filter:
            space['arch'] = [a for a in space['arch'] if a in args.arch_filter.split(',')]
    else:
        space = {}
    keys = list(space)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])] or [{}]
    # mmoe 没有专属专家，去重
    seen, uniq = set(), []
    for c in combos:
        cc = dict(base, **c)
        if cc['arch'] == 'mmoe':
            cc['task_experts'] = 0
        if cc['arch'] == 'ple':
            cc['layers'] = 2
        k = tuple(sorted((str(a), str(b)) for a, b in cc.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(cc)
    print('待扫配置 %d 个（打分只用 train 内 5 折 CV，val 不参与）' % len(uniq), flush=True)

    grid = list(range(args.eval_every, args.max_epochs + 1, args.eval_every))
    results = []
    t0 = time.time()
    for n, c in enumerate(uniq):
        cv, oof = cv_eval(Ztr, Ytr, c, grid)
        j = int(np.argmax(cv))
        results.append((float(cv[j]), grid[j], c, oof[j]))
        print('  [%2d/%d] %-4s E=%d te=%d gate=%-7s tw=%-11s pcgrad=%-5s -> CV %.4f @ep%d (%.0fs)'
              % (n + 1, len(uniq), c['arch'], c['num_experts'], c['task_experts'],
                 c['gate'], c['task_weight'], c['pcgrad'], cv[j], grid[j], time.time() - t0),
              flush=True)
    results.sort(key=lambda r: -r[0])
    best_cv, best_ep, cbest, oof_best = results[0]
    print('\nCV 最优配置: %s' % {k: cbest[k] for k in
                              ['arch', 'num_experts', 'task_experts', 'gate',
                               'task_weight', 'pcgrad', 'layers']})
    print('CV 选择分 %.4f @ epoch %d' % (best_cv, best_ep))
    print('CV 前 5 名:')
    for s, e, c, _ in results[:5]:
        print('  %.4f  %-4s E=%d te=%d %-7s %-11s pcgrad=%s @ep%d'
              % (s, c['arch'], c['num_experts'], c['task_experts'], c['gate'],
                 c['task_weight'], c['pcgrad'], e))

    lam, qthr, lam_sc = pick_lambda(oof_best, Ytr)
    print('\n逐维序数标定 λ（train 折外嵌套选择）:')
    for i, dd in enumerate(DIMS):
        print('  %-24s λ=%.2f  各 λ 的 QWK %s' % (dd, lam[i], lam_sc[i]))

    cs, gs = [], []
    for s in range(args.seeds):
        _, pc, g, _ = fit(Ztr, Ytr, Zva, best_ep, cbest, seed=s)
        cs.append(pc)
        gs.append(g)
    C, gate = np.mean(cs, axis=0), np.mean(gs, axis=0)
    final = np.zeros(C.shape, dtype=int)
    for k in range(C.shape[1]):
        thr = (1 - lam[k]) * BASE_THR + lam[k] * qthr[k]
        final[:, k] = np.clip(np.searchsorted(thr, C[:, k]) + 1, 1, 5)

    npar = sum(p.numel() for p in build(Ztr.shape[1], cbest).parameters())
    res = score(final, Yva, nva, args.tag, args.outdir,
                extra={'chosen_epoch': best_ep, 'cv_score': round(best_cv, 4),
                       'seeds': args.seeds, 'pca': args.pca, 'n_params': int(npar),
                       'config': {k: cbest[k] for k in
                                  ['arch', 'num_experts', 'task_experts', 'gate',
                                   'task_weight', 'pcgrad', 'layers', 'use_probe']},
                       'lambda_per_dim': {dd: lam[i] for i, dd in enumerate(DIMS)},
                       'gate_per_dim': {dd: [round(float(x), 3) for x in gate[i]]
                                        for i, dd in enumerate(DIMS)},
                       'cv_top5': [{'score': round(s, 4), 'epoch': e,
                                    'arch': c['arch'], 'num_experts': c['num_experts'],
                                    'task_experts': c['task_experts'], 'gate': c['gate'],
                                    'task_weight': c['task_weight'], 'pcgrad': c['pcgrad']}
                                   for s, e, c, _ in results[:5]]})
    print('\n===== %s（配置/epoch/λ 全部由 train 内 CV 决定，val 只评一次）=====' % args.tag)
    print(format_report(res))
    print(format_confusion(res))
    print('参数量 %d' % npar)
    print('专家门控（每维平均权重；前 %d 个共享，其后每任务 %d 个专属）:'
          % (cbest['num_experts'], cbest['task_experts']))
    for i, dd in enumerate(DIMS):
        print('  %-24s %s' % (dd, ' '.join('%.3f' % x for x in gate[i])))


if __name__ == '__main__':
    main()
