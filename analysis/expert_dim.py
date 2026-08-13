"""逐专家 × 逐维表现 + 与 sft 的误差互补性，用来解释静态门控为何在某维混合。
train 用折外预测(ridge)/sample-in(sft，已声明)；val 为各专家全量拟合后的预测。"""
import os, io, sys, json
import numpy as np
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import DIMS, K, PROJECT

_ap = argparse.ArgumentParser(); _ap.add_argument('--tag', default='expert_dim')
_ap.parse_known_args()
R = os.path.join(PROJECT, 'revision/evidence/results/sft_grpo')
src, Y, istr, names = RM.load_sources(os.path.join(R, 'feats_4src.npz'), 64)
snames = list(src)
Ytr, Yva = Y[istr], Y[~istr]
ntr = [n for n, t in zip(names, istr) if t]
nva = [n for n, t in zip(names, istr) if not t]
fo = np.arange(len(Ytr)) % 5
Otr, Ova = [], []
for k in snames:
    o1, o2, _ = RM.source_expert_preds(src[k][istr], Ytr, src[k][~istr], fo)
    Otr.append(o1); Ova.append(o2)
Xtr = np.hstack([src[k][istr] for k in snames]); Xva = np.hstack([src[k][~istr] for k in snames])
o1, o2, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
Otr.append(o1); Ova.append(o2); snames += ['joint']


def lp(fn):
    return {r['video']: r['pred_levels'] for r in
            [json.loads(l) for l in io.open(os.path.join(R, fn), encoding='utf-8')]}


pt, pv = lp('pred_sftbalpd3_ck250_train_pd.jsonl'), lp('pred_sftbalpd3_ck250_val_pd.jsonl')
Otr.append(np.array([pt[v] for v in ntr], float))
Ova.append(np.array([pv[v] for v in nva], float))
snames += ['sft']
Ptr, Pva = np.stack(Otr, -1), np.stack(Ova, -1)


def rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float); return r


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


print('=== val 逐专家 × 逐维 Exact ===')
print('%-8s %s' % ('专家', ' '.join('%10s' % d[:10] for d in DIMS)))
for m, nm in enumerate(snames):
    cells = []
    for i in range(K):
        cells.append('%10.3f' % float((FS.lvl(Pva[:, i, m]) == Yva[:, i]).mean()))
    print('%-8s %s' % (nm, ' '.join(cells)))

print('\n=== train(折外) 逐维：各 ridge 专家误差与 sft 误差的**秩相关**（越低越互补）===')
si = snames.index('sft')
print('%-8s %s' % ('专家', ' '.join('%10s' % d[:10] for d in DIMS)))
for m, nm in enumerate(snames):
    if m == si:
        continue
    cells = []
    for i in range(K):
        e1 = np.abs(Ptr[:, i, m] - Ytr[:, i]); e2 = np.abs(Ptr[:, i, si] - Ytr[:, i])
        cells.append('%10.3f' % corr(rank(e1), rank(e2)))
    print('%-8s %s' % (nm, ' '.join(cells)))

print('\n=== train(折外) 逐专家 × 逐维 Exact（选 α 时门控实际看到的信号）===')
print('%-8s %s' % ('专家', ' '.join('%10s' % d[:10] for d in DIMS)))
for m, nm in enumerate(snames):
    cells = ['%10.3f' % float((FS.lvl(Ptr[:, i, m]) == Ytr[:, i]).mean()) for i in range(K)]
    print('%-8s %s' % (nm, ' '.join(cells)))
