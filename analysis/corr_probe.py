# -*- coding: utf-8 -*-
"""证据源之间的**排序误差相关性**——决定融合还有没有上限空间。

ensemble 想在排序指标（QWK/ρ）上超过最强成员，前提是成员的错误模式不同。
实测三路证据与 ck250 的排序误差相关性高达 0.81–0.88（Efficiency 除外 0.55–0.60），
这解释了 QWK 为何卡在 0.56–0.57。这里把两种 ViT 时间采样（f28 / fps4）拆开，
检验"换时间视角"能否带来去相关的证据。
"""
import io, json, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, '/hpc2hdd/home/ce483/hkust_surgery_video_learning/revision/scripts/sft_grpo')
import train_mmoe_lite as T
import ridge_moe as RM

DIMS = ['Depth', 'Bimanual', 'Efficiency', 'Tissue']
src, Y, istr, names = RM.load_sources('feats_4src.npz', 64)
Ytr, Yva = Y[istr], Y[~istr]
nva = [n for n, t in zip(names, istr) if not t]
fo = np.arange(len(Ytr)) % 5

O = {}
for k, v in src.items():
    _, o2, _ = RM.source_expert_preds(v[istr], Ytr, v[~istr], fo)
    O[k] = o2
rr = [json.loads(l) for l in io.open('pred_sftbalpd3_ck250_val_pd.jsonl', encoding='utf-8')]
pm = {r['video']: r['pred_levels'] for r in rr}
O['sft'] = np.array([pm[v] for v in nva], dtype=float)

ks = list(O)
print('各证据源单独在 val 上:')
for k in ks:
    lv = np.clip(np.searchsorted([1.5, 2.5, 3.5, 4.5], O[k]) + 1, 1, 5)
    print('  %-6s Exact %.3f  QWK %.3f' % (k, (lv == Yva).mean(), T.qwk(Yva.ravel(), lv.ravel())))

print('\n排序误差相关矩阵（越低越互补；行列 = %s）' % ks)
for i, d in enumerate(DIMS):
    yt = rankdata(Yva[:, i])
    E = {k: rankdata(O[k][:, i]) - yt for k in ks}
    print('  [%s]' % d)
    print('    %-7s %s' % ('', ' '.join('%6s' % k for k in ks)))
    for a in ks:
        print('    %-7s %s' % (a, ' '.join('%6.2f' % np.corrcoef(E[a], E[b])[0, 1]
                                           for b in ks)))
