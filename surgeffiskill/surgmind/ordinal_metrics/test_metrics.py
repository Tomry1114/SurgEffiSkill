import math
import random
import sys

import numpy as np

from . import metrics as M
from .evaluate import OrdinalEvaluator
from .decode import DIMS, get_rubric

FAIL = []
SKIP = []


def check(name, ok, detail=''):
    print('  %-52s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def close(a, b, tol=1e-9):
    if a != a and b != b:
        return True
    return abs(a - b) <= tol


print('[1] hand-computed edge cases')
y = [1, 2, 3, 4, 5, 3, 3, 2]
check('exact match accuracy=1', close(M.accuracy(y, y), 1.0))
check('exact match mae=0', close(M.mae(y, y), 0.0))
check('exact match macro_f1=1', close(M.macro_f1(y, y), 1.0))
check('exact match kappa=1', close(M.weighted_kappa(y, y), 1.0))
check('exact match spearman=1', close(M.spearman(y, y)[0], 1.0))

check('all off-by-one mae=1', close(M.mae([3, 3, 3, 3], [4, 4, 4, 4]), 1.0))
check('all off-by-one accuracy=0', close(M.accuracy([3, 3, 3, 3], [4, 4, 4, 4]), 0.0))
check('constant pred spearman=nan', M.spearman([1, 2, 3, 4], [2, 2, 2, 2])[0] != M.spearman([1, 2, 3, 4], [2, 2, 2, 2])[0])
check('empty input accuracy=nan', M.accuracy([], []) != M.accuracy([], []))
check('n<3 spearman=nan', M.spearman([1, 2], [1, 2])[0] != M.spearman([1, 2], [1, 2])[0])


yt = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
yp = [1, 2, 2, 3, 3, 4, 4, 5, 5, 1]
k = 5
O = np.zeros((k, k))
for a, b in zip(yt, yp):
    O[a - 1, b - 1] += 1
O /= len(yt)
E = np.outer(O.sum(1), O.sum(0))
i = np.arange(k).reshape(-1, 1)
j = np.arange(k).reshape(1, -1)
W = ((i - j) ** 2) / float((k - 1) ** 2)
manual = 1.0 - (W * O).sum() / (W * E).sum()
check('quadratic kappa matches cell-by-cell manual', close(M.weighted_kappa(yt, yp), manual, 1e-12),
      '= %.6f' % M.weighted_kappa(yt, yp))


_rng = random.Random(7)
_diff = 0.0
for _ in range(200):
    _n = _rng.randint(8, 40)
    _a = [_rng.randint(1, 5) for _ in range(_n)]
    _b = [max(1, min(5, x + _rng.choice([-3, -1, 0, 1, 3]))) for x in _a]
    _q, _l = M.weighted_kappa(_a, _b), M.weighted_kappa(_a, _b, weights='linear')
    if _q == _q and _l == _l:
        _diff = max(_diff, abs(_q - _l))
check('weights arg has effect (linear != quadratic)', _diff > 1e-3, 'max|delta|=%.4f' % _diff)


r = M._average_ranks(np.array([3.0, 1.0, 3.0, 2.0]))
check('ties take average rank', list(r) == [3.5, 1.0, 3.5, 2.0], '= %s' % list(r))

print('\n[2] cross-check vs sklearn / scipy')
try:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 mean_absolute_error)
    have_sk = True
except Exception:
    have_sk = False
try:
    from scipy import stats
    have_sp = True
except Exception:
    have_sp = False

if not have_sk:
    SKIP.append('sklearn cross-check (sklearn absent)')
    print('  skip sklearn cross-check: sklearn not installed')
if not have_sp:
    SKIP.append('scipy cross-check (scipy absent)')
    print('  skip scipy cross-check: scipy not installed')

if have_sk or have_sp:
    rng = random.Random(0)
    worst = {'acc': 0.0, 'f1': 0.0, 'mae': 0.0, 'kappa': 0.0, 'rho': 0.0}
    trials = 300
    for t in range(trials):
        n = rng.randint(5, 60)
        a = [rng.randint(1, 5) for _ in range(n)]

        b = [max(1, min(5, x + rng.choice([-2, -1, 0, 0, 1, 2]))) for x in a]
        if have_sk:
            worst['acc'] = max(worst['acc'], abs(M.accuracy(a, b) - accuracy_score(a, b)))
            worst['f1'] = max(worst['f1'], abs(M.macro_f1(a, b) - f1_score(a, b, average='macro')))
            worst['mae'] = max(worst['mae'], abs(M.mae(a, b) - mean_absolute_error(a, b)))
            sk_k = cohen_kappa_score(a, b, weights='quadratic', labels=[1, 2, 3, 4, 5])
            ours = M.weighted_kappa(a, b)
            if sk_k == sk_k and ours == ours:
                worst['kappa'] = max(worst['kappa'], abs(ours - sk_k))
        if have_sp:
            sp_rho = stats.spearmanr(a, b).statistic
            ours_rho, ours_p = M.spearman(a, b)
            if sp_rho == sp_rho and ours_rho == ours_rho:
                worst['rho'] = max(worst['rho'], abs(ours_rho - sp_rho))
                sp_p = stats.spearmanr(a, b).pvalue
                if ours_p is not None and sp_p == sp_p and abs(ours_rho) < 1.0:
                    worst['p'] = max(worst.get('p', 0.0), abs(ours_p - sp_p))

    if have_sk:
        check('accuracy vs sklearn (%d trials)' % trials, worst['acc'] < 1e-12, 'max|delta|=%.2e' % worst['acc'])
        check('macro_f1 vs sklearn', worst['f1'] < 1e-12, 'max|delta|=%.2e' % worst['f1'])
        check('mae vs sklearn', worst['mae'] < 1e-12, 'max|delta|=%.2e' % worst['mae'])
        check('quadratic kappa vs sklearn', worst['kappa'] < 1e-10, 'max|delta|=%.2e' % worst['kappa'])
    if have_sp:
        check('spearman rho vs scipy', worst['rho'] < 1e-10, 'max|delta|=%.2e' % worst['rho'])
        if 'p' in worst:
            check('spearman p vs scipy', worst['p'] < 1e-9, 'max|delta|=%.2e' % worst['p'])

    if have_sk:

        d = max(abs(M.weighted_kappa(yt, yp, weights='linear')
                    - cohen_kappa_score(yt, yp, weights='linear', labels=[1, 2, 3, 4, 5])), 0)
        check('linear kappa vs sklearn', d < 1e-10, '|delta|=%.2e' % d)

print('\n[3] OrdinalEvaluator end-to-end (rubric-built text)')
R = get_rubric()


def compose(levels):
    return ' '.join('%s: %s' % (d, R.levels[d][levels[i]]) for i, d in enumerate(DIMS))


recs = [
    {'video': 'a.mp4', 'label_score': [3, 3, 3, 3], 'pred': compose([3, 3, 3, 3])},
    {'video': 'b.mp4', 'label_score': [2, 4, 5, 2], 'pred': compose([2, 4, 5, 2])},
    {'video': 'c.mp4', 'label_score': [4, 4, 4, 4], 'pred': compose([4, 4, 4, 4])},
]
res = OrdinalEvaluator().score_records(recs)
check('all correct -> overall accuracy=1', close(res['overall']['accuracy'], 1.0))
check('all correct -> overall mae=0', close(res['overall']['mae'], 0.0))
check('coverage full', res['coverage']['pairs_scored'] == res['coverage']['pairs_total'] == 12,
      '= %d/%d' % (res['coverage']['pairs_scored'], res['coverage']['pairs_total']))

recs2 = list(recs) + [{'video': 'd.mp4', 'label_score': [3, 3, 3, 3], 'pred': 'completely off-topic output'}]
res2 = OrdinalEvaluator(undecodable='skip').score_records(recs2)
check('undecodable pairs skipped and counted', res2['coverage']['pred_decode_failures'] == 4
      and res2['coverage']['pairs_scored'] == 12,
      'fail=%d scored=%d/%d' % (res2['coverage']['pred_decode_failures'],
                                res2['coverage']['pairs_scored'], res2['coverage']['pairs_total']))
res3 = OrdinalEvaluator(undecodable='worst').score_records(recs2)
check("undecodable='worst' counts it and lowers accuracy",
      res3['coverage']['pairs_scored'] == 16 and res3['overall']['accuracy'] < 1.0,
      'acc=%.4f' % res3['overall']['accuracy'])

res4 = OrdinalEvaluator().score_records([{'video': 'x.mp4', 'pred': compose([3, 3, 3, 3])}])
check('no gt -> marked gt_unavailable, not scored',
      res4['coverage']['gt_unavailable_pairs'] == 4 and res4['coverage']['pairs_scored'] == 0)


res5 = OrdinalEvaluator().score_records(
    [{'video': 'y.mp4', 'gt': compose([2, 3, 4, 5]), 'pred': compose([2, 3, 4, 5])}])
check('gt decodable from gt text', res5['gt_source'].get('gt_text') == 1
      and close(res5['overall']['accuracy'], 1.0))

print('\n%s%s' % ('all passed' if not FAIL else 'failed: ' + ', '.join(FAIL),
                 ('  [skipped: %s]' % '; '.join(SKIP)) if SKIP else ''))
sys.exit(1 if FAIL else 0)
