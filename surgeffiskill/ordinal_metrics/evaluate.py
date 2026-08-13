# -*- coding: utf-8 -*-
"""ordinal_metrics.evaluate — 把推理结果（gt/pred 文本）变成 5 个序数指标。

给的是**分层**结果，不是一个数：
* `per_dimension`  —— 4 个技能维度各自的指标（论文表格通常按维度报）
* `overall`        —— 全部 (样本 × 维度) 对混算
* `macro_over_dimensions` —— 4 个维度指标的平均

同时强制给出 coverage：有多少 (样本,维度) 对因为输出解不出等级/没有真值而没参与计算。
默认 `undecodable='skip'`（跳过并如实报数量），也可以设 `'worst'` 把解不出的当作
标尺上离真值最远的一档 —— 报数时必须说明用的哪种，两者不是一回事。
"""

import io
import json
import os
from collections import Counter
from typing import Dict, List, Optional

from . import metrics as M
from .decode import DIMS, decode_vector

DEFAULT_LABEL_SOURCES = ['train_effi_cot.json', 'train_effi_cot_val.json']


def build_label_index(paths: Optional[List[str]] = None, project_root: Optional[str] = None) -> Dict[str, list]:
    """视频文件名 -> label_score [D,B,E,T]，从 CoT 数据里取（那里才有真值等级）。"""
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for p in (paths or DEFAULT_LABEL_SOURCES):
        path = p if os.path.isabs(p) else os.path.join(root, p)
        if not os.path.exists(path):
            continue
        for r in json.load(io.open(path, encoding='utf-8')):
            ls = r.get('label_score')
            if ls and r.get('videos'):
                out[os.path.basename(r['videos'][0])] = [int(x) for x in ls]
    return out


class OrdinalEvaluator:

    def __init__(self, num_levels: int = 5, kappa_weights: str = 'quadratic',
                 undecodable: str = 'skip', macro_f1_labels=None,
                 label_index: Optional[Dict[str, list]] = None):
        if undecodable not in ('skip', 'worst'):
            raise ValueError("undecodable 只支持 'skip' / 'worst'")
        self.num_levels = num_levels
        self.kappa_weights = kappa_weights
        self.undecodable = undecodable
        self.macro_f1_labels = macro_f1_labels
        self.label_index = label_index if label_index is not None else {}

    # ---------------------------------------------------------------- 真值
    def gt_vector(self, rec):
        """-> ([D,B,E,T] or None, 来源标签)。优先级：显式字段 > 视频名索引 > gt 文本解码。"""
        ls = rec.get('label_score')
        if isinstance(ls, (list, tuple)) and len(ls) == len(DIMS):
            try:
                v = [int(x) for x in ls]
                if all(1 <= x <= self.num_levels for x in v):
                    return v, 'label_score'
            except (TypeError, ValueError):
                pass

        vid = rec.get('video') or rec.get('videos')
        if isinstance(vid, (list, tuple)) and vid:
            vid = vid[0]
        if vid:
            hit = self.label_index.get(os.path.basename(str(vid)))
            if hit:
                return list(hit), 'video_index'

        gt_text = rec.get('gt') or rec.get('answer') or rec.get('solution')
        if isinstance(gt_text, str):
            v = decode_vector(gt_text)
            if all(x is not None for x in v):
                return v, 'gt_text'
        return None, 'unavailable'

    # ---------------------------------------------------------------- 主流程
    def score_records(self, records) -> dict:
        pairs = {d: {'y': [], 'p': []} for d in DIMS}
        src_counter = Counter()
        n_samples = 0
        pairs_total = 0
        n_pred_fail = 0
        n_gt_missing = 0

        for rec in records:
            n_samples += 1
            gt, src = self.gt_vector(rec)
            src_counter[src] += 1
            if gt is None:
                n_gt_missing += len(DIMS)
                pairs_total += len(DIMS)
                continue
            pred = decode_vector(rec.get('pred') or '')
            for i, d in enumerate(DIMS):
                pairs_total += 1
                p = pred[i]
                if p is None:
                    n_pred_fail += 1
                    if self.undecodable == 'skip':
                        continue
                    # 'worst'：取标尺上离真值最远的一档
                    p = 1 if gt[i] >= (self.num_levels + 1) / 2.0 else self.num_levels
                pairs[d]['y'].append(gt[i])
                pairs[d]['p'].append(p)

        per_dim = {}
        for d in DIMS:
            per_dim[d] = M.all_metrics(
                pairs[d]['y'], pairs[d]['p'], num_levels=self.num_levels,
                kappa_weights=self.kappa_weights, macro_f1_labels=self.macro_f1_labels)

        all_y = [v for d in DIMS for v in pairs[d]['y']]
        all_p = [v for d in DIMS for v in pairs[d]['p']]
        overall = M.all_metrics(all_y, all_p, num_levels=self.num_levels,
                                kappa_weights=self.kappa_weights,
                                macro_f1_labels=self.macro_f1_labels)

        kappa_key = 'kappa_%s' % self.kappa_weights
        macro = {}
        for key in ['accuracy', 'acc_within_1', 'macro_f1', 'mae', kappa_key, 'spearman_rho']:
            vals = [per_dim[d][key] for d in DIMS
                    if per_dim[d][key] == per_dim[d][key]]      # 剔除 nan
            macro[key] = float(sum(vals) / len(vals)) if vals else float('nan')

        # 逐维混淆矩阵（行=真值 1..L，列=预测 1..L）+ 预测/真值分布
        confusion, pred_dist, gt_dist = {}, {}, {}
        L = self.num_levels
        for d in DIMS:
            m = [[0] * L for _ in range(L)]
            pc, gc = Counter(), Counter()
            for y, p_ in zip(pairs[d]['y'], pairs[d]['p']):
                m[int(y) - 1][int(p_) - 1] += 1
                pc[int(p_)] += 1
                gc[int(y)] += 1
            confusion[d] = m
            pred_dist[d] = {str(k): pc.get(k, 0) for k in range(1, L + 1)}
            gt_dist[d] = {str(k): gc.get(k, 0) for k in range(1, L + 1)}

        return {
            'confusion_matrix': confusion,
            'confusion_axes': {'rows': 'gt level 1..%d' % L, 'cols': 'pred level 1..%d' % L},
            'prediction_distribution': pred_dist,
            'gt_distribution': gt_dist,
            'config': {
                'num_levels': self.num_levels,
                'kappa_weights': self.kappa_weights,
                'undecodable': self.undecodable,
                'macro_f1_labels': (list(self.macro_f1_labels)
                                    if self.macro_f1_labels is not None else 'observed'),
            },
            'n_samples': n_samples,
            'gt_source': dict(src_counter),
            'coverage': {
                'pairs_total': pairs_total,
                'pairs_scored': len(all_y),
                'pred_decode_failures': n_pred_fail,
                'gt_unavailable_pairs': n_gt_missing,
                'scored_rate': (len(all_y) / pairs_total) if pairs_total else float('nan'),
            },
            'per_dimension': per_dim,
            'overall': overall,
            'macro_over_dimensions': macro,
        }


def format_report(res: dict) -> str:
    """把 score_records 的结果排成表。"""
    kappa_key = 'kappa_%s' % res['config']['kappa_weights']
    cov = res['coverage']
    lines = []
    lines.append('样本数 %d | 计分对 %d/%d (%.1f%%) | pred 解码失败 %d | 无真值对 %d'
                 % (res['n_samples'], cov['pairs_scored'], cov['pairs_total'],
                    100.0 * cov['scored_rate'] if cov['scored_rate'] == cov['scored_rate'] else float('nan'),
                    cov['pred_decode_failures'], cov['gt_unavailable_pairs']))
    lines.append('真值来源: %s | undecodable=%s | κ=%s | macro-F1 类集=%s'
                 % (res['gt_source'], res['config']['undecodable'],
                    res['config']['kappa_weights'], res['config']['macro_f1_labels']))
    lines.append('')
    head = '%-24s %6s %9s %9s %9s %7s %9s %9s' % (
        '', 'n', 'Accuracy', '±1 Acc', 'Macro-F1', 'MAE', 'κ(%s)' % res['config']['kappa_weights'][:4], 'Spearman')
    lines.append(head)
    lines.append('-' * len(head))

    def row(name, m):
        def f(x, nd=4):
            return ('%.*f' % (nd, x)) if x == x else '  n/a'
        return '%-24s %6d %9s %9s %9s %7s %9s %9s' % (
            name, m['n'], f(m['accuracy']), f(m.get('acc_within_1', float('nan'))),
            f(m['macro_f1']), f(m['mae']), f(m[kappa_key]), f(m['spearman_rho']))

    for d in DIMS:
        lines.append(row(d, res['per_dimension'][d]))
    lines.append('-' * len(head))
    macro = dict(res['macro_over_dimensions'])
    macro['n'] = res['per_dimension'][DIMS[0]]['n']
    lines.append(row('macro over dimensions', macro))
    lines.append(row('overall (pooled)', res['overall']))

    ps = [(d, res['per_dimension'][d].get('spearman_p')) for d in DIMS]
    ps = [(d, p) for d, p in ps if p is not None]
    if ps:
        text = 'Spearman p: ' + ', '.join('%s=%.3g' % (d.split()[0], p) for d, p in ps)
        overall_p = res['overall'].get('spearman_p')
        if overall_p is not None:
            text += ' | overall p=%.3g' % overall_p
        lines.append('')
        lines.append(text)
    return '\n'.join(lines)


def format_confusion(res: dict) -> str:
    """逐维混淆矩阵 + 预测分布 + 误差幅度分布：一眼看出是否类别塌缩、严重错误多少。"""
    L = res['config']['num_levels']
    lines = ['逐维混淆矩阵（行=真值 t，列=预测 p，1..%d）' % L]
    for d in DIMS:
        m = res['confusion_matrix'][d]
        eh = res['per_dimension'][d].get('error_hist', {})
        lines.append('')
        lines.append('  %s' % d)
        lines.append('        ' + ''.join('%5s' % ('p%d' % (j + 1)) for j in range(L)) + '   | 合计')
        for i in range(L):
            lines.append('    t%d  ' % (i + 1) + ''.join('%5d' % m[i][j] for j in range(L))
                         + '   | %d' % sum(m[i]))
        used = sum(1 for j in range(L) if sum(m[i][j] for i in range(L)) > 0)
        lines.append('    预测分布 %s   用到 %d/%d 个等级%s'
                     % (res['prediction_distribution'][d], used, L,
                        '   <-- 类别塌缩' if used <= 1 else ''))
        if eh:
            lines.append('    误差幅度 |Δ|: ' + '  '.join(
                '%s=%d(%.0f%%)' % (k, eh[k]['n'], 100 * eh[k]['frac'])
                for k in ['0', '1', '2', '3', '4'] if k in eh)
                + '   严重(≥2)=%d(%.0f%%)' % (eh['severe_ge2']['n'], 100 * eh['severe_ge2']['frac']))
    return '\n'.join(lines)
