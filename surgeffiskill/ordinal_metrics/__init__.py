# -*- coding: utf-8 -*-
"""ordinal_metrics — SurgEffiSkill 技能等级（4 维 × 1..5）的序数指标。

现有的 text_metric.py 只报 BLEU/ROUGE/METEOR/CIDEr/SPICE 这类文本指标，衡量的是
生成文本像不像参考答案；这个包补的是**评级本身准不准**：

    Accuracy · Macro-F1 · MAE · 加权 Cohen's κ · Spearman ρ

典型用法：

    from ordinal_metrics import OrdinalEvaluator, build_label_index, format_report

    ev = OrdinalEvaluator(label_index=build_label_index())
    res = ev.score_records(records)      # records 里每条要有 pred，真值给 label_score/video/gt 任一
    print(format_report(res))

命令行（给已经跑好的 eval jsonl 补算指标）：

    python3 eval_ordinal.py eval_results_metrics.jsonl
"""

from .decode import DIMS, decode_levels, decode_vector, get_rubric, split_dimensions
from .metrics import (METRIC_KEYS, acc_within, accuracy, all_metrics, error_hist,
                      macro_f1, mae, spearman, weighted_kappa)
from .evaluate import (OrdinalEvaluator, build_label_index, format_confusion,
                       format_report)

__all__ = [
    'DIMS', 'decode_levels', 'decode_vector', 'get_rubric', 'split_dimensions',
    'accuracy', 'macro_f1', 'mae', 'weighted_kappa', 'spearman', 'all_metrics',
    'METRIC_KEYS', 'OrdinalEvaluator', 'build_label_index', 'format_report', 'format_confusion',
    'acc_within', 'error_hist',
]
