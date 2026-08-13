# ordinal_metrics — 技能等级的序数指标

`text_metric.py` 报的是 BLEU/ROUGE/METEOR/CIDEr/SPICE 这类**文本**指标，衡量生成的评价
像不像参考答案。这个包补的是另一半：**评级本身准不准**。

| 指标 | 含义 | 方向 |
|---|---|---|
| Accuracy | 4 维等级完全命中的比例 | ↑ |
| Macro-F1 | 各等级 F1 的宏平均（不被多数档淹没） | ↑ |
| MAE | 平均绝对等级误差 | ↓ |
| Weighted Cohen's κ | 与专家评分的一致性，按等级距离加权（默认二次） | ↑ |
| Spearman ρ | 与专家评分的秩相关 | ↑ |

## 用法

```bash
# 1) 给已经跑好的推理结果补算（不用重跑推理）
python3 eval_ordinal.py eval_results_metrics.jsonl
python3 eval_ordinal.py eval_*.jsonl --json ordinal_summary.json

# 2) 推理时一起算：infer_metric_ce.py 已接入，跑完会在文本指标之后打印序数指标表，
#    并把每条样本的 gt_levels / pred_levels 写进输出 jsonl
```

```python
from ordinal_metrics import OrdinalEvaluator, build_label_index, format_report

ev = OrdinalEvaluator(label_index=build_label_index())
res = ev.score_records(records)          # 每条要有 pred；真值给 label_score / video / gt 任一
print(format_report(res))
```

## 等级从哪来

模型输出的是 rubric 散文、没有数字等级，所以按 `surg_rubric.json`（来自
`Surgical_Skills_Evaluation_Table.xlsx`）对每个维度的句子做最近邻匹配解码成 1..5，
同时支持显式写法 `Efficiency (4):` / `level 4` / `4/5`。

这条之所以站得住：SFT 目标本身就是 rubric 句子——全部 123 条 CholeSkill 参考答案都能
逐字还原回其 `label_score`（`verify_reward_on_data.py`）。解码器与 GRPO 的 eq(9) 序数
奖励**共用同一份实现**（`ordinal_metrics/decode.py`），不存在两处口径分叉。

真值优先级：记录里的 `label_score` > 按视频文件名从 `train_effi_cot*.json` 索引 > 从 `gt` 文本解码。

## 报数时必须交代的三件事

1. **coverage**：输出里永远带 `pairs_scored / pairs_total` 和 `pred_decode_failures`。
   默认 `undecodable='skip'`——解不出等级的 (样本,维度) 对被**跳过**，所以指标是
   "在能解码的输出上"的条件值。要把它们算作错，用 `--undecodable worst`。两者不是一回事。
2. **Macro-F1 的类集**：默认只对 `y_true ∪ y_pred` 出现过的类求平均（sklearn 默认行为）；
   `--macro-f1-labels all` 强制按整套 1..5——数据里没出现的档会得 F1=0 并显著拉低宏平均。
3. **κ 的标尺**：加权 κ 固定按整条 1..5 标尺建权重矩阵（不是只用观测到的类），
   因为权重取决于两个等级在标尺上的距离。默认二次权重，可切 `--kappa linear`。

结果分三层给，别只挑一个数报：`per_dimension`（4 维各自）、`macro_over_dimensions`
（4 维平均）、`overall`（全部样本×维度混算）。

## 实现与验证

纯 numpy 实现，不依赖 sklearn/scipy（推理用的 `qwen_cpu` 环境没有 sklearn）。
自检：

```bash
python3 -m ordinal_metrics.test_metrics
```

手算边界 + 在装了 sklearn/scipy 的环境（`easyr1`）里与 `sklearn.metrics` /
`scipy.stats` 逐个对拍 300 组随机数据；没装的环境会明确打印跳过了哪几项。
Spearman 的 p 值只在有 scipy 时给出，否则为 `None`（不用近似式糊一个）。
