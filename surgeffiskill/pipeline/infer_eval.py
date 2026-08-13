# -*- coding: utf-8 -*-
"""infer_eval.py — 对一个 LoRA checkpoint 在指定 split 上做推理 + 序数指标。

**用 ms-swift 自己的 PtEngine 推理**，而不是自己拼 `apply_chat_template`：
训练走的是 swift 模板（system、`<video>` 占位、fetch_video 参数都由模板决定），
自己拼很容易和训练端有细微分布差异。用同一套模板从构造上排除这类混淆。

固定评测协议（两个实验臂必须完全一致，只有 --adapters 不同）：
  * 贪心解码（temperature=0），seed 固定 → 同一 checkpoint 重跑可复现
  * max_tokens=256
  * 指标 = ordinal_metrics（Accuracy / Macro-F1 / MAE / 加权 κ / Spearman）+ coverage
  * 额外报「不同预测的个数」——塌成常数时这个是 1，必须一眼看见

    python infer_eval.py --adapters <ckpt> --data <split.json> --tag sft --outdir <dir>
"""
import argparse
import collections
import io
import json
import os
import sys
import time

import torch

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
sys.path.insert(0, PROJECT)
from ordinal_metrics import OrdinalEvaluator, format_confusion, format_report          # noqa: E402
from ordinal_metrics.decode import decode_vector                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct'))
    ap.add_argument('--adapters', default=None, help='LoRA checkpoint；不给则跑基座')
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0, help='0=贪心')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    from swift.llm import InferRequest, PtEngine, RequestConfig

    print('[%s] base=%s\n[%s] adapters=%s' % (a.tag, a.base, a.tag, a.adapters), flush=True)
    engine = PtEngine(a.base, torch_dtype=torch.bfloat16, max_batch_size=1,
                      adapters=[a.adapters] if a.adapters else None)

    data = json.load(io.open(a.data, encoding='utf-8'))
    print('[%s] %d samples from %s' % (a.tag, len(data), a.data), flush=True)

    reqs, meta = [], []
    for rec in data:
        user = [m for m in rec['messages'] if m['role'] == 'user'][0]['content']
        gt_text = [m for m in rec['messages'] if m['role'] == 'assistant'][0]['content']
        reqs.append(InferRequest(messages=[{'role': 'user', 'content': user}],
                                 videos=list(rec['videos'])))
        meta.append((os.path.basename(rec['videos'][0]),
                     [int(x) for x in rec['label_score']], gt_text))

    cfg = RequestConfig(max_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed)
    t0 = time.time()
    resps = engine.infer(reqs, cfg)
    elapsed = time.time() - t0

    results = []
    for (video, levels, gt_text), resp in zip(meta, resps):
        pred_text = resp.choices[0].message.content.strip()
        results.append({'video': video, 'label_score': levels, 'gt': gt_text,
                        'pred': pred_text, 'gt_levels': levels,
                        'pred_levels': decode_vector(pred_text)})

    pred_path = os.path.join(a.outdir, 'pred_%s.jsonl' % a.tag)
    with io.open(pred_path, 'w', encoding='utf-8') as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + u'\n')

    uniq_text = len(set(r['pred'] for r in results))
    uniq_lv = collections.Counter(tuple(r['pred_levels']) for r in results)

    res = OrdinalEvaluator().score_records(results)
    res['run'] = {'tag': a.tag, 'adapters': a.adapters, 'data': a.data, 'seed': a.seed,
                  'engine': 'swift.PtEngine', 'temperature': a.temperature,
                  'decoding': 'greedy' if a.temperature == 0 else 'sample',
                  'max_new_tokens': a.max_new_tokens, 'n': len(results),
                  'elapsed_sec': round(elapsed, 1)}
    res['diversity'] = {'unique_pred_texts': uniq_text,
                        'unique_pred_level_vectors': len(uniq_lv),
                        'top_level_vectors': [[list(k), v] for k, v in uniq_lv.most_common(5)]}
    metric_path = os.path.join(a.outdir, 'ordinal_%s.json' % a.tag)
    with io.open(metric_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print('\n===== %s | %s =====' % (a.tag, os.path.basename(a.data)))
    print(format_report(res))
    print()
    print(format_confusion(res))
    print('\n多样性：不同预测文本 %d/%d，不同等级向量 %d，最常见 %s'
          % (uniq_text, len(results), len(uniq_lv), uniq_lv.most_common(3)))
    if len(uniq_lv) == 1:
        print('!! 预测塌成常数向量 —— κ 必为 0、Spearman 无定义，不是指标计算错误')
    print('\n耗时 %.1f s（%.1f s/样本）' % (elapsed, elapsed / max(len(results), 1)))
    print('预测 => %s\n指标 => %s' % (pred_path, metric_path))


if __name__ == '__main__':
    main()
