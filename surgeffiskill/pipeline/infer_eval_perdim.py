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
from ordinal_metrics import OrdinalEvaluator, format_confusion, format_report
from ordinal_metrics.decode import DIMS, decode_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct'))
    ap.add_argument('--adapters', default=None)
    ap.add_argument('--data', required=True, help='per-dimension dataset, e.g. splits82/val_pd.json')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    from swift.llm import InferRequest, PtEngine, RequestConfig

    print('[%s] adapters=%s' % (a.tag, a.adapters), flush=True)
    engine = PtEngine(a.base, torch_dtype=torch.bfloat16, max_batch_size=1,
                      adapters=[a.adapters] if a.adapters else None)

    data = json.load(io.open(a.data, encoding='utf-8'))
    print('[%s] %d per-dimension samples (%d videos)' % (a.tag, len(data), len(data) // len(DIMS)), flush=True)

    reqs, meta = [], []
    for rec in data:
        user = [m for m in rec['messages'] if m['role'] == 'user'][0]['content']
        gt_line = [m for m in rec['messages'] if m['role'] == 'assistant'][0]['content']
        reqs.append(InferRequest(messages=[{'role': 'user', 'content': user}],
                                 videos=list(rec['videos'])))
        meta.append((os.path.basename(rec['videos'][0]), rec['dim_index'], rec['dim'],
                     [int(x) for x in rec['label_score']], gt_line))

    cfg = RequestConfig(max_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed)
    t0 = time.time()
    resps = engine.infer(reqs, cfg)
    elapsed = time.time() - t0


    merged, order = {}, []
    n_undec = 0
    for (video, di, dim, levels, gt_line), resp in zip(meta, resps):
        line = resp.choices[0].message.content.strip()
        if video not in merged:
            merged[video] = {'video': video, 'label_score': levels, 'gt_levels': levels,
                             'pred_levels': [None] * len(DIMS),
                             '_pred_lines': [''] * len(DIMS), '_gt_lines': [''] * len(DIMS)}
            order.append(video)
        lv = decode_levels(line).get(dim)
        if lv is None:
            n_undec += 1
        merged[video]['pred_levels'][di] = lv
        merged[video]['_pred_lines'][di] = line
        merged[video]['_gt_lines'][di] = gt_line

    results = []
    for v in order:
        r = merged[v]
        r['pred'] = '\n'.join(r.pop('_pred_lines'))
        r['gt'] = '\n'.join(r.pop('_gt_lines'))
        results.append(r)

    with io.open(os.path.join(a.outdir, 'pred_%s.jsonl' % a.tag), 'w', encoding='utf-8') as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + u'\n')

    res = OrdinalEvaluator().score_records(results)
    uniq_lv = collections.Counter(tuple(r['pred_levels']) for r in results)
    res['run'] = {'tag': a.tag, 'adapters': a.adapters, 'data': a.data, 'seed': a.seed,
                  'mode': 'per-dimension (4 queries per video, merged before scoring)',
                  'decoding': 'greedy' if a.temperature == 0 else 'sample',
                  'n_videos': len(results), 'n_queries': len(data),
                  'undecodable_lines': n_undec, 'elapsed_sec': round(elapsed, 1)}
    res['diversity'] = {'unique_pred_texts': len(set(r['pred'] for r in results)),
                        'unique_pred_level_vectors': len(uniq_lv),
                        'top_level_vectors': [[list(k), v] for k, v in uniq_lv.most_common(5)]}
    with io.open(os.path.join(a.outdir, 'ordinal_%s.json' % a.tag), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print('\n===== %s | per-dimension mode =====' % a.tag)
    print(format_report(res))
    print()
    print(format_confusion(res))
    print('\ndistinct level vectors %d, most common %s; single-line decode failures %d/%d'
          % (len(uniq_lv), uniq_lv.most_common(3), n_undec, len(data)))
    print('elapsed %.1f s (%.1f s/query)' % (elapsed, elapsed / max(len(data), 1)))


if __name__ == '__main__':
    main()
