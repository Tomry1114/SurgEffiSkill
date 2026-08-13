# -*- coding: utf-8 -*-
"""把 `cache_llm_tokens.py` 缓存的 LLM 末层隐状态池化成与 `vis_feats_*.npz` 同格式的特征。

池化方式与 `probe_vision_features.py` 逐字一致（空间均值 → 时间三段：均值 / 时间 std /
峰值偏离），这样 ridge 探针、MMoE、评测脚本全部无需改动，两套特征可直接对比：
唯一的差别就是表征来自**冻结视觉塔** 还是 **ck250 LoRA 之后的 LLM**。
"""
import argparse
import io
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    index = json.load(io.open(os.path.join(a.cache, 'index.json'), encoding='utf-8'))
    feats, meta = [], []
    for i, r in enumerate(index):
        arr = np.load(os.path.join(a.cache, r['npy'])).astype(np.float32)   # (T, g, g, D)
        fm = arr.reshape(arr.shape[0], -1, arr.shape[-1]).mean(axis=1)      # (T, D) 空间均值
        feats.append(np.concatenate([fm.mean(axis=0),
                                     fm.std(axis=0),
                                     fm.max(axis=0) - fm.mean(axis=0)]).astype(np.float32))
        meta.append({'split': r['split'], 'video': r['video'], 'label': r['label'],
                     'n_time_chunks': int(arr.shape[0])})
        if i % 20 == 0:
            print('  [%3d/%d] %s T=%d' % (i, len(index), r['video'], arr.shape[0]), flush=True)
    X = np.stack(feats)
    assert np.isfinite(X).all(), '池化后出现 NaN/Inf'
    np.savez_compressed(a.out, X=X, meta=json.dumps(meta, ensure_ascii=False))
    n_tr = sum(1 for m in meta if m['split'] == 'train')
    print('已存 %s  X=%s  train=%d val=%d' % (a.out, X.shape, n_tr, len(meta) - n_tr))


if __name__ == '__main__':
    main()
