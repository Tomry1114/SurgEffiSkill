# -*- coding: utf-8 -*-
"""把 LLM 缓存切成 **clip 级** 样本（每个时间块一条），而不是每视频一条。

动机（v8 Stage-B/C 的实测结论）：在 98 个视频级样本上，MMoE 的逐样本路由测不出
任何作用——CV 选中的操作点门控偏离只有 0.04–0.07，且路由越尖锐增益越低。
98 个样本喂 8 个专家 + 一个逐样本门控，本来就是结构性劣势。

clip 级把训练样本从 98 提到约 2500，并且让门控真正有内容可路由：
"这一段在做组织牵拉" vs "这一段在换器械" 本就该走不同专家——这才是 MMoE 的原意。
标签沿用该视频的四维分数（弱监督），预测在视频内取平均后再按视频级评分。

**分折必须按 video 分组**：同一视频的 clip 高度相关，随机分折会把同一视频劈到
train/test 两边，指标会虚高。下游 `moe_clip.py` 用 GroupKFold(groups=video)。
"""
import argparse
import io
import json
import os

import numpy as np


def load_source(cache):
    index = json.load(io.open(os.path.join(cache, 'index.json'), encoding='utf-8'))
    feats, per_video = [], []
    for i, r in enumerate(index):
        arr = np.load(os.path.join(cache, r['npy'])).astype(np.float32)     # (T,g,g,D)
        f = arr.reshape(arr.shape[0], -1, arr.shape[-1])                    # (T,g*g,D)
        v = np.concatenate([f.mean(axis=1), f.std(axis=1)], axis=1)         # (T,2D)
        feats.append(v)
        per_video.append(int(arr.shape[0]))
        if i % 30 == 0:
            print('  [%3d/%d] %s T=%d' % (i, len(index), r['video'], arr.shape[0]), flush=True)
    return index, feats, per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caches', nargs='+', required=True, help='形如 name=dir')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    names, mats, ref, blocks, off = [], [], None, {}, 0
    counts = None
    for spec in a.caches:
        nm, d = spec.split('=', 1)
        print('读 %s ← %s' % (nm, d), flush=True)
        idx, fs, cnt = load_source(d)
        if ref is None:
            ref, counts = idx, cnt
        else:
            assert len(idx) == len(ref) and counts == cnt, '两个缓存的视频顺序 / 块数不一致'
            for r0, r1 in zip(ref, idx):
                assert r0['video'] == r1['video'] and r0['label'] == r1['label']
        names.append(nm)
        mats.append(np.concatenate(fs, axis=0))
        blocks[nm] = [off, off + mats[-1].shape[1]]
        off += mats[-1].shape[1]

    X = np.hstack(mats)
    assert np.isfinite(X).all(), 'clip 特征出现 NaN/Inf'
    meta = []
    for r, n in zip(ref, counts):
        for t in range(n):
            meta.append({'split': r['split'], 'video': r['video'], 'label': r['label'],
                         'clip': t, 'n_clips': n})
    assert len(meta) == len(X)
    np.savez_compressed(a.out, X=X, meta=json.dumps(meta, ensure_ascii=False),
                        blocks=json.dumps(blocks))
    ntr = sum(1 for m in meta if m['split'] == 'train')
    print('已存 %s  X=%s  clip: train=%d val=%d  视频: %d  每视频块数 %d–%d（中位 %d）'
          % (a.out, X.shape, ntr, len(meta) - ntr, len(ref),
             min(counts), max(counts), int(np.median(counts))))
    print('列区间 blocks = %s' % json.dumps(blocks))


if __name__ == '__main__':
    main()
