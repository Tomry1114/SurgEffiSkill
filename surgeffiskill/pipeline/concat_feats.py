# -*- coding: utf-8 -*-
"""按视频逐条对齐地拼接多个特征文件（`vis_feats_*.npz` / `llm_feats_*.npz` 同格式）。

拼接前**逐条断言** video / split / label 三者一致——特征来自不同的缓存目录，
顺序一致不是可以默认的事情。列区间写进 meta 的 `blocks`，供 `anchor_mmoe.py`
的 `--blocks` 做逐维特征源选择。
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='形如 name=path.npz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    names, Xs, metas = [], [], []
    for spec in a.inputs:
        name, path = spec.split('=', 1)
        d = np.load(path, allow_pickle=True)
        names.append(name)
        Xs.append(d['X'].astype(np.float32))
        metas.append(json.loads(str(d['meta'])))
        print('读入 %-10s %-46s X=%s' % (name, path, Xs[-1].shape))

    ref = metas[0]
    for name, m in zip(names[1:], metas[1:]):
        assert len(m) == len(ref), '%s 记录数 %d != %d' % (name, len(m), len(ref))
        for i, (r0, r1) in enumerate(zip(ref, m)):
            for k in ('video', 'split', 'label'):
                assert r0[k] == r1[k], '第 %d 条 %s 不一致: %r vs %r' % (i, k, r0[k], r1[k])
    print('逐条对齐校验通过：%d 条记录，video/split/label 全等' % len(ref))

    blocks, off = {}, 0
    for name, X in zip(names, Xs):
        blocks[name] = [off, off + X.shape[1]]
        off += X.shape[1]
    X = np.hstack(Xs)
    assert np.isfinite(X).all(), '拼接后出现 NaN/Inf'

    meta = json.dumps(ref, ensure_ascii=False)
    np.savez_compressed(a.out, X=X, meta=meta, blocks=json.dumps(blocks))
    n_tr = sum(1 for r in ref if r['split'] == 'train')
    print('已存 %s  X=%s  train=%d val=%d' % (a.out, X.shape, n_tr, len(ref) - n_tr))
    print('列区间 blocks = %s' % json.dumps(blocks))


if __name__ == '__main__':
    main()
