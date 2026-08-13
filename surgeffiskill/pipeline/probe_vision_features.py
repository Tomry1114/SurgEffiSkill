# -*- coding: utf-8 -*-
"""冻结 vision tower 线性探针：视觉特征里到底有没有各维度（尤其 Tissue）的信息？

只用 Qwen2.5-VL 的视觉塔，不过 LLM。对 123 个视频抽特征，在 train(98) 上拟线性探针，
在 val(25) 上评测。所有预处理统计量（标准化、PCA）只用 train 拟合；超参用 train 内部
5 折 CV 选；val 只在最后评一次。不改验证集、不改评测器。

阶段 1（--stage extract）：抽特征存 npz（需要 GPU）。
阶段 2（--stage probe）：纯 CPU，拟探针 + 出指标。
"""
import argparse
import io
import json
import os
import sys

import numpy as np

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
SPL = os.path.join(PROJECT, 'revision/scripts/sft_grpo/splits82')
DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']


def load_split(name):
    return json.load(io.open(os.path.join(SPL, name), encoding='utf-8'))


# ------------------------------------------------------------------ 阶段 1
def extract(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import fetch_video

    mp = os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct')
    proc = AutoProcessor.from_pretrained(mp)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()
    visual = model.visual
    merge = int(getattr(visual, 'spatial_merge_size', 2))

    recs = []
    for split in ('train.json', 'val.json'):
        for r in load_split(split):
            recs.append((split.replace('.json', ''), r['videos'][0],
                         [int(x) for x in r['label_score']]))
    print('共 %d 个视频' % len(recs), flush=True)

    # 断点续跑：每个视频单独落盘，30 分钟 debug 卡跑不完也不至于全白干
    partdir = args.out + '.parts'
    os.makedirs(partdir, exist_ok=True)

    feats, meta = [], []
    for i, (split, path, lab) in enumerate(recs):
        name = os.path.basename(path).rsplit('.', 1)[0]
        pf = os.path.join(partdir, name + '.npy')
        if os.path.exists(pf):
            vec = np.load(pf)
            t = per_frame = -1
        else:
            ele = {'type': 'video', 'video': path}
            if args.fps:
                ele['fps'] = args.fps
                ele['max_frames'] = args.max_frames
            vid = fetch_video(ele)
            if isinstance(vid, tuple):
                vid = vid[0]
            inputs = proc(text=['<|vision_start|><|video_pad|><|vision_end|>'],
                          videos=[vid], return_tensors='pt')
            pv = inputs['pixel_values_videos'].to('cuda:0', dtype=torch.bfloat16)
            thw = inputs['video_grid_thw'].to('cuda:0')
            with torch.no_grad():
                tok = visual(pv, grid_thw=thw).float().cpu().numpy()   # (N, D)
            t, h, w = [int(x) for x in thw[0].tolist()]
            per_frame = (h // merge) * (w // merge)
            assert tok.shape[0] == t * per_frame, (tok.shape, t, per_frame)
            f = tok.reshape(t, per_frame, tok.shape[-1])
            fm = f.mean(axis=1)                   # (t, D) 每个时间块的空间均值
            vec = np.concatenate([
                fm.mean(axis=0),                  # 全局均值
                fm.std(axis=0),                   # 时间波动（稀有事件的信号在这里）
                fm.max(axis=0) - fm.mean(axis=0),  # 峰值偏离（离散事件）
            ]).astype(np.float32)
            np.save(pf, vec)
        feats.append(vec)
        meta.append({'split': split, 'video': os.path.basename(path), 'label': lab,
                     'n_time_chunks': t, 'tokens_per_frame': per_frame})
        if i % 10 == 0:
            print('  [%3d/%d] %s  vec=%d' % (i, len(recs), name, vec.shape[0]), flush=True)
    X = np.stack(feats)
    np.savez_compressed(args.out, X=X, meta=json.dumps(meta, ensure_ascii=False))
    print('已存 %s  X=%s' % (args.out, X.shape))


# ------------------------------------------------------------------ 阶段 2
def probe(args):
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import cohen_kappa_score, f1_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.out, allow_pickle=True)
    X = d['X']
    meta = json.loads(str(d['meta']))
    y = np.array([m['label'] for m in meta], float)
    istr = np.array([m['split'] == 'train' for m in meta])
    Xtr, Xva, ytr, yva = X[istr], X[~istr], y[istr], y[~istr]
    print('train %s  val %s  特征维 %d' % (Xtr.shape, Xva.shape, X.shape[1]))

    # 三段特征分别标准化后再拼（均值 / 时间std / 峰值偏离）
    D3 = X.shape[1] // 3
    parts_tr, parts_va = [], []
    for s in range(3):
        sc = StandardScaler().fit(Xtr[:, s * D3:(s + 1) * D3])
        parts_tr.append(sc.transform(Xtr[:, s * D3:(s + 1) * D3]))
        parts_va.append(sc.transform(Xva[:, s * D3:(s + 1) * D3]))
    Xtr_s, Xva_s = np.hstack(parts_tr), np.hstack(parts_va)

    ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
    NCOMP = [8, 16, 32, 64]
    rows = []
    for i, dim in enumerate(DIMS):
        best = None
        for nc in NCOMP:
            for al in ALPHAS:
                errs = []
                for tr, te in KFold(5, shuffle=True, random_state=0).split(Xtr_s):
                    p = PCA(n_components=min(nc, len(tr) - 1), random_state=0).fit(Xtr_s[tr])
                    m = Ridge(alpha=al).fit(p.transform(Xtr_s[tr]), ytr[tr, i])
                    errs.append(np.mean((m.predict(p.transform(Xtr_s[te])) - ytr[te, i]) ** 2))
                cv = float(np.mean(errs))
                if best is None or cv < best[0]:
                    best = (cv, nc, al)
        cv, nc, al = best
        p = PCA(n_components=nc, random_state=0).fit(Xtr_s)
        m = Ridge(alpha=al).fit(p.transform(Xtr_s), ytr[:, i])
        raw = m.predict(p.transform(Xva_s))
        pred = np.clip(np.rint(raw), 1, 5)
        g = yva[:, i]
        lab = sorted(set(g.astype(int).tolist()) | set(pred.astype(int).tolist()))
        rho, pv = spearmanr(raw, g)          # 用连续分算 rho，不受取整影响
        base = float(np.bincount(ytr[:, i].astype(int), minlength=6)[1:].argmax() + 1)
        rows.append(dict(dim=dim, n_pca=nc, alpha=al, cv_mse=round(cv, 4),
                         exact=float((pred == g).mean()),
                         within1=float((abs(pred - g) <= 1).mean()),
                         mae=float(abs(pred - g).mean()),
                         f1=float(f1_score(g, pred, labels=lab, average='macro', zero_division=0)),
                         qwk=float(cohen_kappa_score(g.astype(int), pred.astype(int),
                                                     weights='quadratic')),
                         rho=float(rho), p=float(pv),
                         base_exact=float((g == base).mean()),
                         pred_levels=sorted(set(pred.astype(int).tolist()))))

    print('\n%-24s %6s %6s %6s %6s %7s %7s %10s %8s %s'
          % ('dimension', 'Exact', '基线', '±1', 'MAE', 'QWK', 'rho', 'p', 'PCA/α', '用到的级别'))
    for r in rows:
        print('%-24s %6.2f %6.2f %6.2f %6.2f %7.3f %7.3f %10.4g %4d/%-5.0g %s'
              % (r['dim'], r['exact'], r['base_exact'], r['within1'], r['mae'],
                 r['qwk'], r['rho'], r['p'], r['n_pca'], r['alpha'], r['pred_levels']))
    print('\n合并（4 维 100 对）: Exact=%.2f  ±1=%.2f  MAE=%.2f'
          % (np.mean([r['exact'] for r in rows]), np.mean([r['within1'] for r in rows]),
             np.mean([r['mae'] for r in rows])))
    json.dump(rows, io.open(args.report, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('已存 %s' % args.report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['extract', 'probe'], required=True)
    ap.add_argument('--out', default=os.path.join(PROJECT, 'revision/evidence/results/sft_grpo/vis_feats.npz'))
    ap.add_argument('--report', default=os.path.join(PROJECT, 'revision/evidence/results/sft_grpo/vis_probe.json'))
    ap.add_argument('--fps', type=float, default=0.0, help='0 = 用默认采样（28 帧）')
    ap.add_argument('--max-frames', type=int, default=128)
    a = ap.parse_args()
    (extract if a.stage == 'extract' else probe)(a)
