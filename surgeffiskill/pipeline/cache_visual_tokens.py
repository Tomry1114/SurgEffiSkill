# -*- coding: utf-8 -*-
"""缓存 Qwen2.5-VL 视觉塔输出的 token 网格，供 SkillMoE 头离线训练。

与线性探针共用同一条前向路径，但保留时空结构而不是池化成一个向量：
每个视频存成 (T, G, G, D) 的 fp16 张量（G = 空间自适应池化后的边长）。
视觉塔全程冻结，不反传；本脚本只做前向。
"""
import argparse
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
SPL = os.path.join(PROJECT, 'revision/scripts/sft_grpo/splits82')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--grid', type=int, default=8, help='空间自适应池化到 grid×grid')
    ap.add_argument('--fps', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int, default=128)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import fetch_video

    os.makedirs(a.outdir, exist_ok=True)
    mp = os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct')
    proc = AutoProcessor.from_pretrained(mp)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    visual = model.visual
    merge = int(getattr(visual, 'spatial_merge_size', 2))

    recs = []
    for split in ('train.json', 'val.json'):
        for r in json.load(io.open(os.path.join(SPL, split), encoding='utf-8')):
            recs.append((split[:-5], r['videos'][0], [int(x) for x in r['label_score']]))
    print('共 %d 个视频 -> %s' % (len(recs), a.outdir), flush=True)

    index = []
    for i, (split, path, lab) in enumerate(recs):
        name = os.path.basename(path).rsplit('.', 1)[0]
        dst = os.path.join(a.outdir, name + '.npy')
        if not os.path.exists(dst):
            ele = {'type': 'video', 'video': path}
            if a.fps:
                ele['fps'] = a.fps
                ele['max_frames'] = a.max_frames
            vid = fetch_video(ele)
            if isinstance(vid, tuple):
                vid = vid[0]
            inp = proc(text=['<|vision_start|><|video_pad|><|vision_end|>'],
                       videos=[vid], return_tensors='pt')
            pv = inp['pixel_values_videos'].to('cuda:0', dtype=torch.bfloat16)
            thw = inp['video_grid_thw'].to('cuda:0')
            with torch.no_grad():
                tok = visual(pv, grid_thw=thw)                      # (N, D)
            t, h, w = [int(x) for x in thw[0].tolist()]
            gh, gw = h // merge, w // merge
            assert tok.shape[0] == t * gh * gw, (tok.shape, t, gh, gw)
            x = tok.float().reshape(t, gh, gw, -1).permute(0, 3, 1, 2)   # (T,D,gh,gw)
            x = F.adaptive_avg_pool2d(x, (a.grid, a.grid))               # (T,D,G,G)
            arr = x.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float16)
            np.save(dst, arr)
        else:
            arr = np.load(dst, mmap_mode='r')
        index.append({'split': split, 'video': os.path.basename(path), 'npy': name + '.npy',
                      'label': lab, 'shape': list(arr.shape)})
        if i % 10 == 0:
            print('  [%3d/%d] %s %s' % (i, len(recs), name, arr.shape), flush=True)

    json.dump(index, io.open(os.path.join(a.outdir, 'index.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    tot = sum(os.path.getsize(os.path.join(a.outdir, r['npy'])) for r in index)
    print('完成：%d 个文件，共 %.1f MB' % (len(index), tot / 1e6))


if __name__ == '__main__':
    main()
