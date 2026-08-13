# -*- coding: utf-8 -*-
"""缓存 LLM 在**视觉 token 位置**上的最后一层隐状态，供 SkillMoE 使用。

与 `cache_visual_tokens.py` 的区别：后者取视觉塔输出（纯视觉），本脚本取
LoRA-SFT 之后 LLM 末层在视频 token 位置上的表示——语言接地过的特征，
即 `sftbalpd3_ck250` 实际用于打分的东西。

注意：这与从未跑通的 `qwen2_5_vl_mmoe.py` 的关键差别是——
它取的是**单个文本 token**（"Depth perception:" 那个位置）的隐状态，
而塌缩正发生在那里（预测 PC1=0.889）；本脚本取的是**整段视觉 token 序列**，
让 MoE 自己按维度去挑，而不是在一个已经塌成秩-1 的向量上做四份线性投影。
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
PROMPT = ('<video>You are assessing laparoscopic surgical skill from the video above '
          'on four dimensions: Depth perception, Bimanual coordination, Efficiency, '
          'Tissue handling.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--adapters', default=None, help='LoRA checkpoint；不给则用基座')
    ap.add_argument('--grid', type=int, default=8)
    a = ap.parse_args()

    from swift.llm import InferRequest, PtEngine, RequestConfig  # noqa: F401
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import fetch_video

    os.makedirs(a.outdir, exist_ok=True)
    mp = os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct')
    proc = AutoProcessor.from_pretrained(mp)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    if a.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapters).eval()
        print('已加载 adapter %s' % a.adapters, flush=True)
    merge = int(getattr(getattr(model, 'visual', None) or model.base_model.model.visual,
                        'spatial_merge_size', 2))
    tok = proc.tokenizer
    vid_id = tok.convert_tokens_to_ids('<|video_pad|>')
    print('video_pad token id = %s' % vid_id, flush=True)

    recs = []
    for split in ('train.json', 'val.json'):
        for r in json.load(io.open(os.path.join(SPL, split), encoding='utf-8')):
            recs.append((split[:-5], r['videos'][0], [int(x) for x in r['label_score']]))

    index = []
    for i, (split, path, lab) in enumerate(recs):
        name = os.path.basename(path).rsplit('.', 1)[0]
        dst = os.path.join(a.outdir, name + '.npy')
        if not os.path.exists(dst):
            vid = fetch_video({'type': 'video', 'video': path})
            if isinstance(vid, tuple):
                vid = vid[0]
            msgs = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], videos=[vid], return_tensors='pt').to('cuda:0')
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1][0]                      # (L, D)
            pos = (inp['input_ids'][0] == vid_id).nonzero(as_tuple=True)[0]
            hv = h[pos].float()                               # (N_vis, D)
            t, hh, ww = [int(x) for x in inp['video_grid_thw'][0].tolist()]
            gh, gw = hh // merge, ww // merge
            assert hv.shape[0] == t * gh * gw, (hv.shape, t, gh, gw)
            x = hv.reshape(t, gh, gw, -1).permute(0, 3, 1, 2)
            x = F.adaptive_avg_pool2d(x, (a.grid, a.grid))
            arr = x.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float16)
            np.save(dst, arr)
            del out, h, hv, x
            torch.cuda.empty_cache()
        else:
            arr = np.load(dst, mmap_mode='r')
        index.append({'split': split, 'video': os.path.basename(path), 'npy': name + '.npy',
                      'label': lab, 'shape': list(arr.shape)})
        if i % 10 == 0:
            print('  [%3d/%d] %s %s' % (i, len(recs), name, tuple(arr.shape)), flush=True)

    json.dump(index, io.open(os.path.join(a.outdir, 'index.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('完成 %d 个，共 %.1f MB'
          % (len(index), sum(os.path.getsize(os.path.join(a.outdir, r['npy']))
                             for r in index) / 1e6))


if __name__ == '__main__':
    main()
