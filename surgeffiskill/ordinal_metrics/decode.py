# -*- coding: utf-8 -*-
"""ordinal_metrics.decode — 把模型输出的评价文本解码成 4 维 1..5 的等级。

这是 rubric 解码的**唯一实现**，序数指标（ordinal_metrics.evaluate）和 GRPO 的
eq(9) 序数奖励（grpo_reward_plugin.py）都从这里取，避免两处各写一份而慢慢跑偏。

模型输出的是 rubric 散文、没有数字等级，所以按 `surg_rubric.json` 里每个 (维度,等级)
的标准描述做最近邻匹配；同时支持显式写法 `Efficiency (4):` / `level 4` / `4/5`。
这条之所以成立：SFT 目标本身就是 rubric 句子 —— 全部 123 条 CholeSkill 样本的参考
答案都能逐字还原回它的 label_score（见 verify_reward_on_data.py）。
"""

import difflib
import json
import os
import re
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

RUBRIC_PATH = os.environ.get('SURG_RUBRIC_PATH', os.path.join(_PROJECT, 'surg_rubric.json'))
NUM_LEVELS = int(os.environ.get('ESR_NUM_LEVELS', '5'))
MIN_SIM = float(os.environ.get('SURG_ORDINAL_MIN_SIM', '0.30'))

# 顺序 = prompt 提问顺序 = label_score [D, B, E, T] 的顺序
DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']

# 评分表把 B 叫 "Bimanual Dexterity"，prompt 和全部 SFT 目标叫 "Bimanual coordination"
_HEADER_RE = re.compile(
    r'(Depth\s+perception|Bimanual\s+(?:coordination|dexterity)|Efficiency|Tissue\s+handling)'
    r'\s*(\([^)]{0,24}\))?\s*[:：]', re.I)

_LEVEL_IN_TEXT_RES = [
    re.compile(r'\b([1-5])\s*/\s*5\b'),
    re.compile(r'\blevel\s*[:：]?\s*([1-5])\b', re.I),
    re.compile(r'\bscore\s*[:：]?\s*([1-5])\b', re.I),
    re.compile(r'\brating\s*[:：]?\s*([1-5])\b', re.I),
]
_LEVEL_IN_PAREN_RE = re.compile(r'([1-5])')


def canon_dim(name: str) -> str:
    n = re.sub(r'\s+', ' ', (name or '').strip().lower())
    if n.startswith('depth'):
        return DIMS[0]
    if n.startswith('bimanual'):
        return DIMS[1]
    if n.startswith('efficiency'):
        return DIMS[2]
    if n.startswith('tissue'):
        return DIMS[3]
    return name


def norm_text(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]+', ' ', re.sub(r'\s+', ' ', (text or '').lower())).strip()


class Rubric:
    """(维度, 等级) -> 标准描述；由 build_surg_rubric.py 从专家评分表生成。"""

    def __init__(self, levels):
        self.levels = levels
        self._norm = {d: {lv: norm_text(t) for lv, t in m.items()} for d, m in levels.items()}

    @classmethod
    def load(cls, path=None):
        path = path or RUBRIC_PATH
        with open(path, encoding='utf-8') as fh:
            blob = json.load(fh)
        levels = {d: {int(lv): txt for lv, txt in m.items()} for d, m in blob['levels'].items()}
        missing = [d for d in DIMS if d not in levels]
        if missing:
            raise ValueError('rubric %s 缺维度: %s' % (path, missing))
        return cls(levels)

    def match_level(self, dim: str, text: str):
        """-> (等级, 相似度)：与 `text` 最接近的 rubric 档位。"""
        t = norm_text(text)
        best, best_sim = None, -1.0
        for lv, ref in self._norm.get(dim, {}).items():
            sim = difflib.SequenceMatcher(None, t, ref).ratio()
            if sim > best_sim:
                best, best_sim = lv, sim
        return best, best_sim


_RUBRIC: Optional[Rubric] = None


def get_rubric() -> Rubric:
    global _RUBRIC
    if _RUBRIC is None:
        _RUBRIC = Rubric.load()
    return _RUBRIC


def split_dimensions(text: str):
    """-> (segments {维度: {'text', 'explicit_level'}}, 首个标题前的前言, 出现顺序)。"""
    text = text or ''
    matches = list(_HEADER_RE.finditer(text))
    segments, order = {}, []
    for i, m in enumerate(matches):
        dim = canon_dim(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        explicit = None
        if m.group(2):
            hit = _LEVEL_IN_PAREN_RE.search(m.group(2))
            if hit:
                explicit = int(hit.group(1))
        if explicit is None:
            for rex in _LEVEL_IN_TEXT_RES:
                hit = rex.search(body)
                if hit:
                    explicit = int(hit.group(1))
                    break
        order.append(dim)
        segments.setdefault(dim, {'text': body, 'explicit_level': explicit})
    preamble = text[:matches[0].start()] if matches else text
    return segments, preamble, order


def decode_levels(text: str, with_similarity: bool = False):
    """文本 -> {维度: 等级 or None}。显式数字优先，否则 rubric 最近邻。"""
    rubric = get_rubric()
    segments, _, _ = split_dimensions(text)
    out, sims = {}, {}
    for dim in DIMS:
        seg = segments.get(dim)
        if seg is None:
            out[dim], sims[dim] = None, None
            continue
        if seg['explicit_level'] is not None:
            out[dim], sims[dim] = seg['explicit_level'], 1.0
            continue
        lv, sim = rubric.match_level(dim, seg['text'])
        out[dim] = lv if sim >= MIN_SIM else None
        sims[dim] = sim
    return (out, sims) if with_similarity else out


def decode_vector(text: str):
    """文本 -> [D, B, E, T]，未解出的位置是 None。"""
    dec = decode_levels(text)
    return [dec[d] for d in DIMS]
