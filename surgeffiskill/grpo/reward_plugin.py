import json
import math
import os
import re
import sys
from typing import List, Optional


try:
    from swift.plugin import ORM, orms
except Exception:
    ORM = object
    orms = None

_HERE = os.path.dirname(os.path.abspath(__file__))


_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


from ordinal_metrics.decode import (DIMS, MIN_SIM, NUM_LEVELS, RUBRIC_PATH,
                                    decode_levels, get_rubric, split_dimensions)

SIGMA = float(os.environ.get('ESR_SOR_SIGMA', '1.0'))
KERNEL = os.environ.get('SURG_ORDINAL_KERNEL', 'gaussian').lower()
FMT_MIN_WORDS = int(os.environ.get('SURG_FORMAT_MIN_WORDS', '4'))
FMT_EGCOT = os.environ.get('SURG_FORMAT_EGCOT', '0') == '1'
BETA = float(os.environ.get('SURG_REWARD_BETA', '0.5'))


CLASS_BALANCED = os.environ.get('SURG_ORDINAL_CLASS_BALANCED', '0') == '1'
CLASS_WEIGHTS_PATH = os.environ.get(
    'SURG_CLASS_WEIGHTS_PATH', os.path.join(_HERE, 'surg_class_weights.json'))
_CLASS_WEIGHTS = None


def _class_weight(dim: str, level: int) -> float:
    global _CLASS_WEIGHTS
    if _CLASS_WEIGHTS is None:
        with open(CLASS_WEIGHTS_PATH, encoding='utf-8') as fh:
            _CLASS_WEIGHTS = json.load(fh)['weights']
    return float(_CLASS_WEIGHTS.get(dim, {}).get(str(level), 1.0))

_EGCOT_STAGE_RES = [
    re.compile(r'\bobserv', re.I),
    re.compile(r'\bmotion\b', re.I),
    re.compile(r'\bcontrol\b', re.I),
    re.compile(r'\bevidence\b', re.I),
    re.compile(r'\blevel\b', re.I),
]


def _ordinal_credit(delta: int) -> float:
    if KERNEL == 'linear':
        return max(0.0, 1.0 - abs(delta) / float(NUM_LEVELS - 1))
    return math.exp(-(delta ** 2) / (2.0 * SIGMA ** 2))


def _as_gt_levels(label_score) -> Optional[List[int]]:
    if label_score is None:
        return None
    v = label_score
    if hasattr(v, 'tolist'):
        v = v.tolist()
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)) or len(v) != len(DIMS):
        return None
    try:
        levels = [int(x) for x in v]
    except (TypeError, ValueError):
        return None
    if any(lv < 1 or lv > NUM_LEVELS for lv in levels):
        return None
    return levels


def ordinal_reward_one(completion: str, gt_levels: Optional[List[int]]) -> Optional[float]:
    if gt_levels is None:
        return None
    pred = decode_levels(completion)
    num = den = 0.0
    for i, dim in enumerate(DIMS):
        p = pred.get(dim)
        credit = 0.0 if p is None else _ordinal_credit(p - gt_levels[i])
        w = _class_weight(dim, gt_levels[i]) if CLASS_BALANCED else 1.0
        num += w * credit
        den += w
    return (num / den) if den else 0.0


def format_reward_one(completion: str) -> float:
    text = completion or ''
    segments, preamble, order = split_dimensions(text)

    if not order:


        return 0.0

    found = [d for d in DIMS if d in segments]
    c_present = len(found) / float(len(DIMS))

    canonical = [d for d in DIMS if d in segments]
    c_order = 1.0 if [d for d in order if d in canonical][:len(canonical)] == canonical else 0.0

    counts = {d: order.count(d) for d in DIMS}
    c_once = sum(1.0 for d in DIMS if counts[d] == 1) / float(len(DIMS))

    c_body = sum(1.0 for d in DIMS
                 if d in segments and len(segments[d]['text'].split()) >= FMT_MIN_WORDS) / float(len(DIMS))

    c_clean = 1.0 if not preamble.strip() else 0.0

    score = (0.40 * c_present + 0.15 * c_order + 0.15 * c_once
             + 0.20 * c_body + 0.10 * c_clean)

    if FMT_EGCOT:


        stage_hits = []
        for d in DIMS:
            body = segments.get(d, {}).get('text', '')
            stage_hits.append(sum(1.0 for rex in _EGCOT_STAGE_RES if rex.search(body))
                              / float(len(_EGCOT_STAGE_RES)))
        score = 0.5 * score + 0.5 * (sum(stage_hits) / len(stage_hits))

    return min(1.0, max(0.0, score))


class _SurgBase(ORM):
    _warned = False

    def _gt_batch(self, n, kwargs):
        label_score = kwargs.get('label_score')
        solution = (kwargs.get('solution') or kwargs.get('answer')
                    or kwargs.get('answers') or kwargs.get('labels'))
        if label_score is None and solution is None and not _SurgBase._warned:
            _SurgBase._warned = True
            print('[surg-reward] neither `label_score` nor `answer` reached the reward '
                  'function; available kwargs: %s' % sorted(kwargs.keys()), flush=True)
        out = []
        for i in range(n):
            gt = None
            if label_score is not None:
                try:
                    gt = _as_gt_levels(label_score[i])
                except (IndexError, TypeError):
                    gt = None
            if gt is None and solution is not None:
                try:
                    ref = solution[i]
                except (IndexError, TypeError):
                    ref = None
                if isinstance(ref, str):
                    dec = decode_levels(ref)
                    if all(dec[d] is not None for d in DIMS):
                        gt = [dec[d] for d in DIMS]
            out.append(gt)
        return out


class SurgOrdinalReward(_SurgBase):

    def __call__(self, completions, **kwargs) -> List[Optional[float]]:
        gts = self._gt_batch(len(completions), kwargs)
        return [ordinal_reward_one(c, gt) for c, gt in zip(completions, gts)]


class SurgFormatReward(_SurgBase):

    def __call__(self, completions, **kwargs) -> List[float]:
        return [format_reward_one(c) for c in completions]


def exact_reward_one(completion: str, gt_levels: Optional[List[int]]) -> Optional[float]:
    if gt_levels is None:
        return None
    pred = decode_levels(completion)
    num = den = 0.0
    for i, dim in enumerate(DIMS):
        p = pred.get(dim)
        hit = 1.0 if (p is not None and p == gt_levels[i]) else 0.0
        w = _class_weight(dim, gt_levels[i]) if CLASS_BALANCED else 1.0
        num += w * hit
        den += w
    return (num / den) if den else 0.0


class SurgExactReward(_SurgBase):

    def __call__(self, completions, **kwargs) -> List[Optional[float]]:
        gts = self._gt_batch(len(completions), kwargs)
        return [exact_reward_one(c, gt) for c, gt in zip(completions, gts)]


class SurgEq9Reward(_SurgBase):

    def __call__(self, completions, **kwargs) -> List[Optional[float]]:
        gts = self._gt_batch(len(completions), kwargs)
        out = []
        for c, gt in zip(completions, gts):
            r_ord = ordinal_reward_one(c, gt)
            out.append(None if r_ord is None else r_ord + BETA * format_reward_one(c))
        return out


if orms is not None:
    orms['surg_ordinal'] = SurgOrdinalReward
    orms['surg_format'] = SurgFormatReward
    orms['surg_exact'] = SurgExactReward
    orms['surg_eq9'] = SurgEq9Reward
