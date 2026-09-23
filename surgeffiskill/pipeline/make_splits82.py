import argparse
import hashlib
import io
import json
import os
from collections import Counter


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def dump(o, p):
    with io.open(p, 'w', encoding='utf-8') as fh:
        json.dump(o, fh, ensure_ascii=False, indent=1)
        fh.write(u'\n')


def dist(recs):
    return [dict(sorted(Counter(r['label_score'][d] for r in recs).items())) for d in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='train_effi_cot_ce483.json')
    ap.add_argument('--val', default='train_effi_cot_val_ce483.json')
    ap.add_argument('--outdir', default='revision/scripts/sft_grpo/splits82')
    a = ap.parse_args()

    train, val = load(a.train), load(a.val)
    os.makedirs(a.outdir, exist_ok=True)

    def names(rs):
        return set(os.path.basename(r['videos'][0]) for r in rs)
    ntr, nva = names(train), names(val)
    assert not (ntr & nva), 'train/val overlap: %s' % (ntr & nva)

    dump(train, os.path.join(a.outdir, 'train.json'))
    dump(val, os.path.join(a.outdir, 'val.json'))


    for name, recs in [('train', train), ('val', val)]:
        p = os.path.join(a.outdir, '%s_grpo.jsonl' % name)
        with io.open(p, 'w', encoding='utf-8') as fh:
            for r in recs:
                u = [m for m in r['messages'] if m['role'] == 'user']
                s = [m for m in r['messages'] if m['role'] == 'assistant']
                fh.write(json.dumps({'messages': u, 'answer': s[0]['content'],
                                     'videos': r['videos'],
                                     'label_score': [int(x) for x in r['label_score']]},
                                    ensure_ascii=False) + u'\n')

    manifest = {
        'split': '8:2 (paper secIV-A)',
        'source': {'train': a.train, 'val': a.val},
        'sizes': {'train': len(train), 'val': len(val),
                  'ratio': round(len(train) / float(len(train) + len(val)), 3)},
        'level_distribution': {'train': dist(train), 'val': dist(val)},
        'video_lists': {'train': sorted(ntr), 'val': sorted(nva)},
        'sha256': {},
    }
    for n in ['train', 'val']:
        p = os.path.join(a.outdir, '%s.json' % n)
        manifest['sha256'][n] = hashlib.sha256(io.open(p, 'rb').read()).hexdigest()
    dump(manifest, os.path.join(a.outdir, 'manifest.json'))

    print('train=%d val=%d  ratio=%.2f  (no overlap)'
          % (len(train), len(val), len(train) / float(len(train) + len(val))))
    for n, r in [('train', train), ('val', val)]:
        print('  %-5s per-dim level distribution: %s' % (n, dist(r)))
    print('wrote %s/{train,val,train_grpo,val_grpo,manifest}' % a.outdir)


if __name__ == '__main__':
    main()
