import argparse, io, json, os, sys, time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn as nn
import torch.nn.functional as F


_SOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sor')
if _SOR_DIR not in sys.path:
    sys.path.insert(0, _SOR_DIR)
from sor import sor_loss

PROJECT = '/hpc2hdd/home/ce483/hkust_surgery_video_learning'
EXP = os.path.join(PROJECT, 'revision/scripts/sft_grpo')
SPL = os.path.join(EXP, 'splits82')
DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']
K = 4
L = 5
THR = np.array([1.5, 2.5, 3.5, 4.5])


def lvl(y):
    return np.clip(np.searchsorted(THR, np.asarray(y)) + 1, 1, 5).astype(int)


def qwk(p, y, kmin=1, kmax=5):
    n = kmax - kmin + 1
    O = np.zeros((n, n))
    for a, b in zip(np.asarray(y).astype(int), np.asarray(p).astype(int)):
        O[a - kmin, b - kmin] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], float) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    de = float((W * E).sum())
    return 1 - float((W * O).sum()) / de if de > 0 else 0.0


class Expert(nn.Module):

    def __init__(self, d_share, d_bot, p_drop, d_out=L):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p_drop), nn.Linear(d_share, d_bot),
                                 nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_bot, d_out))

    def forward(self, u):
        return self.net(u)


class MMoEHead(nn.Module):

    def __init__(self, d_in, M=4, d_share=64, d_bot=32, d_route=32, p_drop=0.3, tau=1.0):
        super().__init__()
        self.M, self.tau = M, tau
        self.bottom = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(p_drop),
                                    nn.Linear(d_in, d_share), nn.GELU())
        self.experts = nn.ModuleList([Expert(d_share, d_bot, p_drop) for _ in range(M)])

        self.route = nn.Sequential(nn.Linear(d_share, d_route), nn.Tanh())

        self.gate_w = nn.Parameter(torch.zeros(K, d_route, M))
        self.gate_bias = nn.Parameter(torch.zeros(K, M))
        self.register_buffer('levels', torch.arange(1, L + 1, dtype=torch.float32))

    def forward(self, h):
        u = self.bottom(h)
        E = torch.stack([e(u) for e in self.experts], 1)
        r = self.route(u)
        g_logit = torch.einsum('bd,kdm->bkm', r, self.gate_w) + self.gate_bias
        g = F.softmax(g_logit / self.tau, -1)
        z = torch.einsum('bkm,bml->bkl', g, E)
        p = F.softmax(z, -1)
        yhat = (p * self.levels.view(1, 1, -1)).sum(-1)
        return yhat, z, g, E

    def reg(self, g, E, lam_ent, lam_div):
        loss = torch.zeros((), device=g.device)
        if lam_ent > 0:
            ent = -(g.clamp_min(1e-8) * g.clamp_min(1e-8).log()).sum(-1).mean()
            loss = loss - lam_ent * ent
        if lam_div > 0 and self.M > 1:
            Ef = F.normalize(E.reshape(E.shape[0], self.M, -1), dim=-1)
            sim = torch.einsum('bmd,bnd->bmn', Ef, Ef)
            off = sim - torch.eye(self.M, device=E.device).unsqueeze(0)
            loss = loss + lam_div * off.pow(2).mean()
        return loss

def ordinal_loss(z, y):
    B, Kd, Ld = z.shape
    tgt = torch.zeros(B, Kd, Ld, device=z.device)
    yi = (y.long() - 1).clamp(0, Ld - 1)
    tgt.scatter_(2, yi.unsqueeze(-1), 0.8)
    for off, w in ((-1, 0.1), (1, 0.1)):
        idx = (yi + off).clamp(0, Ld - 1)
        tgt.scatter_add_(2, idx.unsqueeze(-1), torch.full_like(tgt[..., :1], w))
    tgt = tgt / tgt.sum(-1, keepdim=True)
    ce = -(tgt * F.log_softmax(z, -1)).sum(-1).mean()
    p = F.softmax(z, -1)
    lv = torch.arange(1, Ld + 1, device=z.device, dtype=torch.float32)
    yhat = (p * lv.view(1, 1, -1)).sum(-1)
    return ce + F.huber_loss(yhat, y.float())


def build_backbone():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    mp = os.path.join(PROJECT, 'Qwen2.5-VL-7B-Instruct')
    proc = AutoProcessor.from_pretrained(mp)
    attn = os.environ.get('MH_ATTN', '')
    if not attn:
        try:
            import flash_attn
            attn = 'flash_attention_2'
        except Exception:
            attn = 'sdpa'
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation=attn)
    vlm = get_peft_model(vlm, LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    vlm.gradient_checkpointing_enable()
    if hasattr(vlm, 'enable_input_require_grads'):
        vlm.enable_input_require_grads()
    vlm.config.use_cache = False
    core = vlm.base_model.model
    vis = core.visual
    for q in vis.parameters():
        q.requires_grad_(False)
    lm = core.model.language_model if hasattr(core.model, 'language_model') else core.model
    norm_mod = lm.norm
    merge = int(getattr(vis, 'spatial_merge_size', 2))
    vid_id = proc.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    d_llm = int(core.config.hidden_size if hasattr(core.config, 'hidden_size')
                else core.config.text_config.hidden_size)
    print('  base=Qwen2.5-VL-7B | LoRA r8 a32 q,k,v,o | vision frozen | attn=%s | hook=%s'
          % (attn, type(norm_mod).__name__), flush=True)
    return proc, vlm, norm_mod, merge, vid_id, d_llm


class EvidenceHead(nn.Module):

    def __init__(self, mu, sd, V, n_lvl, n_ctr, tau=1.0, p_drop=0.0, n_exp=0,
                 scalar=2.0):
        super().__init__()

        self.nl, self.nc, self.tau = n_lvl, n_ctr, tau
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        self.register_buffer('V', V)
        r = V.shape[-1]
        self.drop = nn.Dropout(p_drop)
        self.lvl_ex = nn.ModuleList([nn.Linear(r, L) for _ in range(n_lvl)])
        self.lvl_gate = nn.Parameter(torch.zeros(n_lvl))


        self.n_exp = n_exp if n_exp else n_ctr
        self.ex_src = [j % n_ctr for j in range(self.n_exp)]
        self.ctr_ex = nn.ModuleList([nn.Linear(r, L) for _ in range(self.n_exp)])
        self.ctr_gate = nn.Parameter(self._contrast_basis(K, self.n_exp))
        self.scalar = float(scalar)
        assert self.scalar > 0, 'EvidenceHead keeps only the scalar ordinal-decoding path; beta must be > 0'
        self.head_lvl = nn.ModuleList([nn.Linear(r, 1) for _ in range(n_lvl)])
        self.head_ctr = nn.ModuleList([nn.Linear(r, 1) for _ in range(self.n_exp)])
        self.b_scalar = nn.Parameter(torch.zeros(K))
        self.register_buffer('levels', torch.arange(1, L + 1, dtype=torch.float32))
        self._acc = []

    @staticmethod
    def _contrast_basis(k, n):
        g = torch.Generator().manual_seed(0)
        A = torch.randn(k, n, generator=g)
        A = A - A.mean(0, keepdim=True)
        r = min(n, k - 1)
        Q, _ = torch.linalg.qr(A[:, :r])
        out = torch.zeros(k, n)
        out[:, :r] = Q
        if n > r:
            B = A[:, r:]
            out[:, r:] = B / B.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return out

    def project(self, v):
        z = torch.einsum('bmd,mdr->bmr', (v - self.mu) / self.sd, self.V)
        return z.reshape(z.shape[0], -1)

    def forward(self, v):
        z = self.drop(torch.einsum('bmd,mdr->bmr', (v - self.mu) / self.sd, self.V))
        gl = F.softmax(self.lvl_gate / self.tau, -1)
        zc_in = z[:, self.nl:]
        zc_all = torch.stack([self.ctr_ex[j](zc_in[:, self.ex_src[j]])
                              for j in range(self.n_exp)], 1)
        gc = self.ctr_gate
        ck = torch.einsum('kj,bjl->bkl', gc, zc_all)
        ck = ck - ck.mean(1, keepdim=True)
        sl_all = torch.stack([self.head_lvl[m](z[:, m])[:, 0]
                              for m in range(self.nl)], 1)
        s_lvl = torch.einsum('m,bm->b', gl, sl_all)
        gcs = gc.unsqueeze(0).expand(v.shape[0], -1, -1)
        sc_all = torch.stack([self.head_ctr[j](zc_in[:, self.ex_src[j]])[:, 0]
                              for j in range(self.n_exp)], 1)
        s_ctr = torch.einsum('bkj,bj->bk', gcs, sc_all)
        s_ctr = s_ctr - s_ctr.mean(1, keepdim=True)
        s_k = s_lvl.unsqueeze(1) + self.b_scalar.unsqueeze(0) + s_ctr
        zk = -self.scalar * (self.levels.view(1, 1, -1) - s_k.unsqueeze(-1)) ** 2
        yhat = (F.softmax(zk, -1) * self.levels.view(1, 1, -1)).sum(-1)
        self._observe(gc, ck)
        return yhat, zk, gcs, zc_all

    @torch.no_grad()
    def _observe(self, gc, ck):
        g = gc.float()
        tv = float((g[:, None] - g[None, :]).abs().sum(-1).sum() / 2 / (K * K - K))
        self._acc.append((tv, float(ck.abs().mean()), float(ck.sum(1).abs().max())))

    def pop_stats(self):
        if not self._acc:
            return ''
        a = np.asarray(self._acc, dtype=np.float64); self._acc = []
        return (' | ctr-gate TV %.3f ctr-mag %.4f zero-sum resid %.1e'
                % (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].max()))

    def reg(self, E, lam_div):
        loss = torch.zeros((), device=E.device)
        if lam_div > 0 and E.shape[1] > 1:
            M_ = E.shape[1]
            Ef = F.normalize(E, dim=-1)
            sim = torch.einsum('bml,bnl->bmn', Ef, Ef)
            off = sim - torch.eye(M_, device=E.device).view(1, M_, M_)
            loss = loss + lam_div * off.pow(2).mean()
        return loss


class ESR(nn.Module):

    def __init__(self, d, d_ev, n_exp=5, r=8, tied=False, priv=True, radius=1.0):
        super().__init__()
        assert n_exp >= K + 1, ('ESR requires n_exp >= K+1 (=%d): the zero-sum hyperplane has dim n_exp-1 '
                                'and must be >= K to hold K pairwise-orthogonal gate rows; got n_exp=%d'
                                % (K + 1, n_exp))
        self.n_exp, self.r, self.tied, self.priv = n_exp, r, bool(tied), bool(priv)
        self.radius = float(radius)
        g_ = torch.Generator().manual_seed(0)
        self.B = nn.Parameter(torch.randn(n_exp, r, d, generator=g_) / (d ** 0.5))
        self.A = nn.Parameter(torch.zeros(n_exp, L, r))
        self.s = nn.Parameter(torch.ones(K))


        M = torch.randn(n_exp, n_exp, generator=g_)
        M = M - M.mean(0, keepdim=True)
        Q, _ = torch.linalg.qr(M)
        G0 = Q[:, :K].t().contiguous()
        if tied:
            G0 = G0[0:1].expand_as(G0).clone()
        self.register_buffer('G0', G0)
        self.Wr = nn.Parameter(torch.zeros(K, n_exp, d_ev))
        if tied:
            self.Wr.requires_grad_(False)
        if self.priv:
            self.Ap = nn.Parameter(torch.zeros(K, L, r))
            self.gp = nn.Parameter(torch.ones(K))
        self.priv_off = False
        self._acc = []

    def gate(self, ev):
        g = self.G0 + torch.einsum('kje,e->kj', self.Wr, ev.to(self.Wr.dtype))
        g = g - g.mean(-1, keepdim=True)
        g = self.radius * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return g

    def forward(self, hk, ev, priv_off=False):
        u = torch.einsum('jrd,kd->kjr', self.B, hk.to(self.B.dtype))
        e = torch.einsum('jlr,kjr->kjl', self.A, u)
        g = self.gate(ev)
        out = self.s[:, None] * torch.einsum('kj,kjl->kl', g, e)
        if self.priv and not (priv_off or self.priv_off):
            ub = u.mean(1)
            out = out + self.gp[:, None] * torch.einsum('klr,kr->kl', self.Ap, ub)
        gg = g.detach().float()
        self._acc.append((float((gg[:, None] - gg[None, :]).abs().sum(-1).sum()
                                / 2 / (K * K - K)),
                          float(gg.sum(-1).abs().max()),
                          float(gg.norm(dim=-1).min()),
                          float(gg.norm(dim=-1).max()),
                          float(gg.abs().sum(0).min()),
                          float(gg.abs().max()),
                          float(out.detach().abs().mean())))
        return out, g

    def invariants(self):
        if not self._acc:
            return None
        a = np.asarray(self._acc)
        return dict(gate_tv=float(a[:, 0].mean()),
                    zero_sum_absmax=float(a[:, 1].max()),
                    row_norm_min=float(a[:, 2].min()),
                    row_norm_max=float(a[:, 3].max()),
                    coldest_expert_mass=float(a[:, 4].min()),
                    gate_absmax=float(a[:, 5].max()),
                    gate_absmax_bound=float(self.radius
                                            * ((self.n_exp - 1.0) / self.n_exp) ** 0.5),
                    out_absmean=float(a[:, 6].mean()),
                    n_calls=int(a.shape[0]))

    def pop_stats(self):
        d = self.invariants()
        self._acc = []
        if d is None:
            return ''
        return ('| ESR gate-TV %.4f zero-sum %.2e row-norm[%.4f,%.4f] coldest %.4f |corr| %.4f'
                % (d['gate_tv'], d['zero_sum_absmax'], d['row_norm_min'],
                   d['row_norm_max'], d['coldest_expert_mass'], d['out_absmean']))


class JointModel(nn.Module):
    def __init__(self, vlm, norm_mod, vid_id, merge, d_llm, mm_kw, grid=8):
        super().__init__()
        self.vlm, self.vid_id, self.merge, self.grid = vlm, vid_id, merge, grid
        self.mmoe = MMoEHead(3 * d_llm, **mm_kw)
        self._hs = None
        norm_mod.register_forward_hook(self._grab)

    def _grab(self, m, i, o):
        self._hs = o[0] if isinstance(o, tuple) else o

    def hook_layers(self, lm, idxs, vis_mod, grad=False):
        self._ls = {}
        self._ls_grad = bool(grad)
        def mk(tag):
            def _h(m, i, o):
                _t = (o[0] if isinstance(o, tuple) else o)
                self._ls[tag] = _t.float() if grad else _t.detach().float()
                return o
            return _h
        for i in idxs:
            lm.layers[i].register_forward_hook(mk('L%d' % i))
        def _hv(m, i, o):
            _t = (o[0] if isinstance(o, tuple) else o)
            self._ls['vit'] = _t.float() if grad else _t.detach().float()
            return o
        vis_mod.register_forward_hook(_hv)

    def mlpool(self, input_ids, grid_thw, srcs):
        pos = (input_ids[0] == self.vid_id).nonzero(as_tuple=True)[0]
        t, hh, ww = [int(x) for x in grid_thw[0].tolist()]
        gh, gw = hh // self.merge, ww // self.merge
        out = []
        for sname in srcs:


            sname = sname.split('@')[0]
            if sname == 'norm':
                H = self._hs[0][pos].float()
            elif sname == 'vit':
                H = self._ls['vit']
            else:
                H = self._ls[sname][0][pos]
            fm = H.reshape(t, gh * gw, -1).mean(1)
            sd_ = fm.std(0) if t > 1 else fm.mean(0) * 0
            out.append(torch.cat([fm.mean(0), sd_, fm.max(0).values - fm.mean(0)]))
        return torch.stack(out)


def load_records():
    def rd(fn):
        return json.load(io.open(os.path.join(SPL, fn), encoding='utf-8'))
    pd_ = {}
    for r in rd('train_pd.json') + rd('val_pd.json'):
        v = os.path.basename(r['videos'][0])
        pd_.setdefault(v, {})[int(r['dim_index'])] = r
    out = {}


    _JSET = (('train_joint.json', 'val_joint.json') if os.environ.get('MH_JOINT_RAW')
             else ('train_jointr.json', 'val_jointr.json'))
    jm_ = {}
    for _jf in _JSET:
        for _r in rd(_jf):
            jm_[os.path.basename(_r['videos'][0])] = _r
    for split, fn in (('train', 'train.json'), ('val', 'val.json')):
        rows = []
        for r in rd(fn):
            v = os.path.basename(r['videos'][0])
            assert v in jm_, 'missing joint sample'
            rows.append(dict(name=v, path=r['videos'][0],
                             y=np.array([int(x) for x in r['label_score']], np.float32),
                             msgs=pd_.get(v, {}), jmsg=jm_[v]))
        out[split] = rows
    return out


def main():
    ap = argparse.ArgumentParser(
        description='ESR: spherical zero-sum gate + shared read/write pool + per-skill private write head; '
                    'corrects the LM logits at the four skill-level digit positions. Defaults match the run config.')

    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--warmup', type=int, default=2, help='first N epochs keep LoRA lr=0 (train the head only)')
    ap.add_argument('--lr-lora', dest='lr_lora', type=float, default=3e-5)
    ap.add_argument('--lr-head', dest='lr_head', type=float, default=3e-2)
    ap.add_argument('--accum', type=int, default=8, help='head gradient-accumulation steps')
    ap.add_argument('--accum-lora', dest='accum_lora', type=int, default=2,
                    help='LoRA gradient-accumulation steps (defaults to --accum)')
    ap.add_argument('--drop', type=float, default=0.0)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--lam-text', dest='lam_text', type=float, default=1.0)
    ap.add_argument('--lam-ord', dest='lam_ord', type=float, default=0.0,
                    help='weight of the ordinal loss on the auxiliary numeric head (not in the decode path)')
    ap.add_argument('--lam-div', dest='lam_div', type=float, default=0.05,
                    help='expert-diversity regulariser; no gate-entropy / load-balancing term (see the ESR class doc)')
    ap.add_argument('--fold', type=int, default=-1,
                    help='>=0 uses the f-th 79/19 train-only split; -1 uses all 98 and evaluates val once')
    ap.add_argument('--eval-last', dest='eval_last', action='store_true',
                    help='evaluate only on the last epoch (epoch pre-selected by train-only CV)')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--lm-basis', dest='lm_basis',
                    default=os.path.join(PROJECT, 'revision/evidence/results/sft_grpo',
                                         'basis_layers.npz'),
                    help='per-layer frozen whitening basis .npz (built unsupervised from train features)')
    ap.add_argument('--dc-lvl', dest='dc_lvl', default='L14,L21,L25',
                    help='sources for the level group')
    ap.add_argument('--dc-ctr', dest='dc_ctr', default='vit,L7',
                    help='sources for the profile group')
    ap.add_argument('--dc-nctr', dest='dc_nctr', type=int, default=3,
                    help='number of profile experts (0 = number of sources)')
    ap.add_argument('--dc-rank', dest='dc_rank', type=int, default=8,
                    help='truncate the frozen basis to the first r components (0 = all)')
    ap.add_argument('--dc-scalar', dest='dc_scalar', type=float, default=2.0,
                    help='beta of the scalar ordinal decoder: z_k[l] = -beta*(l - s_k)^2')

    ap.add_argument('--esr-exp', dest='esr_exp', type=int, default=5,
                    help='number of experts J. The zero-sum hyperplane has dim J-1 and must be >= K, so J >= K+1 = 5; '
                         'a smaller value is bumped up automatically (structural, not a hyperparameter)')
    ap.add_argument('--esr-rank', dest='esr_rank', type=int, default=8, help='expert rank r')
    ap.add_argument('--esr-radius', dest='esr_radius', type=float, default=1.0,
                    help='fixed gate radius c (the gate sets only the mix; scale is carried by A and s_k)')
    ap.add_argument('--lr-esr', dest='lr_esr', type=float, default=1e-3)
    ap.add_argument('--esr-ord', dest='esr_ord', type=float, default=2.0,
                    help='weight of the ordinal loss on the ESR-corrected digit logits'
                         ' (numeric supervision stays on the generation path)')
    ap.add_argument('--esr-tied', dest='esr_tied', action='store_true',
                    help='control: all four skills share one gate row (G0 broadcast from row 0, W_r frozen)')
    ap.add_argument('--esr-nopriv', dest='esr_nopriv', action='store_true',
                    help='control: no private write head (shared pool only)')
    ap.add_argument('--no-sor', dest='sor', action='store_false', default=True,
                    help='by default ESR uses SOR (surgeffiskill/sor/sor.py) for ordinal supervision; --no-sor falls back to the built-in ordinal_loss (soft-label CE + Huber)')
    ap.add_argument('--sor-sigma', dest='sor_sigma', type=float, default=1.0,
                    help='SOR discrete-Gaussian width sigma')
    ap.add_argument('--sor-lambda', dest='sor_lambda', type=float, default=0.25,
                    help='SOR KL-term weight lambda (value selected in the ablation)')

    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--ckpt-dir', dest='ckpt_dir', default=None,
                    help='save LoRA + head + optimiser each epoch; default <outdir>/ckpt')
    ap.add_argument('--resume', action='store_true',
                    help='resume from the next epoch if a matching checkpoint exists')
    ap.add_argument('--no-ckpt', dest='no_ckpt', action='store_true', help='disable checkpoint writing')
    ap.add_argument('--max-minutes', dest='max_minutes', type=float, default=0,
                    help='after this many minutes, checkpoint mid-epoch and exit cleanly; 0 = no limit')
    ap.add_argument('--tag', default='esr')
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()
    if a.sor:
        print('  [SOR] ESR ordinal supervision = sor_loss (default; sigma=%.2f, lambda=%.2f)'
              % (a.sor_sigma, a.sor_lambda), flush=True)
    else:
        print('  [SOR] --no-sor: ESR ordinal supervision = built-in ordinal_loss (soft-label CE + Huber)',
              flush=True)
    if a.esr_exp < K + 1:
        print('  [ESR] n_exp %d < K+1; bumped to %d (the zero-sum hyperplane dim n_exp-1 '
              'must be >= K to hold K pairwise-orthogonal gate rows)' % (a.esr_exp, K + 1), flush=True)
        a.esr_exp = K + 1
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random as _random
    _random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _strict = os.environ.get('MH_DET_STRICT', '0') == '1'
    try:
        torch.use_deterministic_algorithms(True, warn_only=not _strict)
        _det = 'on(strict)' if _strict else 'on(warn_only)'
    except Exception as _e:
        _det = 'unavailable: %s' % _e
    print('  deterministic: %s | CUBLAS_WORKSPACE_CONFIG=%s'
          % (_det, os.environ.get('CUBLAS_WORKSPACE_CONFIG')), flush=True)

    print('=== ESR: Qwen2.5-VL-7B + LoRA joint training; ESR enters '
          'forward/backward from step 1 and corrects the decoded digit logits ===', flush=True)
    ds = load_records()
    tr_all, va = ds['train'], ds['val']
    if a.fold >= 0:
        rs = np.random.RandomState(1000 + a.fold)
        idx = rs.permutation(len(tr_all)); nin = max(1, len(tr_all) // 5)
        ev = [tr_all[i] for i in idx[:nin]]; tr = [tr_all[i] for i in idx[nin:]]
        tag2 = 'fold%d' % a.fold
    else:
        tr, ev, tag2 = tr_all, va, 'final'
    if a.smoke:


        tr, ev = tr_all[:3], tr_all[3:6]
        tag2 = 'smoke'
    print('  %s: train %d / eval %d' % (tag2, len(tr), len(ev)), flush=True)

    proc, vlm, norm_mod, merge, vid_id, d_llm = build_backbone()
    _dig = [proc.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, L + 1)]
    assert all(len(x) == 1 for x in _dig), 'level digits are not single tokens: %s' % _dig
    DIG = torch.tensor([x[0] for x in _dig], device='cuda:0')

    model = JointModel(vlm, norm_mod, vid_id, merge, d_llm,
                       dict(M=4, d_share=64, d_bot=32, d_route=32,
                            p_drop=a.drop, tau=a.tau))


    _lb = np.load(a.lm_basis, allow_pickle=True)
    _all = [str(x) for x in _lb['srcs']]
    _lvl = [x.strip() for x in a.dc_lvl.split(',') if x.strip()]
    _ctr = [x.strip() for x in a.dc_ctr.split(',') if x.strip()]
    for _x in _lvl + _ctr:
        assert _x in _all, 'source %s not in the basis file (available: %s)' % (_x, _all)
    _sel = _lvl + _ctr
    _ix = [_all.index(x) for x in _sel]
    model._lm_srcs = _sel
    _idx = sorted({int(x.split('@')[0][1:]) for x in _sel if x.startswith('L')})
    _lmmod = (vlm.base_model.model.model.language_model
              if hasattr(vlm.base_model.model.model, 'language_model')
              else vlm.base_model.model.model)

    model.hook_layers(_lmmod, _idx, vlm.base_model.model.visual, grad=True)
    _Vd = _lb['V'][_ix]
    if a.dc_rank > 0:
        _Vd = _Vd[:, :, :a.dc_rank]
    model.mmoe = EvidenceHead(torch.from_numpy(_lb['mu'][_ix].copy()),
                              torch.from_numpy(_lb['sd'][_ix].copy()),
                              torch.from_numpy(_Vd.copy()),
                              n_lvl=len(_lvl), n_ctr=len(_ctr),
                              tau=a.tau, p_drop=a.drop, n_exp=a.dc_nctr,
                              scalar=a.dc_scalar)
    model.mmoe = model.mmoe.float().cuda()
    print('  evidence head: level %s || profile %s x %d experts; rank %d; trainable params %d'
          % (_lvl, _ctr, model.mmoe.n_exp, _Vd.shape[-1],
             sum(p_.numel() for p_ in model.mmoe.parameters() if p_.requires_grad)),
          flush=True)


    _dev = int(model.mmoe.V.shape[0] * model.mmoe.V.shape[-1])
    model.esr = ESR(d_llm, _dev, n_exp=a.esr_exp, r=a.esr_rank,
                    tied=a.esr_tied, priv=(not a.esr_nopriv),
                    radius=a.esr_radius).float().cuda()
    model._esr_off = False
    _sk = ('Wr', 'Ap', 'gp', 's')
    _nsk = sum(q_.numel() for n_, q_ in model.esr.named_parameters() if n_ in _sk)
    _nall = sum(q_.numel() for q_ in model.esr.parameters())
    print('  [ESR] experts %d (>= K+1=%d) x rank %d | evidence %d-d | radius %.2f | params %d '
          '= shared %d + skill-specific %d (%.0f per skill, %.1f%%) | A/Ap zero-init, '
          's_k=1 => step 0 equals SFT%s%s'
          % (a.esr_exp, K + 1, a.esr_rank, _dev, a.esr_radius, _nall,
             _nall - _nsk, _nsk, _nsk / float(K), 100.0 * _nsk / max(_nall, 1),
             ' | tied-gate control' if a.esr_tied else '',
             ' | no-private control' if a.esr_nopriv else ''), flush=True)
    nlora = sum(q.numel() for n_, q in vlm.named_parameters()
                if q.requires_grad and 'mmoe' not in n_)
    print('  trainable: LoRA %.2fM | evidence head %.2fM | ESR %.3fM'
          % (nlora / 1e6, sum(p.numel() for p in model.mmoe.parameters()) / 1e6,
             _nall / 1e6), flush=True)

    vcache = {}

    def fwd(rec, dim_idx, want_lm):
        model._esr_out = None; model._esr_zlm = None
        from qwen_vl_utils import fetch_video
        r = (rec['jmsg'] if dim_idx < 0 else rec['msgs'][dim_idx]); path = r['videos'][0]
        if path not in vcache:
            v = fetch_video({'type': 'video', 'video': path})
            vcache[path] = v[0] if isinstance(v, tuple) else v
        msgs = []
        for m_ in r['messages']:
            c = m_['content']
            msgs.append({'role': m_['role'],
                         'content': ([{'type': 'video'},
                                      {'type': 'text', 'text': c.replace('<video>', '', 1)}]
                                     if isinstance(c, str) and '<video>' in c
                                     else [{'type': 'text', 'text': c}])})
        if want_lm:
            full = proc.apply_chat_template(msgs, tokenize=False)
        else:


            full = (proc.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
                    + ('' if dim_idx < 0 else DIMS[dim_idx] + ' ('))
        inp = proc(text=[full], videos=[vcache[path]], return_tensors='pt').to('cuda:0')
        lt = None
        if want_lm:
            pr = proc.apply_chat_template(msgs[:-1], tokenize=False,
                                          add_generation_prompt=True)
            n_pr = int(proc(text=[pr], videos=[vcache[path]],
                            return_tensors='pt')['input_ids'].shape[1])
            Ln = int(inp['input_ids'].shape[1]); nk = max(2, Ln - n_pr + 1)
            out = vlm(**inp, use_cache=False, logits_to_keep=nk)
            lg = out.logits[:, :-1]
            tgt = inp['input_ids'][:, Ln - (nk - 1):]
        else:
            n_pr = None
            vlm(**inp, use_cache=False, logits_to_keep=1)

        h = model.mlpool(inp['input_ids'], inp['video_grid_thw'], model._lm_srcs)
        model._tcount = int(inp['video_grid_thw'][0][0])
        if want_lm:

            ans = inp['input_ids'][0][n_pr:]
            hit = (ans[:, None] == DIG[None, :]).any(-1).nonzero()
            assert len(hit) == K, ('answer has %d level-digit tokens, expected %d' % (len(hit), K))
            _off = Ln - (nk - 1)
            _row = (n_pr + hit[:, 0]) - _off
            assert int(_row.min()) >= 0 and int(_row.max()) < lg.shape[1], \
                'digit position falls outside the logits_to_keep window'


            _hrow = (n_pr + hit[:, 0] - 1)
            _hk = model._hs[0][_hrow].float()
            _ev = model.mmoe.project(h[None])[0]
            _dro, _gro = model.esr(_hk, _ev)
            model._esr_out = (_dro, _gro)
            if model._esr_off:
                _dro = _dro * 0.0
            _lg0 = lg[0].float(); _tg0 = tgt[0]
            _ceA = F.cross_entropy(_lg0, _tg0, reduction='none')
            _rows = _lg0[_row]
            _add = torch.zeros_like(_rows)
            _add[torch.arange(K, device=_rows.device)[:, None], DIG[None, :]] = _dro
            _rw = _rows + _add
            _ceK = F.cross_entropy(_rw, _tg0[_row], reduction='none')


            lt = (_ceA.sum() - _ceA[_row].sum() + _ceK.sum()) / _ceA.shape[0]

            model._esr_zlm = _rw[torch.arange(K, device=_rw.device)[:, None],
                                 DIG[None, :]].float()
        return lt, h

    _RB = [None]

    def build_rubric(txt):
        rb, cur = {}, None
        for ln in txt.split('\n'):
            s = ln.strip()
            if s in DIMS:
                cur = DIMS.index(s); rb[cur] = {}
            elif cur is not None and len(s) > 2 and s[0].isdigit() and s[1] == ':':
                rb[cur][int(s[0])] = s[2:].strip()
        assert len(rb) == K and all(len(rb[i]) == L for i in rb), \
            'failed to parse level descriptions (got %s, expected %d dims x %d levels)' \
            % ({i: len(rb[i]) for i in rb}, K, L)
        return rb

    def gen_digits(rec):
        from qwen_vl_utils import fetch_video
        r = rec['jmsg']; path = r['videos'][0]
        if path not in vcache:
            v = fetch_video({'type': 'video', 'video': path})
            vcache[path] = v[0] if isinstance(v, tuple) else v
        msgs, ptxt = [], ''
        for m_ in r['messages']:
            c = m_['content']
            if isinstance(c, str) and '<video>' in c:
                ptxt = c
                msgs.append({'role': m_['role'],
                             'content': [{'type': 'video'},
                                         {'type': 'text', 'text': c.replace('<video>', '', 1)}]})
            else:
                msgs.append({'role': m_['role'], 'content': [{'type': 'text', 'text': c}]})
        if _RB[0] is None:
            _RB[0] = build_rubric(ptxt)
        base = proc.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        pref, ds, es = '', [], []
        for k_ in range(K):
            inp = proc(text=[base + pref + DIMS[k_] + ' ('], videos=[vcache[path]],
                       return_tensors='pt').to('cuda:0')
            o_ = vlm(**inp, use_cache=False, logits_to_keep=1)
            _lgd = o_.logits[0, -1][DIG].float()
            if not model._esr_off:


                _hg = model.mlpool(inp['input_ids'], inp['video_grid_thw'], model._lm_srcs)
                _ev = model.mmoe.project(_hg[None])[0]
                _hk = model._hs[0][-1:].float()
                _d4, _ = model.esr(_hk.expand(K, -1), _ev)
                _lgd = _lgd + _d4[k_]
            q_ = F.softmax(_lgd, -1)
            d_ = int(q_.argmax()) + 1
            ds.append(d_)
            es.append(float((q_ * torch.arange(1, L + 1, device=q_.device,
                                               dtype=q_.dtype)).sum()))
            pref += '%s (%d): %s\n' % (DIMS[k_], d_, _RB[0][k_][d_])
            model._hs = None
        return ds, es

    def evaluate(rows):
        model.eval(); vlm.eval()
        S, G, TC = [], [], []
        with torch.no_grad():
            for r in rows:
                _, h = fwd(r, 0, False)
                yh, _, g, _ = model.mmoe(h[None])
                S.append(yh[0].float().cpu().numpy()); G.append(g[0].float().cpu().numpy())
                TC.append(int(getattr(model, '_tcount', 0)))
                model._hs = None
        S = np.stack(S); Y = np.stack([r['y'] for r in rows]); P = lvl(S)
        _Dg, _Eg = [], []
        with torch.no_grad():
            for r in rows:
                _d, _e = gen_digits(r)
                _Dg.append(_d); _Eg.append(_e); model._hs = None
        Pg = np.asarray(_Dg, np.int64)
        _gx = float((Pg == Y).mean()); _gq = qwk(Pg.reshape(-1), Y.reshape(-1))
        model._eval_gen = Pg.tolist(); model._eval_gen_s = _Eg
        print('    generation-readout (main) Exact %.4f QWK %.4f | numeric head (self-check) Exact %.4f QWK %.4f'
              ' | agree %d/%d'
              % (_gx, _gq, float((P == Y).mean()), qwk(P.reshape(-1), Y.reshape(-1)),
                 int((Pg == P).sum()), P.size), flush=True)
        model._eval_t = list(TC)
        model._eval_s = np.asarray(S).tolist()
        return P, Y, np.mean(G, 0), dict(
            exact=float((P == Y).mean()), mae=float(np.abs(P - Y).mean()),
            qwk=qwk(P.reshape(-1), Y.reshape(-1)),
            perdim=[qwk(P[:, i], Y[:, i]) for i in range(K)],
            gen_exact=_gx, gen_qwk=_gq,
            gen_perdim=[float((Pg[:, i] == Y[:, i]).mean()) for i in range(K)])

    head_p = list(model.mmoe.parameters())
    lora_named = [(n_, q) for n_, q in vlm.named_parameters()
                  if q.requires_grad and 'mmoe' not in n_]
    lora_p = [q for _, q in lora_named]
    if a.accum_lora is None:
        a.accum_lora = a.accum
    opt_lora = torch.optim.AdamW(lora_p, lr=a.lr_lora, weight_decay=0.0)
    opt_head = torch.optim.AdamW(head_p, lr=a.lr_head, weight_decay=1e-2)
    opt_head.add_param_group(dict(params=list(model.esr.parameters()),
                                  lr=a.lr_esr, weight_decay=1e-2))
    head_p = head_p + [q_ for q_ in model.esr.parameters() if q_.requires_grad]
    print('  grad accumulation: LoRA every %d samples | head every %d samples'
          % (a.accum_lora, a.accum), flush=True)

    ck_dir = a.ckpt_dir or os.path.join(a.outdir, 'ckpt')
    ck_path = os.path.join(ck_dir, 'ck_%s_%s.pt' % (a.tag, tag2))
    best_path = os.path.join(ck_dir, 'ck_%s_%s_best.pt' % (a.tag, tag2))
    curve_path = os.path.join(a.outdir, 'val_%s_%s_curve.json' % (a.tag, tag2))
    best_meta_path = os.path.join(ck_dir, 'ck_%s_%s_best.json' % (a.tag, tag2))

    def save_ck(ep, hist, pos=0, partial=None, path=None):
        if a.no_ckpt:
            return
        os.makedirs(ck_dir, exist_ok=True)
        blob = dict(epoch=ep, hist=hist, step=step, pos=pos, partial=partial,
                    args=vars(a),
                    lora={n_: q.detach().to('cpu') for n_, q in lora_named},
                    esr={k_: v_.detach().to('cpu')
                         for k_, v_ in model.esr.state_dict().items()},
                    mmoe={k_: v_.detach().to('cpu')
                          for k_, v_ in model.mmoe.state_dict().items()},
                    opt_lora=opt_lora.state_dict(),
                    opt_head=opt_head.state_dict(),
                    torch_rng=torch.get_rng_state())
        dst = path or ck_path
        tmp = dst + '.tmp'
        torch.save(blob, tmp)
        os.replace(tmp, dst)

    def load_esr(ck):
        sd = ck.get('esr') or ck.get('ro')
        if sd:
            model.esr.load_state_dict({k_: v_.to('cuda:0') for k_, v_ in sd.items()})
        return sd is not None


    _selk = 'gen_exact'
    best = dict(exact=-1.0, epoch=None, m=None, P=None, Y=None, gm=None,
                score=None, t_frames=None, sel_metric=_selk)

    def dump_curve(hist_):
        try:
            json.dump(dict(tag=a.tag, fold=tag2, select_metric=_selk,
                           tie_break='earlier_epoch', best_epoch=best['epoch'],
                           best_exact=best['exact'],
                           select_scope=best.get('scope', 'all_epochs'),
                           history=hist_),
                      io.open(curve_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1, default=float)
        except Exception as _e:
            print('  [warn] failed to write the per-epoch curve: %s' % _e, flush=True)

    def save_best_meta():


        if a.no_ckpt:
            return
        try:
            json.dump(best, io.open(best_meta_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
        except Exception as _e:
            print('  [warn] failed to write best metadata: %s' % _e, flush=True)

    start_ep, hist, t0, step = 1, [], time.time(), 0
    P = Y = gm = m = None
    start_pos, partial = 0, None
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        for n_, q in lora_named:
            q.data.copy_(ck['lora'][n_].to(q.device, q.dtype))
        _r = model.mmoe.load_state_dict({k_: v_.to('cuda:0')
                                         for k_, v_ in ck['mmoe'].items()}, strict=False)
        if _r.missing_keys or _r.unexpected_keys:
            print('  [resume] evidence head loaded non-strictly: missing %s | extra %s'
                  % (list(_r.missing_keys), list(_r.unexpected_keys)), flush=True)
        load_esr(ck)
        if 'opt_lora' in ck:
            try:
                opt_lora.load_state_dict(ck['opt_lora'])
                opt_head.load_state_dict(ck['opt_head'])
            except (ValueError, KeyError) as _oe:
                print('  [warn] optimiser state NOT restored: %s\n'
                      '         usually a param-group size change from added params. '
                      'Pure re-eval is fine; for continued training the optimiser moments are lost and cannot be spliced with the earlier segment.'
                      % _oe, flush=True)
        torch.set_rng_state(ck['torch_rng'].to('cpu', torch.uint8))
        hist = ck['hist']; step = ck.get('step', 0); start_ep = ck['epoch'] + 1
        start_pos = int(ck.get('pos', 0) or 0); partial = ck.get('partial')


        if os.path.exists(best_meta_path) and os.path.exists(best_path):
            try:
                best.update(json.load(io.open(best_meta_path, encoding='utf-8')))
                print('  [resume] restored model-selection state: best = ep %s (%.4f)'
                      % (best['epoch'], best['exact']), flush=True)
            except Exception as _e:
                best['scope'] = 'segment_only'
                print('  [warn] failed to read best metadata (%s); selection scope limited to this segment' % _e, flush=True)
        elif any('exact' in h for h in hist):
            best['scope'] = 'segment_only'
            print('  [warn] missing %s or %s; cannot carry the previous selection threshold; '
                  'best is valid only from epoch %d on (the per-epoch curve is still complete)'
                  % (os.path.basename(best_meta_path), os.path.basename(best_path),
                     start_ep), flush=True)
        if start_pos:
            print('  [resume] intra-epoch position %d/%d, continuing this epoch' % (start_pos, len(tr)), flush=True)
        print('  [resume] restored from %s, %d epochs done, continuing from epoch %d'
              % (ck_path, ck['epoch'], start_ep), flush=True)
        if start_ep > a.epochs:
            print('  [resume] checkpoint already at %d epochs, nothing to continue' % ck['epoch'], flush=True)

    for ep in range(start_ep, a.epochs + 1):
        warm = ep <= a.warmup
        for pg in opt_lora.param_groups:
            pg['lr'] = 0.0 if warm else a.lr_lora
        for q in lora_p:
            q.requires_grad_(not warm)
        if ep == 1 or ep == a.warmup + 1:
            print('  [stage] from ep%d LoRA %s'
                  % (ep, 'warmup frozen (head only)' if warm
                     else 'joint update lr=%g' % a.lr_lora), flush=True)
        model.train(); vlm.train()
        order = np.random.RandomState(a.seed + ep).permutation(len(tr))
        al, ao, ar, nb = 0., 0., 0., 0
        if start_pos and partial and partial.get('ep') == ep:
            al, ao, ar, nb = partial['al'], partial['ao'], partial['ar'], partial['nb']
        opt_lora.zero_grad(); opt_head.zero_grad()
        n_lora = n_head = 0
        for j, i in enumerate(order):
            if j < start_pos:
                continue
            rec = tr[i]
            last = (j == len(order) - 1)
            lt, h = fwd(rec, -1, True)
            yh, z, g, E = model.mmoe(h[None])
            y = torch.as_tensor(rec['y']).cuda()[None]
            lo = ordinal_loss(z, y)
            lr_ = model.mmoe.reg(E, a.lam_div)
            part = a.lam_text * lt + a.lam_ord * lo + lr_
            if a.esr_ord > 0 and model._esr_zlm is not None:
                if a.sor:

                    part = part + a.esr_ord * sor_loss(
                        model._esr_zlm, y[0].long() - 1,
                        sor_sigma=a.sor_sigma, sor_lambda=a.sor_lambda)
                else:
                    part = part + a.esr_ord * ordinal_loss(model._esr_zlm[None], y)
            part.backward()
            model._hs = None
            n_lora += 1
            if n_lora >= a.accum_lora or last:
                for q in lora_p:
                    if q.grad is not None:
                        q.grad /= n_lora
                torch.nn.utils.clip_grad_norm_(lora_p, 1.0)
                opt_lora.step(); opt_lora.zero_grad(); n_lora = 0
            al += float(lt); ao += float(lo); ar += float(lr_); nb += 1; step += 1
            n_head += 1
            if n_head >= a.accum or last:
                for q in head_p:
                    if q.grad is not None:
                        q.grad /= n_head
                torch.nn.utils.clip_grad_norm_(head_p, 1.0)
                opt_head.step(); opt_head.zero_grad(); n_head = 0
            if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes \
                    and j < len(order) - 1:

                if n_lora:
                    for q in lora_p:
                        if q.grad is not None:
                            q.grad /= n_lora
                    torch.nn.utils.clip_grad_norm_(lora_p, 1.0)
                    opt_lora.step(); opt_lora.zero_grad(); n_lora = 0
                if n_head:
                    for q in head_p:
                        if q.grad is not None:
                            q.grad /= n_head
                    torch.nn.utils.clip_grad_norm_(head_p, 1.0)
                    opt_head.step(); opt_head.zero_grad(); n_head = 0
                save_ck(ep - 1, hist, pos=j + 1,
                        partial=dict(ep=ep, al=al, ao=ao, ar=ar, nb=nb))
                print('  [segment] reached %.0f min, checkpointed at %d/%d, exiting to resume later'
                      % (a.max_minutes, j + 1, len(order)), flush=True)
                print('SEGMENT_PAUSED', flush=True)
                return
            if a.smoke and j >= 2:
                break
        if a.eval_last and ep < a.epochs:
            hist.append(dict(epoch=ep, lm=al / nb, ord=ao / nb, reg=ar / nb))
            print('  [ep %d] LM %.4f ord %.4f reg %+.4f | eval deferred to epoch %d | %.1f min%s'
                  % (ep, al / nb, ao / nb, ar / nb, a.epochs, (time.time() - t0) / 60,
                     model.mmoe.pop_stats()), flush=True)
            start_pos, partial = 0, None
            save_ck(ep, hist)
            continue
        P, Y, gm, m = evaluate(ev)
        hist.append(dict(epoch=ep, lm=al / nb, ord=ao / nb, reg=ar / nb, **m))
        print('  [ep %d] LM %.4f ord %.4f reg %+.4f | numeric head Exact %.4f QWK %.4f '
              '| per-dim %s | %.1f min'
              % (ep, al / nb, ao / nb, ar / nb, m['exact'], m['qwk'],
                 np.round(m['perdim'], 3).tolist(), (time.time() - t0) / 60), flush=True)
        print('      gate mean %s%s%s'
              % ({DIMS[i][:9]: np.round(gm[i], 3).tolist() for i in range(K)},
                 model.mmoe.pop_stats(), model.esr.pop_stats()), flush=True)
        _selv = m.get(_selk)
        assert _selv is not None, 'selection metric %s is empty' % _selk
        if float(_selv) > best['exact']:
            best.update(exact=float(_selv), sel_metric=_selk,
                        head_exact=float(m['exact']),
                        gen_exact=float(m['gen_exact']),
                        epoch=ep, m=dict(m),
                        P=np.asarray(P).tolist(), Y=np.asarray(Y).tolist(),
                        gm=np.asarray(gm).tolist(),
                        score=list(getattr(model, '_eval_s', [])),
                        gen_P=list(getattr(model, '_eval_gen', []) or []),
                        t_frames=list(getattr(model, '_eval_t', [])))
            save_ck(ep, hist, path=best_path)
            save_best_meta()
            print('      [select] %s improved -> best = ep %d (%.4f | numeric head %.4f), saved %s'
                  % (_selk, ep, best['exact'], m['exact'],
                     os.path.basename(best_path)), flush=True)
        dump_curve(hist)
        start_pos, partial = 0, None
        save_ck(ep, hist)

    if gm is None:


        P, Y, gm, m = evaluate(ev)
        hist.append(dict(epoch=a.epochs, resumed_eval_only=True, **m))
        print('  [resume] no further training; evaluated from the checkpoint: generation-readout Exact %.4f'
              % m['gen_exact'], flush=True)

    out = dict(args=vars(a), history=hist,
               gate_mean={DIMS[i]: np.round(gm[i], 4).tolist() for i in range(K)})
    out['select'] = dict(metric=_selk, tie_break='earlier_epoch',
                         n_candidates=len(hist), best_epoch=best['epoch'],
                         scope=best.get('scope', 'all_epochs'),
                         best=best['m'], last_epoch=hist[-1].get('epoch') if hist else None,
                         last=dict(m) if m else None)
    dump_curve(hist)
    if best['epoch'] is not None:
        print('\n===== model selection (main metric: generation-readout Exact; ties -> earlier epoch) =====', flush=True)
        print('  candidates %d | best = ep %d' % (len(hist), best['epoch']), flush=True)
        print('  best-epoch : gen Exact %.4f | numeric Exact %.4f QWK %.4f MAE %.4f'
              % (best['m']['gen_exact'], best['m']['exact'], best['m']['qwk'],
                 best['m']['mae']), flush=True)
        print('  last-epoch : gen Exact %.4f | numeric Exact %.4f QWK %.4f MAE %.4f'
              % (m['gen_exact'], m['exact'], m['qwk'], m['mae']), flush=True)
        print('  per-epoch gen Exact %s' % [round(float(h.get('gen_exact', float('nan'))), 3)
                                        for h in hist], flush=True)


    try:
        out['per_video'] = dict(pred=np.asarray(P).tolist(), gt=np.asarray(Y).tolist(),
                                score=list(getattr(model, '_eval_s', [])),
                                gen_pred=list(getattr(model, '_eval_gen', []) or []),
                                t_frames=list(getattr(model, '_eval_t', [])))
        if best['epoch'] is not None:
            out['per_video_best'] = dict(epoch=best['epoch'], pred=best['P'],
                                         gt=best['Y'], score=best['score'],
                                         t_frames=best['t_frames'])
    except Exception as _e:
        out['per_video_error'] = str(_e)


    if ev and best.get('epoch') is not None:
        try:
            _ckb = torch.load(best_path, map_location='cpu', weights_only=False)
            for _n, _q in lora_named:
                _q.data.copy_(_ckb['lora'][_n].to(_q.device, _q.dtype))
            _rb = model.mmoe.load_state_dict({k_: v_.to('cuda:0')
                                              for k_, v_ in _ckb['mmoe'].items()}, strict=False)
            if _rb.missing_keys or _rb.unexpected_keys:
                print('  [ablation] evidence head loaded non-strictly: missing %s | extra %s'
                      % (list(_rb.missing_keys), list(_rb.unexpected_keys)), flush=True)
            load_esr(_ckb)
            model._esr_off = False
            model.esr._acc = []
            _Pon, _Yv, _gon, _mon = evaluate(ev)
            _inv = model.esr.invariants()
            _Gon = np.asarray(_gon)
            _Pgon = np.asarray(getattr(model, '_eval_gen', []))
            model._esr_off = True
            _Pof, _, _, _mof = evaluate(ev)
            _Pgof = np.asarray(getattr(model, '_eval_gen', []))
            model._esr_off = False
            _msh = _Pgsh = None
            if model.esr.priv:
                model.esr.priv_off = True
                _, _, _, _msh = evaluate(ev)
                _Pgsh = np.asarray(getattr(model, '_eval_gen', []))
                model.esr.priv_off = False
            _g0 = model.esr.G0.detach().float().cpu().numpy()
            _iu = np.triu_indices(K, 1)
            out['ablation'] = dict(
                epoch=best['epoch'], n_cells=int(_Pon.size),
                on=dict(_mon), off=dict(_mof),
                drop_gen_exact=float(_mon['gen_exact'] - _mof['gen_exact']),
                drop_head_exact=float(_mon['exact'] - _mof['exact']),
                same_gen_cells=(int((_Pgon == _Pgof).sum())
                                if _Pgon.size and _Pgon.shape == _Pgof.shape else None),
                gt=np.asarray(_Yv).tolist(),
                on_gen_pred=(_Pgon.tolist() if _Pgon.size else None),
                off_gen_pred=(_Pgof.tolist() if _Pgof.size else None),
                esr_invariants=_inv,
                esr_gate_static=_g0.tolist(),
                esr_gate_static_tv=float(
                    np.abs(_g0[:, None, :] - _g0[None, :, :]).sum(-1)[_iu].mean()),
                esr_A_absmax=float(model.esr.A.detach().abs().max()),
                esr_tied=bool(a.esr_tied), esr_priv=bool(model.esr.priv),
                ctr_gate=_Gon.tolist(),
                ctr_gate_tv=float(np.abs(_Gon[:, None, :]
                                         - _Gon[None, :, :]).sum(-1)[_iu].mean()))
            if _msh is not None:
                out['ablation']['shared_only'] = dict(_msh)
                out['ablation']['shared_only_gen_pred'] = (
                    _Pgsh.tolist() if _Pgsh is not None and _Pgsh.size else None)
            print('\n===== ablation (best weights ep %s; on/off at generation time, same checkpoint) ====='
                  % best['epoch'], flush=True)
            print('  generation-readout: ON %.4f | SHARED %s | OFF %.4f | ON-OFF drop %+.4f | '
                  'ON/OFF agree %s/%d'
                  % (_mon['gen_exact'],
                     ('%.4f' % _msh['gen_exact']) if _msh else 'n/a',
                     _mof['gen_exact'], _mon['gen_exact'] - _mof['gen_exact'],
                     out['ablation']['same_gen_cells'], int(_Pon.size)), flush=True)
            print('  numeric head (self-check, ~0): ON %.4f | OFF %.4f | drop %+.4f'
                  % (_mon['exact'], _mof['exact'], _mon['exact'] - _mof['exact']),
                  flush=True)
            if _inv is not None:


                print('  [invariants] zero-sum %.2e | row-norm [%.4f, %.4f] | coldest %.4f | '
                      'max|g| %.4f <= %.4f | pairwise gate-TV %.4f | %d decisions'
                      % (_inv['zero_sum_absmax'], _inv['row_norm_min'],
                         _inv['row_norm_max'], _inv['coldest_expert_mass'],
                         _inv['gate_absmax'], _inv['gate_absmax_bound'],
                         _inv['gate_tv'], _inv['n_calls']), flush=True)
            print('  [mechanism] G0 pairwise TV %.4f | max|A| %.5f (init 0) | tied=%s | priv=%s'
                  % (out['ablation']['esr_gate_static_tv'],
                     out['ablation']['esr_A_absmax'], a.esr_tied, model.esr.priv),
                  flush=True)
            print('  main criterion: generation-readout best Exact %.4f' % best['exact'], flush=True)
        except Exception as _e:
            out['ablation_error'] = str(_e)
            print('  [warn] ablation failed: %s' % _e, flush=True)
    fn = os.path.join(a.outdir, 'esr_%s_%s.json' % (a.tag, tag2))
    json.dump(out, io.open(fn, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\n->', fn, flush=True)
    print('  peak GPU memory %.1f GB' % (torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)
    print('TRAIN_DONE', flush=True)


if __name__ == '__main__':
    main()
