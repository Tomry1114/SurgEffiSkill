import sys, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from esr import ESR, K, L

d, d_ev, J, r = 3584, 40, 5, 8
c = 1.0
ok = []


def chk(name, cond, extra=''):
    ok.append(bool(cond))
    print('  [%s] %-46s %s' % ('PASS' if cond else 'FAIL', name, extra))


print('=== ESR invariants | K=%d L=%d d=%d n_exp=%d r=%d ===' % (K, L, d, J, r))


try:
    ESR(d, d_ev, n_exp=K, r=r); bad = True
except AssertionError:
    bad = False
chk('n_exp < K+1 is rejected', not bad,
    'zero-sum hyperplane has only %d dims at n_exp=%d' % (K - 1, K))

m = ESR(d, d_ev, n_exp=J, r=r, priv=True, radius=c)


SK = ('Wr', 'Ap', 'gp', 's')
nsk = sum(q.numel() for n, q in m.named_parameters() if n in SK)
nall = sum(q.numel() for q in m.parameters())
per = {n: q.numel() for n, q in m.named_parameters()}
print('  params: %s' % per)
print('  total %d = shared %d + skill-specific %d (%.0f per skill, %.2f%%)'
      % (nall, nall - nsk, nsk, nsk / float(K), 100.0 * nsk / nall))
chk('skill-specific params < 2%', nsk / float(nall) < 0.02)
chk('G0 is a buffer (fixed, not a free param)', 'G0' not in per)


G0 = m.G0
chk('G0 rows zero-sum', float(G0.sum(-1).abs().max()) < 1e-5,
    'resid %.2e' % float(G0.sum(-1).abs().max()))
chk('G0 rows unit-norm', float((G0.norm(dim=-1) - 1).abs().max()) < 1e-5)
gram = G0 @ G0.t()
off = (gram - torch.eye(K)).abs().max()
chk('G0 rows pairwise orthogonal', float(off) < 1e-5, 'max|Gram-I| %.2e' % float(off))
mass0 = G0.abs().sum(0)
chk('every expert used by >=1 skill at init', float(mass0.min()) > 1e-3,
    'coldest-expert mass %.4f' % float(mass0.min()))


torch.manual_seed(0)
worst_zs, worst_dn, worst_max, worst_cold = 0.0, 0.0, 0.0, 1e9
BOUND = c * math.sqrt((J - 1) / float(J))
for t in range(2000):
    with torch.no_grad():
        m.Wr.normal_(0, 3.0 * (t % 7 + 1))
        ev = torch.randn(d_ev) * (t % 5 + 1)
        g = m.gate(ev)
    worst_zs = max(worst_zs, float(g.sum(-1).abs().max()))
    worst_dn = max(worst_dn, float((g.norm(dim=-1) - c).abs().max()))
    worst_max = max(worst_max, float(g.abs().max()))
    worst_cold = min(worst_cold, float(g.abs().sum(0).min()))
chk('zero-sum holds throughout training', worst_zs < 1e-5,
    'worst resid over 2000 %.2e' % worst_zs)
chk('row-norm fixed at c (gate cannot collapse to 0)', worst_dn < 1e-5,
    'worst dev %.2e' % worst_dn)
chk('no single expert takes all gate mass', worst_max <= BOUND + 1e-5,
    'max|g| %.4f <= c*sqrt((J-1)/J) = %.4f' % (worst_max, BOUND))
print('  (info) min coldest-expert mass over 2000 draws = %.4f -- once Wr breaks '
      'orthogonality there is no hard lower bound, so it is a measured quantity, '
      'not a guarantee' % worst_cold)
with torch.no_grad():
    m.Wr.zero_()


hk = torch.randn(K, d)
ev = torch.randn(d_ev)
out, g = m(hk, ev)
chk('init output is exactly 0 (strict superset of SFT)',
    float(out.detach().abs().max()) == 0.0, 'max|out| %.1e' % float(out.detach().abs().max()))
chk('init gate is nonzero', float(g.abs().min()) > 0)


m.zero_grad()
out, _ = m(hk, ev)
out.pow(2).sum().backward()
m.zero_grad()
out, _ = m(hk, ev)
(out.sum()).backward()
gA = float(m.A.grad.abs().max()); gAp = float(m.Ap.grad.abs().max())
gs = float(m.s.grad.abs().max()); gWr = float(m.Wr.grad.abs().max())
chk('A has a gradient at step 0', gA > 0, 'max|dA| %.3e' % gA)
chk('Ap has a gradient at step 0', gAp > 0, 'max|dAp| %.3e' % gAp)
chk('s_k init=1 (cannot be zero-init together with A)', float(m.s.detach().min()) == 1.0)
print('  (expected) Wr grad is 0 at step 0: out = s*g*(A u), A=0 => d out/d g = 0; '
      'Wr starts learning once A is nonzero. max|dWr| %.1e' % gWr)


mt = ESR(d, d_ev, n_exp=J, r=r, priv=True, radius=c, tied=True)
gt = mt.gate(torch.randn(d_ev))
chk('tied: all four rows identical', float((gt - gt[0:1]).abs().max()) < 1e-6)
chk('tied: gate norm still c (not degraded to off)',
    abs(float(gt.norm(dim=-1).min()) - c) < 1e-5, '||g|| = %.4f' % float(gt.norm(dim=-1).min()))
chk('tied: Wr frozen', not mt.Wr.requires_grad)


with torch.no_grad():
    m.A.normal_(0, 0.01); m.Ap.normal_(0, 0.01)
o_all, _ = m(hk, ev)
m.priv_off = True
o_sh, _ = m(hk, ev)
m.priv_off = False
chk('priv_off disables only the private write head (output changes)',
    float((o_all - o_sh).abs().max()) > 0, 'delta %.4e' % float((o_all - o_sh).abs().max()))
m2 = ESR(d, d_ev, n_exp=J, r=r, priv=False, radius=c)
chk('--esr-nopriv builds no private params',
    not any(n in ('Ap', 'gp') for n, _ in m2.named_parameters()))

print('\n=== %d/%d passed ===' % (sum(ok), len(ok)))
sys.exit(0 if all(ok) else 1)
