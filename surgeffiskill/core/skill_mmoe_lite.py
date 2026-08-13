# -*- coding: utf-8 -*-
"""SkillMMoE-Lite：池化特征上的多任务 MoE，含 MMoE / CGC / PLE 三种拓扑。

为什么放弃 token 级注意力版（v2–v5）：训练集只有 98 个视频，那些版本 ~10^7 参数，
在同一批冻结特征上始终输给一个 PCA+ridge 线性探针（0.46–0.47 vs 0.51）；
且实测四维注意力已近正交（余弦 0.002）却没换来精度，说明区分维度的是
"如何加权同一组全局统计量"而非"看哪一帧"。

拓扑（`--arch`）：
  mmoe  Ma et al., KDD'18   —— 共享专家 + 逐任务门控 + 逐任务塔
  cgc   Tang et al., RecSys'20 —— **每个任务各有专属专家** + 共享专家，
        任务 k 的门控只在「专属_k ∪ 共享」上选。针对 seesaw：我们四维真值相关
        仅 ρ=0.10–0.27（Tissue vs 其余），纯共享专家会互相拖累。
  ple   同上，多层 CGC 堆叠（`--layers`），共享部分另有一个门控。

门控（`--gate`）：softmax / top-k（Shazeer'17 风格稀疏化，小样本下是强正则）
任务加权（`--task-weight`）：fixed / uncertainty（Kendall et al., CVPR'18，学出来）
梯度冲突（`--pcgrad`）：Yu et al., NeurIPS'20 梯度手术，投影掉任务间冲突分量

**退化性质（v6 更正）**：v5 用 `nn.Linear` 当"残差探针"是错的——那是随机初始化、
被 AdamW+wd 训出来的劣正则线性层，不等于超参经 CV 选定的 ridge，实测反而拖低 QWK
0.065。v6 改为传入**预先拟合好的 ridge 探针分数作为冻结偏置**（train 用折外预测），
塔权重零初始化 ⇒ 训练起点严格等于 ridge 探针本身，MoE 只学残差修正。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']
NUM_LEVELS = 5
TISSUE = 3


def mlp(i, h, o, p):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class GateMix(nn.Module):
    """一层多任务混合：任务 k 在自己可用的专家集合上做门控加权。

    专家用**批量参数张量 + einsum** 实现（不是 ModuleList 循环）：小模型下
    Python 循环会起大量微型 kernel，实测占了绝大部分时间；批量化后单步快约一个量级，
    使 96 配置 × 5 折 CV 的扫描变得可行。
    """

    def __init__(self, in_dim, out_dim, num_dims, n_shared, n_task, hidden, dropout,
                 gate='softmax', topk=2):
        super().__init__()
        self.num_dims, self.n_shared, self.n_task = num_dims, n_shared, n_task
        self.gate_kind, self.topk = gate, topk
        E = n_shared + n_task * num_dims
        self.total = E
        self.w1 = nn.Parameter(torch.empty(E, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(E, hidden))
        self.w2 = nn.Parameter(torch.empty(E, hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(E, out_dim))
        for w in (self.w1, self.w2):
            nn.init.normal_(w, std=(2.0 / w.shape[1]) ** 0.5)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(num_dims, in_dim, E))
        nn.init.normal_(self.gate, std=0.01)
        avail = torch.zeros(num_dims, E, dtype=torch.bool)
        avail[:, :n_shared] = True
        for k in range(num_dims):                      # 任务 k 的专属专家
            st = n_shared + k * n_task
            avail[k, st:st + n_task] = True
        self.register_buffer('avail', avail)

    def forward(self, z_per_dim, z_gate):
        """z_per_dim: (B,K,in) 每维各自的输入；z_gate: (B,in) 门控输入"""
        h = torch.einsum('bki,eih->bkeh', z_per_dim, self.w1) + self.b1
        h = self.drop(F.gelu(h))
        ex = torch.einsum('bkeh,eho->bkeo', h, self.w2) + self.b2      # (B,K,E,out)
        gl = torch.einsum('bd,kde->bke', z_gate, self.gate)
        gl = gl.masked_fill(~self.avail.unsqueeze(0), float('-inf'))
        if self.gate_kind == 'topk':
            k = min(self.topk, int(self.avail[0].sum()))
            thr = gl.topk(k, dim=-1).values[..., -1:]
            gl = gl.masked_fill(gl < thr, float('-inf'))
        a = F.softmax(gl, dim=-1)
        return torch.einsum('bke,bkeo->bko', a, ex), a


class SkillMMoELite(nn.Module):
    def __init__(self, in_dim, num_dims=4, num_experts=4, expert_hidden=16,
                 num_levels=NUM_LEVELS, dropout=0.1, task_experts=1, use_probe=True,
                 arch='cgc', layers=1, gate='softmax', topk=2, task_weight='fixed'):
        super().__init__()
        self.num_dims, self.num_levels, self.arch = num_dims, num_levels, arch
        self.use_probe = use_probe
        n_task = 0 if arch == 'mmoe' else task_experts
        self.n_shared, self.n_task = num_experts, n_task

        dims_in = [in_dim] + [expert_hidden] * (layers - 1)
        self.blocks = nn.ModuleList([
            GateMix(dims_in[i], expert_hidden, num_dims, num_experts, n_task,
                    expert_hidden * 2, dropout, gate, topk) for i in range(layers)])
        self.layers = layers

        self.tower = nn.ModuleList([nn.Linear(expert_hidden, 1) for _ in range(num_dims)])
        for t in self.tower:                       # 零初始化 → 起点等价线性探针
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)
        self.probe = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_dims)]) \
            if use_probe else None
        self.theta0 = nn.Parameter(torch.full((num_dims,), -1.5))
        self.dtheta = nn.Parameter(torch.zeros(num_dims, num_levels - 2))
        self.register_buffer('coral_w', torch.ones(num_dims, num_levels - 1))
        self.task_weight = task_weight
        self.log_var = nn.Parameter(torch.zeros(num_dims))   # Kendall 不确定性加权

    def thresholds(self):
        d = F.softplus(self.dtheta) + 1e-3
        return torch.cat([self.theta0.unsqueeze(-1),
                          self.theta0.unsqueeze(-1) + torch.cumsum(d, dim=-1)], dim=-1)

    def set_class_weights(self, levels_train):
        w = torch.ones(self.num_dims, self.num_levels - 1)
        for k in range(self.num_dims):
            for j in range(self.num_levels - 1):
                pos = float((levels_train[:, k] > j + 1).astype(np.float32).mean())
                pos = min(max(pos, 1e-3), 1 - 1e-3)
                w[k, j] = (1 - pos) / pos
        self.coral_w.copy_(w)

    def forward(self, z, offset=None):
        """offset: (B,K) 冻结的 ridge 探针分数；给了就作为基线，MoE 只学残差。"""
        h = z.unsqueeze(1).expand(-1, self.num_dims, -1)
        gates = []
        for i, blk in enumerate(self.blocks):
            h, a = blk(h, z if i == 0 else h.mean(1))
            gates.append(a)
        s = torch.stack([self.tower[k](h[:, k]).squeeze(-1)
                         for k in range(self.num_dims)], dim=1)
        if self.use_probe:
            s = s + torch.stack([self.probe[k](z).squeeze(-1)
                                 for k in range(self.num_dims)], dim=1)
        if offset is not None:
            s = s + offset
        return {'cum_logits': s.unsqueeze(-1) - self.thresholds().unsqueeze(0),
                'score': s, 'gate': gates[-1], 'gates': gates}

    # ------------------------------------------------------------------ 损失
    def per_dim_loss(self, out, levels, dim_w=None):
        """返回 (K,) 的逐维损失，供 PCGrad / 不确定性加权分别处理。"""
        cl = out['cum_logits']
        thr = torch.arange(1, self.num_levels, device=levels.device).view(1, 1, -1)
        tgt = (levels.unsqueeze(-1) > thr).float()
        w = torch.where(tgt > 0, self.coral_w.unsqueeze(0).expand_as(tgt),
                        torch.ones_like(tgt))
        coral = F.binary_cross_entropy_with_logits(cl, tgt, weight=w, reduction='none').mean((0, 2))
        ev = F.smooth_l1_loss(self.expected_level(cl), levels.float(), reduction='none').mean(0)
        L = coral + ev
        if self.task_weight == 'uncertainty':
            L = L * torch.exp(-self.log_var) + self.log_var
        elif dim_w is not None:
            L = L * torch.as_tensor(dim_w, device=L.device, dtype=L.dtype)
        return L

    def expected_level(self, cum_logits):
        return 1.0 + torch.sigmoid(cum_logits).sum(-1)

    @staticmethod
    def balance_loss(gate):
        imp = gate.mean(dim=(0, 1))
        return (imp.std() / (imp.mean() + 1e-9)) ** 2

    @staticmethod
    def gate_diversity_loss(gate):
        a = F.normalize(gate.mean(0), p=2, dim=-1)
        g = a @ a.T
        off = ~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
        return g[off].mean()

    def decorr_loss(self, score, levels):
        if score.shape[0] < 8:
            return score.sum() * 0.0

        def corr(x):
            x = x - x.mean(0, keepdim=True)
            x = x / (x.std(0, keepdim=True) + 1e-6)
            return x.T @ x / x.shape[0]
        cp, cg = corr(score), corr(levels.float())
        off = ~torch.eye(score.shape[1], dtype=torch.bool, device=score.device)
        return F.relu(cp[off].abs() - cg[off].abs()).mean()

    def aux(self, out, levels, w_bal, w_div, w_dec):
        return (w_bal * self.balance_loss(out['gate'])
                + w_div * self.gate_diversity_loss(out['gate'])
                + w_dec * self.decorr_loss(out['score'], levels))


def pcgrad_step(model, losses, opt):
    """PCGrad（Yu et al., NeurIPS'20）：任务梯度两两冲突时投影掉冲突分量。"""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for i, L in enumerate(losses):
        opt.zero_grad()
        L.backward(retain_graph=(i < len(losses) - 1))
        grads.append([(p.grad.detach().clone() if p.grad is not None else None)
                      for p in params])
    proj = [[None if g is None else g.clone() for g in gi] for gi in grads]
    idx = np.random.permutation(len(grads))
    for i in range(len(grads)):
        for j in idx:
            if i == j:
                continue
            dot = sum(float((a * b).sum()) for a, b in zip(proj[i], grads[j])
                      if a is not None and b is not None)
            if dot < 0:
                nsq = sum(float((b * b).sum()) for b in grads[j] if b is not None) + 1e-12
                for t in range(len(proj[i])):
                    if proj[i][t] is not None and grads[j][t] is not None:
                        proj[i][t] -= dot / nsq * grads[j][t]
    opt.zero_grad()
    for t, p in enumerate(params):
        acc = None
        for i in range(len(proj)):
            if proj[i][t] is not None:
                acc = proj[i][t] if acc is None else acc + proj[i][t]
        p.grad = acc
