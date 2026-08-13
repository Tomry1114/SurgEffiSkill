# -*- coding: utf-8 -*-
"""SkillMoE-v8：把 v6/v7 里**从未生效**的门控机制修好后的多任务 MoE。

v7 的实测事实（`v7_*_1016809*.out`）：四个维度的门控权重全部等于 1/n_avail，
偏离不超过 0.002 —— MMoE 的核心机制（逐任务差异化路由）一次都没发生过。
因此 v7 得到的 "CGC > MMoE (+0.019)" 只能说明**逐维私有参数**有用，
不能说明**路由**有用。v8 逐条修掉导致门控死亡的原因：

1. **零初始化的塔让门控在第 0 步梯度恒为 0**。`score = tower(mix)`，tower 权重为 0
   时 ∂L/∂gate ≡ 0；而 AdamW 的 weight decay 同一步就在把门控参数压向 0。
   → v8 用小随机初始化的塔（`tower_init`），并放弃 "ep0 逐元素等于探针" 这个
   自我设限；锚是否被保住改为**测量**（报 ep0 与探针的偏差）而不是断言。
2. **weight decay 打在门控/阈值上**。→ v8 分参数组：`gate/theta0/dtheta/log_var`
   一律 `weight_decay=0`，门控另给 `gate_lr_mult` 倍学习率。
3. **所有专家看同一个输入**，初始化同分布、损失相同 ⇒ 收敛到同一个函数；
   专家无差别时门控怎么分都一样，梯度自然消失，**自锁**。
   → v8 给每个专家一个固定的随机特征子空间（`subspace`），从结构上破对称。
4. **负载均衡损失 `(std/mean)²` 是在主动逼门控趋于均匀**，与 CGC 的专属专家掩码
   方向相反。→ v8 默认改为**熵惩罚**（鼓励尖锐路由），负载均衡只在共享专家内部
   以极小权重保留，防止死专家。
5. **门控输入是全维 PCA（96 维 × 4 维 × 8 专家 = 3072 个参数 > 样本数）**。
   → v8 的门控只看前 `gate_dim` 个主成分（方差最大、最稳），参数降到 ~256。

拓扑仍是 MMoE (Ma et al., KDD'18) / CGC (Tang et al., RecSys'20)。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = ['Depth perception', 'Bimanual coordination', 'Efficiency', 'Tissue handling']
NUM_LEVELS = 5
TISSUE = 3
BASE_THR = np.array([1.5, 2.5, 3.5, 4.5])
SOFTPLUS_INV = math.log(math.expm1(1.0 - 1e-3))     # 抵消 thresholds() 里的 +1e-3


class GateMixV8(nn.Module):
    """一层多任务混合：任务 k 在「专属_k ∪ 共享」上做门控加权。

    专家用批量参数张量 + einsum 实现；每个专家的第一层权重乘上一个**固定的**
    随机列掩码（`mask`），使专家在结构上就看不同的输入子空间——这是让门控有东西
    可选的前提，v7 缺的就是这个。掩码由 `arch_seed` 决定，与训练 seed 无关，
    因此同一配置的架构在所有 seed / 折上完全一致。
    """

    def __init__(self, in_dim, out_dim, num_dims, n_shared, n_task, hidden, dropout,
                 gate='softmax', topk=1, subspace=0.5, gate_dim=8, arch_seed=1234):
        super().__init__()
        self.num_dims, self.n_shared, self.n_task = num_dims, n_shared, n_task
        self.gate_kind, self.topk = gate, topk
        self.gate_dim = min(gate_dim, in_dim)
        E = n_shared + n_task * num_dims
        self.total = E

        self.w1 = nn.Parameter(torch.empty(E, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(E, hidden))
        self.w2 = nn.Parameter(torch.empty(E, hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(E, out_dim))
        for w in (self.w1, self.w2):
            nn.init.normal_(w, std=(2.0 / w.shape[1]) ** 0.5)
        self.drop = nn.Dropout(dropout)

        g = torch.Generator().manual_seed(arch_seed)
        mask = torch.ones(E, in_dim)
        if subspace < 1.0:
            keep = max(4, int(round(in_dim * subspace)))
            for e in range(E):
                m = torch.zeros(in_dim)
                m[torch.randperm(in_dim, generator=g)[:keep]] = 1.0
                mask[e] = m
        self.register_buffer('mask', mask)

        # 门控只看前 gate_dim 个主成分：参数量 K*gate_dim*E，远小于全维方案
        self.gate = nn.Parameter(torch.zeros(num_dims, self.gate_dim, E))
        nn.init.normal_(self.gate, std=0.1)

        # 逐任务**静态**门控：只有 K*E 个参数，不随样本变化。
        # 依据：本任务的异质性是"哪一维依赖哪类特征"，这是任务的静态属性；
        # MMoE 原版的逐样本门控是为推荐场景设计的（同一用户不同上下文要走不同专家），
        # 在 98 个视频上学逐样本路由基本是纯方差。实测支持这一点：逐样本门控越尖锐
        # 增益越低（corr=−0.65），而冻结均匀门控反而不差。
        self.gate_static = nn.Parameter(torch.zeros(num_dims, E))

        avail = torch.zeros(num_dims, E, dtype=torch.bool)
        avail[:, :n_shared] = True
        for k in range(num_dims):
            st = n_shared + k * n_task
            avail[k, st:st + n_task] = True
        self.register_buffer('avail', avail)

    def forward(self, z_per_dim, z_gate):
        """z_per_dim: (B,K,in)；z_gate: (B,in)。返回 (mix, gate_weights, expert_out)。"""
        w1 = self.w1 * self.mask.unsqueeze(-1)                  # 逐专家子空间
        h = torch.einsum('bki,eih->bkeh', z_per_dim, w1) + self.b1
        h = self.drop(F.gelu(h))
        ex = torch.einsum('bkeh,eho->bkeo', h, self.w2) + self.b2      # (B,K,E,out)

        if self.gate_kind == 'uniform':
            # 受控对照：结构/容量/专家全不变，只把路由**冻结成均匀**。
            # 这是唯一能把"逐任务路由"单独摘出来的消融——v7 是无意中落到了这个状态。
            a = (self.avail.float() / self.avail.float().sum(-1, keepdim=True))
            a = a.unsqueeze(0).expand(z_per_dim.shape[0], -1, -1)
            return torch.einsum('bke,bkeo->bko', a, ex), a, ex

        if self.gate_kind == 'static':
            gl = self.gate_static.unsqueeze(0).expand(z_per_dim.shape[0], -1, -1)
        else:
            gl = torch.einsum('bd,kde->bke', z_gate[:, :self.gate_dim], self.gate)
        gl = gl.masked_fill(~self.avail.unsqueeze(0), float('-inf'))
        if self.gate_kind == 'topk':
            k = min(self.topk, int(self.avail[0].sum()))
            thr = gl.topk(k, dim=-1).values[..., -1:]
            gl = gl.masked_fill(gl < thr, float('-inf'))
        a = F.softmax(gl, dim=-1)
        return torch.einsum('bke,bkeo->bko', a, ex), a, ex


class SkillMoEV8(nn.Module):
    def __init__(self, in_dim, num_dims=4, num_experts=3, expert_hidden=8,
                 num_levels=NUM_LEVELS, dropout=0.1, task_experts=1, arch='cgc',
                 gate='softmax', topk=1, task_weight='fixed', subspace=0.5,
                 gate_dim=8, tower_init=0.05, arch_seed=1234):
        super().__init__()
        self.num_dims, self.num_levels, self.arch = num_dims, num_levels, arch
        n_task = 0 if arch == 'mmoe' else task_experts
        self.n_shared, self.n_task = num_experts, n_task

        self.block = GateMixV8(in_dim, expert_hidden, num_dims, num_experts, n_task,
                               expert_hidden * 2, dropout, gate, topk, subspace,
                               gate_dim, arch_seed)
        self.tower = nn.ModuleList([nn.Linear(expert_hidden, 1) for _ in range(num_dims)])
        for t in self.tower:            # 小随机初始化：门控从第 1 步就有梯度
            nn.init.normal_(t.weight, std=tower_init)
            nn.init.zeros_(t.bias)

        self.theta0 = nn.Parameter(torch.full((num_dims,), float(BASE_THR[0])))
        self.dtheta = nn.Parameter(torch.full((num_dims, num_levels - 2), SOFTPLUS_INV))
        self.register_buffer('coral_w', torch.ones(num_dims, num_levels - 1))
        self.task_weight = task_weight
        self.log_var = nn.Parameter(torch.zeros(num_dims))

    # ------------------------------------------------------------------ 前向
    def thresholds(self):
        d = F.softplus(self.dtheta) + 1e-3
        return torch.cat([self.theta0.unsqueeze(-1),
                          self.theta0.unsqueeze(-1) + torch.cumsum(d, dim=-1)], dim=-1)

    def forward(self, z, offset=None):
        """offset: (B,K) 冻结 ridge 锚（等级尺度）；MoE 学残差。"""
        h = z.unsqueeze(1).expand(-1, self.num_dims, -1)
        mix, a, ex = self.block(h, z)
        s = torch.stack([self.tower[k](mix[:, k]).squeeze(-1)
                         for k in range(self.num_dims)], dim=1)
        if offset is not None:
            s = s + offset
        return {'cum_logits': s.unsqueeze(-1) - self.thresholds().unsqueeze(0),
                'score': s, 'gate': a, 'expert_out': ex}

    def set_class_weights(self, levels_train, mode='none'):
        w = torch.ones(self.num_dims, self.num_levels - 1)
        if mode == 'balanced':
            for k in range(self.num_dims):
                for j in range(self.num_levels - 1):
                    pos = float((levels_train[:, k] > j + 1).astype(np.float32).mean())
                    pos = min(max(pos, 1e-3), 1 - 1e-3)
                    w[k, j] = (1 - pos) / pos
        self.coral_w.copy_(w)

    # ------------------------------------------------------------- 参数组 / 损失
    def param_groups(self, lr, wd, gate_lr_mult=10.0):
        """门控与阈值**不施加 weight decay**；门控另给更高学习率。

        v7 把 wd=1e-2 无差别地打在 gate/theta0/dtheta 上：门控被压成 0（⇒ 均匀 softmax），
        阈值从锚定的 {1.5,2.5,3.5,4.5} 被拖走（1920 步后 theta0≈1.42），
        等于一边锚定一边拆锚。
        """
        gate, nodecay, decay = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith('block.gate') or n.endswith('block.gate_static'):
                gate.append(p)
            elif n in ('theta0', 'dtheta', 'log_var') or n.split('.')[-1] in ('b1', 'b2', 'bias'):
                nodecay.append(p)
            else:
                decay.append(p)
        return [{'params': decay, 'weight_decay': wd, 'lr': lr},
                {'params': nodecay, 'weight_decay': 0.0, 'lr': lr},
                {'params': gate, 'weight_decay': 0.0, 'lr': lr * gate_lr_mult}]

    def loss(self, out, levels, dim_w=None):
        cl = out['cum_logits']
        thr = torch.arange(1, self.num_levels, device=levels.device).view(1, 1, -1)
        tgt = (levels.unsqueeze(-1) > thr).float()
        w = torch.where(tgt > 0, self.coral_w.unsqueeze(0).expand_as(tgt),
                        torch.ones_like(tgt))
        coral = F.binary_cross_entropy_with_logits(cl, tgt, weight=w,
                                                   reduction='none').mean((0, 2))
        reg = F.smooth_l1_loss(out['score'], levels.float(), reduction='none').mean(0)
        L = coral + reg
        if self.task_weight == 'uncertainty':
            L = L * torch.exp(-self.log_var) + self.log_var
        elif dim_w is not None:
            L = L * torch.as_tensor(dim_w, device=L.device, dtype=L.dtype)
        return L

    # --------------------------------------------------------------- 正则项
    @staticmethod
    def gate_entropy(gate):
        """(B,K,E) → 平均归一化熵。作为**惩罚**加进损失即鼓励尖锐（专家化）路由。"""
        p = gate.clamp_min(1e-9)
        h = -(p * p.log()).sum(-1)
        n_avail = (gate > 0).float().sum(-1).clamp_min(2.0)
        return (h / n_avail.log()).mean()

    def shared_load_loss(self, gate):
        """只在共享专家内部做负载均衡，防止共享专家整体死掉（不干预逐维差异化）。"""
        if self.n_shared < 2:
            return gate.sum() * 0.0
        imp = gate[..., :self.n_shared].mean(dim=(0, 1))
        return (imp.std() / (imp.mean() + 1e-9)) ** 2

    @staticmethod
    def expert_decorr(expert_out):
        """(B,K,E,out) → 专家输出的两两余弦相似度均值；惩罚它 = 逼专家学不同的函数。"""
        v = expert_out.mean(1).permute(1, 0, 2).reshape(expert_out.shape[2], -1)
        v = F.normalize(v, p=2, dim=-1)
        g = v @ v.T
        off = ~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
        return g[off].abs().mean()

    def decorr_loss(self, score, levels):
        """预测的维间相关不应超过真值的维间相关（对抗 rank-1 塌缩）。"""
        if score.shape[0] < 8:
            return score.sum() * 0.0

        def corr(x):
            x = x - x.mean(0, keepdim=True)
            x = x / (x.std(0, keepdim=True) + 1e-6)
            return x.T @ x / x.shape[0]
        cp, cg = corr(score), corr(levels.float())
        off = ~torch.eye(score.shape[1], dtype=torch.bool, device=score.device)
        return F.relu(cp[off].abs() - cg[off].abs()).mean()

    def aux(self, out, levels, c):
        return (c['w_ent'] * self.gate_entropy(out['gate'])
                + c['w_load'] * self.shared_load_loss(out['gate'])
                + c['w_exp'] * self.expert_decorr(out['expert_out'])
                + c['w_dec'] * self.decorr_loss(out['score'], levels))


# ---------------------------------------------------------------- routing 统计
def routing_stats(gate, avail):
    """gate: (B,K,E) numpy；返回逐维的路由统计。目标要求的机制证据就出自这里。

    * `dev_l1`   与均匀分布的 L1 距离（0=完全均匀，接近 1=完全尖锐）
    * `entropy`  归一化熵（1=均匀，0=one-hot）
    * `top1`     最常被选中的专家下标及其平均权重
    * `usage`    每个专家在该维上的平均权重
    """
    out = []
    for k in range(gate.shape[1]):
        g = gate[:, k, :]
        av = avail[k].astype(bool)
        n = int(av.sum())
        u = g[:, av]
        unif = 1.0 / n
        dev = float(np.abs(u - unif).sum(-1).mean() / (2 * (1 - unif)))
        p = np.clip(u, 1e-9, None)
        ent = float((-(p * np.log(p)).sum(-1) / np.log(n)).mean())
        idx = np.where(av)[0]
        mu = u.mean(0)
        out.append({'dim': DIMS[k], 'n_avail': n, 'dev_l1': dev, 'entropy': ent,
                    'top1_expert': int(idx[int(mu.argmax())]),
                    'top1_weight': float(mu.max()),
                    'usage': {int(i): round(float(v), 4) for i, v in zip(idx, mu)}})
    return out


def expert_similarity(expert_out):
    """(B,K,E,out) numpy → 专家输出两两余弦相似度矩阵（诊断同质化）。"""
    v = expert_out.mean(1).transpose(1, 0, 2).reshape(expert_out.shape[2], -1)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return v @ v.T
