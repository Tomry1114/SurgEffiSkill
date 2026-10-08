import torch
import torch.nn.functional as F


def sor_loss(logits, y, sor_sigma=1.0, sor_lambda=0.25):
    lv = torch.arange(logits.shape[1], device=y.device, dtype=logits.dtype)
    d = lv.unsqueeze(0) - y.unsqueeze(1).to(logits.dtype)
    q = torch.softmax(-0.5 * d ** 2 / (sor_sigma ** 2), dim=1)
    kl = F.kl_div(F.log_softmax(logits, dim=1), q, reduction="batchmean")
    return F.cross_entropy(logits, y) + sor_lambda * kl
