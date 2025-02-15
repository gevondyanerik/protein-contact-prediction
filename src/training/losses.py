import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _select_contact_output(pred):
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred

def contact_loss_bce(pred, target, mask=None, pos_weight=None, neg_weight=None):
    pred = _select_contact_output(pred)
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=pred.device)
        else:
            mask = mask.bool()
        pred = torch.masked_select(pred, mask)
        target = torch.masked_select(target, mask)
    if pos_weight is not None or neg_weight is not None:
        pos_w = pos_weight if pos_weight is not None else 1.0
        neg_w = neg_weight if neg_weight is not None else 1.0
        weight = target * pos_w + (1 - target) * neg_w
        loss = F.binary_cross_entropy(pred, target, weight=weight)
    else:
        loss = F.binary_cross_entropy(pred, target)
    return loss

def focal_loss(pred, target, gamma=2.0, pos_alpha=None, neg_alpha=None, mask=None):
    pred = _select_contact_output(pred)
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=pred.device)
        else:
            mask = mask.bool()
        pred = torch.masked_select(pred, mask)
        target = torch.masked_select(target, mask)
    eps = 1e-8
    pred = pred.clamp(min=eps, max=1-eps)
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_factor = (1 - pt) ** gamma
    loss = -torch.log(pt)
    if pos_alpha is not None and neg_alpha is not None:
        alpha_weight = target * pos_alpha + (1 - target) * neg_alpha
        loss = alpha_weight * focal_factor * loss
    else:
        loss = focal_factor * loss
    return loss.mean()

def mse_loss_distance(pred, target, mask=None):
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=pred.device)
        else:
            mask = mask.bool()
        pred = torch.masked_select(pred, mask)
        target = torch.masked_select(target, mask)
    return F.mse_loss(pred, target)

def angle_loss(pred, target, mask=None):
    diff = torch.abs(pred - target)
    diff = torch.min(diff, 2 * math.pi - diff)
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=pred.device)
        else:
            mask = mask.bool()
        diff = torch.masked_select(diff, mask)
    return torch.mean(diff ** 2)


class MultiTaskLossWrapper(nn.Module):
    """
    A loss wrapper that learns uncertainty parameters for each task.
    For num_tasks losses L1, L2, ..., L_n the combined loss is:
      L = 0.5 * exp(-s1) * L1 + 0.5 * s1 +
          0.5 * exp(-s2) * L2 + 0.5 * s2 + ... 
    where s_i are learnable log-variances.
    """
    def __init__(self, num_tasks=3):
        super().__init__()
        # Initialize log variances to 0 (i.e. sigma = 1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += 0.5 * torch.exp(-self.log_vars[i]) * loss + 0.5 * self.log_vars[i]
        return total_loss
