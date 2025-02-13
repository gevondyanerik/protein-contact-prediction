import torch
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