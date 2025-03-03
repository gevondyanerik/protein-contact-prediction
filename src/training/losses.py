"""
Loss Functions for Protein Structure Prediction Tasks

This module defines various loss functions used in protein structure prediction tasks,
including contact map prediction, distance distogram prediction, and angle classification.

 Loss Functions Included:
1. Binary Cross-Entropy Loss for Contact Prediction (`contact_bce_loss`)
   - Computes the weighted BCE loss for binary contact maps.
   - Supports masking to exclude invalid positions.
   - Allows for different weights for positive and negative samples.

2. Focal Loss for Contact Prediction (`contact_focal_loss`)
   - A modified version of BCE loss that focuses more on hard-to-classify samples.
   - Uses a modulating factor `(1 - p_t)  gamma` to adjust the influence of easy vs. hard examples.
   - Suitable for highly imbalanced contact maps.

3. Distance Distogram Loss (`distance_distogram_loss`)
   - A categorical cross-entropy loss applied to distance bin classification.
   - Converts continuous distance values into discrete bins.
   - Supports masking to ignore padding positions.
   - If no bin edges are provided, it defaults to 21 bins spanning distances from 0 to 20 Å.

4. Angle Classification Loss (`angle_crossentropy_loss`)
   - A cross-entropy loss for predicting dihedral angles of protein backbones.
   - Discretizes angles into a fixed number of bins.
   - Supports masking to ignore invalid residues.

 Usage:
Each loss function is designed to be used within a PyTorch training pipeline
for multi-task learning in protein structure prediction models.

"""

import torch
import torch.nn.functional as F


def contact_bce_loss(
    contact_logits, contact_targets, mask=None, pos_weight=1.0, neg_weight=1.0
):
    """Computes weighted binary cross-entropy loss for contact map prediction.

    Args:
        contact_logits (Tensor): Logits from the model, shape (B, L, L).
        contact_targets (Tensor): Binary target contact map, shape (B, L, L).
        mask (Tensor, optional): Mask tensor indicating valid positions, shape (B, L, L). Defaults to None.
        pos_weight (float, optional): Weight for positive samples (contacts). Defaults to 1.0.
        neg_weight (float, optional): Weight for negative samples (non-contacts). Defaults to 1.0.

    Returns:
        Tensor: Scalar loss value.
    """

    if mask is not None:
        contact_logits = contact_logits[mask]
        contact_targets = contact_targets[mask]

    target = contact_targets.float()
    loss = F.binary_cross_entropy_with_logits(contact_logits, target, reduction="none")

    weight = torch.where(
        target == 1,
        torch.tensor(pos_weight, dtype=loss.dtype, device=loss.device),
        torch.tensor(neg_weight, dtype=loss.dtype, device=loss.device),
    )

    weighted_loss = loss * weight

    return weighted_loss.mean()


def contact_focal_loss(
    contact_logits, contact_targets, mask=None, gamma=2.0, alpha=0.25
):
    """Computes focal loss for contact map prediction.

    Args:
        contact_logits (Tensor): Logits from the model, shape (B, L, L).
        contact_targets (Tensor): Binary target contact map, shape (B, L, L).
        mask (Tensor, optional): Mask tensor indicating valid positions, shape (B, L, L). Defaults to None.
        gamma (float, optional): Modulation factor for difficult samples. Defaults to 2.0.
        alpha (float, optional): Scaling factor for positive class weighting. Defaults to 0.25.

    Returns:
        Tensor: Scalar loss value.
    """

    if mask is not None:
        contact_logits = contact_logits[mask]
        contact_targets = contact_targets[mask]

    bce_loss = F.binary_cross_entropy_with_logits(
        contact_logits,
        contact_targets.float(),
        reduction="none",
    )

    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss

    return focal_loss.mean()


def distance_distogram_loss(distance_preds, distance_targets, max_distance, mask=None):
    """Computes multi-class cross-entropy loss for distance distogram prediction.

    Args:
        distance_preds (Tensor): Logits over distance bins, shape (B, L, L, num_bins).
        distance_targets (Tensor): True distance values, shape (B, L, L).
        mask (Tensor, optional): Mask tensor indicating valid positions, shape (B, L, L). Defaults to None.

    Returns:
        Tensor: Scalar loss value.
    """

    num_bins = distance_preds.shape[-1]
    # Create default bin edges from 0 to max_distance.
    bins = torch.linspace(
        0, max_distance, steps=num_bins + 1, device=distance_targets.device
    )
    # Convert continuous distances into bin indices.
    target_bins = torch.bucketize(distance_targets, bins) - 1
    target_bins = target_bins.clamp(min=0, max=num_bins - 1)

    if mask is not None:
        distance_preds = distance_preds[mask]  # shape: [N, num_bins]
        target_bins = target_bins[mask]  # shape: [N]
    else:
        distance_preds = distance_preds.view(-1, num_bins)
        target_bins = target_bins.view(-1)

    loss = F.cross_entropy(distance_preds, target_bins.long(), reduction="mean")
    return loss


def angle_crossentropy_loss(angle_logits, angle_targets, mask=None):
    """Computes cross-entropy loss for angle prediction using discrete bins.

    Args:
        angle_logits (Tensor): Logits over angle bins, shape (B, L, 2, num_angle_bins).
        angle_targets (Tensor): True angle labels, shape (B, L, 2).
        mask (Tensor, optional): Mask tensor indicating valid positions, shape (B, L). Defaults to None.

    Returns:
        Tensor: Scalar loss value.
    """
    # Unpack shape: B=batch size, L=sequence length, 2=number of angles per residue, num_angle_bins=number of angle bins.
    B, L, two, num_angle_bins = angle_logits.shape

    if mask is not None:
        # Ensure mask is of shape (B, L) and then expand it to (B, L, 2).
        if mask.dim() == 3:
            mask = mask.squeeze(-1)

        mask = mask.unsqueeze(-1).expand(-1, -1, 2)

    # Flatten angle logits to shape (B * L * 2, num_angle_bins)
    angle_logits_flat = angle_logits.view(-1, num_angle_bins)
    # Flatten angle targets to shape (B * L * 2, )
    angle_targets_flat = angle_targets.view(-1)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        angle_logits_flat = angle_logits_flat[mask_flat]
        angle_targets_flat = angle_targets_flat[mask_flat]

    loss = F.cross_entropy(
        angle_logits_flat, angle_targets_flat.long(), reduction="mean"
    )
    return loss
