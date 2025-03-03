"""
Contact Map Evaluation Metrics

This module provides functions for evaluating the performance of contact map predictions
in protein structure prediction tasks. The evaluation includes common classification
metrics as well as specialized ranking-based metrics used in structural biology.

Metrics Computed:
1. Accuracy - Overall correctness of contact map predictions.
2. Precision - Ratio of correctly predicted contacts (True Positives) to all predicted contacts.
3. Recall - Ratio of correctly predicted contacts (True Positives) to all actual contacts.
4. F1 Score - Harmonic mean of precision and recall, balancing false positives and false negatives.
5. Precision at Top-L (`top_L_precision`) - Measures the fraction of correct contacts in the top L predictions,
   where L is the sequence length divided by a configurable divisor.
6. Precision-Recall AUC (`pr_auc`) - Area under the precision-recall curve, capturing predictive performance.
7. ROC AUC (`roc_auc`) - Area under the Receiver Operating Characteristic curve, evaluating model discrimination.

Usage:
The function `compute_contact_metrics` is designed to work with contact map predictions
from deep learning models trained for protein contact prediction.

"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_contact_metrics(
    contact_preds, contact_targets, mask=None, threshold=0.5, top_L_divisor=2.0
):
    """Computes various metrics for evaluating predicted protein contact maps.

    Args:
        contact_preds (Tensor): Predicted contact map probabilities, shape (B, L, L).
        contact_targets (Tensor): Ground truth contact map, shape (B, L, L).
        mask (Tensor, optional): Mask indicating valid positions, shape (B, L, L). Defaults to None.
        threshold (float, optional): Probability threshold for converting predictions into binary contacts. Defaults to 0.5.
        top_L_divisor (float, optional): Controls `top_L_precision`, where L is divided by this value. Defaults to 2.0.

    Returns:
        dict: A dictionary containing the computed evaluation metrics:
            - `"accuracy"`: Overall classification accuracy.
            - `"precision"`: Contact prediction precision.
            - `"recall"`: Contact prediction recall.
            - `"f1"`: F1 score.
            - `"top_L_precision"`: Precision at top-L ranked contacts.
            - `"pr_auc"`: Precision-recall AUC.
            - `"roc_auc"`: ROC AUC score.
    """
    # Move tensors to CPU and detach (ensuring shapes remain: (B, L, L))
    contact_preds = contact_preds.cpu().detach()  # shape: (B, L, L)
    contact_targets = contact_targets.cpu()  # shape: (B, L, L)
    if mask is not None:
        mask = mask.cpu()  # shape: (B, L, L)

    # Flatten valid predictions for each batch.
    # For each batch b, if mask is provided, select elements where mask[b] is True.
    # Otherwise, flatten the entire (L, L) matrix.
    preds_flat = torch.cat(
        [
            contact_preds[b][mask[b]] if mask is not None else contact_preds[b].view(-1)
            for b in range(contact_preds.size(0))
        ]
    )  # shape: (N, ), where N is total number of valid positions across the batch.
    targs_flat = torch.cat(
        [
            (
                contact_targets[b][mask[b]]
                if mask is not None
                else contact_targets[b].view(-1)
            )
            for b in range(contact_targets.size(0))
        ]
    )  # shape: (N, )

    # Convert flattened tensors to numpy arrays.
    preds_np = preds_flat.numpy()  # shape: (N, )
    targs_np = targs_flat.numpy()  # shape: (N, )
    # Binarize predictions using the given threshold.
    pred_bin = (preds_np >= threshold).astype(np.int32)  # shape: (N, )
    targ_bin = targs_np.astype(np.int32)  # shape: (N, )

    # Compute confusion matrix components.
    tp = np.sum((pred_bin == 1) & (targ_bin == 1))
    fp = np.sum((pred_bin == 1) & (targ_bin == 0))
    tn = np.sum((pred_bin == 0) & (targ_bin == 0))
    fn = np.sum((pred_bin == 0) & (targ_bin == 1))
    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    # Compute PR-AUC and ROC-AUC.
    pr_auc = average_precision_score(targ_bin, preds_np)
    roc_auc = roc_auc_score(targ_bin, preds_np)
    # Compute top-L precision:
    # Sort predictions in descending order and compute precision for top L predictions.
    sorted_indices = np.argsort(-preds_np)  # shape: (N, )
    sorted_targs = targ_bin[sorted_indices]  # shape: (N, )
    total_pairs = sorted_targs.shape[0]
    top_L = max(1, min(total_pairs, int(total_pairs / top_L_divisor)))
    top_hits = np.sum(sorted_targs[:top_L])
    top_L_precision = top_hits / top_L

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top_L_precision": top_L_precision,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
