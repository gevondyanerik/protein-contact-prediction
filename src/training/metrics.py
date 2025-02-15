import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def compute_metrics(preds_list, targets_list, threshold=0.5, k=10):
    """
    Computes overall accuracy, precision, recall, F1, Top L/k Precision, ROC-AUC, and PR-AUC.
    
    For each sample, Top L/k Precision is computed as follows:
      - Let L be the number of residues (i.e. the first dimension of the prediction).
      - Compute n = max(1, L/k).
      - Take the top n predictions (by raw score, not thresholded) and calculate the fraction 
        of those that are true contacts.
    
    ROC-AUC and PR-AUC are computed per sample on the flattened prediction and target matrices.
    
    Args:
        preds_list (list[Tensor]): List of predicted contact maps (shape: (L, L)).
        targets_list (list[Tensor]): List of ground truth contact maps (shape: (L, L)), binary.
        threshold (float): Threshold to binarize predictions (for standard metrics).
        k (int): Divisor to determine how many top predictions to consider for Top L/k Precision.
    
    Returns:
        dict: A dictionary with keys 'accuracy', 'precision', 'recall', 'f1',
              'Top L/k Precision (k={k})', 'ROC-AUC', and 'PR-AUC'.
    """
    total_tp = total_fp = total_tn = total_fn = 0
    top_lk_precisions = []
    auc_list = []
    pr_auc_list = []
    
    for preds, targets in zip(preds_list, targets_list):
        # Standard metrics: binarize predictions using threshold.
        bin_preds = (preds >= threshold).float()
        tp = ((bin_preds == 1) & (targets == 1)).sum().item()
        fp = ((bin_preds == 1) & (targets == 0)).sum().item()
        tn = ((bin_preds == 0) & (targets == 0)).sum().item()
        fn = ((bin_preds == 0) & (targets == 1)).sum().item()
        
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        
        # Compute Top L/k Precision for this sample.
        L = preds.shape[0]
        n = max(1, int(L / k))
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        top_indices = torch.topk(preds_flat, n).indices
        top_targets = targets_flat[top_indices]
        sample_top_lk_precision = top_targets.sum().item() / n
        top_lk_precisions.append(sample_top_lk_precision)
        
        # Compute ROC-AUC for this sample.
        preds_np = preds_flat.detach().cpu().numpy()
        targets_np = targets_flat.detach().cpu().numpy()
        if len(np.unique(targets_np)) < 2:
            auc_val = 0.0
            pr_auc_val = 0.0
        else:
            auc_val = roc_auc_score(targets_np, preds_np)
            pr_auc_val = average_precision_score(targets_np, preds_np)
        auc_list.append(auc_val)
        pr_auc_list.append(pr_auc_val)
    
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_top_lk_precision = sum(top_lk_precisions) / len(top_lk_precisions) if top_lk_precisions else 0
    avg_auc = sum(auc_list) / len(auc_list) if auc_list else 0
    avg_pr_auc = sum(pr_auc_list) / len(pr_auc_list) if pr_auc_list else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc-auc": avg_auc,
        "pr-auc": avg_pr_auc,
        "top-L/10-precision": avg_top_lk_precision,
    }