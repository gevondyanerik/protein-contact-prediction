import torch

def compute_metrics(preds_list, targets_list, threshold=0.5):
    total_tp = total_fp = total_tn = total_fn = 0
    for preds, targets in zip(preds_list, targets_list):
        bin_preds = (preds >= threshold).float()
        tp = ((bin_preds == 1) & (targets == 1)).sum().item()
        fp = ((bin_preds == 1) & (targets == 0)).sum().item()
        tn = ((bin_preds == 0) & (targets == 0)).sum().item()
        fn = ((bin_preds == 0) & (targets == 1)).sum().item()
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}