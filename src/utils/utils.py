# src/utils/utils.py
import yaml
import random
import torch
import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

# src/utils/collate_fn.py
import torch


def train_val_collate(batch):
    """
    Collates a list of items returned by ProteinDataset.
    Each item is a 7-tuple:
      (chain_key, start_idx, seq, contact_matrix, distance_matrix, mask, full_padded_size)
    Returns:
      - chain_keys: list of chain keys.
      - starts: list of start indices.
      - seqs: list of sequences (strings).
      - cmat_tensor: Tensor of shape (B, L+2, L+2)
      - dmat_tensor: Tensor of shape (B, L+2, L+2)
      - mask_tensor: Boolean tensor of shape (B, L+2, L+2)
      - full_sizes: list of full padded sizes.
    """
    chain_keys, starts, seqs, cmat_list, dmat_list, mask_list, full_sizes = [], [], [], [], [], [], []
    for (chain_key, start, seq, cmat, dmat, mask, full_size) in batch:
        chain_keys.append(chain_key)
        starts.append(start)
        seqs.append(seq)
        cmat_list.append(cmat)
        dmat_list.append(dmat)
        mask_list.append(mask)
        full_sizes.append(full_size)
    cmat_tensor = torch.from_numpy(np.array(cmat_list)).float()
    dmat_tensor = torch.from_numpy(np.array(dmat_list)).float()
    mask_tensor = torch.from_numpy(np.array(mask_list)).float()
    return chain_keys, starts, seqs, cmat_tensor, dmat_tensor, mask_tensor, full_sizes

def convert_seqs_to_tokens(seq_batch, alphabet):
    """
    Accepts a list of sequences (strings) and tokenizes them using the provided ESM alphabet.
    The ESM batch_converter returns a tuple (labels, strings, tokens); we return tokens.
    """
    batch_converter = alphabet.get_batch_converter()
    labeled = [(f"seq_{i}", seq) for i, seq in enumerate(seq_batch)]
    _, _, tokens = batch_converter(labeled)
    return tokens