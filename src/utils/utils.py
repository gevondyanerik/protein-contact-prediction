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


def train_val_collate(batch):
    chain_keys, starts, seqs, cmat_list, dmat_list, mask_list, full_sizes, msa_list = [], [], [], [], [], [], [], []
    for (chain_key, start, seq, cmat, dmat, mask, full_size, msa) in batch:
         chain_keys.append(chain_key)
         starts.append(start)
         seqs.append(seq)
         cmat_list.append(cmat)
         dmat_list.append(dmat)
         mask_list.append(mask)
         full_sizes.append(full_size)
         msa_list.append(msa)
    cmat_tensor = torch.from_numpy(np.array(cmat_list)).float()
    dmat_tensor = torch.from_numpy(np.array(dmat_list)).float()
    mask_tensor = torch.from_numpy(np.array(mask_list)).float()
    return chain_keys, starts, seqs, cmat_tensor, dmat_tensor, mask_tensor, full_sizes, msa_list

def convert_seqs_to_tokens(seq_batch, alphabet, msa_batch=None, msa_alphabet=None):
    """
    Tokenizes a list of sequences using the provided ESM alphabet.
    If msa_batch and msa_alphabet are provided, tokenizes the MSA as well.
    
    Args:
        seq_batch (list[str]): List of reference sequences.
        alphabet: The ESM alphabet for the reference model.
        msa_batch (list[list[str]], optional): List (per sample) of MSA sequences.
        msa_alphabet (optional): The ESM alphabet for the MSA model.
    
    Returns:
        If msa_batch is provided: (ref_tokens, msa_tokens)
        Otherwise: ref_tokens
    """
    batch_converter = alphabet.get_batch_converter()
    labeled = [(f"seq_{i}", seq) for i, seq in enumerate(seq_batch)]
    _, _, ref_tokens = batch_converter(labeled)
    if msa_batch is not None and msa_alphabet is not None:
        # For each sample, if the msa list is empty, default to the reference sequence.
        labeled_msa = []
        for i, msa in enumerate(msa_batch):
            if not msa:
                msa = [seq_batch[i]]
            # Create a list of (id, sequence) tuples for this sample.
            labeled_msa.append([(f"msa_{i}_{j}", s) for j, s in enumerate(msa)])
        msa_converter = msa_alphabet.get_batch_converter()
        _, _, msa_tokens = msa_converter(labeled_msa)
        return torch.tensor(ref_tokens), torch.tensor(msa_tokens)
    else:
        return torch.tensor(ref_tokens)