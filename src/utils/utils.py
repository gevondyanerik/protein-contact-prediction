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
    Converts a batch of sequences into tokens using the provided alphabet’s batch_converter.
    
    If msa_batch and msa_alphabet are provided, then for each sample in the batch, the
    corresponding MSA (a list of sequences) is tokenized using the msa_alphabet’s batch_converter.
    
    Args:
        seq_batch (list of str): List of reference sequences.
        alphabet: ESM alphabet (for the reference model).
        msa_batch (list of list of str, optional): For each sample, a list of MSA sequences.
        msa_alphabet: ESM alphabet for the MSA model.
    
    Returns:
        If msa_batch and msa_alphabet are provided:
            (ref_tokens, msa_tokens) where:
                - ref_tokens is a tensor of shape (B, L+2)
                - msa_tokens is a tensor of shape (B, num_msa, L+2)
        Otherwise:
            ref_tokens (tensor) of shape (B, L+2)
    """
    # Tokenize reference sequences.
    batch_converter = alphabet.get_batch_converter()
    labeled = [(f"seq_{i}", seq) for i, seq in enumerate(seq_batch)]
    _, _, ref_tokens = batch_converter(labeled)
    
    if msa_batch is not None and msa_alphabet is not None:
        msa_tokens_list = []
        msa_batch_converter = msa_alphabet.get_batch_converter()
        # For each sample, tokenize its list of MSA sequences.
        for i, msa in enumerate(msa_batch):
            # Each msa is expected to be a list of sequences.
            labeled_msa = [(f"msa_{i}_{j}", seq) for j, seq in enumerate(msa)]
            # The batch converter returns a tuple; we only need the tokens.
            _, _, msa_tokens = msa_batch_converter(labeled_msa)
            # msa_tokens shape: (num_msa, L+2)
            msa_tokens_list.append(msa_tokens)
        # Stack the token tensors into shape (B, num_msa, L+2)
        msa_tokens = torch.stack(msa_tokens_list, dim=0)
        return ref_tokens, msa_tokens
    else:
        return ref_tokens