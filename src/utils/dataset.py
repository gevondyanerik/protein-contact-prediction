import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import h5py
from tqdm import tqdm

class ProteinDataset(Dataset):
    """
    A dataset for protein contact/distance/angle prediction.
    Reads from an HDF5 file (group "chains") where each chain has:
      - Attribute "sequence"
      - Datasets "target_matrix" (contact map) and "distance_matrix" (distance map)
    The dataset builds an index (or sliding windows for validation) and returns a tuple:
      (chain_key, start_idx, seq, contact_matrix, distance_matrix, mask, full_padded_size)
    Here the target matrices are padded from (L, L) to (L+2, L+2) to reserve positions for special tokens.
    
    Training mode:
      - If len(seq) > max_input_length, truncate to max_input_length (pad_special behavior).
      - Otherwise, apply random crop.
    
    Validation mode:
      The behavior is determined by `val_mode`:
        - "sliding_window": index_list contains windows and full_padded_size is returned.
        - "random_crop": random crop is applied.
        - "truncate_seq": sequence is simply truncated.
    """
    def __init__(self, h5_file, max_input_length=100, mode="train", val_mode="truncate_seq", step=20):
        self.h5_file = h5_file
        self.max_input_length = max_input_length
        self.mode = mode  # "train" or "val"
        self.val_mode = val_mode  
        self.step = step

        self.index_list = []
        with h5py.File(h5_file, 'r') as f:
            chains_group = f["chains"]
            for key in tqdm(chains_group.keys(), desc=f"Indexing {h5_file}"):
                grp = chains_group[key]
                seq = grp.attrs["sequence"]
                if isinstance(seq, bytes):
                    seq = seq.decode("utf-8")
                L = len(seq)
                full_padded_size = L + 2  # full size after special token padding
                if self.mode == "val" and self.val_mode == "sliding_window" and L > self.max_input_length:
                    start = 0
                    while start + self.max_input_length <= L:
                        self.index_list.append((key, start, self.max_input_length, True, full_padded_size))
                        start += self.step
                else:
                    self.index_list.append((key, 0, L, False, full_padded_size))
        self.length = len(self.index_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chain_key, start_idx, sub_len, is_subwin, full_padded_size = self.index_list[idx]
        with h5py.File(self.h5_file, 'r') as f:
            grp = f["chains"][chain_key]
            seq = grp.attrs["sequence"]
            if isinstance(seq, bytes):
                seq = seq.decode("utf-8")
            cmat = grp["target_matrix"]
            dmat = grp["distance_matrix"]
            if is_subwin:
                end_idx = start_idx + sub_len
                sub_seq = seq[start_idx:end_idx]
                sub_cmat = cmat[start_idx:end_idx, start_idx:end_idx]
                sub_dmat = dmat[start_idx:end_idx, start_idx:end_idx]
                seq = sub_seq
                cmat_np = sub_cmat[()]
                dmat_np = sub_dmat[()]
            else:
                cmat_np = cmat[()]
                dmat_np = dmat[()]
        
        if self.mode == "train":
            if len(seq) > self.max_input_length:
                # Truncate (pad_special behavior)
                seq = seq[:self.max_input_length]
                cmat_np = cmat_np[:self.max_input_length, :self.max_input_length]
                dmat_np = dmat_np[:self.max_input_length, :self.max_input_length]
            else:
                seq, cmat_np, dmat_np = self._random_crop(seq, cmat_np, dmat_np)
            seq = self._pad_seq(seq)
            cmat_np, dmat_np = self._pad_matrices(cmat_np, dmat_np)
        else:
            if self.val_mode == "truncate_seq" and len(seq) > self.max_input_length:
                seq = seq[:self.max_input_length]
                cmat_np = cmat_np[:self.max_input_length, :self.max_input_length]
                dmat_np = dmat_np[:self.max_input_length, :self.max_input_length]
            elif self.val_mode == "random_crop":
                seq, cmat_np, dmat_np = self._random_crop(seq, cmat_np, dmat_np)
            # In sliding_window mode, index_list already provides windows.
            seq = self._pad_seq(seq)
            cmat_np, dmat_np = self._pad_matrices(cmat_np, dmat_np)
        
        cmat_np, dmat_np = self._pad_target_special(cmat_np, dmat_np)
        mask = self._build_special_mask(cmat_np.shape[0])
        return chain_key, start_idx, seq, cmat_np, dmat_np, mask, full_padded_size

    def _random_crop(self, seq, cmat_np, dmat_np):
        L = len(seq)
        if L > self.max_input_length:
            start_idx = random.randint(0, L - self.max_input_length)
            end_idx = start_idx + self.max_input_length
            sub_seq = seq[start_idx:end_idx]
            sub_cmat = cmat_np[start_idx:end_idx, start_idx:end_idx]
            sub_dmat = dmat_np[start_idx:end_idx, start_idx:end_idx]
            return sub_seq, sub_cmat, sub_dmat
        else:
            return seq, cmat_np, dmat_np

    def _pad_seq(self, seq):
        if len(seq) < self.max_input_length:
            seq = seq + ("X" * (self.max_input_length - len(seq)))
        return seq

    def _pad_matrices(self, cmat_np, dmat_np):
        size = cmat_np.shape[0]
        if size < self.max_input_length:
            pad_width = ((0, self.max_input_length - size), (0, self.max_input_length - size))
            cmat_np = np.pad(cmat_np, pad_width, mode='constant', constant_values=0)
            dmat_np = np.pad(dmat_np, pad_width, mode='constant', constant_values=0)
        return cmat_np, dmat_np

    def _pad_target_special(self, cmat_np, dmat_np):
        L = cmat_np.shape[0]
        new_size = L + 2
        new_cmat = np.zeros((new_size, new_size), dtype=cmat_np.dtype)
        new_dmat = np.zeros((new_size, new_size), dtype=dmat_np.dtype)
        new_cmat[1:L+1, 1:L+1] = cmat_np
        new_dmat[1:L+1, 1:L+1] = dmat_np
        return new_cmat, new_dmat

    def _build_special_mask(self, size):
        mask = np.ones((size, size), dtype=bool)
        mask[0, :] = False
        mask[size-1, :] = False
        mask[:, 0] = False
        mask[:, size-1] = False
        return mask