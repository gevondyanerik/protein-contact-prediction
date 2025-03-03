"""
Protein Structure Dataset for ESM2 Model

This dataset class handles protein sequences, multiple sequence alignments (MSA),
and structural features such as contact maps, distance matrices, and angle matrices
for use with the ESM2 model.

Features:
- Sequence Processing: Extracts and tokenizes protein sequences.
- Sliding Window Support: Handles long sequences by applying sliding window cropping.
- MSA Support: Loads and processes multiple sequence alignments.
- Structural Features:
  - Contact map (binary interaction matrix).
  - Distance matrix (pairwise residue distances).
  - Angle matrix (backbone dihedral angles).
- Dynamic Masking: Generates valid region masks for training.

Arguments:
- `h5_file (str)`: Path to the HDF5 dataset file.
- `mode (str)`: Dataset mode (`"train"` or `"val"`).
- `train_mode (str)`: Training cropping strategy (`"truncate_seq"` or `"random_crop"`).
- `val_mode (str)`: Validation cropping strategy (`"truncate_seq"` or `"sliding_window"`).
- `max_length (int)`: Maximum sequence length for processing.
- `max_distance (int)`: Maximum distance threshold for structural features.
- `sliding_window_step (int)`: Step size for sliding window in validation mode.
- `use_msa (bool)`: Whether to use multiple sequence alignments.
- `num_msa_rows (int)`: Number of MSA sequences to include.
- `num_angle_bins (int)`: Number of bins for discretizing angle values.

Workflow:
1. Loads sequences and MSAs from an HDF5 file.
2. Applies truncation, cropping, or sliding window strategies.
3. Extracts contact, distance, and angle matrices.
4. Tokenizes sequences and MSAs for input to ESM2.
5. Returns structured data for model training.
"""

import random

import esm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ESM2Dataset(Dataset):
    def __init__(
        self,
        h5_file: str,
        mode: str,
        train_mode: str,
        val_mode: str,
        max_length: int,
        max_distance: int,
        sliding_window_step: int,
        use_msa: bool,
        num_msa_rows: int,
        num_angle_bins: int,
    ):
        self.h5_file = h5_file
        self.mode = mode
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.max_length = max_length
        self.max_distance = max_distance
        self.sliding_window_step = sliding_window_step
        self.use_msa = use_msa
        self.num_msa_rows = num_msa_rows
        self.num_angle_bins = num_angle_bins
        self.alphabet = esm.Alphabet.from_architecture("ESM-1")

        self.index_list = []
        # Open the HDF5 file and build an index of available chains.
        with h5py.File(self.h5_file, "r") as file:
            chains_group = file["chains"]

            for key in tqdm(chains_group.keys(), desc=f"Indexing {h5_file}"):
                group = chains_group[key]
                # Decode sequence attribute.
                sequence = (
                    group.attrs["sequence"].decode("utf-8")
                    if isinstance(group.attrs["sequence"], bytes)
                    else group.attrs["sequence"]
                )

                length_sequence = len(sequence)

                # Apply cropping or sliding window strategy.
                if length_sequence > self.max_length:
                    if self.mode == "train":
                        if self.train_mode in ["truncate_seq", "random_crop"]:
                            self.index_list.append((key, 0, self.max_length))

                        else:
                            raise Exception(
                                "'train_mode' expected to be in ['truncate_seq', 'random_crop']"
                            )

                    elif self.mode == "val":
                        if self.val_mode == "truncate_seq":
                            self.index_list.append((key, 0, self.max_length))

                        elif self.val_mode == "sliding_window":
                            start = 0
                            while start + self.max_length <= length_sequence:
                                self.index_list.append((key, start, self.max_length))
                                start += self.sliding_window_step

                        else:
                            raise Exception(
                                "'train_mode' expected to be in ['truncate_seq', 'random_crop']"
                            )

                    else:
                        raise Exception("'mode' expected to be in ['train', 'val']")

                else:
                    self.index_list.append((key, 0, length_sequence))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        # Retrieve index information: key, start_index, and effective length.
        key, start_index, length = self.index_list[index]
        # Open the HDF5 file to load data for the given chain.
        with h5py.File(self.h5_file, "r") as file:
            group = file["chains"][key]
            sequence = (
                group.attrs["sequence"].decode("utf-8")
                if isinstance(group.attrs["sequence"], bytes)
                else group.attrs["sequence"]
            )
            msa_list = [
                msa.decode("utf-8") if isinstance(msa, bytes) else msa
                for msa in group.attrs["msa"]
            ]

            cmat = group["contact_matrix"][
                ()
            ]  # Shape: [original_length, original_length]
            dmat = group["distance_matrix"][
                ()
            ]  # Shape: [original_length, original_length]
            amat = group["angle_matrix"][()]  # Shape: [original_length, 2]

        # Crop or random crop if sequence length exceeds max_length.
        if len(sequence) > self.max_length:
            if self.mode == "train" and self.train_mode == "random_crop":
                start_index = random.randint(0, len(sequence) - self.max_length)

            sequence = sequence[start_index : start_index + self.max_length]
            cmat = cmat[
                start_index : start_index + self.max_length,
                start_index : start_index + self.max_length,
            ]
            dmat = dmat[
                start_index : start_index + self.max_length,
                start_index : start_index + self.max_length,
            ]
            amat = amat[start_index : start_index + self.max_length, :]

            if self.use_msa:
                msa_list = [
                    msa[start_index : start_index + self.max_length] for msa in msa_list
                ]

        # Tokenize sequence: add CLS and EOS tokens.
        sequence_tokens = [self.alphabet.cls_idx]
        sequence_tokens += [
            self.alphabet.tok_to_idx.get(amino_acid, self.alphabet.unk_idx)
            for amino_acid in sequence
        ]
        pad_length = self.max_length - len(sequence)
        sequence_tokens += [self.alphabet.unk_idx] * pad_length
        sequence_tokens.append(self.alphabet.eos_idx)

        # Process MSA tokens if enabled.
        if self.use_msa:
            msa_tokens_list = []

            for msa in msa_list:
                msa_row = msa[: self.max_length]
                msa_tokens = (
                    [self.alphabet.cls_idx]
                    + [
                        self.alphabet.tok_to_idx.get(aa, self.alphabet.unk_idx)
                        for aa in msa_row
                    ]
                    + [self.alphabet.eos_idx]
                )

                pad_length = (self.max_length + 2) - len(msa_tokens)

                if pad_length > 0:
                    msa_tokens += [self.alphabet.unk_idx] * pad_length

                msa_tokens_list.append(msa_tokens)

            # Ensure exactly num_msa_rows are returned.
            if len(msa_tokens_list) < self.num_msa_rows:
                pad_row = (
                    [self.alphabet.cls_idx]
                    + [self.alphabet.unk_idx] * self.max_length
                    + [self.alphabet.eos_idx]
                )

                while len(msa_tokens_list) < self.num_msa_rows:
                    msa_tokens_list.append(pad_row)

            else:
                msa_tokens_list = msa_tokens_list[: self.num_msa_rows]

        # Determine padded matrix size: original sequence length (after cropping) + 2 (for CLS and EOS).
        padded_size = self.max_length + 2
        # Pad contact matrix and distance matrix to shape [padded_size, padded_size].
        padded_cmat = np.zeros((padded_size, padded_size), dtype=cmat.dtype)
        valid_length = len(sequence)
        padded_cmat[1 : valid_length + 1, 1 : valid_length + 1] = cmat
        padded_dmat = np.zeros((padded_size, padded_size), dtype=dmat.dtype)
        padded_dmat[1 : valid_length + 1, 1 : valid_length + 1] = dmat

        # Discretize the angle matrix into bins.
        # bin_edges: array of shape [num_angle_bins + 1]
        bin_edges = np.linspace(-np.pi, np.pi, self.num_angle_bins + 1)
        # discrete_amat: shape [S, 2] with values in [0, num_angle_bins - 1]
        discrete_amat = np.digitize(amat, bin_edges) - 1
        # Pad angle matrix to shape [padded_size, 2].
        padded_amat = np.zeros((padded_size, 2), dtype=np.int64)
        padded_amat[1 : valid_length + 1, :] = discrete_amat
        # Create masks for contact/distance matrices (shape: [padded_size, padded_size])
        mask_cmat_dmat = np.zeros((padded_size, padded_size), dtype=bool)
        mask_cmat_dmat[1 : valid_length + 1, 1 : valid_length + 1] = True
        # Create mask for angle matrix (shape: [padded_size])
        mask_amat = np.zeros((padded_size,), dtype=bool)
        mask_amat[1 : valid_length + 1] = True

        # Convert everything to torch tensors.
        sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.long)

        if self.use_msa:
            msa_tokens = torch.tensor(msa_tokens_list, dtype=torch.long)

        padded_cmat = torch.from_numpy(padded_cmat)
        padded_dmat = torch.from_numpy(padded_dmat)
        padded_amat = torch.from_numpy(padded_amat)
        mask_cmat_dmat = torch.from_numpy(mask_cmat_dmat)
        mask_amat = torch.from_numpy(mask_amat)

        sample = {
            "sequence_tokens": sequence_tokens,  # Shape: [S + 2]
            "msa_tokens": msa_tokens if self.use_msa else [],
            "contact_map": padded_cmat,  # Shape: [S + 2, S + 2]
            "distance_map": padded_dmat,  # Shape: [S + 2, S + 2]
            "angle_map": padded_amat,  # Shape: [S + 2, 2]
            "contact_distance_mask": mask_cmat_dmat,  # Shape: [S + 2, S + 2]
            "angle_mask": mask_amat,  # Shape: [S + 2]
        }

        return sample
