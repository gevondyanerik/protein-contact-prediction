"""
Utility Functions for Training and Configuration Management

This module provides helper functions for:
- Reproducibility: Setting random seeds for deterministic behavior.
- Configuration Handling: Loading and validating YAML-based configuration files.
- Model Training Setup: Ensuring required parameters are correctly defined.

Functions:

1. set_seed(seed: int) -> None
   - Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

2. load_config(path: str) -> dict
   - Loads a YAML configuration file and returns it as a Python dictionary.

3. check_block(config: dict, expected_types: dict, block_name: str) -> None
   - Validates the structure of a specific block in the config file.
   - Ensures the block contains the expected keys and values of correct types.

4. check_config(config: dict) -> None
   - Performs a comprehensive validation of the entire configuration file.
   - Checks for required fields and validates type consistency.

Configuration Validation:
The `check_config` function enforces the following:
- Dataset Configuration (`"dataset"`)
  - Includes sequence length, MSA settings, and batch sizes.
- Model Configuration (`"model"`)
  - Defines architecture details like transformer layers and fusion parameters.
- Training Configuration (`"train"`)
  - Controls loss functions, learning rate schedules, and optimization settings.
- Checkpointing Configuration (`"checkpoints"`)
  - Specifies paths and metrics for saving the best model checkpoints.
- MLflow Logging Configuration (`"mlflow"`)
  - Enables logging for experiment tracking.
"""

import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def check_block(config: dict, expected_types: dict, block_name: str) -> None:
    assert block_name in config, f"'{block_name}' block is expected in the config file."
    assert set(config[block_name].keys()) == set(
        expected_types.keys()
    ), f"The list of arguments in the {block_name} config block doesn't match the expected list: {list(expected_types.keys())}."
    for key, value in config[block_name].items():
        assert (
            type(config[block_name][key]) in expected_types[key]
        ), f"The '{key}' argument type doesn't match the expected list: {expected_types[key]}."


def check_config(config: dict) -> None:
    assert (
        "train_data_path" in config
    ), "'train_data_path' argument is expected in the config file."
    assert (
        "test_data_path" in config
    ), "'test_data_path' argument is expected in the config file."

    dataset_expected_types = {
        "max_length": [int],
        "max_distance": [int],
        "sliding_window_step": [int],
        "train_mode": [str],
        "val_mode": [str],
        "use_msa": [bool],
        "num_msa_rows": [int],
        "num_angle_bins": [int],
        "train_batch_size": [int],
        "val_batch_size": [int],
    }

    check_block(
        config=config, expected_types=dataset_expected_types, block_name="dataset"
    )

    model_expected_types = {
        "esm2_name": [str],
        "esm_msa_name": [str],
        "main_task": [str],
        "use_contact": [bool],
        "use_distance": [bool],
        "use_angle": [bool],
        "num_layers_to_freeze_esm2": [int],
        "num_layers_to_freeze_msa": [int],
        "fusion_dim": [int],
        "fusion_num_layers": [int],
        "fusion_num_heads": [int],
    }

    check_block(config=config, expected_types=model_expected_types, block_name="model")

    train_expected_types = {
        "seed": [int],
        "num_epochs": [int],
        "contact_loss_type": [str],
        "bce_contact_class_weights": [list],
        "learning_rate": [float],
        "cosine_annealing_warm_restart_t_0": [int, float],
        "cosine_annealing_warm_restart_t_mult": [int, float],
        "cosine_annealing_warm_restart_eta_min": [int, float],
        "focal_gamma": [float],
        "focal_alpha": [float],
        "contact_loss_weight": [float],
        "distance_loss_weight": [float],
        "angle_loss_weight": [float],
        "use_scheduler": [bool],
        "overfit_one_batch": [bool],
    }

    check_block(config=config, expected_types=train_expected_types, block_name="train")

    checkpoints_expected_types = {
        "checkpoint_path_to_load": [str, type(None)],
        "checkpoints_dir_path": [str],
        "checkpoints_metric": [str],
        "maximize_metric": [bool],
        "save_best_k_checkpoints": [int],
    }

    check_block(
        config=config,
        expected_types=checkpoints_expected_types,
        block_name="checkpoints",
    )

    mlflow_expected_types = {
        "enabled": [bool],
        "tracking_uri": [str],
        "experiment_name": [str],
        "run_name": [str],
    }

    check_block(
        config=config, expected_types=mlflow_expected_types, block_name="mlflow"
    )


def flatten_dict(d, parent_key="", sep="."):
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())

        else:
            items.append((new_key, value))

    return dict(items)
