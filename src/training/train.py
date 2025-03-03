"""
Protein Contact Prediction Training Script

This script trains a multi-task deep learning model for protein structure prediction using the ESM2 model.
It supports multiple tasks, including:
- Contact prediction: Predicts binary contact maps for protein residues.
- Distance prediction: Predicts distance distributions between residues.
- Angle prediction: Predicts backbone dihedral angles.

Features:
- Dataset Handling: Loads sequences, MSAs, contact maps, distance matrices, and angle matrices from an HDF5 dataset.
- Model Training: Uses a transformer-based ESM2 model with optional MSA-based embeddings.
- Loss Functions: Supports BCE, focal loss, and distogram loss.
- Evaluation Metrics: Computes precision, recall, F1 score, AUC, and precision at top L.
- Checkpointing: Saves and manages the best model checkpoints.
- MLflow Integration: Logs training metrics if enabled.

Training Pipeline:
1. Loads configuration settings.
2. Initializes the dataset and data loaders.
3. Loads the ESM2-based multi-task model.
4. Runs the training loop with loss computation and optimization.
5. Evaluates the model on a validation set after each epoch.
6. Logs metrics using MLflow (if enabled).
7. Saves model checkpoints based on validation performance.

Usage:
This script is executed from the command line and requires a YAML configuration file.
It supports optional one-batch overfitting for debugging.
"""

import os
import sys

import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import compute_contact_metrics

from models.multi_task_esm2_model import MultiTaskESM2Model
from training.losses import (
    angle_crossentropy_loss,
    contact_bce_loss,
    contact_focal_loss,
    distance_distogram_loss,
)
from utils.dataset import ESM2Dataset
from utils.utils import check_config, flatten_dict, load_config, set_seed


def train_pipeline(config):
    """
    Trains the multi-task ESM2 model on protein structure prediction tasks.

    The training pipeline performs the following steps:
      1. Validate and load configuration.
      2. Set random seeds for reproducibility.
      3. Initialize the training and validation datasets and data loaders.
         - Sequences are tokenized to shape [B, S] (B=batch, S=sequence length).
         - Contact and distance matrices are padded to shape [S + 2, S + 2].
         - Angle matrices are discretized and padded to shape [S + 2, 2].
      4. Instantiate the MultiTaskESM2Model and move it to the appropriate device.
      5. Optionally load a checkpoint and restore model and optimizer states.
      6. Define the optimizer and Cosine Annealing Warm Restarts scheduler.
      7. Run the training loop for a specified number of epochs.
         - Compute losses for contacts, distances, and angles.
         - Update the model weights.
      8. Evaluate on the validation set and compute evaluation metrics.
         - For the main task "contact", metrics are computed on the contact map.
         - For "distance", the distance prediction is converted into a binary contact map.
      9. Log metrics using MLflow and save checkpoints.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
    """

    check_config(config)

    set_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow_args = config["mlflow"]

    use_mlflow = mlflow_args["enabled"]
    print("MLflow Enabled..." if use_mlflow else "MLflow Disabled...")

    if use_mlflow:
        mlflow.set_tracking_uri(mlflow_args["tracking_uri"])
        mlflow.set_experiment(mlflow_args["experiment_name"])
        mlflow.start_run(run_name=mlflow_args["run_name"])
        mlflow.log_params(flatten_dict(config))

    dataset_args = config["dataset"]

    # Initialize training and validation datasets.
    train_dataset = ESM2Dataset(
        h5_file=config["train_data_path"],
        max_length=dataset_args["max_length"],
        max_distance=dataset_args["max_distance"],
        mode="train",
        train_mode=dataset_args["train_mode"],
        val_mode=None,
        sliding_window_step=dataset_args["sliding_window_step"],
        use_msa=dataset_args["use_msa"],
        num_msa_rows=dataset_args["num_msa_rows"],
        num_angle_bins=dataset_args["num_angle_bins"],
    )

    val_dataset = ESM2Dataset(
        h5_file=config["test_data_path"],
        max_length=dataset_args["max_length"],
        max_distance=dataset_args["max_distance"],
        mode="val",
        train_mode=None,
        val_mode=dataset_args["val_mode"],
        sliding_window_step=dataset_args["sliding_window_step"],
        use_msa=dataset_args["use_msa"],
        num_msa_rows=dataset_args["num_msa_rows"],
        num_angle_bins=dataset_args["num_angle_bins"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_args["train_batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_args["val_batch_size"],
        shuffle=False,
    )

    model_args = config["model"]

    model = MultiTaskESM2Model(
        esm2_name=model_args["esm2_name"],
        esm_msa_name=model_args["esm_msa_name"],
        use_msa=dataset_args["use_msa"],
        use_distance=model_args["use_distance"],
        use_angle=model_args["use_angle"],
        num_layers_to_freeze_esm2=model_args["num_layers_to_freeze_esm2"],
        num_layers_to_freeze_msa=model_args["num_layers_to_freeze_msa"],
        max_distance=dataset_args["max_distance"],
        num_angle_bins=dataset_args["num_angle_bins"],
        fusion_num_heads=model_args["fusion_num_heads"],
        fusion_dim=model_args["fusion_dim"],
    )

    model.to(device)

    checkpoints_args = config["checkpoints"]

    checkpoint_path_to_load = checkpoints_args.get("checkpoint_path_to_load", None)

    train_args = config["train"]

    overfit_one_batch = train_args["overfit_one_batch"]

    if overfit_one_batch:
        one_batch = next(iter(train_loader))
        print("One batch overfitting enabled...")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["learning_rate"])

    if checkpoint_path_to_load:
        checkpoint = torch.load(checkpoint_path_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path_to_load}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_args["cosine_annealing_warm_restart_t_0"],
        T_mult=train_args["cosine_annealing_warm_restart_t_mult"],
        eta_min=train_args["cosine_annealing_warm_restart_eta_min"],
    )

    best_checkpoints = []

    for epoch in range(1, train_args["num_epochs"] + 1):
        model.train()
        train_total_loss = 0.0

        train_loader = [one_batch] if overfit_one_batch else train_loader
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)

        for batch in train_pbar:
            sequence_tokens = batch["sequence_tokens"].to(device)

            if dataset_args["use_msa"]:
                msa_tokens = batch["msa_tokens"].to(device)

            else:
                msa_tokens = None

            contact_map = batch["contact_map"].to(device)
            distance_map = batch["distance_map"].to(device)
            angle_map = batch["angle_map"].to(device)
            contact_distance_mask = batch["contact_distance_mask"].to(device)
            angle_mask = batch["angle_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(sequence_tokens, msa_tokens)

            final_loss = 0

            if model_args["use_contact"]:
                c_preds = outputs["contact_logits"]

                if train_args["contact_loss_type"] == "focal":
                    c_loss = contact_focal_loss(
                        c_preds,
                        contact_map,
                        mask=contact_distance_mask,
                        gamma=train_args["focal_gamma"],
                        alpha=train_args["focal_alpha"],
                    )

                elif train_args["contact_loss_type"] == "bce":
                    c_loss = contact_bce_loss(
                        c_preds,
                        contact_map,
                        mask=contact_distance_mask,
                        pos_weight=train_args["bce_contact_class_weights"][1],
                        neg_weight=train_args["bce_contact_class_weights"][0],
                    )

                else:
                    raise Exception(
                        "'contact_loss_type' expected to be in ['bce', 'focal']"
                    )

                final_loss += train_args["contact_loss_weight"] * c_loss

            if model_args["use_distance"]:
                d_preds = outputs["distance_logits"]
                d_loss = distance_distogram_loss(
                    d_preds,
                    distance_map,
                    max_distance=dataset_args["max_distance"],
                    mask=contact_distance_mask,
                )
                final_loss += train_args["distance_loss_weight"] * d_loss

            else:
                d_loss = 0

            if model_args["use_angle"]:
                a_preds = outputs["angle_logits"]
                a_loss = angle_crossentropy_loss(a_preds, angle_map, mask=angle_mask)
                final_loss += train_args["angle_loss_weight"] * a_loss

            else:
                a_loss = 0

            final_loss.backward()
            optimizer.step()

            train_total_loss += final_loss.item()
            train_pbar.set_postfix(loss=f"{final_loss.item():.4f}")

        avg_train_loss = train_total_loss / (len(train_loader))
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")
        if use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        if train_args["use_scheduler"]:
            scheduler.step()

        model.eval()
        all_preds = []
        all_targs = []
        all_masks = []
        val_total_loss = 0.0

        with torch.no_grad():
            val_loader = [one_batch] if overfit_one_batch else val_loader
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", leave=False)
            for batch in val_pbar:
                sequence_tokens = batch["sequence_tokens"].to(device)

                if dataset_args["use_msa"]:
                    msa_tokens = batch["msa_tokens"].to(device)

                else:
                    msa_tokens = None

                contact_map = batch["contact_map"].to(device)
                distance_map = batch["distance_map"].to(device)
                angle_map = batch["angle_map"].to(device)
                contact_distance_mask = batch["contact_distance_mask"].to(device)
                angle_mask = batch["angle_mask"].to(device)

                optimizer.zero_grad()

                outputs = model(sequence_tokens, msa_tokens)

                c_preds = outputs["contact_logits"]

                if c_preds.dim() == 4 and c_preds.size(0) == 1:
                    c_preds = c_preds.squeeze(0)

                if train_args["contact_loss_type"] == "focal":
                    c_loss = contact_focal_loss(
                        c_preds,
                        contact_map,
                        mask=contact_distance_mask,
                        gamma=train_args["focal_gamma"],
                        alpha=train_args["focal_alpha"],
                    )

                elif train_args["contact_loss_type"] == "bce":
                    c_loss = contact_bce_loss(
                        c_preds, contact_map, mask=contact_distance_mask
                    )

                else:
                    raise Exception(
                        "'contact_loss_type' expected to be in ['bce', 'focal']"
                    )

                if model_args["use_distance"]:
                    d_preds = outputs["distance_logits"]
                    d_loss = distance_distogram_loss(
                        d_preds,
                        distance_map,
                        max_distance=dataset_args["max_distance"],
                        mask=contact_distance_mask,
                    )

                else:
                    d_loss = 0

                if model_args["use_angle"]:
                    a_preds = outputs["angle_logits"]
                    a_loss = angle_crossentropy_loss(
                        a_preds, angle_map, mask=angle_mask
                    )

                else:
                    a_loss = 0

                final_loss = (
                    train_args["contact_loss_weight"] * c_loss
                    + train_args["distance_loss_weight"] * d_loss
                    + train_args["angle_loss_weight"] * a_loss
                )

                val_total_loss += final_loss.item()
                val_pbar.set_postfix(loss=f"{final_loss.item():.4f}")

            avg_val_loss = val_total_loss / (len(val_loader))
            print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

            if model_args["main_task"] == "contact":
                preds = torch.sigmoid(outputs["contact_logits"])

            elif model_args["main_task"] == "distance":
                d_probs = F.softmax(outputs["distance_logits"], dim=-1)
                bin_width = 20.0 / d_probs.shape[-1]
                num_contact_bins = int(8.0 // bin_width)
                preds = d_probs[..., :num_contact_bins].sum(dim=-1)

            else:
                raise Exception("'main_task' expected to be in ['contact', 'distance']")

            all_preds.append(preds.cpu())
            all_targs.append(contact_map.cpu())
            all_masks.append(contact_distance_mask.cpu())

            if use_mlflow:
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            if overfit_one_batch:
                preds_tensor = all_preds[0]
                targs_tensor = all_targs[0]
                mask_tensor = all_masks[0]

            else:
                preds_tensor = torch.cat(all_preds, dim=0)
                targs_tensor = torch.cat(all_targs, dim=0)
                mask_tensor = torch.cat(all_masks, dim=0)

            metrics_dict = compute_contact_metrics(
                preds_tensor,
                targs_tensor,
                mask=mask_tensor,
                threshold=0.5,
            )

            for metric_name, metric_value in metrics_dict.items():
                print(f"Epoch {epoch} - Val {metric_name}: {metric_value:.4f}")
                if use_mlflow:
                    mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)

            checkpoints_dir = checkpoints_args["checkpoints_dir_path"]
            os.makedirs(checkpoints_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoints_dir, f"checkpoint_epoch_{epoch}.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics_dict,
                },
                checkpoint_path,
            )

            print(f"Saved checkpoint: {checkpoint_path}")

            current_metric_value = metrics_dict[checkpoints_args["checkpoints_metric"]]
            maximize = checkpoints_args["maximize_metric"]
            best_checkpoints.append((current_metric_value, epoch, checkpoint_path))

            if maximize:
                best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            else:
                best_checkpoints.sort(key=lambda x: x[0], reverse=False)

            save_best_k = checkpoints_args["save_best_k_checkpoints"]

            if len(best_checkpoints) > save_best_k:
                worst = best_checkpoints.pop()
                worst_path = worst[2]

                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"Removed old checkpoint: {worst_path}")

    if use_mlflow:
        mlflow.end_run()


def main():
    config = load_config("./src/config.yaml")
    train_pipeline(config)


if __name__ == "__main__":
    main()
