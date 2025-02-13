#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the src folder to sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import load_config, set_seed, train_val_collate, convert_seqs_to_tokens
from utils.dataset import ProteinDataset
from training.losses import contact_loss_bce, focal_loss, mse_loss_distance, angle_loss
from training.metrics import compute_metrics

# Import model types.
from models.esm2_contact_predictor import ESM2ContactPredictor
from models.esm2_contact_distance_predictor import ESM2ContactDistancePredictor
from models.esm2_contact_distance_angle_predictor import ESM2ContactDistanceAnglePredictor
from models.esm2_contact_angle_predictor import ESM2ContactAnglePredictor


def train_pipeline(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])
    
    mlflow_cfg = config.get("mlflow", {})
    use_mlflow = mlflow_cfg.get("enabled", False)
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "Default"))
        mlflow.start_run(run_name=mlflow_cfg.get("run_name", None))
        mlflow.log_params(config)
    
    # Create datasets.
    train_dataset = ProteinDataset(
        h5_file=config["train_data_path"],
        max_input_length=config["max_input_length"],
        mode="train",
        val_mode=config.get("val_mode", "truncate_seq")
    )
    val_dataset = ProteinDataset(
        h5_file=config["test_data_path"],
        max_input_length=config["max_input_length"],
        mode="val",
        val_mode=config.get("val_mode", "sliding_window")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=train_val_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=train_val_collate
    )
    
    overfit_one_batch = config.get("overfit_one_batch", False)
    if overfit_one_batch:
        one_batch_train = next(iter(train_loader), None)
        one_batch_val = one_batch_train
    else:
        one_batch_train = None
        one_batch_val = None
    
    model_type = config.get("model_type", "esm2_contact_predictor")
    num_layers_to_freeze = config.get("num_layers_to_freeze", None)
    if model_type == "esm2_contact_predictor":
        model = ESM2ContactPredictor(
            esm_model_name=config["esm_model_name"],
            num_layers_to_freeze=num_layers_to_freeze
        )
    elif model_type == "esm2_contact_distance_predictor":
        model = ESM2ContactDistancePredictor(
            esm_model_name=config["esm_model_name"],
            num_layers_to_freeze=num_layers_to_freeze,
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0)
        )
    elif model_type == "esm2_contact_distance_angle_predictor":
        model = ESM2ContactDistanceAnglePredictor(
            esm_model_name=config["esm_model_name"],
            num_layers_to_freeze=num_layers_to_freeze,
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0),
            fusion_source=config.get("fusion_source", "distance"),
            fusion_d_model=config.get("fusion_d_model", 64),
            fusion_num_heads=config.get("fusion_num_heads", 4)
        )
    elif model_type == "esm2_contact_angle_predictor":
        model = ESM2ContactAnglePredictor(
            esm_model_name=config["esm_model_name"],
            num_layers_to_freeze=num_layers_to_freeze,
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0)
        )
    else:
        raise ValueError("Unsupported model_type: " + model_type)

    model.to(device)
    mlflow.pytorch.log_model(model, "model")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_type = config.get("loss_type", "bce")
    gamma = config.get("focal_gamma", 2.0)

    checkpoint_path_to_load = config.get("checkpoint_path_to_load", None)
    if checkpoint_path_to_load:
        checkpoint = torch.load(checkpoint_path_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    cw = config.get("contact_class_weights", None)
    if cw and len(cw) == 2:
        neg_weight, pos_weight = cw
    else:
        neg_weight, pos_weight = None, None
    
    contact_loss_weight = config.get("contact_loss_weight", 1.0)
    distance_loss_weight = config.get("distance_loss_weight", 1.0)
    angle_loss_weight = config.get("angle_loss_weight", 0.0)
    
    reweight_contact_loss = config.get("reweight_contact_loss", False)
    distance_to_weight_scale = config.get("distance_to_weight_scale", 10.0)
    
    num_epochs = config["num_epochs"]
    eval_interval = config["eval_interval"]
    
    # For sliding_window validation, precompute full sizes per chain.
    full_sizes = {}
    if config.get("val_mode", "truncate_seq") == "sliding_window":
        for idx in range(len(val_dataset)):
            chain_key, _, seq, cmat, dmat, mask, full_size = val_dataset[idx]
            full_sizes[chain_key] = full_size
    
    best_checkpoints = []

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_iter = [one_batch_train] if (overfit_one_batch and one_batch_train is not None) else train_loader
        train_pbar = tqdm(train_iter, desc=f"Epoch {epoch} [TRAIN]", leave=False)
        for chain_keys, starts, seq_batch, cmat_batch, dmat_batch, mask_batch, _ in train_pbar:
            cmat_batch = cmat_batch.to(device)
            dmat_batch = dmat_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            tokens = convert_seqs_to_tokens(seq_batch, model.alphabet).to(device)
            
            if model_type in ["esm2_contact_distance_predictor", "esm2_contact_distance_angle_predictor", "esm2_contact_angle_predictor"]:
                outputs = model(tokens)
                if model_type == "esm2_contact_distance_predictor":
                    contact_probs, distance_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=gamma,
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    d_loss = mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                    loss = contact_loss_weight * c_loss + distance_loss_weight * d_loss
                elif model_type == "esm2_contact_distance_angle_predictor":
                    contact_probs, distance_preds, angle_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=gamma,
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    d_loss = mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                    a_loss = angle_loss(angle_preds, cmat_batch, mask=mask_batch)
                    loss = (contact_loss_weight * c_loss +
                            distance_loss_weight * d_loss +
                            angle_loss_weight * a_loss)
                elif model_type == "esm2_contact_angle_predictor":
                    contact_probs, angle_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=gamma,
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    a_loss = angle_loss(angle_preds, cmat_batch, mask=mask_batch)
                    loss = contact_loss_weight * c_loss + angle_loss_weight * a_loss
            else:
                preds = model(tokens)
                if loss_type == "focal":
                    loss = focal_loss(preds, cmat_batch, gamma=gamma,
                                      pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                else:
                    loss = contact_loss_bce(preds, cmat_batch, mask=mask_batch,
                                            pos_weight=pos_weight, neg_weight=neg_weight)
            
            if reweight_contact_loss and model_type in ["esm2_contact_distance_predictor", "esm2_contact_distance_angle_predictor", "esm2_contact_angle_predictor"]:
                reweight_factor = torch.exp(-dmat_batch / distance_to_weight_scale)
                reweight_factor = torch.masked_select(reweight_factor, mask_batch)
                factor = reweight_factor.mean()
                if model_type == "esm2_contact_distance_predictor":
                    loss = contact_loss_weight * (loss / contact_loss_weight * factor) + distance_loss_weight * mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                elif model_type == "esm2_contact_distance_angle_predictor":
                    loss = contact_loss_weight * (c_loss * factor) + distance_loss_weight * d_loss + angle_loss_weight * a_loss
                elif model_type == "esm2_contact_angle_predictor":
                    loss = contact_loss_weight * (c_loss * factor) + angle_loss_weight * a_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")
        if use_mlflow:
            import mlflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        
        model.eval()
        all_preds = []
        all_targs = []
        val_iter = [one_batch_val] if (overfit_one_batch and one_batch_val is not None) else val_loader
        val_pbar = tqdm(val_iter, desc=f"Epoch {epoch} [VAL]", leave=False)
        with torch.no_grad():
            for chain_keys, starts, seq_batch, cmat_batch, dmat_batch, mask_batch, full_sizes_batch in val_pbar:
                cmat_batch = cmat_batch.to(device)
                mask_batch = mask_batch.to(device)
                tokens = convert_seqs_to_tokens(seq_batch, model.alphabet).to(device)
                if model_type in ["esm2_contact_distance_predictor", "esm2_contact_distance_angle_predictor", "esm2_contact_angle_predictor"]:
                    outputs = model(tokens)
                    contact_preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                else:
                    contact_preds = model(tokens)
                # Treat each sliding-window as an independent sample.
                for key, start, pred, target, full_size in zip(chain_keys, starts, contact_preds.cpu(), cmat_batch.cpu(), full_sizes_batch):
                    all_preds.append(pred)
                    all_targs.append(target)
        
        # Concatenate all window predictions (each treated as a separate sample)
        merged_preds = torch.cat(all_preds, dim=0)
        merged_targs = torch.cat(all_targs, dim=0)
        metrics = compute_metrics(
            [merged_preds[i] for i in range(merged_preds.shape[0])],
            [merged_targs[i] for i in range(merged_targs.shape[0])],
            threshold=0.5
        )
        for k, v in metrics.items():
            print(f"Epoch {epoch} - Val {k}: {v:.4f}")
            if use_mlflow:
                import mlflow
                mlflow.log_metric(k, v, step=epoch)

        # --- Checkpoint Saving ---
        # Get new config values:
        checkpoints_metric = config.get("checkpoints_metric", "f1")
        checkpoints_dir_path = config.get("checkpoints_dir_path", "checkpoints")
        save_best_k = config.get("save_best_k_checkpoints", 1)
        maximize = config.get("maximize", True)
        os.makedirs(checkpoints_dir_path, exist_ok=True)
        
        current_metric = metrics.get(checkpoints_metric, 0)
        checkpoint_path = os.path.join(checkpoints_dir_path, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path} with {checkpoints_metric}: {current_metric:.4f}")
        
        best_checkpoints.append((current_metric, epoch, checkpoint_path))

        # Sort best_checkpoints based on the metric value.
        if maximize:
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0], reverse=True)
        else:
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0])
        if len(best_checkpoints) > save_best_k:
            worst = best_checkpoints.pop()  # Remove the worst checkpoint.
            try:
                if os.path.exists(worst[2]):
                    os.remove(worst[2])
                    print(f"Removed checkpoint {worst[2]} with {checkpoints_metric}: {worst[0]:.4f}")
                else:
                    print(f"Checkpoint {worst[2]} does not exist.")
            except Exception as e:
                print("Error removing checkpoint:", e)

    if use_mlflow:
        import mlflow
        mlflow.end_run()

if __name__ == "__main__":
    config_path = os.path.join("src", "config.yaml")
    from utils.utils import load_config
    config = load_config(config_path)
    train_pipeline(config)