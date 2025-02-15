#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import load_config, set_seed, train_val_collate, convert_seqs_to_tokens
from utils.dataset import ProteinDataset
from training.losses import contact_loss_bce, focal_loss, mse_loss_distance, angle_loss, MultiTaskLossWrapper
from training.metrics import compute_metrics

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
    
    train_dataset = ProteinDataset(
        h5_file=config["train_data_path"],
        max_input_length=config["max_input_length"],
        mode="train",
    )
    val_dataset = ProteinDataset(
        h5_file=config["test_data_path"],
        max_input_length=config["max_input_length"],
        mode="val",
        val_mode=config.get("val_mode", "sliding_window")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=train_val_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        collate_fn=train_val_collate,
    )
    
    overfit_one_batch = config.get("overfit_one_batch", False)
    if overfit_one_batch:
        one_batch_train = next(iter(train_loader), None)
        one_batch_val = one_batch_train
    else:
        one_batch_train = None
        one_batch_val = None
    
    model_type = config.get("model_type", "esm2_contact_predictor_msa")

    if model_type == "esm2_contact_predictor_msa":
        model = ESM2ContactPredictorMSA(
            esm_model_name=config.get("esm_model_name", "esm2_t6_8M_UR50D"),
            msa_model_name=config.get("msa_model_name", "esm_msa1b_t12_100M_UR50S"),
            num_layers_to_freeze=config.get("num_layers_to_freeze", None)
        )
    elif model_type == "esm2_contact_distance_predictor_msa":
        from models.esm2_contact_distance_predictor_msa import ESM2ContactDistancePredictorMSA
        model = ESM2ContactDistancePredictorMSA(
            esm_model_name=config["esm_model_name"],
            msa_model_name=config.get("msa_model_name", "esm_msa1b_t12_100M_UR50S"),
            num_layers_to_freeze=config.get("num_layers_to_freeze", None),
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0),
            fusion_source=config.get("fusion_source", "distance"),
            fusion_d_model=config.get("fusion_d_model", 64),
            fusion_num_heads=config.get("fusion_num_heads", 4)
        )
    elif model_type == "esm2_contact_distance_angle_predictor_msa":
        from models.esm2_contact_distance_angle_predictor_msa import ESM2ContactDistanceAnglePredictorMSA
        model = ESM2ContactDistanceAnglePredictorMSA(
            esm_model_name=config["esm_model_name"],
            msa_model_name=config.get("msa_model_name", "esm_msa1b_t12_100M_UR50S"),
            num_layers_to_freeze=config.get("num_layers_to_freeze", None),
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0),
            fusion_source=config.get("fusion_source", "distance"),
            fusion_d_model=config.get("fusion_d_model", 64),
            fusion_num_heads=config.get("fusion_num_heads", 4)
        )
    elif model_type == "esm2_contact_angle_predictor_msa":
        from models.esm2_contact_angle_predictor_msa import ESM2ContactAnglePredictorMSA
        model = ESM2ContactAnglePredictorMSA(
            esm_model_name=config["esm_model_name"],
            msa_model_name=config.get("msa_model_name", "esm_msa1b_t12_100M_UR50S"),
            num_layers_to_freeze=config.get("num_layers_to_freeze", None),
            feature_fusion=config.get("feature_fusion", False),
            fusion_scale=config.get("fusion_scale", 1.0),
            fusion_source=config.get("fusion_source", "distance"),
            fusion_d_model=config.get("fusion_d_model", 64),
            fusion_num_heads=config.get("fusion_num_heads", 4)
        )
    else:
        raise ValueError("Unsupported model_type: " + model_type)

    model.to(device)
    if use_mlflow:
        mlflow.pytorch.log_model(model, "model")
    
    use_uncertainty_weighting = config.get("use_uncertainty_weighting", False)
    if use_uncertainty_weighting:
        from training.losses import MultiTaskLossWrapper
        num_tasks = 3 if model_type == "esm2_contact_distance_angle_predictor_msa" else 2
        loss_wrapper = MultiTaskLossWrapper(num_tasks=num_tasks).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_wrapper.parameters()),
                                     lr=config["learning_rate"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    loss_type = config.get("loss_type", "bce")
    gamma = config.get("focal_gamma", 2.0)
    
    neg_weight, pos_weight = config.get("contact_class_weights", None)
    
    contact_loss_weight = config.get("contact_loss_weight", 1.0)
    distance_loss_weight = config.get("distance_loss_weight", 1.0)
    angle_loss_weight = config.get("angle_loss_weight", 1.0)
    
    reweight_contact_loss = config.get("reweight_contact_loss", False)
    distance_to_weight_scale = config.get("distance_to_weight_scale", 10.0)

    scheduler_type = config.get("scheduler_type", "step")
    if scheduler_type == "step":
        step_size = config.get("scheduler_step_size", 10)
        gamma_val = config.get("scheduler_gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=config.get("focal_gamma", 2.0))
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config.get("scheduler_factor", 0.1), 
            factor=config.get("scheduler_factor", 0.1), 
            patience=config.get("scheduler_patience", 5), 
            verbose=True,
        )
    else:
        scheduler = None
    
    checkpoint_path_to_load = config.get("checkpoint_path_to_load", None)
    if checkpoint_path_to_load:
        checkpoint = torch.load(checkpoint_path_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    best_checkpoints = []
    for epoch in range(1, config["num_epochs"]+1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_iter = [next(iter(train_loader))] if (overfit_one_batch and one_batch_train is not None) else train_loader
        train_pbar = tqdm(train_iter, desc=f"Epoch {epoch} [TRAIN]", leave=False)
        for chain_keys, starts, seq_batch, cmat_batch, dmat_batch, mask_batch, full_sizes, msa_batch in train_pbar:
            cmat_batch = cmat_batch.to(device)
            dmat_batch = dmat_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()

            if config.get("use_msa", False) and model_type == "esm2_contact_predictor_msa":
                tokens, msa_tokens = convert_seqs_to_tokens(seq_batch, model.ref_alphabet, msa_batch=[s for _, s in seq_batch], msa_alphabet=model.msa_alphabet)
                tokens = tokens.to(device)
                msa_tokens = msa_tokens.to(device)
                outputs = model(tokens, msa_tokens)
            else:
                tokens = convert_seqs_to_tokens(seq_batch, model.alphabet).to(device)
                outputs = model(tokens)
            
            if model_type in ["esm2_contact_distance_predictor_msa", "esm2_contact_distance_angle_predictor_msa", "esm2_contact_angle_predictor_msa"]:
                if model_type == "esm2_contact_distance_predictor_msa":
                    contact_probs, distance_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=config.get("focal_gamma", 2.0),
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    d_loss = mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                    loss = contact_loss_weight * c_loss + distance_loss_weight * d_loss
                elif model_type == "esm2_contact_distance_angle_predictor_msa":
                    contact_probs, distance_preds, angle_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=config.get("focal_gamma", 2.0),
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    d_loss = mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                    a_loss = angle_loss(angle_preds, cmat_batch, mask=mask_batch)
                    loss = (contact_loss_weight * c_loss +
                            distance_loss_weight * d_loss +
                            angle_loss_weight * a_loss)
                elif model_type == "esm2_contact_angle_predictor_msa":
                    contact_probs, angle_preds = outputs
                    if loss_type == "focal":
                        c_loss = focal_loss(contact_probs, cmat_batch, gamma=config.get("focal_gamma", 2.0),
                                            pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                    else:
                        c_loss = contact_loss_bce(contact_probs, cmat_batch, mask=mask_batch,
                                                  pos_weight=pos_weight, neg_weight=neg_weight)
                    a_loss = angle_loss(angle_preds, cmat_batch, mask=mask_batch)
                    loss = contact_loss_weight * c_loss + angle_loss_weight * a_loss
            else:
                preds = outputs
                if loss_type == "focal":
                    loss = focal_loss(preds, cmat_batch, gamma=config.get("focal_gamma", 2.0),
                                      pos_alpha=pos_weight, neg_alpha=neg_weight, mask=mask_batch)
                else:
                    loss = contact_loss_bce(preds, cmat_batch, mask=mask_batch,
                                            pos_weight=pos_weight, neg_weight=neg_weight)
            
            if reweight_contact_loss and model_type in ["esm2_contact_distance_predictor_msa", "esm2_contact_distance_angle_predictor_msa", "esm2_contact_angle_predictor_msa"]:
                reweight_factor = torch.exp(-dmat_batch / distance_to_weight_scale)
                reweight_factor = torch.masked_select(reweight_factor, mask_batch.bool())
                factor = reweight_factor.mean()
                if model_type == "esm2_contact_distance_predictor_msa":
                    loss = contact_loss_weight * (loss / contact_loss_weight * factor) + distance_loss_weight * mse_loss_distance(distance_preds, dmat_batch, mask=mask_batch)
                elif model_type == "esm2_contact_distance_angle_predictor_msa":
                    loss = contact_loss_weight * (c_loss * factor) + distance_loss_weight * d_loss + angle_loss_weight * a_loss
                elif model_type == "esm2_contact_angle_predictor_msa":
                    loss = contact_loss_weight * (c_loss * factor) + angle_loss_weight * a_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")
        if use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        
        model.eval()
        all_preds = []
        all_targs = []
        val_iter = [one_batch_val] if (overfit_one_batch and one_batch_val is not None) else val_loader
        val_pbar = tqdm(val_iter, desc=f"Epoch {epoch} [VAL]", leave=False)
        with torch.no_grad():
            for chain_keys, starts, seq_batch, cmat_batch, dmat_batch, mask_batch, full_sizes, msa_batch in val_pbar:
                cmat_batch = cmat_batch.to(device)
                mask_batch = mask_batch.to(device)
                if config.get("use_msa", False) and "msa" in model_type:
                    tokens, msa_tokens = convert_seqs_to_tokens(seq_batch, model.ref_alphabet, 
                                                                  msa_batch=msa_batch, 
                                                                  msa_alphabet=model.msa_alphabet)
                    tokens = tokens.to(device)
                    msa_tokens = msa_tokens.to(device)
                    outputs = model(tokens, msa_tokens)
                    contact_preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                else:
                    tokens = convert_seqs_to_tokens(seq_batch, model.alphabet).to(device)
                    outputs = model(tokens)
                    contact_preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                # Treat each sliding-window as an independent sample.
                for key, start, pred, target, full_size in zip(chain_keys, starts, 
                                                               contact_preds.cpu(), 
                                                               cmat_batch.cpu(), 
                                                               full_sizes_batch):
                    all_preds.append(pred)
                    all_targs.append(target)
        
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
        
        if scheduler is not None:
            if scheduler_type == "step":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(metrics.get("f1", 0))
        
        checkpoints_metric = config.get("checkpoints_metric", "roc-auc")
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
        if maximize:
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0], reverse=True)
        else:
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0])
        if len(best_checkpoints) > save_best_k:
            worst = best_checkpoints.pop()
            try:
                if os.path.exists(worst[2]):
                    os.remove(worst[2])
                    print(f"Removed checkpoint {worst[2]} with {checkpoints_metric}: {worst[0]:.4f}")
                else:
                    print(f"Checkpoint {worst[2]} does not exist.")
            except Exception as e:
                print("Error removing checkpoint:", e)
    
    if use_mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    config_path = os.path.join("src", "config.yaml")
    config = load_config(config_path)
    train_pipeline(config)