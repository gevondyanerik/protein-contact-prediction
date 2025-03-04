# Protein Contact Prediction with Multi-task ESM2
This repository contains a deep learning solution for predicting residue-residue contacts in proteins, while also incorporating distance and angle (dihedral) predictions as auxiliary tasks. The model is built upon the pre-trained ESM2 transformer architecture and can optionally leverage multiple sequence alignments (MSA) to improve predictions.

## Features
- **Multi-task Learning:** Simultaneously predicts contacts, distances, and angles.
- **Flexible Input Processing:** Handles variable-length protein sequences with options for random cropping, truncation, or sliding window sampling.
- **MSA Integration:** Optionally uses multiple sequence alignments to enhance the sequence representation.
- **Custom Loss Functions:** Supports binary cross-entropy, focal loss for contact prediction, cross-entropy for distogram (distance) prediction, and angle cross-entropy loss.
- **Evaluation Metrics:** Computes standard metrics such as accuracy, precision, recall, F1 score, ROC AUC, and top-L precision.
- **MLflow Integration:** Tracks experiments, logs hyperparameters, and stores performance metrics.

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/protein-contact-prediction.git
   cd protein-contact-prediction
   ```
2. **Set up pre-commit hooks:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y pre-commit
   pre-commit install
   ```
3. **Set up mlflow tunnel:**
   ```bash
   chmod +x ./scripts/setup_mlflow_tunnel.sh
   sudo ./scripts/setup_mlflow_tunnel.sh
   ```
4. **Set up jupyter tunnel:**
   ```bash
   chmod +x ./scripts/setup_jupyter_tunnel.sh
   sudo ./scripts/setup_jupyter_tunnel.sh
   ```
5. **Set up docker container:**
   ```bash
   sudo make build-container
   sudo make start-container
   ```

### Dataset preparation
1. **Set up .env file::**
   ```txt
    PDB_DATA_URL=https://drive.google.com/file/d/{link}/view?usp=sharing
    PDB_DATA_OUTPUT_DIR=../data/pdb
    PDB_DATA_ZIP_NAME=pdb_data.zip
    UNZIP_FILE=true
    REMOVE_ZIP=true
   ```
2. **Enter into container:**
   ```bash
   sudo make enter-container
   ```
3. **Download dataset pdb files:**
   ```bash
    chmod +x ./scripts/download_pdb_data.sh
    sh ./scripts/download_pdb_data.sh
   ```
3. **Convert dataset pdb files to .h5 dataset:**
   ```bash
    python ./scripts/pdbs_to_h5_dataset.py --pdb_folder *path-to-train-pdb-folder* --output_dataset *path-to-output-folder* --contact_threshold 8.0
   ```

### Start mlflow (if you set mlflow.enabled is true in training yaml configuration file, SSH tunnel accessible via http://<your_server_ip>:5000)
   ```bash
   sudo make start-mlflow
   ```


### Start jupyter (optional, SSH tunnel accessible via http://<your_server_ip>:8888)
   ```bash
   sudo make start-jupyter
   ```

### Training
1. **Set up training yaml configuration file (example in src/config.yaml):**
   ```yaml
    train_data_path: `path to training HDF5 dataset`
    test_data_path: `path to validation HDF5 dataset`

    dataset:
        max_length: `maximum protein sequence length used for cropping/truncation/padding`
        max_distance: `maximum distance value for generating distance bins`
        sliding_window_step: `step size when applying a sliding window on long sequences during validation`
        train_mode: `cropping strategy during training; "random_crop" selects a random part, "truncate_seq" always takes the first part`
        val_mode: `cropping strategy during validation; "sliding_window" uses overlapping windows, "truncate_seq" uses the first part`
        use_msa: `whether to use multiple sequence alignment (msa) data for training`
        num_msa_rows: `number of MSA sequences to use per protein`
        num_angle_bins: `number of discrete bins for angle classification`
        train_batch_size: `number of training samples per batch`
        val_batch_size: `number of validation samples per batch`

    model:
        esm2_name: `identifier for pre-trained esm2 model used for sequence embeddings`
        esm_msa_name: `identifier for pre-trained msa model used for msa embeddings`
        main_task: `primary prediction task; "contact" uses contact head predictions, "distance" uses distance head predictions (converted to binary contact map)`
        use_contact: `whether to train on contact map prediction task`
        use_distance: `whether to train on distance map prediction task`
        use_angle: `whether to train on angle map prediction task`
        num_layers_to_freeze_esm2: `number of initial layers to freeze in esm2 model`
        num_layers_to_freeze_msa: `number of initial layers to freeze in msa model`
        fusion_dim: `size of embedding space for fused features`
        fusion_num_layers: `number of layer in fusion module`
        fusion_num_heads: `number of heads in fusion module`

    train:
        seed: `random seed for reproducibility`
        num_epochs: `number of training epochs`
        contact_loss_type: `loss type for contact prediction ("bce" or "focal")`
        bce_contact_class_weights: `weights for negative and positive classes in bce loss (e.g., [1.0, 30.0])`
        learning_rate: `optimizer’s learning rate`
        cosine_annealing_warm_restart_t_0: `cosine annealing scheduler t_0 parameter`
        cosine_annealing_warm_restart_t_mult: `cosine annealing scheduler t_mult parameter`
        cosine_annealing_warm_restart_eta_min: `cosine annealing scheduler eta_min parameter`
        focal_gamma: `focal loss annealing scheduler gamma parameter`
        focal_alpha: `focal loss annealing scheduler alpha parameter`
        contact_loss_weight: `weighting factor for the contribution of contact loss to the total loss`
        distance_loss_weight: `weighting factor for the contribution of distance loss to the total loss`
        angle_loss_weight: `weighting factor for the contribution of angle loss to the total loss`
        use_scheduler: `weither to use learning rate scheduler`
        overfit_one_batch: `weither to use one batch overfitting`

    checkpoints:
        checkpoint_path_to_load: `path to checkpoint that you wanna load`
        checkpoints_dir_path: `path to checkpoints folder`
        checkpoints_metric: `metric used to determine best checkpoint`
        maximize_metric: `indicates if higher values of the checkpoint metric are better`
        save_best_k_checkpoints: `maximum number of top checkpoints to retain`

    mlflow:
        enabled: `enable mlflow logging`
        tracking_uri: `mlflow tracking server uri (e.g., "http://localhost:5000")`
        experiment_name: `mlflow experiment name`
        run_name: `mlflow run name`
   ```
2. **Enter into container:**
   ```bash
   sudo make enter-container
   ```
3. **Run training script:**
   ```bash
   python src/training/train.py
   ```

### One batch overfit (optional)
1. **Change these arguments in training yaml configuration file:**
   ```yaml
    dataset:
        max_length: 5
        train_mode: "truncate_seq"
        val_mode: "truncate_seq"
        train_batch_size: 1
        val_batch_size: 1
    train:
        num_epochs: 1000
        overfit_one_batch: true
    ```
