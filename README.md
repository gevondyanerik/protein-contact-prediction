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
    PDB_DATA_URL=*link to the dataset (google disk, for example)*
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
1. **Set up training yaml configuration file:**
   ```yaml
    train_data_path: "data/train_dataset.h5"
    test_data_path: "data/test_dataset.h5"

    dataset:
        max_length: 50
        max_distance: 40
        sliding_window_step: 50
        train_mode: "random_crop"    # "truncate_seq" or "random_crop"
        val_mode: "sliding_window"   # "truncate_seq" or "sliding_window"
        use_msa: true
        num_msa_rows: 4
        num_angle_bins: 24
        train_batch_size: 8
        val_batch_size: 4

    model:
        esm2_name: "esm2_t6_8M_UR50D"
        esm_msa_name: "esm_msa1b_t12_100M_UR50S"
        main_task: "contact"   # "contact" or "distance"
        use_contact: true
        use_distance: true
        use_angle: true
        num_layers_to_freeze_esm2: 6
        num_layers_to_freeze_msa: 12
        fusion_dim: 256
        fusion_num_layers: 4
        fusion_num_heads: 8

    train:
        seed: 42
        num_epochs: 15
        contact_loss_type: "bce"    # "bce" or "focal"
        bce_contact_class_weights: [1.0, 30.0]
        learning_rate: 0.0001
        cosine_annealing_warm_restart_t_0: 10
        cosine_annealing_warm_restart_t_mult: 1
        cosine_annealing_warm_restart_eta_min: 0.000001
        focal_gamma: 2.0
        focal_alpha: 2.0
        contact_loss_weight: 1.
        distance_loss_weight: 0.25
        angle_loss_weight: 0.1
        use_scheduler: true
        overfit_one_batch: false

    checkpoints:
        checkpoint_path_to_load: null
        checkpoints_dir_path: "checkpoints/"
        checkpoints_metric: "roc_auc"
        maximize_metric: true
        save_best_k_checkpoints: 2

    mlflow:
        enabled: true
        tracking_uri: "http://localhost:5000"
        experiment_name: "exp_name"
        run_name: "v1"
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
