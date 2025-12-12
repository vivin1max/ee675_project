# Neural Speech BCI Decoder - Final Project

**Course**: EE-675: Data Analysis and Control Techniques for Neurotechnology Des
 
**Project Title**: Neural Speech Decoding: Classical State-Space Models vs. Deep Learning Approaches  
**Student**: Vivin Thiyagarajan

Deep learning models for decoding attempted speech from neural activity in a paralyzed participant.

---

## Architecture

```
Neural Data (256D) -> Preprocessing -> RNN/LSTM/Transformer -> CTC Loss -> Phonemes
      |                    |                                        |
  Area 6v          Day Adaptation                            Beam Search -> Text
                   Blockwise Norm
```

**Components**: PyTorch models (RNN/LSTM/Transformer), Kalman filter baselines, CTC loss, Adam optimizer, data augmentation (Gaussian noise), day-specific adaptation

---

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA 11.7+
- NVIDIA GPU (12+ GB VRAM recommended)
- 32 GB RAM
- **Dataset**: Neural recordings from Willett et al. (2023)
  - Download: https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
  - Extract to `data/competitionData/`

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python preprocess_data.py --data_dir data/competitionData
```

### 3. Train Models

```bash
# Best deep learning model (RNN Baseline - 700 epochs)
python -m src.training.train --config configs/rnn_baseline.yaml

# Best classical model (Kalman Ridge)
python -m src.training.train_kalman --config configs/kalman_ridge_test.yaml

# Other architectures (100 epochs each)
python -m src.training.train --config configs/arch_lstm_base.yaml
python -m src.training.train --config configs/arch_lstm_regularised.yaml
python -m src.training.train --config configs/arch_transformer_base.yaml
python -m src.training.train --config configs/arch_transformer_regularised.yaml

# RNN ablations (100 epochs each)
python -m src.training.train --config configs/rnn_fastemit.yaml
python -m src.training.train --config configs/rnn_coord_dropout.yaml
python -m src.training.train --config configs/rnn_layernorm.yaml
python -m src.training.train --config configs/rnn_step_lr.yaml
python -m src.training.train --config configs/rnn_best_combo.yaml
```

### 4. Generate Plots

```bash
python scripts/plot_rnn_ablations_final.py
python scripts/plot_dl_architectures_comparison.py
python scripts/plot_deep_vs_classical_comparison.py
```

---

## Models Implemented

**Deep Learning:**
- **RNN Baseline**: 4-layer bidirectional GRU, hidden=768, dropout=0.5, 700 epochs
- **LSTM Base**: 5-layer LSTM, hidden=1024, dropout=0.40, constant LR, 100 epochs
- **LSTM Regularised**: 4-layer LSTM, hidden=768, dropout=0.5, 100 epochs  
- **Transformer Base**: 6-layer, hidden=1024, 8 heads, FFN=2048, dropout=0.10, 100 epochs
- **Transformer Regularised**: 3-layer, hidden=512, 8 heads, FFN=1024, dropout=0.2, 100 epochs

**RNN Ablations** (100 epochs each):
- **FastEmit** (λ=0.01): Reduces blank token emissions
- **CoordDrop** (0.1): Coordinated dropout across layers
- **LayerNorm**: Layer normalization for training stability
- **StepLR**: Step learning rate schedule (γ=0.5, step=30)
- **BestCombo**: FastEmit + CoordDrop combined

**Classical Baselines:**
- **Kalman Baseline**: 16D latent space, ~70K parameters
- **Kalman Ridge**: 32D latent, ridge regularization (α=0.001), ~77K parameters

---

## Features

- Bidirectional GRU/LSTM/Transformer architectures
- CTC loss for sequence-to-sequence decoding
- Day-specific adaptation for neural drift
- Blockwise normalization for within-day drift
- Data augmentation (Gaussian noise σ=1.0, offset σ=0.3)
- Learning rate scheduling (linear decay 0.01→0.001)
- Gradient clipping (max_norm=1.0)
- FastEmit regularization for blank reduction
- Coordinated dropout for temporal consistency
- Tensorboard logging
- Weights & Biases (W&B) support (optional)

**Metrics**: CER (Character Error Rate), WER (Word Error Rate) 

---

## Testing

Pre-computed results available in `results/final_results.csv` and `results/plots/`

**To reproduce**:
1. Train model using configs in `configs/`
2. Evaluate on test set (automatically done during training)
3. Results saved to `experiments/<model_name>/`
4. Generate plots using provided plotting scripts

**Monitoring**:
```bash
tensorboard --logdir experiments/
```

---

## Configuration

### Model Configs (configs/)
All hyperparameters specified in YAML files:
- `hidden_dim`: Hidden layer size (512-768)
- `layer_dim`: Number of layers (3-4)
- `dropout`: Dropout rate (0.2-0.5)
- `num_epochs`: Training epochs (100-700)
- `learning_rate`: Initial LR (0.01)
- `batch_size`: Batch size (8)
- `fastemit_lambda`: FastEmit regularization (0.01)
- `coordinated_dropout`: CoordDrop rate (0.1)

### Data Preprocessing
- Input: 256D neural features from Area 6v (ventral premotor cortex)
- Sampling: 20ms bins (50 Hz)
- Normalization: Blockwise z-score (250-sentence blocks)
- Augmentation: Gaussian noise + constant offset per channel
- Output: 40 phoneme classes (39 phones + blank for CTC)

---

## File Structure

```
FINAL_SUBMISSION/
├── README.txt                          # This file
├── requirements.txt                    # Python dependencies
├── preprocess_data.py                  # Data preprocessing script
├── LICENSE                             # MIT License
│
├── src/
│   ├── dataio/
│   │   ├── dataset.py                  # PyTorch dataset
│   │   └── preprocessing.py            # Data normalization
│   ├── models/
│   │   ├── base_ctc.py                 # Base CTC model
│   │   ├── baseline_gru.py             # RNN/GRU model
│   │   ├── lstm_ctc.py                 # LSTM model
│   │   └── transformer_ctc.py          # Transformer model
│   ├── classical/
│   │   └── kalman_decoder.py           # Kalman filter baselines
│   ├── training/
│   │   ├── train.py                    # DL training script
│   │   ├── train_kalman.py             # Kalman training script
│   │   ├── utils.py                    # Training utilities
│   │   ├── scheduler_utils.py          # LR schedulers
│   │   ├── fastemit_loss.py            # FastEmit regularization
│   │   └── coordinated_dropout.py      # Coordinated dropout
│   └── evaluation/
│       └── metrics.py                  # CER/WER computation
│
├── configs/                            # 12 YAML configuration files
│   ├── rnn_baseline.yaml
│   ├── rnn_fastemit.yaml
│   ├── rnn_coord_dropout.yaml
│   ├── rnn_layernorm.yaml
│   ├── rnn_step_lr.yaml
│   ├── rnn_best_combo.yaml
│   ├── arch_lstm_base.yaml
│   ├── arch_lstm_regularised.yaml
│   ├── arch_transformer_base.yaml
│   ├── arch_transformer_regularised.yaml
│   ├── kalman_baseline.yaml
│   └── kalman_ridge_test.yaml
│
├── results/
│   ├── final_results.csv               # Performance metrics
│   └── plots/                          # 5 publication plots (300 DPI)
│       ├── rnn_ablations_side_by_side.png
│       ├── dl_architectures_cer_wer.png
│       ├── dl_architectures_params_vs_wer.png
│       ├── deep_vs_classical_cer_wer.png
│       └── deep_vs_classical_size_vs_wer.png
│
├── scripts/                            # Plotting/visualization scripts
│   ├── plot_rnn_ablations_final.py
│   ├── plot_dl_architectures_comparison.py
│   └── plot_deep_vs_classical_comparison.py
│
└── data/                               # Dataset directory
    └── competitionData/                # Extract dataset here
        ├── train/
        ├── test/
        └── competitionHoldOut/
```

---

## Technical Details

**Data**: 256D neural features (Area 6v), 20ms bins, ~8800 train/880 test sentences  
**Loss**: CTC (Connectionist Temporal Classification)  
**Optimizer**: Adam (lr: 0.01→0.001, β₁=0.9, β₂=0.999, weight_decay: 5e-5, grad_clip: 1.0)  
**Augmentation**: Gaussian white noise (σ=1.0) + constant offset (σ=0.3)  
**Hardware**: NVIDIA RTX 4080 (12GB VRAM), 32GB RAM, 12-core CPU  

**Reference**: Willett, F. R., et al. (2023) "A high-performance speech neuroprosthesis" *Nature* 620(7976), 1031-1036

**Dataset**: Willett, Francis R. et al. (2023). Data from: A high-performance speech neuroprosthesis. *Dryad*. https://doi.org/10.5061/dryad.x69p8czpq

---
