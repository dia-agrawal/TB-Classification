# Lightning Combined Training Script - Help Documentation

## Overview

`lightning_combined.py` is a unified PyTorch Lightning training script that supports three different training modes for audio processing using masked autoencoders:

1. **Autoencoder Mode (`auto`)**: Standard masked autoencoder training for unsupervised learning
2. **Classifier Mode (`classifier`)**: Binary classification using the encoder with a classification head
3. **Triplet Loss Mode (`tripletloss`)**: Triplet margin loss training for similarity learning

## Features

- **Unified Interface**: Single script handles all three training modes
- **Dynamic Filtering**: Optional reconstruction loss-based filtering of training data
- **Energy-Based Filtering**: Signal energy-based filtering for additional data quality control
- **Multiple Loss Functions**: Support for various loss types in classifier mode
- **SpecAugment**: Audio augmentation for improved generalization
- **Checkpoint Management**: Automatic model checkpointing and early stopping
- **Comprehensive Logging**: Detailed training logs and model summaries
- **Configurable Architecture**: Flexible model parameters via YAML config

## Training Modes

### 1. Autoencoder Mode (`auto`)
- **Purpose**: Unsupervised learning of audio representations
- **Input**: Mel spectrograms from good quality audio
- **Output**: Reconstructed mel spectrograms
- **Loss**: Reconstruction loss (MSE)
- **Masking**: Configurable mask ratio (default: 0.8)

### 2. Classifier Mode (`classifier`)
- **Purpose**: Binary classification of audio quality
- **Input**: Mel spectrograms from good (label=1) and bad (label=0) audio
- **Output**: Binary classification (good/bad)
- **Loss**: Multiple options (BCE, Focal, Asymmetric, AUC, etc.)
- **Features**: Class weighting, SpecAugment, data extension

### 3. Triplet Loss Mode (`tripletloss`)
- **Purpose**: Learning audio similarity representations
- **Input**: Triplets of anchor, positive, and negative samples
- **Output**: Embeddings for similarity comparison
- **Loss**: Triplet margin loss with optional reconstruction loss
- **Features**: Learnable margin, optional reconstruction loss

## Command Line Arguments

### Basic Arguments
```bash
--config PATH              # Path to YAML config file (default: default_config.yaml)
--mode_type {auto,classifier,tripletloss}  # Training mode
--batch_size INT           # Batch size (overrides config)
--base_path PATH           # Base path for data and outputs (default: .)
--checkpoint_path PATH     # Checkpoint path to resume training
--prefix STR               # Prefix for output/log files
--num_workers INT          # Number of DataLoader workers (default: 0, was 3)
```

### Training Parameters
```bash
--start_epoch INT          # Starting epoch (overrides config)
--max_epochs INT           # Maximum number of epochs (overrides config)
--patience INT             # Early stopping patience (overrides config)
--mask_ratio FLOAT         # Mask ratio for autoencoder (0.0 to 1.0)
```

### Data and Augmentation
```bash
--base_dir PATH            # Base directory for data (default: data/dia_tmp_2)
--apply_specaugment        # Enable SpecAugment for classifier mode
--extend_bad               # Extend bad data samples for classifier mode
```

### Classifier Mode Options
```bash
--class_weight FLOAT FLOAT # Class weights [bad_weight, good_weight]
--loss_type STR            # Loss function: who, focal, asymmetric, auc, bce
```

### Triplet Loss Mode Options
```bash
--triplet_recon_loss STR   # Reconstruction loss: none, mse, logcosh
--triplet_recon_weight FLOAT # Weight for reconstruction loss (default: 1.0)
```

### Dynamic Filtering Options
```bash
--use_filter_dir PATH      # Directory containing train_recon.csv and val_recon.csv
--filter_mode {median,mean,percentile,hardcode}  # Filtering mode
--filter_params ARGS       # Filter parameters (varies by mode)
--hard_low_cut FLOAT       # Hard lower cutoff (median/mean modes only)
--hard_high_cut FLOAT      # Hard upper cutoff (median/mean modes only)
--energy_filter_mode {median,mean,percentile,hardcode}  # Energy filtering mode
--energy_filter_params ARGS # Energy filter parameters (same format as filter_params)
--energy_hard_low_cut FLOAT # Energy hard lower cutoff (median/mean modes only)
--energy_hard_high_cut FLOAT # Energy hard upper cutoff (median/mean modes only)

### Memory Optimization Options
```bash
--memory_efficient          # Enable memory-efficient settings (recommended for memory issues)
--pin_memory                # Enable pin_memory for faster GPU transfer (uses more RAM)
--persistent_workers        # Enable persistent workers (uses more RAM)
```

## Configuration File (YAML)

The script supports YAML configuration files. Example structure:

```yaml
# Training parameters
mode_type: classifier
batch_size: 128
max_epochs: 500
patience: 25
mask_ratio: 0.8
base_dir: data/dia_tmp_2

# Model parameters
model:
  num_mels: 256
  mel_len: 256
  patch_size: 16
  in_chans: 1
  embed_dim: 256
  encoder_depth: 6
  num_heads: 6
  decoder_embed_dim: 256
  decoder_depth: 6
  decoder_num_heads: 6
  mlp_ratio: 1

# Mode-specific parameters
apply_specaugment: false
class_weight: [4.0, 1.0]
loss_type: who
triplet_recon_loss: none
triplet_recon_weight: 1.0
```

## Usage Examples

### 1. Autoencoder Training
```bash
python lightning_combined.py \
    --mode_type auto \
    --base_dir data/audio_dataset \
    --batch_size 64 \
    --max_epochs 100 \
    --mask_ratio 0.8
```

### 2. Classifier Training
```bash
python lightning_combined.py \
    --mode_type classifier \
    --base_dir data/audio_dataset \
    --batch_size 32 \
    --apply_specaugment \
    --class_weight 4.0 1.0 \
    --loss_type focal \
    --extend_bad
```

### 3. Triplet Loss Training
```bash
python lightning_combined.py \
    --mode_type tripletloss \
    --base_dir data/audio_dataset \
    --batch_size 16 \
    --triplet_recon_loss mse \
    --triplet_recon_weight 0.5
```

### 4. With Dynamic Filtering
```bash
python lightning_combined.py \
    --mode_type classifier \
    --use_filter_dir evaluations/recon_loss_analysis \
    --filter_mode median \
    --filter_params 1.5 \
    --hard_low_cut 1.5
```

### 5. With Independent Filtering (Both Reconstruction Loss and Energy)
```bash
# Apply both filters independently
python lightning_combined.py \
    --mode_type classifier \
    --use_filter_dir evaluations/recon_loss_analysis \
    --filter_mode median \
    --filter_params 1.5 \
    --energy_filter_mode percentile \
    --energy_filter_params 0.1 0.9 \
    --energy_hard_low_cut 0.5

# Only reconstruction loss filtering
python lightning_combined.py \
    --mode_type classifier \
    --use_filter_dir evaluations/recon_loss_analysis \
    --filter_mode median \
    --filter_params 1.5

# Only energy filtering
python lightning_combined.py \
    --mode_type classifier \
    --use_filter_dir evaluations/recon_loss_analysis \
    --energy_filter_mode hardcode \
    --energy_filter_params 0.5 2.0
```

### 6. Memory-Efficient Training
```bash
# Enable memory-efficient mode
python lightning_combined.py \
    --mode_type classifier \
    --memory_efficient \
    --batch_size 32 \
    --num_workers 0
```

### 7. Resume Training from Checkpoint
```bash
python lightning_combined.py \
    --mode_type classifier \
    --checkpoint_path checkpoints/epoch=10-train_acc=0.8500-val_acc=0.8200.ckpt \
    --max_epochs 200
```

## Data Directory Structure

The script expects the following directory structure:

```
data/
├── dia_tmp_2/
│   ├── train/
│   │   ├── good/          # Good quality audio (.pt files)
│   │   └── bad/           # Bad quality audio (.pt files)
│   └── val/
│       ├── good/          # Good quality audio (.pt files)
│       └── bad/           # Bad quality audio (.pt files)
```

## CSV File Format

When using `--create_ignored_recon` with `evalmain.py`, the generated CSV files now include both reconstruction loss and signal energy data:

### CSV Structure
```
mean,std,histogram_bins,histogram_counts,energy_mean,energy_std,energy_histogram_bins,energy_histogram_counts
1.234567,0.123456,"[10, 20, 30, ...]","[0, 5, 10, ...]",0.987654,0.098765,"[5, 15, 25, ...]","[0, 3, 8, ...]"

filename,reconstruction_loss,signal_energy
file1.pt,1.123456,0.876543
file2.pt,1.234567,0.987654
...
```

### Energy Calculation
Signal energy is calculated as the mean square energy of the mel spectrogram:
```
energy = mean(mel_spectrogram²)
```

This provides a measure of the overall signal strength that can be used for additional filtering criteria.

## Output Structure

The script creates the following output structure:

```
{base_path}/
├── {prefix}/
│   ├── logs/
│   │   ├── config_used.yaml      # Configuration used for training
│   │   ├── model_summary.txt     # Model architecture summary
│   │   └── {prefix}/             # Training logs (CSV format)
│   └── checkpoints/
│       └── *.ckpt               # Model checkpoints
```

## Dynamic Filtering

The script supports dynamic filtering based on reconstruction loss analysis and signal energy:

### Filter Modes
- **median**: Uses interquartile range (IQR) with parameter `k` (default: 1.5)
- **mean**: Uses standard deviation with parameter `k` (default: 1.5)
- **percentile**: Uses percentile-based filtering with lower/upper bounds
- **hardcode**: Uses fixed reconstruction loss bounds

### Filter Parameters
- **median/mean**: `k` parameter (default: 1.5)
- **percentile**: `lower_percentile upper_percentile` (default: 0.05 0.95)
- **hardcode**: `lower_bound upper_bound` (default: 1.5 7.0)

### Hard Cutoffs
For median and mean modes, you can specify hard cutoffs:
- `--hard_low_cut`: Minimum reconstruction loss threshold
- `--hard_high_cut`: Maximum reconstruction loss threshold

## Energy-Based Filtering

The script also supports energy-based filtering using signal energy (mean square energy) calculated from mel spectrograms:

### Energy Filter Modes
- **median**: Uses interquartile range (IQR) with parameter `energy_k` (default: same as reconstruction loss)
- **mean**: Uses standard deviation with parameter `energy_k` (default: same as reconstruction loss)
- **percentile**: Uses percentile-based filtering with energy-specific bounds
- **hardcode**: Uses fixed energy bounds

### Energy Filter Parameters
- **median/mean**: `energy_k` parameter (default: same as reconstruction loss k)
- **percentile**: `energy_lower_percentile energy_upper_percentile` (default: same as reconstruction loss)
- **hardcode**: `energy_lower_bound energy_upper_bound` (default: same as reconstruction loss)

### Energy Hard Cutoffs
For median and mean energy modes, you can specify energy-specific hard cutoffs:
- `--energy_hard_low_cut`: Minimum signal energy threshold
- `--energy_hard_high_cut`: Maximum signal energy threshold

### Independent Filtering
When both reconstruction loss and energy filtering are enabled, each filter is applied independently based on their respective command line options. Files are ignored if they fail **either** filter (union operation). This provides comprehensive filtering by removing files that are outliers in reconstruction loss OR signal energy.

### Filter Combination Logic
- **Reconstruction loss filtering**: Applied if `--filter_mode` is specified
- **Energy filtering**: Applied if `--energy_filter_mode` is specified and energy data is available
- **Combined result**: Union of both filters (files ignored by either filter)
- **Independent parameters**: Each filter uses its own mode and parameters

### Patient-Level Filtering Behavior
- **Partially filtered patients**: Use remaining files for prediction (majority vote)
- **Completely filtered patients**: All files filtered out → treated as "good" class (prediction = 1)
- **Rationale**: If all files from a patient are filtered out, it suggests the patient's data quality is poor, which typically indicates a "good" (healthy) patient with low-quality recordings

## Loss Functions (Classifier Mode)

- **who**: Weighted Binary Cross-Entropy
- **focal**: Focal Loss for handling class imbalance
- **asymmetric**: Asymmetric Loss
- **auc**: AUC-based Loss
- **bce**: Standard Binary Cross-Entropy

## Model Architecture

The script uses `AudioMaskedAutoencoderViT` with configurable parameters:

- **num_mels**: Number of mel frequency bins (default: 256)
- **mel_len**: Length of mel spectrogram (default: 256)
- **patch_size**: Size of patches for Vision Transformer (default: 16)
- **embed_dim**: Embedding dimension (default: 256)
- **encoder_depth**: Number of encoder layers (default: 6)
- **num_heads**: Number of attention heads (default: 6)
- **decoder_depth**: Number of decoder layers (default: 6)

## Monitoring and Logging

The script automatically logs:
- Training and validation losses
- Accuracy metrics (for classifier and triplet modes)
- Model checkpoints (top 5 best models)
- Early stopping based on validation metrics
- Model architecture summary

## Tips and Best Practices

1. **Data Preparation**: Ensure your mel spectrograms are saved as `.pt` files
2. **Batch Size**: Adjust based on your GPU memory and dataset size
3. **Class Imbalance**: Use `--class_weight` and `--extend_bad` for imbalanced datasets
4. **Augmentation**: Enable `--apply_specaugment` for better generalization
5. **Filtering**: Use dynamic filtering to remove noisy samples
6. **Checkpointing**: Always save checkpoints for resuming training
7. **Monitoring**: Watch validation metrics to prevent overfitting

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Data Loading Errors**: Check file paths and ensure `.pt` files exist
3. **Filtering Errors**: Verify reconstruction loss CSV files exist and are properly formatted
4. **Checkpoint Loading**: Ensure checkpoint mode matches current mode

### Performance Optimization
1. Use appropriate `num_workers` for your system
2. Enable `pin_memory=True` for faster data loading (disabled by default for memory efficiency)
3. Use gradient accumulation for larger effective batch sizes
4. Monitor GPU utilization and adjust batch size accordingly
5. Use `--memory_efficient` flag for reduced memory usage
6. Reduce `prefetch_factor` for lower memory consumption

## Dependencies

Required packages (see `requirements.txt`):
- PyTorch
- PyTorch Lightning
- torchaudio
- pandas
- numpy
- PyYAML
- torchinfo

Install dependencies:
```bash
pip install -r requirements.txt
```

## File Dependencies

The script depends on several other files in the project:
- `audio_mae.py`: Audio Masked Autoencoder implementation
- `loss_functions.py`: Custom loss functions
- `Recon_loss_filtering.py`: Dynamic filtering functionality
- `default_config.yaml`: Default configuration file
