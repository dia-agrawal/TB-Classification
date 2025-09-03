# TB-Classification
TB classification using AudioMAE Facebook 

# Unified Lightning Training Script

This repository provides a **unified PyTorch Lightning** training script that supports three modes:

1. **Autoencoding** (`auto`)  
2. **Binary Classification** (`classifier`)  
3. **Triplet-loss Embedding** (`tripletloss`)  

It builds on Facebook’s AudioMAE implementation of Masked Autoencoders for audio, extending it with classification and triplet loss workflows.

## Requirements

Reconstruction Loss Filtering Usage Examples
This document shows how to use the new filtering functionality to filter audio files based on reconstruction loss.

Basic Usage
The filtering functionality is integrated into evalmain.py and requires:

--create_ignored_recon: Enable ignored reconstruction list creation
--filter_mode: Choose filtering method (optional, defaults to median with k=1.5 and hard low cutoff of 1.5)
--filter_params: Optional parameters for the filter (uses defaults if not provided)
--hard_low_cut: Optional hard lower cutoff (only for median/mean modes)
--hard_high_cut: Optional hard upper cutoff (only for median/mean modes)
Filter Modes
1. Median + IQR-based filtering (Recommended for skewed distributions)
# Default behavior (median with k=1.5 and hard low cutoff of 1.5)
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon

# Explicit median mode with default k=1.5
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode median

# Custom k value
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode median --filter_params 2.0

# With hard low cutoff (safety net for positively skewed data)
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode median --hard_low_cut 1.5

# With both hard cutoffs
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode median --hard_low_cut 1.5 --hard_high_cut 8.0
How it works: Computes Q1 and Q3, then filters within [Q1 - kIQR, Q3 + kIQR]

Pros: Robust to outliers, doesn't assume normality
Cons: May keep too much low-end data unless you add a custom lower bound
2. Mean ± k*std filtering (For roughly normal distributions)
# Default k=1.5
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode mean

# Custom k value
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode mean --filter_params 2.0

# With hard low cutoff
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode mean --hard_low_cut 1.5

# With both hard cutoffs
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode mean --hard_low_cut 1.5 --hard_high_cut 8.0
How it works: Keeps values within [mean - kstd, mean + kstd]

Pros: Easy to interpret
Cons: Not ideal for long-tailed distributions, skewed tail inflates mean and std
3. Percentile-based clipping
# Default 5th to 95th percentile
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode percentile

# Custom percentiles (2.5th to 97.5th)
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode percentile --filter_params 0.025 0.975
How it works: Manually clips below lower percentile and above upper percentile

Pros: Fully data-driven, handles skew well
Cons: Doesn't use "spread" like IQR/std, less interpretable
4. Fixed range cutoff
# Default range [1.5, 7.0]
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode hardcode

# Custom range
python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode hardcode --filter_params 1.0 8.0
How it works: Hardcoded range cutoff

Pros: Most interpretable, good if you know noise/silence thresholds
Cons: Not adaptive, might miss subtle distribution shifts
Output Files
The filtering process creates:

train_recon.csv: Original training reconstruction data (mean, std, histogram, per-file losses)
val_recon.csv: Original validation reconstruction data (mean, std, histogram, per-file losses) - if validation data exists
test_recon.csv: Original test reconstruction data (mean, std, histogram, per-file losses)
Recommendations
For your use case (very low values = noise/silence, very high values = outliers):

Best option: Use default behavior (median with k=1.5 and hard low cutoff of 1.5)

python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon
For more control: Use explicit median with custom parameters

python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode median --filter_params 2.0 --hard_low_cut 1.5
Alternative: Use percentile to ignore both tails

python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode percentile --filter_params 0.05 0.95
If you know your thresholds: Use hardcode

python evalmain.py --evaluate_using_dir /path/to/model/dir --create_ignored_recon --filter_mode hardcode --filter_params 1.5 7.0
Integration with Existing Workflow
The filtering functionality integrates seamlessly with the existing evaluation pipeline:

Run normal evaluation with --create_ignored_recon --filter_mode <mode>
Get reconstruction CSV files in your output directory
Use the reconstruction CSV files to filter out problematic files in subsequent processing

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- PyTorch Lightning ≥ 1.7  
- torchaudio  
- torchinfo  
- `audio_mae` module (wrapping Facebook’s AudioMAE)  

Install dependencies:

pip install torch torchvision torchaudio pytorch-lightning torchinfo pyyaml


### **Reference** 
Our AudioMae model is referenced from the github link below and developed for our use. 
Facebook AudioMae: https://github.com/facebookresearch/AudioMAE/tree/main

