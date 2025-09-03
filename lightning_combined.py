# ------------------------------------------------------
# Unified Lightning Training Script
# ------------------------------------------------------
import argparse
import yaml
import os, csv
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from audio_mae import AudioMaskedAutoencoderViT
from torchaudio.transforms import FrequencyMasking, TimeMasking
import random
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pandas as pd
from torchinfo import summary
import gc  # For garbage collection

# Memory optimization settings
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit CUDA memory split size
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = False  # Disable cudnn benchmark to save memory
torch.backends.cudnn.deterministic = True  # Enable deterministic mode

from loss_functions import GeneralizedLoss

# ------------------------------------------------------
# --- DATASETS ---
# ------------------------------------------------------

def read_ignored_files_from_recon_csv(filter_dir, filter_mode='median', filter_params=None, energy_filter_mode=None):
    """
    Read ignored files from reconstruction loss CSV files and apply filtering
    
    Args:
        filter_dir: Directory containing train_recon.csv and val_recon.csv
        filter_mode: Filter mode ('median', 'mean', 'percentile', 'hardcode')
        filter_params: Filter parameters dictionary
        energy_filter_mode: Energy filter mode (if None, uses same as filter_mode)
    
    Returns:
        tuple: (train_ignored_files, val_ignored_files) - lists of filenames to ignore
    """
    train_ignored_files = []
    val_ignored_files = []
    
    # Import filtering functions
    try:
        from Recon_loss_filtering import apply_filtering_from_directory
    except ImportError:
        print("Warning: Recon_loss_filtering module not found. Skipping filtering.")
        return train_ignored_files, val_ignored_files
    
    # Check if energy data is available in CSV files
    train_recon_csv = os.path.join(filter_dir, 'train_recon.csv')
    val_recon_csv = os.path.join(filter_dir, 'val_recon.csv')
    
    has_energy_data = False
    if os.path.exists(train_recon_csv):
        try:
            train_df = pd.read_csv(train_recon_csv, skiprows=3)
            has_energy_data = 'signal_energy' in train_df.columns
        except Exception:
            has_energy_data = False
    
    # Apply reconstruction loss filtering if filter_mode is specified
    recon_train_ignored = []
    recon_val_ignored = []
    if filter_mode is not None:
        print(f"Applying reconstruction loss filtering...")
        try:
            recon_train_ignored, recon_val_ignored, _, _, _, _ = apply_filtering_from_directory(
                filter_dir, filter_mode, filter_params
            )
            print(f"Reconstruction loss filtering: {len(recon_train_ignored)} train, {len(recon_val_ignored)} val files ignored")
        except Exception as e:
            print(f"Error in reconstruction loss filtering: {e}")
    
    # Apply energy filtering if energy data is available and energy filter mode is specified
    energy_train_ignored = []
    energy_val_ignored = []
    if has_energy_data and energy_filter_mode is not None:
        print(f"Energy data detected. Applying energy-based filtering...")
        try:
            # Create energy-only filter params
            energy_filter_params = {}
            if 'energy_k' in filter_params:
                energy_filter_params['k'] = filter_params['energy_k']
            if 'energy_lower_percentile' in filter_params:
                energy_filter_params['lower_percentile'] = filter_params['energy_lower_percentile']
            if 'energy_upper_percentile' in filter_params:
                energy_filter_params['upper_percentile'] = filter_params['energy_upper_percentile']
            if 'energy_lower_bound' in filter_params:
                energy_filter_params['lower_bound'] = filter_params['energy_lower_bound']
            if 'energy_upper_bound' in filter_params:
                energy_filter_params['upper_bound'] = filter_params['energy_upper_bound']
            if 'energy_hard_low_cut' in filter_params:
                energy_filter_params['hard_low_cut'] = filter_params['energy_hard_low_cut']
            if 'energy_hard_high_cut' in filter_params:
                energy_filter_params['hard_high_cut'] = filter_params['energy_hard_high_cut']
            
            energy_train_ignored, energy_val_ignored, _, _, _, _ = apply_filtering_from_directory(
                filter_dir, energy_filter_mode, energy_filter_params
            )
            print(f"Energy filtering: {len(energy_train_ignored)} train, {len(energy_val_ignored)} val files ignored")
        except Exception as e:
            print(f"Error in energy-based filtering: {e}")
    
    # Combine ignored files from both filters (union)
    if recon_train_ignored and energy_train_ignored:
        train_ignored_files = list(set(recon_train_ignored) | set(energy_train_ignored))
        print(f"Combined filtering: {len(train_ignored_files)} train files ignored (union of both filters)")
    elif recon_train_ignored:
        train_ignored_files = recon_train_ignored
    elif energy_train_ignored:
        train_ignored_files = energy_train_ignored
    else:
        train_ignored_files = []
    
    if recon_val_ignored and energy_val_ignored:
        val_ignored_files = list(set(recon_val_ignored) | set(energy_val_ignored))
        print(f"Combined filtering: {len(val_ignored_files)} val files ignored (union of both filters)")
    elif recon_val_ignored:
        val_ignored_files = recon_val_ignored
    elif energy_val_ignored:
        val_ignored_files = energy_val_ignored
    else:
        val_ignored_files = []
    
    return train_ignored_files, val_ignored_files


class AudioDataset(Dataset):
    """
    AudioDataset for all modes (auto, classifier, triplet).

    Modes:
      - 'auto': Loads all .pt files from data_dir_good, returns mel spectrograms.
      - 'classifier': Loads all .pt files from data_dir_good (label=1) and data_dir_bad (label=0), returns (mel, label) pairs for binary classification.
      - 'triplet': Loads triplets from data_dir_good and data_dir_bad for triplet loss training.

    Args:
        mode (str): One of 'auto', 'classifier', or 'triplet'.
        data_dir_good (str or list): Directory or list of directories for 'good' samples.
        data_dir_bad (str or list): Directory or list of directories for 'bad' samples (used in classifier/triplet).
        apply_specaugment (bool): Whether to apply SpecAugment (only for classifier mode).
        

    Shuffling:
        At the end of each training epoch, the dataset is reshuffled automatically.

    Returns:
        - 'auto': mel (Tensor)
        - 'classifier': (mel, label) (Tensor, int)
        - 'triplet': (anchor, positive, negative) (Tensor, Tensor, Tensor)
    """
    def __init__(self, mode, data_dir_good=None, data_dir_bad=None, apply_specaugment=False, shuffle=True, extend_bad=False, 
                 filter_dir=None, filter_mode='median', filter_params=None, memory_efficient=False, energy_filter_mode=None):
        self.mode = mode
        self.apply_specaugment = apply_specaugment
        self.memory_efficient = memory_efficient
        
        # Memory-efficient augmentation setup
        if not self.memory_efficient:
            self.freq_mask = FrequencyMasking(freq_mask_param=8)
            self.time_mask = TimeMasking(time_mask_param=20)
        else:
            # Lazy initialization for memory efficiency
            self.freq_mask = None
            self.time_mask = None
        
        # Read ignored files if filtering is enabled
        train_ignored_files = []
        val_ignored_files = []
        if filter_dir is not None:
            print(f"Applying reconstruction loss filtering from directory: {filter_dir}")
            train_ignored_files, val_ignored_files = read_ignored_files_from_recon_csv(
                filter_dir, filter_mode, filter_params, energy_filter_mode
            )
        
        ignored = []
        if mode == 'auto':
            # Load all .pt files from data_dir_good
            if type(data_dir_good) is not list:
                data_dir_good = [data_dir_good]
            self.files = []
            for d in data_dir_good:
                if d is not None:  # Type guard
                    t = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pt')]
                    self.files.extend(t)
            
            # Apply filtering if enabled
            if filter_dir is not None:
                # Determine if this is training or validation data based on directory path
                is_training_data = any('train' in d.lower() for d in data_dir_good)
                is_validation_data = any('val' in d.lower() for d in data_dir_good)
                
                if is_training_data and train_ignored_files:
                    print(f"Filtering training data: removing {len(train_ignored_files)} ignored files")
                    self.files = [f for f in self.files if os.path.basename(f) not in train_ignored_files]
                elif is_validation_data and val_ignored_files:
                    print(f"Filtering validation data: removing {len(val_ignored_files)} ignored files")
                    self.files = [f for f in self.files if os.path.basename(f) not in val_ignored_files]
                
                print(f"Final file count after filtering: {len(self.files)}")




        elif mode == 'classifier':
            # Build a list of (filepath, label) for binary classification
            self.samples = []
            if type(data_dir_good) is not list:
                data_dir_good = [data_dir_good]
            if type(data_dir_bad) is not list:
                data_dir_bad = [data_dir_bad]
                
            for d in data_dir_good:
                if d is not None:  # Type guard
                    tmp = [(os.path.join(d, f), 1) for f in os.listdir(d) if f.endswith('.pt')]
                    self.samples.extend(tmp)

                    if extend_bad: # VA: REVERT
                        self.samples.extend(tmp*2)
                                            
            for d in data_dir_bad:
                if d is not None:  # Type guard
                    tmp = [(os.path.join(d, f), 0) for f in os.listdir(d) if f.endswith('.pt')]
                    self.samples.extend(tmp)
                    if extend_bad:
                        self.samples.extend(tmp*3)
            
            # Apply filtering if enabled
            if filter_dir is not None:
                # Determine if this is training or validation data based on directory paths
                is_training_data = any('train' in d.lower() for d in data_dir_good + data_dir_bad)
                is_validation_data = any('val' in d.lower() for d in data_dir_good + data_dir_bad)
                
                if is_training_data and train_ignored_files:
                    print(f"Filtering training data: removing {len(train_ignored_files)} ignored files")
                    self.samples = [(f, l) for f, l in self.samples if os.path.basename(f) not in train_ignored_files]
                elif is_validation_data and val_ignored_files:
                    print(f"Filtering validation data: removing {len(val_ignored_files)} ignored files")
                    self.samples = [(f, l) for f, l in self.samples if os.path.basename(f) not in val_ignored_files]
                
                print(f"Final sample count after filtering: {len(self.samples)}")
                        
            if shuffle:
                random.shuffle(self.samples)

            
        elif mode == 'triplet':
            # Prepare triplet dataset (anchor, positive, negative)
            if type(data_dir_good) is not list:
                data_dir_good = [data_dir_good]
            if type(data_dir_bad) is not list:
                data_dir_bad = [data_dir_bad]
            self.goodfile = []
            for d in data_dir_good:
                if d is not None:  # Type guard
                    t = [(os.path.join(d, f)) for f in os.listdir(d) if f.endswith('.pt')]
                    self.goodfile.extend(t)
            self.badfile = []
            for d in data_dir_bad:
                if d is not None:  # Type guard
                    tmp = [(os.path.join(d, f)) for f in os.listdir(d) if f.endswith('.pt')]
                    self.badfile.extend(tmp)
            
            # Apply filtering if enabled
            if filter_dir is not None:
                # Determine if this is training or validation data based on directory paths
                is_training_data = any('train' in d.lower() for d in data_dir_good + data_dir_bad)
                is_validation_data = any('val' in d.lower() for d in data_dir_good + data_dir_bad)
                
                if is_training_data and train_ignored_files:
                    print(f"Filtering training data: removing {len(train_ignored_files)} ignored files")
                    self.goodfile = [f for f in self.goodfile if os.path.basename(f) not in train_ignored_files]
                    self.badfile = [f for f in self.badfile if os.path.basename(f) not in train_ignored_files]
                elif is_validation_data and val_ignored_files:
                    print(f"Filtering validation data: removing {len(val_ignored_files)} ignored files")
                    self.goodfile = [f for f in self.goodfile if os.path.basename(f) not in val_ignored_files]
                    self.badfile = [f for f in self.badfile if os.path.basename(f) not in val_ignored_files]
                
                print(f"Final good file count after filtering: {len(self.goodfile)}")
                print(f"Final bad file count after filtering: {len(self.badfile)}")
            
            self.anchorfile = self.goodfile.copy()
            random.shuffle(self.anchorfile)
            if shuffle:
                random.shuffle(self.badfile)


        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def __len__(self):
        if self.mode == 'auto':
            return len(self.files)
        elif self.mode == 'classifier':
            return len(self.samples)
        else:
            return len(self.goodfile)
            
    def __getitem__(self, idx):
        if self.mode == 'auto':
            # Return a single mel spectrogram
            path = self.files[idx]
            mel = torch.load(path, map_location='cpu')  # Force CPU loading
            mel = mel[:, 1:]
            mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
            return mel.unsqueeze(0).float(), path  # Ensure float32 precision
        elif self.mode == 'classifier':
            # Return (mel, label) for binary classification
            path, label = self.samples[idx]
            mel = torch.load(path, map_location='cpu')  # Force CPU loading
            mel = mel[:, 1:]
            mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
            if self.apply_specaugment: # and label == 0: # VA: REVERT
                # Lazy initialization of augmentation transforms
                if self.memory_efficient and self.freq_mask is None:
                    self.freq_mask = FrequencyMasking(freq_mask_param=8)
                    self.time_mask = TimeMasking(time_mask_param=20)
                
                r = random.random()
                if r < 0.15:
                    mel = self.freq_mask(self.time_mask(mel))
                elif r < 0.45:
                    mel = mel + torch.randn_like(mel, dtype=torch.float32) * 0.01
                elif r < 0.66:
                    gain = 10 ** (random.uniform(-6, 6) / 20)
                    mel = mel * gain
            return mel.unsqueeze(0).float(), torch.tensor(label, dtype=torch.long), path  # Ensure float32 precision
        else:
            # Return (anchor, positive, negative) for triplet loss
            goodpath = self.goodfile[idx]
            badpath = self.badfile[idx % len(self.badfile)]
            anchorpath = self.anchorfile[idx]
            try:
                mel_good = torch.load(goodpath, map_location='cpu')  # Force CPU loading
                mel_bad = torch.load(badpath, map_location='cpu')  # Force CPU loading
                mel_anchor = torch.load(anchorpath, map_location='cpu')  # Force CPU loading
                mel_good = mel_good[:, 1:]
                mel_good = F.pad(input=mel_good, pad=(0, 0, 0, 59), mode='constant', value=0)
                if self.apply_specaugment: # and label == 0: # VA: REVERT
                    # Lazy initialization of augmentation transforms
                    if self.memory_efficient and self.freq_mask is None:
                        self.freq_mask = FrequencyMasking(freq_mask_param=8)
                        self.time_mask = TimeMasking(time_mask_param=20)
                    
                    r = random.random()
                    if r < 0.15:
                        mel_good = self.freq_mask(self.time_mask(mel_good))
                    elif r < 0.45:
                        mel_good = mel_good + torch.randn_like(mel_good, dtype=torch.float32) * 0.01
                    elif r < 0.66:
                        gain = 10 ** (random.uniform(-6, 6) / 20)
                        mel_good = mel_good * gain
                mel_bad = mel_bad[:, 1:]
                mel_bad = F.pad(input=mel_bad, pad=(0, 0, 0, 59), mode='constant', value=0)
                if self.apply_specaugment: # and label == 0: # VA: REVERT
                    # Lazy initialization of augmentation transforms
                    if self.memory_efficient and self.freq_mask is None:
                        self.freq_mask = FrequencyMasking(freq_mask_param=8)
                        self.time_mask = TimeMasking(time_mask_param=20)
                    
                    r = random.random()
                    if r < 0.10:
                        mel_bad = self.freq_mask(self.time_mask(mel_bad))
                    elif r < 0.20:
                        mel_bad = mel_good + torch.randn_like(mel_bad, dtype=torch.float32) * 0.01
                    elif r < 0.30:
                        gain = 10 ** (random.uniform(-6, 6) / 20)
                        mel_bad = mel_bad * gain
                mel_anchor = mel_anchor[:, 1:]
                mel_anchor = F.pad(input=mel_anchor, pad=(0, 0, 0, 59), mode='constant', value=0)
                return (mel_anchor.unsqueeze(0).float(), mel_good.unsqueeze(0).float(), mel_bad.unsqueeze(0).float())  # Ensure float32 precision
            except Exception as e:
                print(f"Failed loading data at index {idx}: {e}")
                raise

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        if self.mode == 'auto':
            random.shuffle(self.files)
        elif self.mode == 'classifier':
            random.shuffle(self.samples)
        elif self.mode == 'triplet':
            random.shuffle(self.goodfile)
            random.shuffle(self.badfile)
            self.anchorfile = self.goodfile.copy()
            random.shuffle(self.anchorfile)

# ------------------------------------------------------
# --- MODELS ---
# ------------------------------------------------------
class LearnableMarginTripletLoss(nn.Module):
    def __init__(self, init_margin: float = 0.2):
        super().__init__()
        # raw_margin is unconstrained; softplus(raw_margin) will be ≥ 0
        self.raw_margin = nn.Parameter(torch.tensor(init_margin))

    def forward(self, anchor, positive, negative):
        # pairwise distances
        pos_d = F.pairwise_distance(anchor, positive, p=2)
        neg_d = F.pairwise_distance(anchor, negative, p=2)
        # ensure positive margin
        margin = F.softplus(self.raw_margin)
        # classic triplet loss
        return F.relu(pos_d - neg_d + margin).mean()


class UnifiedLightningModel(pl.LightningModule):
    """
    Unified LightningModule supporting autoencoder, classifier, and triplet loss modes.

    - 'auto': Standard autoencoder training.
    - 'classifier': Binary classification using softmax and CrossEntropyLoss.
    - 'tripletloss': Triplet margin loss, with optional reconstruction loss (MSE or logcosh).

    Args:
        model_params (dict): Parameters for AudioMaskedAutoencoderViT.
        mode (str): One of 'auto', 'classifier', or 'tripletloss'.
        triplet_recon_loss (str): Optional, 'mse', 'logcosh', or 'none'.
        triplet_recon_weight (float): Weight for reconstruction loss in triplet mode.
        mask_ratio (float): Mask ratio for autoencoder training (0.0 to 1.0).
    """
    def __init__(self, model_params, mode, triplet_recon_loss=None, triplet_recon_weight=1.0, class_weight=None, mask_ratio=0.8, init_margin: float = 0.2, loss_type = "bce", memory_efficient=False):
        super().__init__()
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.loss_type = loss_type
        self.memory_efficient = memory_efficient 

        print(f"model_params: {model_params}")
        
        # if mode == 'classifier': mask_ratio = 0.0
        # elif mode == "tripletloss": mask_ratio = 0.0
        # else: mask_ratio = 0.8

        
        # self.model = AudioMaskedAutoencoderViT(**model_params)
        self.model = AudioMaskedAutoencoderViT(
            num_mels=model_params.get('num_mels', 256), # Same
            mel_len=model_params.get('mel_len', 256), # 
            patch_size=model_params.get('patch_size', 16),
            in_chans=model_params.get('in_chans', 1),
            embed_dim=model_params.get('embed_dim', 256), #192, #256
            encoder_depth= model_params.get('encoder_depth', 6), #8
            num_heads=model_params.get('num_heads', 6), #6
            decoder_embed_dim=model_params.get('decoder_embed_dim', 256), #192, #256
            decoder_depth= model_params.get('decoder_depth', 6), #4, #8
            decoder_num_heads= model_params.get('decoder_num_heads', 6), #4, #6
            mlp_ratio= model_params.get('mlp_ratio', 1), #2
            norm_pix_loss=False,
            num_classes=2
            )
        # Ensure model is in float32 precision
        self.model = self.model.float()

        self.triplet_recon_loss = triplet_recon_loss
        self.triplet_recon_weight = triplet_recon_weight

        if False:
            for i in range(model_params.get('encoder_depth', 6)):
                for param in self.model.encoder.layers[i].parameters():
                    param.requires_grad = False


        if mode == 'classifier':
            # Freeze encoder layers and set up classifier head for 2-class softmax
            # Add class weights to handle class imbalance
            if class_weight is None:
                class_weight = [4, 1.0]  # Default 4:1 ratio
            self.class_weights = torch.tensor(class_weight, dtype=torch.float32)  # [Bad, Good]
            print(f"Using class weights for classifier: Bad={self.class_weights[0]}, Good={self.class_weights[1]} ({self.class_weights[0]}:{self.class_weights[1]} ratio)")
            # Replace the classifier head to output 2 logits
            # in_features = self.model.linear2.in_features if hasattr(self.model, 'linear2') else model_params.get('embed_dim', 256) // 3
            # self.model.linear2 = nn.Linear(in_features, 2, bias=True)
            
        elif mode == 'tripletloss':
            if init_margin is not None:
                self.triplet_loss = LearnableMarginTripletLoss(init_margin)
            else:
                self.triplet_loss = LearnableMarginTripletLoss(0.2)  # Default margin


    def _step_base(self, batch):
        """Forward and loss for autoencoder mode."""
        # imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs, _ = batch
        imgs = imgs.to(self.device)
        loss, _, _ = self.model(imgs, mask_ratio=self.mask_ratio)
        return loss

    def _step_classifier(self, batch):
        """Forward and loss for classifier mode (softmax, CrossEntropyLoss)."""
        mel, label, path_ = batch
        mel = mel.to(self.device)
        label = label.to(self.device)
        logits = self.model.forward_encoder_nomasking_classification(mel)
        
        # For GeneralizedLoss, we need to convert logits to probabilities for the positive class
        if self.loss_type in ['who', 'focal', 'asymmetric', 'auc', 'bce']:
            # Convert logits to probabilities for the positive class (class 1)
            probabilities = torch.sigmoid(logits[:, 1])  # Take probability of positive class
            _loss = GeneralizedLoss(mode=self.loss_type)
            loss = _loss(probabilities, label.float())  # Convert label to float for binary loss
        else:
            assert False, "Unknown loss type: " + self.loss_type
            # Fallback to standard CrossEntropyLoss
            class_weights_device = self.class_weights.to(self.device, dtype=torch.float32)
            ce_loss = nn.CrossEntropyLoss(weight=class_weights_device)
            loss = ce_loss(logits, label)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == label).float().mean().item()
        return loss, accuracy

    def _logcosh_loss(self, pred, target):
        """Numerically stable log-cosh loss."""
        return torch.mean(torch.log(torch.cosh(pred - target)))

    def _step_tripletloss(self, batch):
        """Forward and loss for triplet loss mode, with optional reconstruction loss."""
        anchor, positive, negative = batch
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        a = self.model.forward_encoder_nomasking(anchor)
        p = self.model.forward_encoder_nomasking(positive)
        n = self.model.forward_encoder_nomasking(negative)
        triplet_loss = self.triplet_loss(a, p, n)
        d_pos = (a - p).norm(dim=1)
        d_neg = (a - n).norm(dim=1)
        acc = ((d_pos + 0.2) < d_neg).float().mean().item()
        recon_loss = None
        total_loss = triplet_loss
        # Optional reconstruction loss
        if self.triplet_recon_loss and self.triplet_recon_loss != 'none':
            imgs = anchor
            imgs = imgs.to(self.device)
            loss, pred, mask = self.model(imgs, mask_ratio=0)
            if self.triplet_recon_loss == 'mse':
                recon_loss = F.mse_loss(pred.unsqueeze(1), imgs)
            elif self.triplet_recon_loss == 'logcosh':
                recon_loss = self._logcosh_loss(pred.unsqueeze(1), imgs)
            else:
                raise ValueError(f"Unknown triplet_recon_loss: {self.triplet_recon_loss}")
            total_loss = triplet_loss + self.triplet_recon_weight * recon_loss
        return total_loss, acc, triplet_loss, recon_loss

    def _shared_step(self, batch, step_type):
        """
        Shared logic for training and validation steps.
        Handles logging and returns for all modes.
        """
        on_step = (step_type == "train")
        on_epoch = True
        prog_bar = True
        
        # Get batch size from the batch
        if isinstance(batch, (list, tuple)):
            batch_size = batch[0].size(0) if hasattr(batch[0], 'size') else len(batch[0])
        else:
            batch_size = batch.size(0) if hasattr(batch, 'size') else len(batch)
        
        if self.mode == 'auto':
            loss = self._step_base(batch)
            self.log(f"{step_type}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
            return loss
        elif self.mode == 'classifier':
            celoss, accuracy = self._step_classifier(batch)
            self.log(f"{step_type}_acc", accuracy, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
            if step_type == "val":
                return {"val_loss": celoss, "val_acc": accuracy}
            return celoss
        elif self.mode == 'tripletloss':
            total_loss, acc, triplet_loss, recon_loss = self._step_tripletloss(batch)
            self.log(f"{step_type}_triplet_acc", acc, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
            self.log(f"{step_type}_loss", total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)
            self.log(f"{step_type}_triplet_loss", triplet_loss, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
            if recon_loss is not None:
                self.log(f"{step_type}_recon_loss", recon_loss, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
            if step_type == "val":
                return {"val_loss": total_loss, "val_triplet_acc": acc}
            return total_loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Configure optimizer for all modes."""
        return optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)

    def on_train_epoch_end(self):
        """Shuffle the training dataset at the end of each epoch."""
        train_loader = self.trainer.train_dataloader
        if train_loader is not None and hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'on_epoch_end'):
            train_loader.dataset.on_epoch_end()
        
        # Force garbage collection to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Print memory usage if in memory-efficient mode
        if hasattr(self, 'memory_efficient') and self.memory_efficient:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            # Always print memory usage for debugging
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# ------------------------------------------------------
# --- CONFIG/UTILITY ---
# ------------------------------------------------------
def load_config(yaml_path, cli_args):
    """Load YAML config and override with CLI args (top-level keys only). Model parameters must be under 'model:' in YAML."""
    config = {}
    if yaml_path:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    cli_dict = vars(cli_args)
    # Only override top-level keys, not nested model params
    for k, v in cli_dict.items():
        if v is not None and k != 'model':
            config[k] = v
    return config

def get_output_paths(prefix, base_path):
    """Create output directories for logs and checkpoints."""
    log_dir = os.path.join(base_path, prefix, "logs")
    checkpoint_dir = os.path.join(base_path, prefix, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return {"log_dir": log_dir, "checkpoint_dir": checkpoint_dir}

def save_model_summary(model, log_dir, mode, model_params, mask_ratio=None):
    """Generate and save model summary to log directory."""
    try:
        # Create a dummy input for summary
        if mode == 'auto':
            # For autoencoder, use a single mel spectrogram
            dummy_input = torch.randn(1, 1, 256, 256)  # [batch, channels, mel_bins, time_steps]
        elif mode == 'classifier':
            # For classifier, use a single mel spectrogram
            dummy_input = torch.randn(1, 1, 256, 256)
        elif mode == 'tripletloss':
            # For triplet loss, we'll use a single mel spectrogram for encoder summary
            dummy_input = torch.randn(1, 1, 256, 256)
        else:
            dummy_input = torch.randn(1, 1, 256, 256)
        
        # Generate model summary
        model_summary = summary(
            model.model,
            input_data=dummy_input,
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            col_width=20,
            row_settings=["var_names"]
        )
        
        # Save summary to file
        summary_file = os.path.join(log_dir, "model_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Model Summary for {mode} mode\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Parameters:\n")
            for key, value in model_params.items():
                f.write(f"  {key}: {value}\n")
            if mask_ratio is not None:
                f.write(f"  mask_ratio: {mask_ratio}\n")
            f.write(f"\nModel Architecture Summary:\n")
            f.write(str(model_summary))
            f.write(f"\n\nTotal Parameters: {model_summary.total_params:,}")
            f.write(f"\nTrainable Parameters: {model_summary.trainable_params:,}")
            
            # Handle different torchinfo versions
            try:
                non_trainable = getattr(model_summary, 'non_trainable_params', None)
                if non_trainable is not None:
                    f.write(f"\nNon-trainable Parameters: {non_trainable:,}")
                else:
                    # Calculate non-trainable parameters manually
                    total_params = sum(p.numel() for p in model.model.parameters())
                    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                    non_trainable = total_params - trainable_params
                    f.write(f"\nNon-trainable Parameters: {non_trainable:,}")
            except AttributeError:
                # Calculate non-trainable parameters manually
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                non_trainable = total_params - trainable_params
                f.write(f"\nNon-trainable Parameters: {non_trainable:,}")
        
        print(f"Model summary saved to: {summary_file}")
        print(f"Total parameters: {model_summary.total_params:,}")
        print(f"Trainable parameters: {model_summary.trainable_params:,}")
        try:
            non_trainable_print = getattr(model_summary, 'non_trainable_params', None)
            if non_trainable_print is not None:
                print(f"Non-trainable parameters: {non_trainable_print:,}")
            else:
                print(f"Non-trainable parameters: {non_trainable:,}")
        except AttributeError:
            print(f"Non-trainable parameters: {non_trainable:,}")
        
        return model_summary
        
    except Exception as e:
        print(f"Warning: Could not generate model summary: {e}")
        return None

# ------------------------------------------------------
# --- MAIN ENTRY POINT ---
# ------------------------------------------------------
def main():
    """Main function to parse arguments, load config, set up data, model, and run training."""
    parser = argparse.ArgumentParser(description="Combined Lightning Training Script")
    parser.add_argument('--config', type=str, default="default_config.yaml", help='Path to YAML config file')
    parser.add_argument('--mode_type', type=str, default=None, choices=['auto', 'classifier', 'tripletloss'], help='Training mode')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--base_path', type=str, default='.', help='Base path for data and outputs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path (overrides config)')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for all output/log files')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 for main process, reduces memory usage)')
    parser.add_argument('--apply_specaugment', action='store_true', help='Override apply_specaugment for classifier mode (False/True)')
    parser.add_argument('--triplet_recon_loss', type=str, default=None, help='Reconstruction loss for tripletloss mode: none, mse, logcosh')
    parser.add_argument('--triplet_recon_weight', type=float, default=1.0, help='Weight for reconstruction loss in tripletloss mode')
    parser.add_argument('--class_weight', type=float, nargs=2, default=None, help='Class weights for classifier mode [bad_class_weight, good_class_weight]')
    parser.add_argument('--start_epoch', type=int, default=None, help='Starting epoch (overrides config)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs (overrides config)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (overrides config)')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Mask ratio for autoencoder training (0.0 to 1.0)')
    parser.add_argument('--extend_bad', action='store_true', help='Extend bad data for classifier mode (True/False)')
    parser.add_argument('--base_dir', type=str, default='data/dia_tmp_2', help='Base directory for data (overrides config)')
    parser.add_argument('--loss_type', type = str, default='who',help= 'Options: who, focal, asymetrical, auc, bce. Refer to loss_functions for more info on what choices mean ' )
    parser.add_argument('--use_filter_dir', type=str, default=None, 
                       help="Use evaluation directory with dynamic filtering for training. Path to directory containing train_recon.csv and val_recon.csv.")
    parser.add_argument('--filter_mode', type=str, choices=['median', 'mean', 'percentile', 'hardcode'], 
                       help='Filtering mode for dynamic filtering: median (IQR-based), mean (std-based), percentile (percentile-based), hardcode (fixed range)')
    parser.add_argument('--filter_params', type=str, nargs='+', 
                       help='Filter parameters: median/mean use "k" (default 1.5), percentile uses "lower upper" (default 0.05 0.95), hardcode uses "min max" (default 1.5 7.0)')
    parser.add_argument('--hard_low_cut', type=float, help='Hard lower cutoff (only for median/mean modes)')
    parser.add_argument('--hard_high_cut', type=float, help='Hard upper cutoff (only for median/mean modes)')
    parser.add_argument('--energy_filter_mode', type=str, choices=['median', 'mean', 'percentile', 'hardcode'], 
                       help='Energy filtering mode (if different from filter_mode). If not specified, uses same mode as filter_mode.')
    parser.add_argument('--energy_filter_params', type=str, nargs='+', 
                       help='Energy filter parameters (same format as filter_params but for energy filtering)')
    parser.add_argument('--energy_hard_low_cut', type=float, help='Energy hard lower cutoff (only for median/mean modes)')
    parser.add_argument('--energy_hard_high_cut', type=float, help='Energy hard upper cutoff (only for median/mean modes)')
    parser.add_argument('--memory_efficient', action='store_true', help='Enable memory-efficient mode (reduces DataLoader memory usage)')
    args = parser.parse_args()

    config = load_config(args.config, args)

    filtered_data_path = None

    paths = get_output_paths(config.get('prefix', ''), config.get('base_path', '.'))
    model_params = config.get('model', {})

    mode = config['mode_type'] if 'mode_type' in config else config.get('mode_type', 'auto')
    # mode = config['mode_type'] if 'mode_type' in config else config.get('mode_type', 'auto')
    print(f"Using mode: {mode}")
    print(f"Using precision: float32")
    num_workers = config.get('num_workers', 0)  # Changed default from 3 to 0
    persistent_workers = False  # Disable persistent workers to save memory
    apply_specaugment = config.get('apply_specaugment', False)
    triplet_recon_loss = config.get('triplet_recon_loss', None)
    triplet_recon_weight = config.get('triplet_recon_weight', 1.0)
    class_weight = config.get('class_weight', [4.0, 1.0])
    mask_ratio = config.get('mask_ratio', 0.8)
    loss_type = config.get('loss_type', 'who')
    # Validate mask ratio
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"Mask ratio must be between 0.0 and 1.0, got {mask_ratio}")
    
    # Get training parameters with CLI overrides
    start_epoch = config.get('start_epoch', 0)
    max_epochs = config.get('max_epochs', 500)
    patience = config.get('patience', 25)
    
    # --- DATASET/MODEL SELECTION --
    base_dir = config.get('base_dir', 'data/dia_tmp_2')
    
    print(f"Training parameters:")
    print(f"  - Start epoch: {start_epoch}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Patience: {patience}")
    print(f"  - Mask ratio: {mask_ratio}")
    print(f"  - Mode: {mode}")
    print(f"  - Base directory: {base_dir}")
    
    # Print memory usage info
    if torch.cuda.is_available():
        print(f"  - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  - GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")



    # Save config at the start
    with open(os.path.join(paths['log_dir'], "config_used.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

    # --- DATASET/MODEL SELECTION --
    data_dir_good_train = [f"{base_dir}/train/good"]
    data_dir_bad_train  = [f"{base_dir}/train/bad"]
    data_dir_good_val = [f"{base_dir}/val/good"] 
    data_dir_bad_val = [f"{base_dir}/val/bad"]

    data_dir_train = [f"{base_dir}/train/good", f"{base_dir}/train/bad"]
    data_dir_val = [f"{base_dir}/val/good", f"{base_dir}/val/bad"]


    # Parse filter parameters if provided
    filter_params = None
    if args.filter_params:
        if args.filter_mode in ['median', 'mean']:
            if len(args.filter_params) >= 1:
                filter_params = {'k': float(args.filter_params[0])}
        elif args.filter_mode == 'percentile':
            if len(args.filter_params) >= 2:
                filter_params = {'lower_percentile': float(args.filter_params[0]), 'upper_percentile': float(args.filter_params[1])}
        elif args.filter_mode == 'hardcode':
            if len(args.filter_params) >= 2:
                filter_params = {'lower_bound': float(args.filter_params[0]), 'upper_bound': float(args.filter_params[1])}
    
    # Add hard cutoffs if specified (only for median/mean)
    if filter_params and args.filter_mode in ['median', 'mean']:
        if args.hard_low_cut is not None:
            filter_params['hard_low_cut'] = args.hard_low_cut
        if args.hard_high_cut is not None:
            filter_params['hard_high_cut'] = args.hard_high_cut
    
    # Parse energy filter parameters if provided
    energy_filter_mode = args.energy_filter_mode
    if args.energy_filter_params:
        if energy_filter_mode in ['median', 'mean']:
            if len(args.energy_filter_params) >= 1:
                filter_params['energy_k'] = float(args.energy_filter_params[0])
        elif energy_filter_mode == 'percentile':
            if len(args.energy_filter_params) >= 2:
                filter_params['energy_lower_percentile'] = float(args.energy_filter_params[0])
                filter_params['energy_upper_percentile'] = float(args.energy_filter_params[1])
        elif energy_filter_mode == 'hardcode':
            if len(args.energy_filter_params) >= 2:
                filter_params['energy_lower_bound'] = float(args.energy_filter_params[0])
                filter_params['energy_upper_bound'] = float(args.energy_filter_params[1])
    
    # Add energy hard cutoffs if specified (only for median/mean)
    if energy_filter_mode in ['median', 'mean']:
        if args.energy_hard_low_cut is not None:
            filter_params['energy_hard_low_cut'] = args.energy_hard_low_cut
        if args.energy_hard_high_cut is not None:
            filter_params['energy_hard_high_cut'] = args.energy_hard_high_cut

    if mode == 'classifier' :
        train_ds = AudioDataset(
            mode='classifier',
            data_dir_good = data_dir_good_train,
            data_dir_bad  = data_dir_bad_train,
            apply_specaugment=apply_specaugment, 
            extend_bad = args.extend_bad,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
        val_ds = AudioDataset(
            mode='classifier',
            data_dir_good = data_dir_good_val,
            data_dir_bad  = data_dir_bad_val,
            apply_specaugment= False,
            extend_bad = False,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
    elif mode == 'tripletloss':
        train_ds = AudioDataset(
            mode='triplet',
            data_dir_good = data_dir_good_train,
            data_dir_bad  = data_dir_bad_train,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
        val_ds = AudioDataset(
            mode='triplet',
            data_dir_good = data_dir_good_val,
            data_dir_bad  = data_dir_bad_val,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
    elif mode == 'auto':
        train_ds = AudioDataset(
            mode='auto',
            data_dir_good = data_dir_train,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
        val_ds = AudioDataset(
            mode='auto',
            data_dir_good= data_dir_val,
            filter_dir = args.use_filter_dir,
            filter_mode = args.filter_mode,
            filter_params = filter_params,
            memory_efficient = args.memory_efficient,
            energy_filter_mode = energy_filter_mode
        )
    else:
        raise ValueError(f"Unknown mode_type: {mode}")

    model = UnifiedLightningModel(model_params, mode, triplet_recon_loss=triplet_recon_loss, triplet_recon_weight=triplet_recon_weight, class_weight=class_weight, loss_type=loss_type, mask_ratio=mask_ratio, init_margin=0.2, memory_efficient=args.memory_efficient)

    # Generate and save model summary
    print("Generating model summary...")
    try:
        save_model_summary(model, paths['log_dir'], mode, model_params, mask_ratio)
    except Exception as e:
        print(f"Warning: Could not generate model summary: {e}")

    batch_size = config.get('batch_size', 128)
    
    # Memory-efficient DataLoader settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': False,  # Disable pin_memory to save RAM
        'persistent_workers': persistent_workers,
        'drop_last': False,  # Keep all samples
        'prefetch_factor': 2 if num_workers > 0 else None,  # Reduce prefetch factor
    }
    
    # Add memory-efficient options
    if args.memory_efficient:
        dataloader_kwargs.update({
            'prefetch_factor': 1,  # Minimal prefetching
            'generator': torch.Generator(device='cpu'),  # Use CPU generator
        })
        print("Memory-efficient DataLoader settings enabled")
    
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dataloader_kwargs
    )

    # --- CHECKPOINT LOGIC ---
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("--------------------------------")
        print("Loading checkpoint")
        print("--------------------------------")
        print(f"Loading checkpoint {checkpoint_path} ...")
        model = UnifiedLightningModel.load_from_checkpoint(checkpoint_path, model_params=model_params, mode=mode, triplet_recon_loss=triplet_recon_loss, triplet_recon_weight=triplet_recon_weight, class_weight=class_weight, loss_type=loss_type, mask_ratio=mask_ratio, init_margin=0.2, memory_efficient=args.memory_efficient, strict=False)
        # Generate model summary even when loading from checkpoint
        print("Generating model summary for loaded checkpoint...")
        try:
            save_model_summary(model, paths['log_dir'], mode, model_params, mask_ratio)
        except Exception as e:
            print(f"Warning: Could not generate model summary for checkpoint: {e}")
    else:         
        print(f"No checkpoint found")
    
    # --- CALLBACKS ---
    callbacks = []
    if mode == 'classifier':
        monitor_metric = "val_acc"
        filename_str = f"{{epoch:02d}}-{{train_acc:.4f}}-{{val_acc:.4f}}"
        print(f"DEBUG: Classifier mode - monitoring {monitor_metric} with filename {filename_str}")
    elif mode == 'tripletloss':
        monitor_metric = "val_triplet_acc"
        filename_str = f"{{epoch:02d}}-{{train_triplet_acc:.4f}}-{{val_triplet_acc:.4f}}"
        print(f"DEBUG: Tripletloss mode - monitoring {monitor_metric} with filename {filename_str}")
    else:
        monitor_metric = "val_loss"
        filename_str = f"{{epoch:02d}}-{{train_loss:.4f}}-{{val_loss:.4f}}"
        print(f"DEBUG: {mode} mode - monitoring {monitor_metric} with filename {filename_str}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths['checkpoint_dir'],
        filename=filename_str,
        save_top_k=5,
        monitor=monitor_metric,
        mode='max' if mode in ['classifier', 'tripletloss'] else 'min',
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=patience, verbose=True, mode='max' if mode in ['classifier', 'tripletloss'] else 'min')
    callbacks.append(early_stopping)

    # --- TRAINER ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,  # Use float32 precision for consistency
        accumulate_grad_batches=2,
        max_epochs=max_epochs,
        fast_dev_run=False,
        num_sanity_val_steps=0,  # Disable sanity check to save memory
        log_every_n_steps=30,
        # batch_size=batch_size,
        callbacks=callbacks,
        logger=CSVLogger(paths['log_dir']),
        enable_model_summary=False,  # Disable model summary to save memory
        enable_progress_bar=True,
        enable_checkpointing=True
    )
    
    # Handle starting from a specific epoch
    # if start_epoch > 0:
    #     print(f"Starting training from epoch {start_epoch}")
    #     # Note: PyTorch Lightning doesn't directly support starting from a specific epoch
    #     # This would require custom logic or using a checkpoint that starts from that epoch
    #     # For now, we'll just log the intention
    #     print(f"Note: start_epoch={start_epoch} is set but not implemented. Use checkpoint_path to resume from a specific checkpoint.")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()

