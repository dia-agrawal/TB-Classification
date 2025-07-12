# ------------------------------------------------------
# Unified Lightning Training Script
# ------------------------------------------------------
import argparse
import yaml
import os
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
from torchinfo import summary

torch.set_float32_matmul_precision('high')

# ------------------------------------------------------
# --- DATASETS ---
# ------------------------------------------------------
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
    def __init__(self, mode, data_dir_good=None, data_dir_bad=None, apply_specaugment=False, shuffle=True, extend_bad = False):
        self.mode = mode
        self.apply_specaugment = apply_specaugment
        self.freq_mask = FrequencyMasking(freq_mask_param=8)
        self.time_mask = TimeMasking(time_mask_param=20)
        if mode == 'auto':
            # Load all .pt files from data_dir_good
            if type(data_dir_good) is not list:
                data_dir_good = [data_dir_good]
            self.files = []
            for d in data_dir_good:
                if d is not None:  # Type guard
                    t = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pt')]
                    self.files.extend(t)
        elif mode == 'classifier':
            # Build a list of (filepath, label) for binary classification
            self.samples = []
            if type(data_dir_good) is not list:
                data_dir_good = [data_dir_good]
            if type(data_dir_bad) is not list:
                data_dir_bad = [data_dir_bad]
            for d in data_dir_good:
                if d is not None:  # Type guard
                    self.samples.extend([(os.path.join(d, f), 1) for f in os.listdir(d) if f.endswith('.pt')])
            for d in data_dir_bad:
                if d is not None:  # Type guard
                    tmp = [(os.path.join(d, f), 0) for f in os.listdir(d) if f.endswith('.pt')]
                    self.samples.extend(tmp)
                    if extend_bad:
                        self.samples.extend(tmp*2)
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
            self.anchorfile = self.goodfile.copy()
            if shuffle:
                random.shuffle(self.anchorfile)
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
            mel = torch.load(path)
            mel = mel[:, 1:]
            mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
            return mel.unsqueeze(0).float()  # Ensure float32 precision
        elif self.mode == 'classifier':
            # Return (mel, label) for binary classification
            path, label = self.samples[idx]
            mel = torch.load(path)
            mel = mel[:, 1:]
            mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
            if self.apply_specaugment and label == 0:
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
                mel_good = torch.load(goodpath)
                mel_bad = torch.load(badpath)
                mel_anchor = torch.load(anchorpath)
                mel_good = mel_good[:, 1:]
                mel_good = F.pad(input=mel_good, pad=(0, 0, 0, 59), mode='constant', value=0)
                mel_bad = mel_bad[:, 1:]
                mel_bad = F.pad(input=mel_bad, pad=(0, 0, 0, 59), mode='constant', value=0)
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
    def __init__(self, model_params, mode, triplet_recon_loss=None, triplet_recon_weight=1.0, class_weight=None, mask_ratio=0.8):
        super().__init__()
        self.mode = mode
        self.mask_ratio = mask_ratio

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
                class_weight = [1.5, 1.0]  # Default 4:1 ratio
            self.class_weights = torch.tensor(class_weight, dtype=torch.float32)  # [Bad, Good]
            print(f"Using class weights for classifier: Bad={self.class_weights[0]}, Good={self.class_weights[1]} ({self.class_weights[0]}:{self.class_weights[1]} ratio)")
            # Replace the classifier head to output 2 logits
            # in_features = self.model.linear2.in_features if hasattr(self.model, 'linear2') else model_params.get('embed_dim', 256) // 3
            # self.model.linear2 = nn.Linear(in_features, 2, bias=True)
            
        elif mode == 'tripletloss':
            self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.7, p=2.0)

    def _step_base(self, batch):
        """Forward and loss for autoencoder mode."""
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(self.device)
        loss, _, _ = self.model(imgs, mask_ratio=self.mask_ratio)
        return loss

    def _step_classifier(self, batch):
        """Forward and loss for classifier mode (softmax, CrossEntropyLoss)."""
        mel, label, path_ = batch
        mel = mel.to(self.device)
        label = label.to(self.device)
        logits = self.model.forward_encoder_nomasking_classification(mel)
        
        # Create loss function with class weights on the correct device and precision
        class_weights_device = self.class_weights.to(self.device, dtype=torch.float32)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights_device)
        celoss = ce_loss(logits, label)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == label).float().mean().item()
        return celoss, accuracy

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
        acc = (d_pos < d_neg).float().mean().item()
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
        if self.mode == 'auto':
            loss = self._step_base(batch)
            self.log(f"{step_type}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
            return loss
        elif self.mode == 'classifier':
            celoss, accuracy = self._step_classifier(batch)
            self.log(f"{step_type}_acc", accuracy, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
            if step_type == "val":
                return {"val_loss": celoss, "val_acc": accuracy}
            return celoss
        elif self.mode == 'tripletloss':
            total_loss, acc, triplet_loss, recon_loss = self._step_tripletloss(batch)
            self.log(f"{step_type}_triplet_acc", acc, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
            self.log(f"{step_type}_loss", total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
            self.log(f"{step_type}_triplet_loss", triplet_loss, on_step=on_step, on_epoch=on_epoch)
            if recon_loss is not None:
                self.log(f"{step_type}_recon_loss", recon_loss, on_step=on_step, on_epoch=on_epoch)
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
    parser.add_argument('--num_workers', type=int, default=3, help='Number of DataLoader workers')
    parser.add_argument('--apply_specaugment', type=int, help='Override apply_specaugment for classifier mode (1=True, 0=False)')
    parser.add_argument('--triplet_recon_loss', type=str, default=None, help='Reconstruction loss for tripletloss mode: none, mse, logcosh')
    parser.add_argument('--triplet_recon_weight', type=float, default=1.0, help='Weight for reconstruction loss in tripletloss mode')
    parser.add_argument('--class_weight', type=float, nargs=2, default=None, help='Class weights for classifier mode [bad_class_weight, good_class_weight]')
    parser.add_argument('--start_epoch', type=int, default=None, help='Starting epoch (overrides config)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs (overrides config)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (overrides config)')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Mask ratio for autoencoder training (0.0 to 1.0)')
    parser.add_argument('--extend_bad', action='store_true', help='Extend bad data for classifier mode (True/False)')
    parser.add_argument('--base_dir', type=str, default='data/dia_tmp_2', help='Base directory for data (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config, args)

    paths = get_output_paths(config.get('prefix', ''), config.get('base_path', '.'))
    model_params = config.get('model', {})

    mode = config['mode_type'] if 'mode_type' in config else config.get('mode_type', 'auto')
    # mode = config['mode_type'] if 'mode_type' in config else config.get('mode_type', 'auto')
    print(f"Using mode: {mode}")
    print(f"Using precision: float32")
    num_workers = config.get('num_workers', 3)
    persistent_workers = num_workers > 0
    apply_specaugment = config.get('apply_specaugment', False)
    triplet_recon_loss = config.get('triplet_recon_loss', None)
    triplet_recon_weight = config.get('triplet_recon_weight', 1.0)
    class_weight = config.get('class_weight', [4.0, 1.0])
    mask_ratio = config.get('mask_ratio', 0.8)
    
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


    if mode == 'classifier' :
        train_ds = AudioDataset(
            mode='classifier',
            data_dir_good = data_dir_good_train,
            data_dir_bad  = data_dir_bad_train,
            apply_specaugment=apply_specaugment if apply_specaugment is not None else True, 
            extend_bad = args.extend_bad
        )
        val_ds = AudioDataset(
            mode='classifier',
            data_dir_good = data_dir_good_val,
            data_dir_bad  = data_dir_bad_val,
            apply_specaugment= False,
            extend_bad = False
        )
    elif mode == 'tripletloss':
        train_ds = AudioDataset(
            mode='triplet',
            data_dir_good = data_dir_good_train,
            data_dir_bad  = data_dir_bad_train,
        )
        val_ds = AudioDataset(
            mode='triplet',
            data_dir_good = data_dir_good_val,
            data_dir_bad  = data_dir_bad_val,
        )
    elif mode == 'auto':
        train_ds = AudioDataset(
            mode='auto',
            data_dir_good = data_dir_train,
        )
        val_ds = AudioDataset(
            mode='auto',
            data_dir_good= data_dir_val,
        )
    else:
        raise ValueError(f"Unknown mode_type: {mode}")

    model = UnifiedLightningModel(model_params, mode, triplet_recon_loss=triplet_recon_loss, triplet_recon_weight=triplet_recon_weight, class_weight=class_weight, mask_ratio=mask_ratio)

    # Generate and save model summary
    print("Generating model summary...")
    save_model_summary(model, paths['log_dir'], mode, model_params, mask_ratio)

    batch_size = config.get('batch_size', 64 if mode != 'auto' else 512)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    # --- CHECKPOINT LOGIC ---
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("--------------------------------")
        print("Loading checkpoint")
        print("--------------------------------")
        print(f"Loading checkpoint {checkpoint_path} ...")
        model = UnifiedLightningModel.load_from_checkpoint(checkpoint_path, model_params=model_params, mode=mode, triplet_recon_loss=triplet_recon_loss, triplet_recon_weight=triplet_recon_weight, class_weight=class_weight, mask_ratio=mask_ratio, strict=False)
        # Generate model summary even when loading from checkpoint
        print("Generating model summary for loaded checkpoint...")
        save_model_summary(model, paths['log_dir'], mode, model_params, mask_ratio)
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
        num_sanity_val_steps=0,
        log_every_n_steps=30,
        callbacks=callbacks,
        logger=CSVLogger(paths['log_dir'])
    )
    
    # Handle starting from a specific epoch
    if start_epoch > 0:
        print(f"Starting training from epoch {start_epoch}")
        # Note: PyTorch Lightning doesn't directly support starting from a specific epoch
        # This would require custom logic or using a checkpoint that starts from that epoch
        # For now, we'll just log the intention
        print(f"Note: start_epoch={start_epoch} is set but not implemented. Use checkpoint_path to resume from a specific checkpoint.")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Save config used for this run (again at end)
    with open(os.path.join(paths['log_dir'], f"{config.get('prefix', '')}_config_used.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

if __name__ == "__main__":
    main()

