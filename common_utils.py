#!/usr/bin/env python3
"""
Common Utilities for Audio Classification Models
===============================================

This module contains shared utilities used across training and inference scripts.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path


# Add the Audio_mae path
sys.path.append('/mnt/ssd_data/data/dia/ast/Audio_mae')


class AudioDataProcessor:
    """Utilities for processing audio data."""
    
    @staticmethod
    def preprocess_mel(mel):
        """
        Standard preprocessing for mel spectrograms.
        
        Args:
            mel (torch.Tensor): Raw mel spectrogram
            
        Returns:
            torch.Tensor: Preprocessed mel spectrogram
        """
        mel = mel[:, 1:]  # Remove first column
        mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
        return mel
    
    @staticmethod
    def setup_device(device='auto'):
        """
        Setup device for model inference/training.
        
        Args:
            device (str): Device specification ('auto', 'cuda', 'cpu')
            
        Returns:
            str: Device to use
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device


class ModelLoader:
    """Utilities for loading and managing models."""
    
    @staticmethod
    def load_checkpoint(checkpoint_path, device, mode='classifier', **kwargs):
        """
        Load model from checkpoint with proper parameter extraction.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            device (str): Device to load model on
            mode (str): Model mode
            **kwargs: Additional model parameters
            
        Returns:
            tuple: (model, extracted_parameters)
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model parameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            model_params = hparams.get('model_params', {})
            mode = hparams.get('mode', mode)
            triplet_recon_loss = hparams.get('triplet_recon_loss', None)
            triplet_recon_weight = hparams.get('triplet_recon_weight', 1.0)
        else:
            # Fallback to default parameters
            model_params = {}
            mode = mode
            triplet_recon_loss = None
            triplet_recon_weight = 1.0
        
        # Override with kwargs if provided
        if 'triplet_recon_loss' in kwargs:
            triplet_recon_loss = kwargs['triplet_recon_loss']
        if 'triplet_recon_weight' in kwargs:
            triplet_recon_weight = kwargs['triplet_recon_weight']
        
        extracted_params = {
            'model_params': model_params,
            'mode': mode,
            'triplet_recon_loss': triplet_recon_loss,
            'triplet_recon_weight': triplet_recon_weight,
            'state_dict': checkpoint['state_dict']
        }
        
        return extracted_params


class ConfigManager:
    """Utilities for managing configuration."""
    
    # Default model parameters
    DEFAULT_MODEL_PARAMS = {
        'num_mels': 256,
        'mel_len': 256,
        'patch_size': 16,
        'in_chans': 1,
        'embed_dim': 256,
        'encoder_depth': 6,
        'num_heads': 6,
        'decoder_embed_dim': 256,
        'decoder_depth': 6,
        'decoder_num_heads': 6,
        'mlp_ratio': 1,
        'norm_pix_loss': False,
    }
    
    # Default data paths
    DATA_PATHS = {
        'train_good': "data/dia_tmp/train/good",
        'train_bad': "data/dia_tmp/train/bad",
        'val_good': "data/dia_tmp/val/good",
        'val_bad': "data/dia_tmp/val/bad",
        'test_good': "data/dia_tmp/test/good",
        'test_bad': "data/dia_tmp/test/bad",
    }
    
    @staticmethod
    def load_config(yaml_path, cli_args):
        """
        Load YAML config and override with CLI args.
        
        Args:
            yaml_path (str): Path to YAML config file
            cli_args: Command line arguments
            
        Returns:
            dict: Combined configuration
        """
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
    
    @staticmethod
    def get_output_paths(prefix, base_path):
        """
        Create output directories for logs and checkpoints.
        
        Args:
            prefix (str): Prefix for output directories
            base_path (str): Base path for outputs
            
        Returns:
            dict: Dictionary with log_dir and checkpoint_dir
        """
        log_dir = os.path.join(base_path, prefix, "logs")
        checkpoint_dir = os.path.join(base_path, prefix, "checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return {"log_dir": log_dir, "checkpoint_dir": checkpoint_dir}


class DataPathManager:
    """Utilities for managing data paths."""
    
    @staticmethod
    def get_data_paths(split='train'):
        """
        Get data paths for a specific split.
        
        Args:
            split (str): Data split ('train', 'val', 'test')
            
        Returns:
            dict: Dictionary with good and bad data paths
        """
        base_paths = ConfigManager.DATA_PATHS
        
        if split == 'train':
            return {
                'good': [base_paths['train_good']],
                'bad': [base_paths['train_bad']]
            }
        elif split == 'val':
            return {
                'good': [base_paths['val_good']],
                'bad': [base_paths['val_bad']]
            }
        elif split == 'test':
            return {
                'good': [base_paths['test_good']],
                'bad': [base_paths['test_bad']]
            }
        else:
            raise ValueError(f"Unknown split: {split}")
    
    @staticmethod
    def validate_data_paths(data_paths):
        """
        Validate that data paths exist.
        
        Args:
            data_paths (dict): Dictionary with good and bad data paths
            
        Returns:
            bool: True if all paths exist
        """
        for split, paths in data_paths.items():
            for path in paths:
                if not os.path.exists(path):
                    print(f"Warning: Data path does not exist: {path}")
                    return False
        return True


class ModelUtils:
    """Utilities for model operations."""
    
    @staticmethod
    def get_model_forward_method(mode):
        """
        Get the appropriate forward method name for a given mode.
        
        Args:
            mode (str): Model mode ('auto', 'classifier', 'tripletloss')
            
        Returns:
            str: Forward method name
        """
        if mode == 'classifier':
            return 'forward_encoder_nomasking_classification'
        elif mode == 'tripletloss':
            return 'forward_encoder_nomasking'
        elif mode == 'auto':
            return 'forward'  # Standard forward method
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @staticmethod
    def get_monitor_metric(mode):
        """
        Get the appropriate monitor metric for a given mode.
        
        Args:
            mode (str): Model mode
            
        Returns:
            str: Monitor metric name
        """
        if mode == 'classifier':
            return 'val_acc'
        else:
            return 'val_loss'
    
    @staticmethod
    def get_checkpoint_mode(mode):
        """
        Get the checkpoint mode (max/min) for a given model mode.
        
        Args:
            mode (str): Model mode
            
        Returns:
            str: Checkpoint mode ('max' or 'min')
        """
        if mode == 'classifier':
            return 'max'  # Maximize accuracy
        else:
            return 'min'  # Minimize loss


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    def find_audio_files(directory, extension='.pt'):
        """
        Find all audio files in a directory recursively.
        
        Args:
            directory (str): Directory to search
            extension (str): File extension to search for
            
        Returns:
            list: List of file paths
        """
        audio_files = []
        for ext in [extension]:
            audio_files.extend(Path(directory).rglob(f'*{ext}'))
        return [str(f) for f in audio_files]
    
    @staticmethod
    def save_config(config, output_path):
        """
        Save configuration to file.
        
        Args:
            config (dict): Configuration dictionary
            output_path (str): Output file path
        """
        with open(output_path, 'w') as f:
            yaml.safe_dump(config, f)
    
    @staticmethod
    def ensure_directory(path):
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            path (str): Directory path
        """
        os.makedirs(path, exist_ok=True)


class MetricsUtils:
    """Utilities for metrics calculation."""
    
    @staticmethod
    def calculate_accuracy(predictions, labels):
        """
        Calculate accuracy from predictions and labels.
        
        Args:
            predictions (np.ndarray): Model predictions
            labels (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        return np.mean(np.array(predictions) == np.array(labels))
    
    @staticmethod
    def calculate_embedding_stats(embeddings):
        """
        Calculate statistics for embeddings.
        
        Args:
            embeddings (np.ndarray): Embedding vectors
            
        Returns:
            dict: Embedding statistics
        """
        norms = np.linalg.norm(embeddings, axis=1)
        return {
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'mean_embedding': np.mean(embeddings, axis=0),
            'std_embedding': np.std(embeddings, axis=0),
            'min_norm': np.min(norms),
            'max_norm': np.max(norms)
        }
    
    @staticmethod
    def calculate_loss_stats(losses):
        """
        Calculate statistics for loss values.
        
        Args:
            losses (list): List of loss values
            
        Returns:
            dict: Loss statistics
        """
        losses = np.array(losses)
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            'median_loss': np.median(losses)
        }


# Convenience functions for backward compatibility
def preprocess_mel(mel):
    """Convenience function for mel preprocessing."""
    return AudioDataProcessor.preprocess_mel(mel)


def setup_device(device='auto'):
    """Convenience function for device setup."""
    return AudioDataProcessor.setup_device(device)


def load_checkpoint(checkpoint_path, device, mode='classifier', **kwargs):
    """Convenience function for checkpoint loading."""
    return ModelLoader.load_checkpoint(checkpoint_path, device, mode, **kwargs)


def get_data_paths(split='train'):
    """Convenience function for getting data paths."""
    return DataPathManager.get_data_paths(split)


def get_monitor_metric(mode):
    """Convenience function for getting monitor metric."""
    return ModelUtils.get_monitor_metric(mode)


def get_checkpoint_mode(mode):
    """Convenience function for getting checkpoint mode."""
    return ModelUtils.get_checkpoint_mode(mode) 