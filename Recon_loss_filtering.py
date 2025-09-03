import numpy as np
import pandas as pd
from pathlib import Path
import os

def filter_median_iqr(losses, k=1.5, hard_low_cut=None, hard_high_cut=None):
    """
    Median + IQR-based filtering (Robust to outliers)
    
    Args:
        losses: List of reconstruction losses
        k: IQR multiplier (default 1.5)
        hard_low_cut: Hard lower cutoff (overrides IQR lower bound if provided)
        hard_high_cut: Hard upper cutoff (overrides IQR upper bound if provided)
    
    Returns:
        mask: Boolean array indicating which samples to keep
    """
    losses = np.array(losses)
    q1 = np.percentile(losses, 25)
    q3 = np.percentile(losses, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Apply hard cutoffs if provided
    if hard_low_cut is not None:
        lower_bound = max(lower_bound, hard_low_cut)
    if hard_high_cut is not None:
        upper_bound = min(upper_bound, hard_high_cut)
    
    mask = (losses >= lower_bound) & (losses <= upper_bound)
    return mask

def filter_mean_std(losses, k=1.5, hard_low_cut=None, hard_high_cut=None):
    """
    Mean ± k*std filtering (Assumes roughly normal)
    
    Args:
        losses: List of reconstruction losses
        k: Standard deviation multiplier (default 1.5)
        hard_low_cut: Hard lower cutoff (overrides std lower bound if provided)
        hard_high_cut: Hard upper cutoff (overrides std upper bound if provided)
    
    Returns:
        mask: Boolean array indicating which samples to keep
    """
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    lower_bound = mean_loss - k * std_loss
    upper_bound = mean_loss + k * std_loss
    
    # Apply hard cutoffs if provided
    if hard_low_cut is not None:
        lower_bound = max(lower_bound, hard_low_cut)
    if hard_high_cut is not None:
        upper_bound = min(upper_bound, hard_high_cut)
    
    mask = (losses >= lower_bound) & (losses <= upper_bound)
    return mask

def filter_percentile(losses, lower_percentile=0.05, upper_percentile=0.95):
    """
    Percentile-based clipping
    
    Args:
        losses: List of reconstruction losses
        lower_percentile: Lower percentile cutoff (default 0.05 = 5th percentile)
        upper_percentile: Upper percentile cutoff (default 0.95 = 95th percentile)
    
    Returns:
        mask: Boolean array indicating which samples to keep
    """
    losses = np.array(losses)
    lower_bound = np.percentile(losses, lower_percentile * 100)
    upper_bound = np.percentile(losses, upper_percentile * 100)
    
    mask = (losses >= lower_bound) & (losses <= upper_bound)
    return mask

def filter_hardcode(losses, lower_bound=1.5, upper_bound=7.0):
    """
    Fixed range cutoff
    
    Args:
        losses: List of reconstruction losses
        lower_bound: Lower cutoff value (default 1.5)
        upper_bound: Upper cutoff value (default 7.0)
    
    Returns:
        mask: Boolean array indicating which samples to keep
    """
    losses = np.array(losses)
    mask = (losses >= lower_bound) & (losses <= upper_bound)
    return mask

def apply_filtering(losses, paths, filter_mode, **filter_params):
    """
    Apply filtering to losses and paths and return filtered data
    
    Args:
        losses: List of reconstruction losses
        paths: List of corresponding file paths
        filter_mode: Filtering mode ('median', 'mean', 'percentile', 'hardcode')
        **filter_params: Additional parameters for the specific filter mode
    
    Returns:
        tuple: (filtered_losses, filtered_paths, mask)
    """
    losses = np.array(losses)
    paths = np.array(paths)
    
    # Apply the appropriate filter
    if filter_mode == 'median':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        mask = filter_median_iqr(losses, k, hard_low_cut, hard_high_cut)
        
        q1 = np.percentile(losses, 25)
        q3 = np.percentile(losses, 75)
        iqr = q3 - q1
        iqr_lower = q1 - k * iqr
        iqr_upper = q3 + k * iqr
        
        # Calculate final bounds
        final_lower = max(iqr_lower, hard_low_cut) if hard_low_cut is not None else iqr_lower
        final_upper = min(iqr_upper, hard_high_cut) if hard_high_cut is not None else iqr_upper
        print(f"Applied median+IQR filtering with k={k}")
        print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        print(f"IQR bounds: [{iqr_lower:.4f}, {iqr_upper:.4f}]")
        if hard_low_cut is not None:
            print(f"Hard low cutoff: {hard_low_cut:.4f}")
        if hard_high_cut is not None:
            print(f"Hard high cutoff: {hard_high_cut:.4f}")
        print(f"Final bounds: [{final_lower:.4f}, {final_upper:.4f}]")
        
    elif filter_mode == 'mean':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        mask = filter_mean_std(losses, k, hard_low_cut, hard_high_cut)
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        std_lower = mean_loss - k * std_loss
        std_upper = mean_loss + k * std_loss
        
        # Calculate final bounds
        final_lower = max(std_lower, hard_low_cut) if hard_low_cut is not None else std_lower
        final_upper = min(std_upper, hard_high_cut) if hard_high_cut is not None else std_upper
        
        print(f"Applied mean±std filtering with k={k}")
        print(f"Mean: {mean_loss:.4f}, Std: {std_loss:.4f}")
        print(f"Std bounds: [{std_lower:.4f}, {std_upper:.4f}]")
        if hard_low_cut is not None:
            print(f"Hard low cutoff: {hard_low_cut:.4f}")
        if hard_high_cut is not None:
            print(f"Hard high cutoff: {hard_high_cut:.4f}")
        print(f"Final bounds: [{final_lower:.4f}, {final_upper:.4f}]")
        
    elif filter_mode == 'percentile':
        lower_p = filter_params.get('lower_percentile', 0.05)
        upper_p = filter_params.get('upper_percentile', 0.95)
        mask = filter_percentile(losses, lower_p, upper_p)
        print(f"Applied percentile filtering with bounds [{lower_p*100:.1f}%, {upper_p*100:.1f}%]")
        print(f"Lower bound: {np.percentile(losses, lower_p * 100):.4f}")
        print(f"Upper bound: {np.percentile(losses, upper_p * 100):.4f}")
        
    elif filter_mode == 'hardcode':
        lower_bound = filter_params.get('lower_bound', 1.5)
        upper_bound = filter_params.get('upper_bound', 7.0)
        mask = filter_hardcode(losses, lower_bound, upper_bound)
        print(f"Applied hardcode filtering with bounds [{lower_bound}, {upper_bound}]")
        
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")
    
    # Apply mask to get filtered data
    filtered_losses = losses[mask]
    filtered_paths = paths[mask]
    
    print(f"Total files: {len(losses)}, Kept: {np.sum(mask)}, Filtered out: {len(losses) - np.sum(mask)}")
    
    return filtered_losses.tolist(), filtered_paths.tolist(), mask





def apply_filtering_from_directory(eval_dir, filter_mode=None, filter_params=None, 
                                  current_train_path=None, current_test_path=None):
    """
    Apply filtering using existing train_recon.csv, val_recon.csv, and test_recon.csv files in an evaluation directory
    
    Args:
        eval_dir: Directory containing train_recon.csv, val_recon.csv, and test_recon.csv
        filter_mode: Filter mode to apply (if None, uses default median)
        filter_params: Filter parameters (if None, uses defaults)
        current_train_path: Current training data path for validation
        current_test_path: Current test data path for validation
    
    Returns:
        tuple: (train_ignored_files, val_ignored_files, test_ignored_files, train_metadata, val_metadata, test_metadata)
    """
    train_recon_csv = os.path.join(eval_dir, 'train_recon.csv')
    val_recon_csv = os.path.join(eval_dir, 'val_recon.csv')
    test_recon_csv = os.path.join(eval_dir, 'test_recon.csv')
    
    if not os.path.exists(train_recon_csv):
        raise FileNotFoundError(f"train_recon.csv not found in {eval_dir}")
    if not os.path.exists(test_recon_csv):
        raise FileNotFoundError(f"test_recon.csv not found in {eval_dir}")
    
    # Check if validation CSV exists
    val_df = None
    val_losses = []
    val_paths = []
    if os.path.exists(val_recon_csv):
        val_df = pd.read_csv(val_recon_csv, skiprows=3)  # Skip header section
        val_losses = val_df['reconstruction_loss'].tolist()
        val_paths = val_df['filename'].tolist()
        print(f"Found validation data: {len(val_losses)} files")
    else:
        print(f"Warning: val_recon.csv not found in {eval_dir}")
    
    # Read existing CSV files
    train_df = pd.read_csv(train_recon_csv, skiprows=3)  # Skip header section
    test_df = pd.read_csv(test_recon_csv, skiprows=3)    # Skip header section
    
    # Extract losses and paths
    train_losses = train_df['reconstruction_loss'].tolist()
    train_paths = train_df['filename'].tolist()
    test_losses = test_df['reconstruction_loss'].tolist()
    test_paths = test_df['filename'].tolist()
    
    # Use provided parameters or defaults
    if filter_mode is None:
        filter_mode = 'median'
    if filter_params is None:
        filter_params = {'k': 1.5, 'hard_low_cut': 1.5}
    
    print(f"Applying {filter_mode} filtering with parameters: {filter_params}")
    
    # Apply filtering to get masks
    train_filtered_losses, train_filtered_paths, train_mask = apply_filtering(
        train_losses, train_paths, filter_mode, **filter_params
    )
    
    test_filtered_losses, test_filtered_paths, test_mask = apply_filtering(
        test_losses, test_paths, filter_mode, **filter_params
    )
    
    # Apply filtering to validation data if available
    val_filtered_losses = []
    val_filtered_paths = []
    val_mask = []
    if val_losses:
        val_filtered_losses, val_filtered_paths, val_mask = apply_filtering(
            val_losses, val_paths, filter_mode, **filter_params
        )
    
    # Get ignored files (where mask is False)
    train_ignored_indices = np.where(~train_mask)[0]
    test_ignored_indices = np.where(~test_mask)[0]
    
    train_ignored_files = [os.path.basename(train_paths[i]) for i in train_ignored_indices]
    test_ignored_files = [os.path.basename(test_paths[i]) for i in test_ignored_indices]
    
    # Get validation ignored files if available
    val_ignored_files = []
    if val_losses:
        val_ignored_indices = np.where(~val_mask)[0]
        val_ignored_files = [os.path.basename(val_paths[i]) for i in val_ignored_indices]
    
    # Create metadata
    train_metadata = {
        'filter_mode': filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(train_losses),
        'kept_files': np.sum(train_mask),
        'ignored_files': len(train_ignored_files)
    }
    
    val_metadata = {
        'filter_mode': filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(val_losses),
        'kept_files': np.sum(val_mask) if val_losses else 0,
        'ignored_files': len(val_ignored_files)
    }
    
    test_metadata = {
        'filter_mode': filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(test_losses),
        'kept_files': np.sum(test_mask),
        'ignored_files': len(test_ignored_files)
    }
    
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Train: {len(train_ignored_files)} files ignored out of {len(train_losses)} total")
    if val_losses:
        print(f"Validation: {len(val_ignored_files)} files ignored out of {len(val_losses)} total")
    print(f"Test: {len(test_ignored_files)} files ignored out of {len(test_losses)} total")
    
    return train_ignored_files, val_ignored_files, test_ignored_files, train_metadata, val_metadata, test_metadata


def apply_energy_filtering(losses, energies, paths, filter_mode, energy_filter_mode=None, **filter_params):
    """
    Apply filtering to losses and energies, returning filtered data based on both criteria
    
    Args:
        losses: List of reconstruction losses
        energies: List of signal energies
        paths: List of corresponding file paths
        filter_mode: Filtering mode for reconstruction losses ('median', 'mean', 'percentile', 'hardcode')
        energy_filter_mode: Filtering mode for energies (if None, uses same as filter_mode)
        **filter_params: Additional parameters for the specific filter mode
    
    Returns:
        tuple: (filtered_losses, filtered_energies, filtered_paths, mask)
    """
    losses = np.array(losses)
    energies = np.array(energies)
    paths = np.array(paths)
    
    # Use same filter mode for energy if not specified
    if energy_filter_mode is None:
        energy_filter_mode = filter_mode
    
    # Apply filtering to reconstruction losses
    if filter_mode == 'median':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        loss_mask = filter_median_iqr(losses, k, hard_low_cut, hard_high_cut)
    elif filter_mode == 'mean':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        loss_mask = filter_mean_std(losses, k, hard_low_cut, hard_high_cut)
    elif filter_mode == 'percentile':
        lower_p = filter_params.get('lower_percentile', 0.05)
        upper_p = filter_params.get('upper_percentile', 0.95)
        loss_mask = filter_percentile(losses, lower_p, upper_p)
    elif filter_mode == 'hardcode':
        lower_bound = filter_params.get('lower_bound', 1.5)
        upper_bound = filter_params.get('upper_bound', 7.0)
        loss_mask = filter_hardcode(losses, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode}")
    
    # Apply filtering to energies
    if energy_filter_mode == 'median':
        k = filter_params.get('energy_k', filter_params.get('k', 1.5))
        hard_low_cut = filter_params.get('energy_hard_low_cut', None)
        hard_high_cut = filter_params.get('energy_hard_high_cut', None)
        energy_mask = filter_median_iqr(energies, k, hard_low_cut, hard_high_cut)
    elif energy_filter_mode == 'mean':
        k = filter_params.get('energy_k', filter_params.get('k', 1.5))
        hard_low_cut = filter_params.get('energy_hard_low_cut', None)
        hard_high_cut = filter_params.get('energy_hard_high_cut', None)
        energy_mask = filter_mean_std(energies, k, hard_low_cut, hard_high_cut)
    elif energy_filter_mode == 'percentile':
        lower_p = filter_params.get('energy_lower_percentile', filter_params.get('lower_percentile', 0.05))
        upper_p = filter_params.get('energy_upper_percentile', filter_params.get('upper_percentile', 0.95))
        energy_mask = filter_percentile(energies, lower_p, upper_p)
    elif energy_filter_mode == 'hardcode':
        lower_bound = filter_params.get('energy_lower_bound', filter_params.get('lower_bound', 1.5))
        upper_bound = filter_params.get('energy_upper_bound', filter_params.get('upper_bound', 7.0))
        energy_mask = filter_hardcode(energies, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unknown energy_filter_mode: {energy_filter_mode}")
    
    # Combine masks (keep files that pass both filters)
    combined_mask = loss_mask & energy_mask
    
    # Apply combined mask to get filtered data
    filtered_losses = losses[combined_mask]
    filtered_energies = energies[combined_mask]
    filtered_paths = paths[combined_mask]
    
    print(f"Reconstruction loss filtering: {np.sum(loss_mask)}/{len(losses)} files passed")
    print(f"Energy filtering: {np.sum(energy_mask)}/{len(energies)} files passed")
    print(f"Combined filtering: {np.sum(combined_mask)}/{len(losses)} files passed")
    
    return filtered_losses.tolist(), filtered_energies.tolist(), filtered_paths.tolist(), combined_mask


def apply_energy_filtering_from_directory(eval_dir, filter_mode=None, energy_filter_mode=None, filter_params=None):
    """
    Apply energy-based filtering using existing CSV files with both reconstruction loss and energy data
    
    Args:
        eval_dir: Directory containing train_recon.csv, val_recon.csv, and test_recon.csv
        filter_mode: Filtering mode for reconstruction losses
        energy_filter_mode: Filtering mode for energies (if None, uses same as filter_mode)
        filter_params: Filter parameters dictionary
    
    Returns:
        tuple: (train_ignored_files, val_ignored_files, test_ignored_files, train_metadata, val_metadata, test_metadata)
    """
    train_recon_csv = os.path.join(eval_dir, 'train_recon.csv')
    val_recon_csv = os.path.join(eval_dir, 'val_recon.csv')
    test_recon_csv = os.path.join(eval_dir, 'test_recon.csv')
    
    if not os.path.exists(train_recon_csv):
        raise FileNotFoundError(f"train_recon.csv not found in {eval_dir}")
    if not os.path.exists(test_recon_csv):
        raise FileNotFoundError(f"test_recon.csv not found in {eval_dir}")
    
    # Read CSV files
    train_df = pd.read_csv(train_recon_csv, skiprows=3)  # Skip header section
    test_df = pd.read_csv(test_recon_csv, skiprows=3)    # Skip header section
    
    # Check if energy data is available
    has_energy_data = 'signal_energy' in train_df.columns and 'signal_energy' in test_df.columns
    
    if not has_energy_data:
        print("Warning: Energy data not found in CSV files. Falling back to reconstruction loss only filtering.")
        return apply_filtering_from_directory(eval_dir, filter_mode, filter_params)
    
    # Extract losses, energies, and paths
    train_losses = train_df['reconstruction_loss'].tolist()
    train_energies = train_df['signal_energy'].tolist()
    train_paths = train_df['filename'].tolist()
    
    test_losses = test_df['reconstruction_loss'].tolist()
    test_energies = test_df['signal_energy'].tolist()
    test_paths = test_df['filename'].tolist()
    
    # Check for validation data
    val_losses = []
    val_energies = []
    val_paths = []
    if os.path.exists(val_recon_csv):
        val_df = pd.read_csv(val_recon_csv, skiprows=3)
        if 'signal_energy' in val_df.columns:
            val_losses = val_df['reconstruction_loss'].tolist()
            val_energies = val_df['signal_energy'].tolist()
            val_paths = val_df['filename'].tolist()
            print(f"Found validation data with energy: {len(val_losses)} files")
        else:
            print(f"Warning: val_recon.csv found but no energy data")
    else:
        print(f"Warning: val_recon.csv not found in {eval_dir}")
    
    # Use provided parameters or defaults
    if filter_mode is None:
        filter_mode = 'median'
    if filter_params is None:
        filter_params = {'k': 1.5, 'hard_low_cut': 1.5}
    
    print(f"Applying energy-based filtering:")
    print(f"  Reconstruction loss mode: {filter_mode}")
    print(f"  Energy mode: {energy_filter_mode if energy_filter_mode else filter_mode}")
    print(f"  Parameters: {filter_params}")
    
    # Apply filtering to get masks
    train_filtered_losses, train_filtered_energies, train_filtered_paths, train_mask = apply_energy_filtering(
        train_losses, train_energies, train_paths, filter_mode, energy_filter_mode, **filter_params
    )
    
    test_filtered_losses, test_filtered_energies, test_filtered_paths, test_mask = apply_energy_filtering(
        test_losses, test_energies, test_paths, filter_mode, energy_filter_mode, **filter_params
    )
    
    # Apply filtering to validation data if available
    val_filtered_losses = []
    val_filtered_energies = []
    val_filtered_paths = []
    val_mask = []
    if val_losses:
        val_filtered_losses, val_filtered_energies, val_filtered_paths, val_mask = apply_energy_filtering(
            val_losses, val_energies, val_paths, filter_mode, energy_filter_mode, **filter_params
        )
    
    # Get ignored files (where mask is False)
    train_ignored_indices = np.where(~train_mask)[0]
    test_ignored_indices = np.where(~test_mask)[0]
    
    train_ignored_files = [train_paths[i] for i in train_ignored_indices]
    test_ignored_files = [test_paths[i] for i in test_ignored_indices]
    
    # Get validation ignored files if available
    val_ignored_files = []
    if val_losses:
        val_ignored_indices = np.where(~val_mask)[0]
        val_ignored_files = [val_paths[i] for i in val_ignored_indices]
    
    # Create metadata
    train_metadata = {
        'filter_mode': filter_mode,
        'energy_filter_mode': energy_filter_mode if energy_filter_mode else filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(train_losses),
        'kept_files': np.sum(train_mask),
        'ignored_files': len(train_ignored_files)
    }
    
    val_metadata = {
        'filter_mode': filter_mode,
        'energy_filter_mode': energy_filter_mode if energy_filter_mode else filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(val_losses),
        'kept_files': np.sum(val_mask) if val_losses else 0,
        'ignored_files': len(val_ignored_files)
    }
    
    test_metadata = {
        'filter_mode': filter_mode,
        'energy_filter_mode': energy_filter_mode if energy_filter_mode else filter_mode,
        'filter_params': str(filter_params),
        'total_files': len(test_losses),
        'kept_files': np.sum(test_mask),
        'ignored_files': len(test_ignored_files)
    }
    
    print(f"\n=== ENERGY-BASED FILTERING RESULTS ===")
    print(f"Train: {len(train_ignored_files)} files ignored out of {len(train_losses)} total")
    if val_losses:
        print(f"Validation: {len(val_ignored_files)} files ignored out of {len(val_losses)} total")
    print(f"Test: {len(test_ignored_files)} files ignored out of {len(test_losses)} total")
    
    return train_ignored_files, val_ignored_files, test_ignored_files, train_metadata, val_metadata, test_metadata
