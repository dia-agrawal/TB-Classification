#!/usr/bin/env python3
"""
Test script to verify the new filtering approach works correctly.
"""

import numpy as np
from Recon_loss_filtering import apply_filtering

def test_filtering():
    """Test the apply_filtering function with different filter modes."""
    
    # Create sample data
    np.random.seed(42)
    losses = np.random.normal(3.0, 1.0, 100)  # Normal distribution with some outliers
    losses[0:5] = np.random.uniform(0.1, 0.5, 5)  # Add some low outliers
    losses[95:100] = np.random.uniform(8.0, 12.0, 5)  # Add some high outliers
    
    paths = [f"file_{i}.pt" for i in range(len(losses))]
    
    print("Original data statistics:")
    print(f"Mean: {np.mean(losses):.4f}")
    print(f"Std: {np.std(losses):.4f}")
    print(f"Min: {np.min(losses):.4f}")
    print(f"Max: {np.max(losses):.4f}")
    print(f"Total samples: {len(losses)}")
    print()
    
    # Test median filtering
    print("=== Testing Median Filtering ===")
    filtered_losses, filtered_paths, mask = apply_filtering(
        losses.tolist(), paths, 'median', k=1.5, hard_low_cut=1.5
    )
    print(f"Filtered data statistics:")
    print(f"Mean: {np.mean(filtered_losses):.4f}")
    print(f"Std: {np.std(filtered_losses):.4f}")
    print(f"Min: {np.min(filtered_losses):.4f}")
    print(f"Max: {np.max(filtered_losses):.4f}")
    print(f"Kept samples: {len(filtered_losses)}")
    print()
    
    # Test mean filtering
    print("=== Testing Mean Filtering ===")
    filtered_losses, filtered_paths, mask = apply_filtering(
        losses.tolist(), paths, 'mean', k=1.5
    )
    print(f"Filtered data statistics:")
    print(f"Mean: {np.mean(filtered_losses):.4f}")
    print(f"Std: {np.std(filtered_losses):.4f}")
    print(f"Min: {np.min(filtered_losses):.4f}")
    print(f"Max: {np.max(filtered_losses):.4f}")
    print(f"Kept samples: {len(filtered_losses)}")
    print()
    
    # Test percentile filtering
    print("=== Testing Percentile Filtering ===")
    filtered_losses, filtered_paths, mask = apply_filtering(
        losses.tolist(), paths, 'percentile', lower_percentile=0.05, upper_percentile=0.95
    )
    print(f"Filtered data statistics:")
    print(f"Mean: {np.mean(filtered_losses):.4f}")
    print(f"Std: {np.std(filtered_losses):.4f}")
    print(f"Min: {np.min(filtered_losses):.4f}")
    print(f"Max: {np.max(filtered_losses):.4f}")
    print(f"Kept samples: {len(filtered_losses)}")
    print()
    
    # Test hardcode filtering
    print("=== Testing Hardcode Filtering ===")
    filtered_losses, filtered_paths, mask = apply_filtering(
        losses.tolist(), paths, 'hardcode', lower_bound=1.5, upper_bound=7.0
    )
    print(f"Filtered data statistics:")
    print(f"Mean: {np.mean(filtered_losses):.4f}")
    print(f"Std: {np.std(filtered_losses):.4f}")
    print(f"Min: {np.min(filtered_losses):.4f}")
    print(f"Max: {np.max(filtered_losses):.4f}")
    print(f"Kept samples: {len(filtered_losses)}")
    print()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_filtering()
