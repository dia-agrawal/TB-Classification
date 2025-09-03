import argparse
import os

import numpy as np
from sklearn.metrics import classification_report, roc_curve
import yaml
import json
from pathlib import Path
from lightning_combined import AudioDataset, load_config

from InferEng import AudioInferenceEngine 
from config import convert_numpy_types, generate_report
from evaluator import AudioEvaluator

import csv

def find_optimal_threshold(probabilities, labels, optimization_metric='balanced'):
    """
    Find optimal threshold to maximize sensitivity and specificity.
    
    Args:
        probabilities: Array of probabilities (positive class)
        labels: Array of true labels
        optimization_metric: 'balanced' (geometric mean), 'sensitivity', 'specificity', or 'f1'
    
    Returns:
        dict: Optimal threshold and corresponding metrics
    """
    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    
    # Calculate specificity (1 - FPR)
    specificity = 1 - fpr
    
    # Calculate different optimization metrics
    balanced_scores = np.sqrt(tpr * specificity)  # Geometric mean
    f1_scores = 2 * (tpr * specificity) / (tpr + specificity + 1e-8)  # F1 score
    
    # Find optimal threshold based on metric
    if optimization_metric == 'balanced':
        optimal_idx = np.argmax(balanced_scores)
    elif optimization_metric == 'sensitivity':
        optimal_idx = np.argmax(tpr)
    elif optimization_metric == 'specificity':
        optimal_idx = np.argmax(specificity)
    elif optimization_metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown optimization metric: {optimization_metric}")
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = specificity[optimal_idx]
    optimal_balanced = balanced_scores[optimal_idx]
    
    return {
        'threshold': optimal_threshold,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'balanced_score': optimal_balanced,
        'optimization_metric': optimization_metric
    }

def find_optimal_thresholds(original_results, filtered_results, patient_result=False, optimization_metric='balanced'):
    """
    Find optimal thresholds for both original and filtered results.
    
    Args:
        original_results: Original evaluation results
        filtered_results: Filtered evaluation results
        patient_result: Whether to optimize for patient-level results
    
    Returns:
        dict: Optimal thresholds and metrics for both scenarios
    """
    print("\n=== FINDING OPTIMAL THRESHOLDS ===")
    
    results = {}
    
    # File-level optimization
    if 'probabilities' in original_results and 'labels' in original_results:
        print("Finding optimal threshold for file-level predictions (original)...")
        probs = np.array(original_results['probabilities'])
        labels = np.array(original_results['labels'])
        
        # Handle 2D probability arrays
        if probs.ndim == 2:
            probs = probs[:, 1]  # Use positive class probability
        
        # Find optimal threshold
        optimal = find_optimal_threshold(probs, labels, optimization_metric)
        results['file_level_original'] = optimal
        print(f"  Optimal threshold: {optimal['threshold']:.4f}")
        print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
        print(f"  Specificity: {optimal['specificity']:.4f}")
        print(f"  Balanced score: {optimal['balanced_score']:.4f}")
    
    if filtered_results is not None and 'probabilities' in filtered_results and 'labels' in filtered_results:
        print("Finding optimal threshold for file-level predictions (filtered)...")
        probs = np.array(filtered_results['probabilities'])
        labels = np.array(filtered_results['labels'])
        
        # Handle 2D probability arrays
        if probs.ndim == 2:
            probs = probs[:, 1]  # Use positive class probability
        
        # Find optimal threshold
        optimal = find_optimal_threshold(probs, labels, optimization_metric)
        results['file_level_filtered'] = optimal
        print(f"  Optimal threshold: {optimal['threshold']:.4f}")
        print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
        print(f"  Specificity: {optimal['specificity']:.4f}")
        print(f"  Balanced score: {optimal['balanced_score']:.4f}")
    
    # Patient-level optimization
    if patient_result:
        if 'patient_probabilities' in original_results and 'patient_labels' in original_results:
            print("Finding optimal threshold for patient-level predictions (original)...")
            probs = np.array(original_results['patient_probabilities'])
            labels = np.array(original_results['patient_labels'])
            
            # Handle 2D probability arrays
            if probs.ndim == 2:
                probs = probs[:, 1]  # Use positive class probability
            
            # Find optimal threshold
            optimal = find_optimal_threshold(probs, labels, optimization_metric)
            results['patient_level_original'] = optimal
            print(f"  Optimal threshold: {optimal['threshold']:.4f}")
            print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
            print(f"  Specificity: {optimal['specificity']:.4f}")
            print(f"  Balanced score: {optimal['balanced_score']:.4f}")
        
        if filtered_results is not None and 'patient_probabilities' in filtered_results and 'patient_labels' in filtered_results:
            print("Finding optimal threshold for patient-level predictions (filtered)...")
            probs = np.array(filtered_results['patient_probabilities'])
            labels = np.array(filtered_results['patient_labels'])
            
            # Handle 2D probability arrays
            if probs.ndim == 2:
                probs = probs[:, 1]  # Use positive class probability
            
            # Find optimal threshold
            optimal = find_optimal_threshold(probs, labels, optimization_metric)
            results['patient_level_filtered'] = optimal
            print(f"  Optimal threshold: {optimal['threshold']:.4f}")
            print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
            print(f"  Specificity: {optimal['specificity']:.4f}")
            print(f"  Balanced score: {optimal['balanced_score']:.4f}")
    
    return results

def print_threshold_recommendations(optimal_thresholds, optimization_metric):
    """
    Print recommendations for threshold usage based on optimal thresholds.
    
    Args:
        optimal_thresholds: Dictionary containing optimal threshold results
        optimization_metric: The metric used for optimization
    """
    print(f"\n=== THRESHOLD RECOMMENDATIONS ===")
    print(f"Based on {optimization_metric} optimization:")
    
    if 'file_level_original' in optimal_thresholds and 'file_level_filtered' in optimal_thresholds:
        orig_thresh = optimal_thresholds['file_level_original']['threshold']
        filt_thresh = optimal_thresholds['file_level_filtered']['threshold']
        print(f"File-level threshold recommendation:")
        print(f"  - Original data: {orig_thresh:.4f}")
        print(f"  - Filtered data: {filt_thresh:.4f}")
        print(f"  - Difference: {abs(orig_thresh - filt_thresh):.4f}")
    
    if 'patient_level_original' in optimal_thresholds and 'patient_level_filtered' in optimal_thresholds:
        orig_thresh = optimal_thresholds['patient_level_original']['threshold']
        filt_thresh = optimal_thresholds['patient_level_filtered']['threshold']
        print(f"Patient-level threshold recommendation:")
        print(f"  - Original data: {orig_thresh:.4f}")
        print(f"  - Filtered data: {filt_thresh:.4f}")
        print(f"  - Difference: {abs(orig_thresh - filt_thresh):.4f}")
    
    print(f"\nUsage:")
    print(f"  - For file-level evaluation: Use the file-level threshold")
    print(f"  - For patient-level evaluation: Use the patient-level threshold")
    print(f"  - For filtered scenarios: Use the filtered threshold")
    print(f"  - For non-filtered scenarios: Use the original threshold")

def main():
    """Main function for running inference and evaluation."""
    required_bool = False
    
    parser = argparse.ArgumentParser(description="Audio Model Inference and Evaluation")
    parser.add_argument('--mode', type=str, required=required_bool, 
                       choices=['auto', 'classifier', 'tripletloss'],
                       help='Model mode')
    parser.add_argument('--config', type=str, default="default_config.yaml", help='Path to YAML config file')
    parser.add_argument('--checkpoint_path', type=str, required=required_bool,
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Threshold for patient-level prediction (default: 0.55)')
    parser.add_argument('--data_path', type=str, required=required_bool,
                       help='Path to test data directory')
    parser.add_argument('--train_data_path', type=str, help='Path to training data directory (for KNN reference)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run only inference, skip evaluation')
    parser.add_argument('--patient_result', action='store_true', help='windowed results averaged per patient')
    parser.add_argument('--run_knn_MJ', action='store_true', help='Run KNN evaluation')  #. added
    parser.add_argument('--K', type=int, default=5, help='Number of nearest neighbors for KNN using Majority Rule')  #. added
    parser.add_argument('--max_ref_samples', type=int, default=None, help='Maximum number of reference samples for KNN (faster evaluation)')    
    parser.add_argument('--evaluate_using_dir', type=str)
    parser.add_argument('--create_ignored_recon', action='store_true', help='Create reconstruction loss CSV files for train, val, and test data for manual evaluation (no filtering applied)')
    parser.add_argument('--filter_mode', type=str, choices=['median', 'mean', 'percentile', 'hardcode'], 
                       help='Filtering mode for ignored files: median (IQR-based), mean (std-based), percentile (percentile-based), hardcode (fixed range)')
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
    parser.add_argument('--use_filter_dir', type=str, 
                       help="Use evaluation directory with dynamic filtering. Path to directory containing train_recon.csv, val_recon.csv, and test_recon.csv. Will dynamically calculate ignored files based on filter parameters.")
    parser.add_argument('--optimization_metric', type=str, default='balanced', 
                       choices=['balanced', 'sensitivity', 'specificity', 'f1'],
                       help='Metric to optimize for threshold finding: balanced (geometric mean), sensitivity, specificity, or f1')
    args = parser.parse_args()

    config = load_config(args.config, args)

    model_params = config.get('model', {})
    if args.evaluate_using_dir: 
        # Validate directory exists
        if not os.path.exists(args.evaluate_using_dir):
            raise ValueError(f"Directory {args.evaluate_using_dir} does not exist")
        
        checkpoint_dir = os.path.join(args.evaluate_using_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
            
        current_max_val = 0
        current_max_epoch = None
        for ckpt in os.listdir(checkpoint_dir): 
            try: 
                txt = ckpt.split('acc=')
                val_acc = (txt[2]).split('.ckpt')
                if float(val_acc[0]) >= current_max_val: 
                    current_max_val = float(val_acc[0])
                    current_max_epoch = os.path.join(args.evaluate_using_dir, 'checkpoints',ckpt)
            except IndexError: 
                try:
                    txt = ckpt.split('val_loss=')
                    val_acc = (txt[1]).split('.ckpt')
                    if float(val_acc[0]) >= current_max_val: 
                        current_max_val = float(val_acc[0])
                        current_max_epoch = os.path.join(args.evaluate_using_dir, 'checkpoints',ckpt)
                except (IndexError, ValueError):
                    # Skip files that don't match expected naming pattern
                    continue
            except Exception as e:
                print(f"Unexpected error processing {ckpt}: {e}")
                continue
        yaml_file = os.path.join(args.evaluate_using_dir, 'logs', 'config_used.yaml')
        if not os.path.exists(yaml_file):
            raise ValueError(f"Config file {yaml_file} does not exist")
            
        with open(yaml_file, 'r') as file: 
            data = yaml.safe_load(file)
        
        base_dir = None
        batch_size = args.batch_size
        mode_type = None 
        train_data_path = None
        for key, value in data.items(): 
            if key == 'base_dir':
                base_dir = os.path.join(value, 'test')
                train_data_path = os.path.join(value, 'train')
            elif key == 'mode_type':
                mode_type = value
        checkpoint = current_max_epoch
        assert base_dir is not None, "config_used.yaml doesn't contain a data path (base_dir)"
        assert mode_type is not None, "config_used.yaml doesn't contain a mode type (mode_type)"
        
        # Validate data directory exists
        if not os.path.exists(base_dir):
            raise ValueError(f"Test data directory {base_dir} does not exist")
        if args.create_ignored_recon:
            if mode_type != 'auto':
                raise ValueError("to create ignored reconstruction list, mode type should be auto")
            if train_data_path is None:
                raise ValueError("config_used.yaml doesnt have a datapath")
            if not os.path.exists(train_data_path):
                raise ValueError(f"Training data directory {train_data_path} does not exist")
        elif mode_type == 'auto': 
            mode_type = 'tripletloss'
    else: 
        checkpoint=args.checkpoint_path
        mode_type = args.mode
        base_dir = args.data_path  
        train_data_path = None
        if args.mode == 'tripletloss': 
            train_data_path = args.train_data_path
            if train_data_path is None: 
                raise ValueError('need train data path for triplet mode')
            if not os.path.exists(train_data_path):
                raise ValueError(f"Training data directory {train_data_path} does not exist")
        if checkpoint is None or mode_type is None or base_dir is None:
            print('_____________INVALID PARAMETER INPUT_________________')
            raise ValueError('must use --checkpoint_path , --mode, --data_path')
        if args.create_ignored_recon:
            if mode_type != 'auto':
                raise ValueError("to create ignored reconstruction list, mode type should be auto")
            
    # Validate checkpoint file exists
    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint file {checkpoint} does not exist")
        
    print("Loading model...")
    engine = AudioInferenceEngine(
        checkpoint_path=checkpoint,
        mode=mode_type,
        model_params=model_params,
        device=args.device
    )
    
    if args.inference_only:
        # Run inference on single file or directory
        if os.path.isfile(base_dir):
            # Single file inference
            result = engine.predict_ifiles(base_dir, 1)
            if result:
                print(f"Prediction for {base_dir}:")
                print(json.dumps(convert_numpy_types(result), indent=2))
        else:
            # Directory inference
            audio_files = []
            for ext in ['*.pt']:
                audio_files.extend(Path(base_dir).rglob(ext))
            
            if audio_files:
                results = engine.predict_ifiles([str(f) for f in audio_files], args.batch_size if batch_size is None else batch_size)
                
                # Save results
                os.makedirs(args.output_dir, exist_ok=True)
                results_file = os.path.join(args.output_dir, 'inference_results.json')
                
                with open(results_file, 'w') as f:
                    json.dump(convert_numpy_types(results), f, indent=2)
                
                print(f"Inference results saved to {results_file}")
            else:
                print(f"No .pt files found in {base_dir}")
    else:
        # Run full evaluation
        print("Running evaluation...")
        evaluator = AudioEvaluator(engine, base_dir, mode_type)

        if mode_type == 'tripletloss':
            # Check if train_data_path is available
            if train_data_path is None:
                raise ValueError("train_data_path is required for tripletloss mode but was not provided")
            
            # Solved: Change data mode
            train_ds = AudioDataset(
                mode='classifier',
                data_dir_good=[os.path.join(train_data_path, 'good')],
                data_dir_bad=[os.path.join(train_data_path, 'bad')],
            )
            results = evaluator.run_evaluation(
                patient_result=args.patient_result,
                train_dataset=train_ds,
                threshold=args.threshold,
                run_mj= args.run_knn_MJ, 
                K = args.K
            )
        elif mode_type == 'classifier':  #DA FIXED VA CHECK 
            results = evaluator.run_evaluation( #. 
                patient_result=args.patient_result, #. 
                threshold=args.threshold #. 
            )
        else:
            print(f"======Running in Auto Mode========")
            data_dir_train = [f"{base_dir}/good", f"{base_dir}/bad"]
            
            assert not args.create_ignored_recon or train_data_path is not None, "train_data_path is required for auto mode"
            
            # Detect validation data path
            val_data_path = None
            if train_data_path is not None:
                # If train_data_path is /path/to/data/train, val_data_path should be /path/to/data/val
                base_data_dir = os.path.dirname(train_data_path)
                potential_val_path = os.path.join(base_data_dir, 'val')
                if os.path.exists(potential_val_path):
                    val_data_path = potential_val_path
                    print(f"Found validation data path: {val_data_path}")
                else:
                    print(f"Validation data path not found: {potential_val_path}")

            if train_data_path is not None: 
                train_ds = AudioDataset(
                        mode='auto',
                        data_dir_good = data_dir_train,
                    )
            
            # Create validation dataset if validation path exists
            val_ds = None
            if val_data_path is not None:
                data_dir_val = [f"{val_data_path}/good", f"{val_data_path}/bad"]
                val_ds = AudioDataset(
                        mode='auto',
                        data_dir_good = data_dir_val,
                    )
                print(f"Created validation dataset with {len(val_ds)} files")
            
            # Generate reconstruction losses for all files (no filtering applied)
            # All files are included for manual evaluation purposes
            results = evaluator.run_evaluation(
                patient_result=args.patient_result,
                train_dataset= train_ds,
                val_dataset= val_ds,
                threshold=args.threshold, 
                generate_ignored_recon = args.create_ignored_recon
            )      
            
              
        # Create reconstruction loss CSV files for manual evaluation (no filtering applied)
        if args.create_ignored_recon and mode_type == 'auto':
            print(f"\n=== RECONSTRUCTION LOSS DATA CREATION ===")
            print(f"Creating reconstruction loss CSV files for manual evaluation...")
            print(f"No filtering applied - all files included for manual review.")
            
            # Ensure we have the reconstruction data
            if 'test_corresponding_recon_loss' in results and 'train_corresponding_recon_loss' in results:
                print(f"Test data: {len(results['test_corresponding_recon_loss'])} files processed")
                print(f"Training data: {len(results['train_corresponding_recon_loss'])} files processed")
                
                # Check for validation data
                if 'val_corresponding_recon_loss' in results:
                    print(f"Validation data: {len(results['val_corresponding_recon_loss'])} files processed")
                
                # Update statistics to reflect all data (no filtering)
                results['mean_loss'] = np.mean(results['test_corresponding_recon_loss'])
                results['std_loss'] = np.std(results['test_corresponding_recon_loss'])
                results['train_mean_recon_loss'] = np.mean(results['train_corresponding_recon_loss'])
                results['train_std_recon_loss'] = np.std(results['train_corresponding_recon_loss'])
                
                print(f"\n=== RECONSTRUCTION LOSS SUMMARY ===")
                print(f"Test data mean loss: {results['mean_loss']:.4f}")
                print(f"Test data std loss: {results['std_loss']:.4f}")
                print(f"Training data mean loss: {results['train_mean_recon_loss']:.4f}")
                print(f"Training data std loss: {results['train_std_recon_loss']:.4f}")
                
                if 'val_corresponding_recon_loss' in results:
                    results['val_mean_recon_loss'] = np.mean(results['val_corresponding_recon_loss'])
                    results['val_std_recon_loss'] = np.std(results['val_corresponding_recon_loss'])
                    print(f"Validation data mean loss: {results['val_mean_recon_loss']:.4f}")
                    print(f"Validation data std loss: {results['val_std_recon_loss']:.4f}")
                
                print(f"All reconstruction losses calculated and saved for manual evaluation.")
            else:
                print(f"Warning: Reconstruction loss data not found in results.")
        
        # Handle directory-based filtering for classifier/triplet modes
        if args.use_filter_dir and mode_type in ['classifier', 'tripletloss']:
            
            print(f"\n=== NEW FILTERING APPROACH ===")
            print(f"Using evaluation directory: {args.use_filter_dir}")
            
            # Import the filtering functions
            from Recon_loss_filtering import apply_filtering_from_directory, apply_energy_filtering_from_directory
            
            # Determine current data paths
            current_train_path = train_data_path if 'train_data_path' in locals() else None
            current_test_path = base_dir if 'base_dir' in locals() else None
            
            # Parse filter parameters if provided
            filter_mode = None
            filter_params = None
            
            if args.filter_mode:
                filter_mode = args.filter_mode
                filter_params = {}
                
                if args.filter_params:
                    if filter_mode in ['median', 'mean']:
                        if len(args.filter_params) >= 1:
                            filter_params['k'] = float(args.filter_params[0])
                    elif filter_mode == 'percentile':
                        if len(args.filter_params) >= 2:
                            filter_params['lower_percentile'] = float(args.filter_params[0])
                            filter_params['upper_percentile'] = float(args.filter_params[1])
                    elif filter_mode == 'hardcode':
                        if len(args.filter_params) >= 2:
                            filter_params['lower_bound'] = float(args.filter_params[0])
                            filter_params['upper_bound'] = float(args.filter_params[1])
                
                # Add hard cutoffs if specified (only for median/mean)
                if filter_mode in ['median', 'mean']:
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
            
            # Run evaluation on original data (no filtering)
            print(f"\n=== RUNNING EVALUATION ON ORIGINAL DATA ===")
            original_results = evaluator.run_evaluation(
                patient_result=args.patient_result,
                train_dataset=train_ds if 'train_ds' in locals() else None,
                threshold=args.threshold,
                run_mj=args.run_knn_MJ,
                K=args.K
            )
            
            # Apply filtering to get ignored files
            print(f"\n=== APPLYING FILTERING ===")
            try:
                # Check if energy data is available in CSV files
                train_recon_csv = os.path.join(args.use_filter_dir, 'train_recon.csv')
                has_energy_data = False
                if os.path.exists(train_recon_csv):
                    try:
                        import pandas as pd
                        train_df = pd.read_csv(train_recon_csv, skiprows=3)
                        has_energy_data = 'signal_energy' in train_df.columns
                    except Exception:
                        has_energy_data = False
                
                # Apply reconstruction loss filtering if filter_mode is specified
                recon_ignored_files = []
                if filter_mode is not None:
                    print(f"Applying reconstruction loss filtering:")
                    print(f"  Filter mode: {filter_mode}")
                    print(f"  Parameters: {filter_params}")
                    
                    recon_train_ignored, recon_val_ignored, recon_test_ignored, _, _, _ = apply_filtering_from_directory(
                        args.use_filter_dir,
                        filter_mode=filter_mode,
                        filter_params=filter_params,
                        current_train_path=current_train_path,
                        current_test_path=current_test_path
                    )
                    recon_ignored_files = recon_test_ignored
                    print(f"Reconstruction loss filtering: {len(recon_test_ignored)} test files ignored")
                
                # Apply energy filtering if energy_filter_mode is specified and energy data is available
                energy_ignored_files = []
                if energy_filter_mode is not None and has_energy_data:
                    print(f"Applying energy filtering:")
                    print(f"  Energy mode: {energy_filter_mode}")
                    print(f"  Energy parameters: {filter_params}")
                    
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
                    
                    # Apply energy filtering using the energy-only function
                    energy_train_ignored, energy_val_ignored, energy_test_ignored, _, _, _ = apply_filtering_from_directory(
                        args.use_filter_dir,
                        filter_mode=energy_filter_mode,
                        filter_params=energy_filter_params,
                        current_train_path=current_train_path,
                        current_test_path=current_test_path
                    )
                    energy_ignored_files = energy_test_ignored
                    print(f"Energy filtering: {len(energy_test_ignored)} test files ignored")
                elif energy_filter_mode is not None and not has_energy_data:
                    print(f"Warning: Energy filtering requested but no energy data found in CSV files")
                
                # Combine ignored files from both filters
                if recon_ignored_files and energy_ignored_files:
                    # Union of both sets (files ignored by either filter)
                    combined_ignored = list(set(recon_ignored_files) | set(energy_ignored_files))
                    print(f"Combined filtering: {len(combined_ignored)} test files ignored (union of both filters)")
                    test_ignored_files = combined_ignored
                    train_ignored_files = list(set(recon_train_ignored) | set(energy_train_ignored)) if 'recon_train_ignored' in locals() and 'energy_train_ignored' in locals() else []
                    val_ignored_files = list(set(recon_val_ignored) | set(energy_val_ignored)) if 'recon_val_ignored' in locals() and 'energy_val_ignored' in locals() else []
                elif recon_ignored_files:
                    test_ignored_files = recon_ignored_files
                    train_ignored_files = recon_train_ignored if 'recon_train_ignored' in locals() else []
                    val_ignored_files = recon_val_ignored if 'recon_val_ignored' in locals() else []
                elif energy_ignored_files:
                    test_ignored_files = energy_ignored_files
                    train_ignored_files = energy_train_ignored if 'energy_train_ignored' in locals() else []
                    val_ignored_files = energy_val_ignored if 'energy_val_ignored' in locals() else []
                else:
                    test_ignored_files = []
                    train_ignored_files = []
                    val_ignored_files = []
                
                print(f"Train files ignored: {len(train_ignored_files)}")
                if val_ignored_files:
                    print(f"Validation files ignored: {len(val_ignored_files)}")
                print(f"Test files ignored: {len(test_ignored_files)}")
                
            except Exception as e:
                print(f"Error applying filtering: {e}")
                raise
            
            # Filter original results
            print(f"\n=== FILTERING RESULTS ===")
            filtered_results = filter_evaluation_results(original_results, test_ignored_files, args.patient_result, args.threshold)
            
            # Generate reports and comparison
            print(f"\n=== GENERATING REPORTS ===")
            original_output_dir = os.path.join(args.output_dir, "original_results")
            filtered_output_dir = os.path.join(args.output_dir, "filtered_results")
            os.makedirs(original_output_dir, exist_ok=True)
            os.makedirs(filtered_output_dir, exist_ok=True)
            
            # Generate reports
            evaluator.results = original_results
            generate_report(evaluator, original_output_dir, "original", args.threshold)
            
            temp_evaluator = type(evaluator)(evaluator.model_engine, evaluator.test_data_path, evaluator.mode)
            temp_evaluator.results = filtered_results
            generate_report(temp_evaluator, filtered_output_dir, "filtered", args.threshold)
            
            # Compare results
            compare_results(original_results, filtered_results, args.output_dir)
            
            # Find optimal thresholds for both scenarios
            optimal_thresholds = find_optimal_thresholds(original_results, filtered_results, args.patient_result, args.optimization_metric)
            
            # Save optimal thresholds to file
            optimal_threshold_file = os.path.join(args.output_dir, "optimal_thresholds.json")
            with open(optimal_threshold_file, 'w') as f:
                json.dump(optimal_thresholds, f, indent=2, default=str)
            print(f"Optimal thresholds saved to: {optimal_threshold_file}")
            
            # Print optimal threshold summary
            print(f"\n=== OPTIMAL THRESHOLD SUMMARY ===")
            print(f"Optimization metric: {args.optimization_metric}")
            if 'file_level_original' in optimal_thresholds:
                print(f"File-level (original): {optimal_thresholds['file_level_original']['threshold']:.4f}")
            if 'file_level_filtered' in optimal_thresholds:
                print(f"File-level (filtered): {optimal_thresholds['file_level_filtered']['threshold']:.4f}")
            if 'patient_level_original' in optimal_thresholds:
                print(f"Patient-level (original): {optimal_thresholds['patient_level_original']['threshold']:.4f}")
            if 'patient_level_filtered' in optimal_thresholds:
                print(f"Patient-level (filtered): {optimal_thresholds['patient_level_filtered']['threshold']:.4f}")
            
            # Print recommendations
            print_threshold_recommendations(optimal_thresholds, args.optimization_metric)
            
            # Use filtered results for final output
            results = filtered_results
        elif mode_type not in ['auto']:
            # No filtering - run normal evaluation (only for non-auto modes)
            print(f"\n=== STANDARD EVALUATION (NO FILTERING) ===")
            results = evaluator.run_evaluation(
                patient_result=args.patient_result,
                train_dataset=train_ds if 'train_ds' in locals() else None,
                threshold=args.threshold,
                run_mj=args.run_knn_MJ,
                K=args.K
            )
            
            # Find optimal thresholds for non-filtered scenario
            if results is not None and (mode_type == 'classifier' or mode_type == 'tripletloss'):
                print("\n=== FINDING OPTIMAL THRESHOLDS (NO FILTERING) ===")
                optimal_thresholds = find_optimal_thresholds(results, None, args.patient_result, args.optimization_metric)
                
                # Save optimal thresholds to file
                optimal_threshold_file = os.path.join(args.output_dir, "optimal_thresholds_no_filtering.json")
                with open(optimal_threshold_file, 'w') as f:
                    json.dump(optimal_thresholds, f, indent=2, default=str)
                print(f"Optimal thresholds saved to: {optimal_threshold_file}")
                
                # Print optimal threshold summary
                print(f"\n=== OPTIMAL THRESHOLD SUMMARY (NO FILTERING) ===")
                print(f"Optimization metric: {args.optimization_metric}")
                if 'file_level_original' in optimal_thresholds:
                    print(f"File-level: {optimal_thresholds['file_level_original']['threshold']:.4f}")
                if 'patient_level_original' in optimal_thresholds:
                    print(f"Patient-level: {optimal_thresholds['patient_level_original']['threshold']:.4f}")
        
        # Generate detailed report and plots
        print(f"\nGenerating detailed evaluation report...")
        # Set results in evaluator for report generation
        evaluator.results = results
        generate_report(evaluator, args.output_dir, "default", args.threshold)
        
        # Print summary
        if results is not None and (mode_type == 'classifier' or (mode_type == 'tripletloss' and not args.run_knn_MJ)):
            print(f"\n=== CLASSIFIER EVALUATION SUMMARY ===")
            print(f"Overall Accuracy: {results['accuracy']:.4f}")
            print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
            print(f"Sensitivity (True Positive Rate): {results['sensitivity']:.4f}")
            print(f"Specificity (True Negative Rate): {results['specificity']:.4f}")
            print(f"ROC AUC: {results['roc_curve']['auc']:.4f}")
            print(f"Average Precision: {results['pr_curve']['avg_precision']:.4f}")
            print(f"\nPer-Class Accuracy:")
            for class_name, acc in results['per_class_accuracy'].items():
                count = results['per_class_counts'][class_name]
                print(f"  {class_name}: {acc:.4f} ({count} samples)")
            print(f"\nPer-Class Metrics (from classification report):")
            for class_name in ['Bad', 'Good']:
                if class_name in results['classification_report']:
                    metrics = results['classification_report'][class_name]
                    print(f"  {class_name}:")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1-Score: {metrics['f1-score']:.4f}")
                    print(f"    Support: {metrics['support']}")
            if args.patient_result:
                print(f"================================================")
                print(f"PATIENT-LEVEL EVALUATION RESULTS (threshold: {args.threshold}):")
                print(f"================================================")
                print(f"Patient Accuracy: {results['total_patient_accuracy']:.4f}")
                print(f"Patient Sensitivity (True Positive Rate): {results['patient_sensitivity']:.4f}")
                print(f"Patient Specificity (True Negative Rate): {results['patient_specificity']:.4f}")
                print(f"Patient ROC AUC: {results['patient_roc_curve']['auc']:.4f}")
                print(f"Patient Average Precision: {results['patient_pr_curve']['avg_precision']:.4f}")
                
                print(f"\nPatient-Level Confusion Matrix:")
                print(results['patient_confusion_matrix'])
                print(f"\nPatient-Level Classification Report:")
                print(classification_report(results['patient_labels'], results['patient_predictions'], 
                                          target_names=['Bad', 'Good'], zero_division=0))
                
                print(f"\nPatient Statistics:")
                print(f"Total Number of Patients: {len(results['unique_patients'])}")
                print(f"Number of Bad Patients: {np.sum(results['patient_labels'] == 0)}")
                print(f"Number of Good Patients: {np.sum(results['patient_labels'] == 1)}")

                
                print(f"\nPatient-Level Metrics (from classification report):")
                for class_name in ['Bad', 'Good']:
                    if class_name in results['patient_classification_report']:
                        metrics = results['patient_classification_report'][class_name]
                        print(f"  {class_name}:")
                        print(f"    Precision: {metrics['precision']:.4f}")
                        print(f"    Recall: {metrics['recall']:.4f}")
                        print(f"    F1-Score: {metrics['f1-score']:.4f}")
                        print(f"    Support: {metrics['support']}")
        
        # CHANGED: print this for tripletloss mode anyway                
        if results is not None and mode_type == 'tripletloss':
            print(f"\nTriplet Loss Evaluation Summary:")
            print(f"Mean embedding norm: {results['embedding_stats']['mean_norm']:.4f}")
            print(f"Std embedding norm: {results['embedding_stats']['std_norm']:.4f}")
                
            print("=============Stats that matter=============")
            print(f"Positive mean distances from anchor: {results['positive_distance_mean']}")
            print(f"Positive std distances from anchor {results['positive_distance_std']}")
            print(f"Negative mean distances from anchor {results['negative_distance_mean']}")
            print(f"Negative std distances from anchor {results['negative_distance_std']}")
            print(f"\n")
            # print(f"Test Positive mean distances from anchor {results['test_positive_distance_mean']}")
            # print(f"Test Positive std distances from anchor {results['test_positive_distance_std']}")
            # print(f"Test Negative mean distances from anchor {results['test_negative_distance_mean']}")
            # print(f"Test Negative std distances from anchor {results['test_negative_distance_std']}")
            # print(f"\n")
            print(f"Kmeans Cluster Accuracy: {results['Kmeans_cluster_accuracy']}")

        elif results is not None and mode_type == 'auto':
            print(f"\nAutoencoder Evaluation Summary:")
            print(f"Mean reconstruction loss: {results['mean_loss']:.4f}")
            print(f"Std reconstruction loss: {results['std_loss']:.4f}")
            if args.create_ignored_recon: 
                print(f"Mean training reconstruction loss: {results['train_mean_recon_loss']:.4f}")
                print(f"Std training reconstruction loss: {results['train_std_recon_loss']:.4f}")   

def filter_evaluation_results(original_results, ignored_files, patient_result=False, threshold=0.55):
    """
    Filter evaluation results by removing data for ignored files.
    
    Args:
        original_results: Original evaluation results
        ignored_files: List of filenames to ignore
        patient_result: Whether patient-level results are included
        threshold: Threshold for patient-level predictions (file-level predictions are preserved)
    
    Returns:
        dict: Filtered evaluation results
    """
    print(f"Filtering results: removing {len(ignored_files)} ignored files")
    
    # Convert ignored files to set for faster lookup
    ignored_set = set(ignored_files)
    
    # Filter file-level results
    filtered_results = original_results.copy()
    
    # Get file paths from original results
    if 'patient_files' in original_results:
        file_paths = original_results['patient_files']
    else:
        # If no patient_files, we need to reconstruct from other data
        print("Warning: No patient_files found in results, cannot filter file-level metrics")
        return original_results
    
    # Find indices of files to keep
    keep_indices = []
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        if filename not in ignored_set:
            keep_indices.append(i)
    
    print(f"Keeping {len(keep_indices)} out of {len(file_paths)} files")
    
    # Filter arrays
    if 'labels' in original_results:
        filtered_results['labels'] = [original_results['labels'][i] for i in keep_indices]
    if 'probabilities' in original_results:
        filtered_results['probabilities'] = [original_results['probabilities'][i] for i in keep_indices]
    if 'patient_files' in original_results:
        filtered_results['patient_files'] = [original_results['patient_files'][i] for i in keep_indices]
    
    # Preserve original file-level predictions (no threshold recalculation)
    # This prevents sudden shifts in file-level confusion matrix when threshold changes
    if 'predictions' in original_results:
        filtered_results['predictions'] = [original_results['predictions'][i] for i in keep_indices]
        print(f"Preserved original file-level predictions for {len(keep_indices)} remaining files")
        
        # Debug: Show prediction distribution
        pred_array = np.array(filtered_results['predictions'])
        print(f"File-level prediction distribution: {np.sum(pred_array == 0)} Bad, {np.sum(pred_array == 1)} Good")
    else:
        print(f"Warning: No original predictions found in results")
    
    # Recalculate file-level metrics
    if 'predictions' in filtered_results and 'labels' in filtered_results:
        predictions = np.array(filtered_results['predictions'])
        labels = np.array(filtered_results['labels'])
        
        # Recalculate basic metrics
        filtered_results['accuracy'] = np.mean(predictions == labels)
        
        # Recalculate per-class accuracy
        unique_labels = np.unique(labels)
        class_names = ['Bad', 'Good']
        per_class_accuracy = {}
        per_class_counts = {}
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_count = np.sum(mask)
            class_correct = np.sum((labels == label) & (predictions == label))
            per_class_accuracy[class_names[i]] = class_correct / class_count if class_count > 0 else 0.0
            per_class_counts[class_names[i]] = int(class_count)
        
        filtered_results['per_class_accuracy'] = per_class_accuracy
        filtered_results['per_class_counts'] = per_class_counts
        filtered_results['balanced_accuracy'] = np.mean(list(per_class_accuracy.values()))
        
        # Recalculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        filtered_results['confusion_matrix'] = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            filtered_results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            filtered_results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            filtered_results['sensitivity'] = per_class_accuracy.get('Good', 0.0)
            filtered_results['specificity'] = per_class_accuracy.get('Bad', 0.0)
        
        # Recalculate ROC and PR curves
        if 'probabilities' in filtered_results:
            probabilities = np.array(filtered_results['probabilities'])
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            
            # Handle both 1D and 2D probability arrays
            if probabilities.ndim == 2:
                prob_for_roc = probabilities[:, 1]  # Use positive class probability
            else:
                prob_for_roc = probabilities  # Assume it's already positive class probability
            
            fpr, tpr, _ = roc_curve(labels, prob_for_roc)
            roc_auc = auc(fpr, tpr)
            filtered_results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            
            precision, recall, _ = precision_recall_curve(labels, prob_for_roc)
            avg_precision = average_precision_score(labels, prob_for_roc)
            filtered_results['pr_curve'] = {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}
    
    # Handle patient-level results
    # Note: Patient-level predictions are recalculated using threshold on mean probabilities
    # This allows threshold tuning for patient-level decisions while preserving file-level predictions
    if patient_result and 'patient_predictions' in original_results:
        print(f"Recalculating patient-level results using threshold {threshold}...")
        
        # Get all original file paths to identify completely filtered patients
        all_original_files = original_results['patient_files']
        all_original_labels = original_results['labels']
        
        # Get remaining file paths and their corresponding predictions/labels
        remaining_files = filtered_results['patient_files']
        remaining_predictions = filtered_results['predictions']
        remaining_labels = filtered_results['labels']
        remaining_probabilities = filtered_results['probabilities']
        
        # First, identify all patients in the original dataset
        all_patients = set()
        original_patient_files = {}
        
        for i, file_path in enumerate(all_original_files):
            try:
                filename = os.path.basename(file_path)
                if '_' in filename:
                    patient_id = filename.split('_')[1]
                else:
                    patient_id = next(
                        (part for part in Path(file_path).parts
                        if part.startswith('R2D2') or (len(part) > 8 and part.isalnum())),
                        filename
                    )
                
                all_patients.add(patient_id)
                if patient_id not in original_patient_files:
                    original_patient_files[patient_id] = []
                original_patient_files[patient_id].append(i)
                
            except Exception:
                # Fallback to filename as patient ID
                patient_id = os.path.basename(file_path)
                all_patients.add(patient_id)
                if patient_id not in original_patient_files:
                    original_patient_files[patient_id] = []
                original_patient_files[patient_id].append(i)
        
        # Group remaining files by patient ID
        patient_data = {}
        remaining_patients = set()
        
        for i, file_path in enumerate(remaining_files):
            try:
                filename = os.path.basename(file_path)
                if '_' in filename:
                    patient_id = filename.split('_')[1]
                else:
                    patient_id = next(
                        (part for part in Path(file_path).parts
                        if part.startswith('R2D2') or (len(part) > 8 and part.isalnum())),
                        filename
                    )
                
                remaining_patients.add(patient_id)
                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'predictions': [],
                        'labels': [],
                        'probabilities': [],
                        'files': []
                    }
                
                patient_data[patient_id]['predictions'].append(remaining_predictions[i])
                patient_data[patient_id]['labels'].append(remaining_labels[i])
                patient_data[patient_id]['probabilities'].append(remaining_probabilities[i])
                patient_data[patient_id]['files'].append(file_path)
                
            except Exception:
                # Fallback to filename as patient ID
                patient_id = os.path.basename(file_path)
                remaining_patients.add(patient_id)
                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'predictions': [],
                        'labels': [],
                        'probabilities': [],
                        'files': []
                    }
                patient_data[patient_id]['predictions'].append(remaining_predictions[i])
                patient_data[patient_id]['labels'].append(remaining_labels[i])
                patient_data[patient_id]['probabilities'].append(remaining_probabilities[i])
                patient_data[patient_id]['files'].append(file_path)
        
        # Identify completely filtered patients
        completely_filtered_patients = all_patients - remaining_patients
        if completely_filtered_patients:
            print(f"Found {len(completely_filtered_patients)} patients with all files filtered out. Treating them as 'good' class.")
            print(f"Completely filtered patients: {list(completely_filtered_patients)}")
        
        # Calculate patient-level predictions
        patient_predictions = []
        patient_labels = []
        patient_probabilities = []
        unique_patients = []
        
        # Process patients with remaining files
        for patient_id, data in patient_data.items():
            if len(data['predictions']) > 0:
                # Calculate mean probability (handle both 1D and 2D arrays)
                prob_array = np.array(data['probabilities'])
                if prob_array.ndim == 2:
                    prob = np.mean(prob_array[:, 1])  # Use positive class probability
                else:
                    prob = np.mean(prob_array)  # Assume it's already positive class probability
                
                # Apply threshold to mean probability for patient-level prediction
                pred = 1 if prob >= threshold else 0
                
                # Use first label (should be same for all files of same patient)
                label = data['labels'][0]
                
                patient_predictions.append(pred)
                patient_labels.append(label)
                patient_probabilities.append(prob)
                unique_patients.append(patient_id)
        
        # Add completely filtered patients as "good" class
        for patient_id in completely_filtered_patients:
            # Get the original label for this patient (should be same for all files)
            original_file_indices = original_patient_files[patient_id]
            original_label = all_original_labels[original_file_indices[0]]  # Use first file's label
            
            # Treat completely filtered patients as "good" class (prediction = 1)
            patient_predictions.append(1)  # Always predict "good" for completely filtered patients
            patient_labels.append(original_label)  # Keep original label for evaluation
            patient_probabilities.append(1.0)  # High confidence for "good" prediction
            unique_patients.append(patient_id)
            
            print(f"Patient {patient_id}: All files filtered out, treating as 'good' class (original label: {original_label})")
        
        # Update filtered results with recalculated patient-level data
        filtered_results['patient_predictions'] = patient_predictions
        filtered_results['patient_labels'] = patient_labels
        filtered_results['patient_probabilities'] = patient_probabilities
        filtered_results['unique_patients'] = unique_patients
        
        # Recalculate patient-level metrics
        if len(patient_predictions) > 0:
            from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
            
            # Basic accuracy
            filtered_results['total_patient_accuracy'] = np.mean(np.array(patient_predictions) == np.array(patient_labels))
            
            # Confusion matrix
            cm = confusion_matrix(patient_labels, patient_predictions)
            filtered_results['patient_confusion_matrix'] = cm
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                filtered_results['patient_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                filtered_results['patient_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                # Handle single class case
                filtered_results['patient_sensitivity'] = 1.0 if 1 in patient_labels else 0.0
                filtered_results['patient_specificity'] = 1.0 if 0 in patient_labels else 0.0
            
            # ROC and PR curves
            if len(patient_probabilities) > 0:
                prob_array = np.array(patient_probabilities)
                if prob_array.ndim == 2:
                    prob_for_roc = prob_array[:, 1]
                else:
                    prob_for_roc = prob_array
                
                fpr, tpr, _ = roc_curve(patient_labels, prob_for_roc)
                roc_auc = auc(fpr, tpr)
                filtered_results['patient_roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                
                precision, recall, _ = precision_recall_curve(patient_labels, prob_for_roc)
                avg_precision = average_precision_score(patient_labels, prob_for_roc)
                filtered_results['patient_pr_curve'] = {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}
            
            # Calculate patient-level classification report
            from sklearn.metrics import classification_report
            patient_report = classification_report(
                patient_labels, patient_predictions, 
                target_names=['Bad', 'Good'], 
                output_dict=True,
                zero_division=0
            )
            filtered_results['patient_classification_report'] = patient_report
        
        filtered_results['patient_results_recalculated'] = True
        print(f"Recalculated patient-level results for {len(unique_patients)} patients using threshold {threshold}")
        
        # Debug: Show patient-level prediction distribution
        patient_pred_array = np.array(patient_predictions)
        print(f"Patient-level prediction distribution: {np.sum(patient_pred_array == 0)} Bad, {np.sum(patient_pred_array == 1)} Good")
    
    return filtered_results

def compare_results(original_results, filtered_results, output_dir):
    """
    Compare original and filtered results and save comparison report.
    
    Args:
        original_results: Original evaluation results
        filtered_results: Filtered evaluation results
        output_dir: Output directory for comparison report
    """
    print("=== COMPARISON REPORT ===")
    
    # File-level comparison
    if 'accuracy' in original_results and 'accuracy' in filtered_results:
        print(f"File-level Accuracy:")
        print(f"  Original: {original_results['accuracy']:.4f}")
        print(f"  Filtered: {filtered_results['accuracy']:.4f}")
        print(f"  Difference: {filtered_results['accuracy'] - original_results['accuracy']:.4f}")
    
    if 'sensitivity' in original_results and 'sensitivity' in filtered_results:
        print(f"File-level Sensitivity:")
        print(f"  Original: {original_results['sensitivity']:.4f}")
        print(f"  Filtered: {filtered_results['sensitivity']:.4f}")
        print(f"  Difference: {filtered_results['sensitivity'] - original_results['sensitivity']:.4f}")
    
    if 'specificity' in original_results and 'specificity' in filtered_results:
        print(f"File-level Specificity:")
        print(f"  Original: {original_results['specificity']:.4f}")
        print(f"  Filtered: {filtered_results['specificity']:.4f}")
        print(f"  Difference: {filtered_results['specificity'] - original_results['specificity']:.4f}")
    
    if 'roc_curve' in original_results and 'roc_curve' in filtered_results:
        print(f"File-level ROC AUC:")
        print(f"  Original: {original_results['roc_curve']['auc']:.4f}")
        print(f"  Filtered: {filtered_results['roc_curve']['auc']:.4f}")
        print(f"  Difference: {filtered_results['roc_curve']['auc'] - original_results['roc_curve']['auc']:.4f}")
    
    # Patient-level comparison
    if 'total_patient_accuracy' in original_results and 'total_patient_accuracy' in filtered_results:
        print(f"Patient-level Accuracy:")
        print(f"  Original: {original_results['total_patient_accuracy']:.4f}")
        print(f"  Filtered: {filtered_results['total_patient_accuracy']:.4f}")
        print(f"  Difference: {filtered_results['total_patient_accuracy'] - original_results['total_patient_accuracy']:.4f}")
    
    if 'patient_sensitivity' in original_results and 'patient_sensitivity' in filtered_results:
        print(f"Patient-level Sensitivity:")
        print(f"  Original: {original_results['patient_sensitivity']:.4f}")
        print(f"  Filtered: {filtered_results['patient_sensitivity']:.4f}")
        print(f"  Difference: {filtered_results['patient_sensitivity'] - original_results['patient_sensitivity']:.4f}")
    
    if 'patient_specificity' in original_results and 'patient_specificity' in filtered_results:
        print(f"Patient-level Specificity:")
        print(f"  Original: {original_results['patient_specificity']:.4f}")
        print(f"  Filtered: {filtered_results['patient_specificity']:.4f}")
        print(f"  Difference: {filtered_results['patient_specificity'] - original_results['patient_specificity']:.4f}")
    
    if 'patient_roc_curve' in original_results and 'patient_roc_curve' in filtered_results:
        print(f"Patient-level ROC AUC:")
        print(f"  Original: {original_results['patient_roc_curve']['auc']:.4f}")
        print(f"  Filtered: {filtered_results['patient_roc_curve']['auc']:.4f}")
        print(f"  Difference: {filtered_results['patient_roc_curve']['auc'] - original_results['patient_roc_curve']['auc']:.4f}")
    
    # File-level confusion matrix comparison
    if 'confusion_matrix' in original_results and 'confusion_matrix' in filtered_results:
        print(f"\nFile-level Confusion Matrix Comparison:")
        print(f"Original:")
        print(original_results['confusion_matrix'])
        print(f"Filtered:")
        print(filtered_results['confusion_matrix'])
    else:
        print("File-level confusion matrix not available for comparison")
    
    # Patient-level confusion matrix comparison
    if 'patient_confusion_matrix' in original_results and 'patient_confusion_matrix' in filtered_results:
        print(f"\nPatient-level Confusion Matrix Comparison:")
        print(f"Original:")
        print(original_results['patient_confusion_matrix'])
        print(f"Filtered:")
        print(filtered_results['patient_confusion_matrix'])
    else:
        print("Patient-level confusion matrix not available for comparison")
    
    # Sample counts
    if 'labels' in original_results and 'labels' in filtered_results:
        print(f"Sample Counts:")
        print(f"  Original: {len(original_results['labels'])}")
        print(f"  Filtered: {len(filtered_results['labels'])}")
        print(f"  Removed: {len(original_results['labels']) - len(filtered_results['labels'])}")
    
    # Save comparison to file
    comparison_file = os.path.join(output_dir, "comparison_report.txt")
    with open(comparison_file, 'w') as f:
        f.write("=== EVALUATION RESULTS COMPARISON ===\n\n")
        
        if 'accuracy' in original_results and 'accuracy' in filtered_results:
            f.write(f"File-level Accuracy:\n")
            f.write(f"  Original: {original_results['accuracy']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['accuracy']:.4f}\n")
            f.write(f"  Difference: {filtered_results['accuracy'] - original_results['accuracy']:.4f}\n\n")
        
        if 'sensitivity' in original_results and 'sensitivity' in filtered_results:
            f.write(f"File-level Sensitivity:\n")
            f.write(f"  Original: {original_results['sensitivity']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['sensitivity']:.4f}\n")
            f.write(f"  Difference: {filtered_results['sensitivity'] - original_results['sensitivity']:.4f}\n\n")
        
        if 'specificity' in original_results and 'specificity' in filtered_results:
            f.write(f"File-level Specificity:\n")
            f.write(f"  Original: {original_results['specificity']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['specificity']:.4f}\n")
            f.write(f"  Difference: {filtered_results['specificity'] - original_results['specificity']:.4f}\n\n")
        
        if 'roc_curve' in original_results and 'roc_curve' in filtered_results:
            f.write(f"File-level ROC AUC:\n")
            f.write(f"  Original: {original_results['roc_curve']['auc']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['roc_curve']['auc']:.4f}\n")
            f.write(f"  Difference: {filtered_results['roc_curve']['auc'] - original_results['roc_curve']['auc']:.4f}\n\n")
        
        if 'total_patient_accuracy' in original_results and 'total_patient_accuracy' in filtered_results:
            f.write(f"Patient-level Accuracy:\n")
            f.write(f"  Original: {original_results['total_patient_accuracy']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['total_patient_accuracy']:.4f}\n")
            f.write(f"  Difference: {filtered_results['total_patient_accuracy'] - original_results['total_patient_accuracy']:.4f}\n\n")
        
        if 'patient_sensitivity' in original_results and 'patient_sensitivity' in filtered_results:
            f.write(f"Patient-level Sensitivity:\n")
            f.write(f"  Original: {original_results['patient_sensitivity']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['patient_sensitivity']:.4f}\n")
            f.write(f"  Difference: {filtered_results['patient_sensitivity'] - original_results['patient_sensitivity']:.4f}\n\n")
        
        if 'patient_specificity' in original_results and 'patient_specificity' in filtered_results:
            f.write(f"Patient-level Specificity:\n")
            f.write(f"  Original: {original_results['patient_specificity']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['patient_specificity']:.4f}\n")
            f.write(f"  Difference: {filtered_results['patient_specificity'] - original_results['patient_specificity']:.4f}\n\n")
        
        if 'patient_roc_curve' in original_results and 'patient_roc_curve' in filtered_results:
            f.write(f"Patient-level ROC AUC:\n")
            f.write(f"  Original: {original_results['patient_roc_curve']['auc']:.4f}\n")
            f.write(f"  Filtered: {filtered_results['patient_roc_curve']['auc']:.4f}\n")
            f.write(f"  Difference: {filtered_results['patient_roc_curve']['auc'] - original_results['patient_roc_curve']['auc']:.4f}\n\n")
        
        if 'labels' in original_results and 'labels' in filtered_results:
            f.write(f"Sample Counts:\n")
            f.write(f"  Original: {len(original_results['labels'])}\n")
            f.write(f"  Filtered: {len(filtered_results['labels'])}\n")
            f.write(f"  Removed: {len(original_results['labels']) - len(filtered_results['labels'])}\n\n")
    
    print(f"Comparison report saved to: {comparison_file}")

if __name__ == "__main__":
    main()
