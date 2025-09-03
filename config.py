import os 
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import json
from sklearn.manifold import TSNE
import csv
from pathlib import Path

def generate_report(evaluator, output_dir='evaluation_results', file_prefix="default", threshold=0.55):
    """Generate comprehensive evaluation report."""
    if evaluator.results is None:
        print("No evaluation results available. Run evaluation first.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    results_file = os.path.join(output_dir, f'{evaluator.mode}_{file_prefix}_evaluation_results.json')
    
    # Convert all numpy types to JSON-serializable
    # cm = results['confusion_matrix']
    
    filtered = dict(evaluator.results)
    filtered.pop('embeddings', None)
    filtered.pop('labels', None)
    filtered.pop('mean_embedding', None)
    filtered.pop('predictions', None)
    filtered.pop('probabilities', None)

    json_results = convert_numpy_types(filtered)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Generate visualizations
    if evaluator.mode == 'classifier':
        _plot_classifier_results(evaluator, output_dir, threshold)
        # If patient results are available, also plot patient-level results
        if 'patient_confusion_matrix' in evaluator.results:
            _plot_patient_results(evaluator, output_dir)
    elif evaluator.mode == 'tripletloss':
        _plot_triplet_results(evaluator, output_dir)
        _plot_classifier_results(evaluator, output_dir, threshold)
        # If patient results are available, also plot patient-level results
        if 'patient_confusion_matrix' in evaluator.results:
            _plot_patient_results(evaluator, output_dir)
    elif evaluator.mode == 'auto': 
        _plot_autoencoder_results(evaluator, output_dir)

def _plot_classifier_results(evaluator, output_dir, threshold=0.55):
    """Generate plots for classifier results."""
    results = evaluator.results
    if results is None:
        print("No classifier results available for plotting.")
        return
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    roc_data = results['roc_curve']
    plt.plot(roc_data['fpr'], roc_data['tpr'], 
            label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    pr_data = results['pr_curve']
    plt.plot(pr_data['recall'], pr_data['precision'], 
            label=f'PR Curve (AP = {pr_data["avg_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-Class Accuracy Bar Plot
    plt.figure(figsize=(8, 6))
    classes = list(results['per_class_accuracy'].keys())
    accuracies = list(results['per_class_accuracy'].values())
    counts = list(results['per_class_counts'].values())
    
    bars = plt.bar(classes, accuracies, color=['#ff7f7f', '#7fbf7f'], alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}\n({count} samples)', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Key Metrics Summary Plot
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'Sensitivity', 'Specificity', 'ROC AUC', 'Avg\nPrecision']
    values = [
        results['accuracy'],
        results['balanced_accuracy'],
        results['sensitivity'],
        results['specificity'],
        results['roc_curve']['auc'],
        results['pr_curve']['avg_precision']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Key Classification Metrics Summary')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save classification report as text
    report_file = os.path.join(output_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("=== CLASSIFIER EVALUATION RESULTS ===\n\n")
        if 'patient_confusion_matrix' in results:
            f.write(f"Patient-level threshold: {threshold}\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"Sensitivity (True Positive Rate): {results['sensitivity']:.4f}\n")
        f.write(f"Specificity (True Negative Rate): {results['specificity']:.4f}\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for class_name, acc in results['per_class_accuracy'].items():
            count = results['per_class_counts'][class_name]
            f.write(f"  {class_name}: {acc:.4f} ({count} samples)\n")
        f.write("\n")
        
        f.write("Detailed Classification Report:\n")
        f.write(str(classification_report(results['labels'], results['predictions'], 
                                        target_names=['Bad', 'Good'], zero_division='warn')))
        
        f.write(f"\nROC AUC: {results['roc_curve']['auc']:.4f}\n")
        f.write(f"Average Precision: {results['pr_curve']['avg_precision']:.4f}\n")
    
    print(f"Classifier plots saved to {output_dir}")

def _plot_triplet_results(evaluator, output_dir):
    """Generate plots for triplet loss results."""
    results = evaluator.results
    if results is None:
        print("No triplet results available for plotting.")
        return
    
    # Embedding distribution
    plt.figure(figsize=(10, 6))
    embedding_norms = np.linalg.norm(results['embeddings'], axis=1)
    plt.hist(embedding_norms, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Embedding Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Embedding Norms')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_norms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save embedding statistics
    stats_file = os.path.join(output_dir, 'embedding_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Embedding Statistics:\n")
        f.write(f"Mean norm: {results['embedding_stats']['mean_norm']:.4f}\n")
        f.write(f"Std norm: {results['embedding_stats']['std_norm']:.4f}\n")
        f.write(f"Number of embeddings: {len(results['embeddings'])}\n")
    
    # t-SNE visualization
    if len(results['embeddings']) > 1:
        try:
            plt.figure(figsize=(10, 8))
            tsne = TSNE(n_components=2, perplexity=min(30, len(results['embeddings'])-1), random_state=42)
            embeddings_2d = tsne.fit_transform(results['embeddings'])
            
            # Color by labels if available
            if 'labels' in results and len(results['labels']) == len(results['embeddings']):
                colors = ['darkred' if label == 1 else 'darkblue' for label in results['labels']]
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7, s=50)
                plt.title('t-SNE Visualization of Triplet Embeddings')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='darkred', label='Good'),
                                    Patch(facecolor='darkblue', label='Bad')]
                plt.legend(handles=legend_elements)
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
                plt.title('t-SNE Visualization of Triplet Embeddings')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tsne_embeddings.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create t-SNE visualization: {e}")
    
    # If patient-level results are available, also plot patient-level results
    if 'avg_patient_embeddings' in results:
        _plot_triplet_patient_results(evaluator, output_dir)
    
    print(f"Triplet plots saved to {output_dir}")

def _plot_triplet_patient_results(evaluator, output_dir):
    """Generate plots for triplet loss patient-level results."""
    results = evaluator.results
    if results is None:
        print("No triplet patient results available for plotting.")
        return
    
    # Patient-level embedding norm distribution
    plt.figure(figsize=(10, 6))
    patient_embedding_norms = np.linalg.norm(results['avg_patient_embeddings'], axis=1)
    patient_labels = results['avg_patient_labels']
    
    # Separate norms by patient label
    bad_norms = patient_embedding_norms[patient_labels == 0]
    good_norms = patient_embedding_norms[patient_labels == 1]
    
    plt.hist(bad_norms, bins=20, alpha=0.7, label='Bad Patients', color='red', edgecolor='black')
    plt.hist(good_norms, bins=20, alpha=0.7, label='Good Patients', color='green', edgecolor='black')
    plt.xlabel('Patient Embedding Norm')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient-Level Embedding Norms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_embedding_norms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient-level embedding statistics summary
    plt.figure(figsize=(10, 6))
    metrics = ['Patient\nMean Norm', 'Patient\nStd Norm', 'Sample\nMean Norm', 'Sample\nStd Norm']
    values = [
        results['patient_level_stats']['mean_norm'],
        results['patient_level_stats']['std_norm'],
        results['embedding_stats']['mean_norm'],
        results['embedding_stats']['std_norm']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.xlabel('Embedding Statistics')
    plt.ylabel('Norm Value')
    plt.title('Patient vs Sample Level Embedding Statistics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_vs_sample_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient-level t-SNE visualization
    if len(results['avg_patient_embeddings']) > 1:
        try:
            plt.figure(figsize=(10, 8))
            tsne = TSNE(n_components=2, perplexity=min(30, len(results['avg_patient_embeddings'])-1), random_state=42)
            patient_embeddings_2d = tsne.fit_transform(results['avg_patient_embeddings'])
            
            # Color by patient labels
            colors = ['darkred' if label == 1 else 'darkblue' for label in results['avg_patient_labels']]
            plt.scatter(patient_embeddings_2d[:, 0], patient_embeddings_2d[:, 1], c=colors, alpha=0.7, s=100)
            plt.title('t-SNE Visualization of Patient-Level Triplet Embeddings')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='darkred', label='Good Patients'),
                                Patch(facecolor='darkblue', label='Bad Patients')]
            plt.legend(handles=legend_elements)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tsne_patient_embeddings.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create patient-level t-SNE visualization: {e}")
    
    # Save patient-level embedding statistics
    stats_file = os.path.join(output_dir, 'patient_embedding_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("=== PATIENT-LEVEL TRIPLET LOSS EVALUATION RESULTS ===\n\n")
        f.write("Patient-Level Embedding Statistics:\n")
        f.write(f"Mean norm: {results['patient_level_stats']['mean_norm']:.4f}\n")
        f.write(f"Std norm: {results['patient_level_stats']['std_norm']:.4f}\n")
        f.write(f"Number of patients: {len(results['unique_patients'])}\n")
        f.write(f"Number of bad patients: {np.sum(results['avg_patient_labels'] == 0)}\n")
        f.write(f"Number of good patients: {np.sum(results['avg_patient_labels'] == 1)}\n\n")
        
        f.write("Sample-Level Embedding Statistics:\n")
        f.write(f"Mean norm: {results['embedding_stats']['mean_norm']:.4f}\n")
        f.write(f"Std norm: {results['embedding_stats']['std_norm']:.4f}\n")
        f.write(f"Number of samples: {len(results['embeddings'])}\n")
        
        f.write("\nPer-Patient Statistics:\n")
        for patient in results['unique_patients']:
            stats = results['patient_embedding_stats'][patient]
            f.write(f"Patient {patient}:\n")
            f.write(f"  Mean norm: {stats['mean_norm']:.4f}\n")
            f.write(f"  Std norm: {stats['std_norm']:.4f}\n")
            f.write(f"  Number of samples: {stats['num_samples']}\n")
            f.write(f"  Label: {'Good' if results['patient_labels'][patient][0] == 1 else 'Bad'}\n\n")
            
    
    
    print(f"Triplet patient-level plots saved to {output_dir}")

def create_recon_csv(results, output_dir, mode='train'):
    """Create CSV file with reconstruction data and energy data for training or testing."""
    if f'{mode}_mean_recon_loss' not in results:
        print(f"No {mode} reconstruction data available for CSV creation.")
        return
    
    csv_file = os.path.join(output_dir, f'{mode}_recon.csv')
    
    # Extract data
    losses = results[f'{mode}_corresponding_recon_loss']
    paths = results[f'{mode}_corresponding_path']
    
    # Extract energy data if available
    energies = None
    if f'{mode}_corresponding_energy' in results:
        energies = results[f'{mode}_corresponding_energy']
        
    # Calculate histogram for losses
    hist_bins, hist_counts = np.histogram(losses, bins=20)
    hist_bins_str = str(hist_bins.tolist())
    hist_counts_str = str(hist_counts.tolist())
    
    # Calculate histogram for energies if available
    energy_hist_bins_str = ""
    energy_hist_counts_str = ""
    if energies is not None:
        energy_hist_bins, energy_hist_counts = np.histogram(energies, bins=20)
        energy_hist_bins_str = str(energy_hist_bins.tolist())
        energy_hist_counts_str = str(energy_hist_counts.tolist())
    
    # Write CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with statistics
        if energies is not None:
            writer.writerow(['mean', 'std', 'histogram_bins', 'histogram_counts', 'energy_mean', 'energy_std', 'energy_histogram_bins', 'energy_histogram_counts'])
            writer.writerow([
                f"{results[f'{mode}_mean_recon_loss']:.6f}",
                f"{results[f'{mode}_std_recon_loss']:.6f}",
                hist_bins_str,
                hist_counts_str,
                f"{results[f'{mode}_mean_energy']:.6f}",
                f"{results[f'{mode}_std_energy']:.6f}",
                energy_hist_bins_str,
                energy_hist_counts_str
            ])
        else:
            writer.writerow(['mean', 'std', 'histogram_bins', 'histogram_counts'])
            writer.writerow([
                f"{results[f'{mode}_mean_recon_loss']:.6f}",
                f"{results[f'{mode}_std_recon_loss']:.6f}",
                hist_bins_str,
                hist_counts_str
            ])
        
        # Add empty line for separation
        writer.writerow([])
        
        # Write individual file data
        if energies is not None:
            writer.writerow(['filename', 'reconstruction_loss', 'signal_energy'])
            for path, loss, energy in zip(paths, losses, energies):
                if isinstance(path, list):
                    path = path[0] if path else "unknown"
                elif hasattr(path, 'tolist'):
                    path = path.tolist()
                    if isinstance(path, list) and path:
                        path = path[0]
                
                path_str = str(path)
                if path_str == "unknown":
                    filename = "unknown"
                else:
                    try:
                        filename = Path(path_str).name
                    except:
                        filename = path_str
                
                writer.writerow([filename, f"{loss:.6f}", f"{energy:.6f}"])
        else:
            writer.writerow(['filename', 'reconstruction_loss'])
            for path, loss in zip(paths, losses):
                if isinstance(path, list):
                    path = path[0] if path else "unknown"
                elif hasattr(path, 'tolist'):
                    path = path.tolist()
                    if isinstance(path, list) and path:
                        path = path[0]
                
                path_str = str(path)
                if path_str == "unknown":
                    filename = "unknown"
                else:
                    try:
                        filename = Path(path_str).name
                    except:
                        filename = path_str
                
                writer.writerow([filename, f"{loss:.6f}"])
    
    print(f"{mode.capitalize()} reconstruction CSV saved to {csv_file}")

def _plot_filtering_statistics(evaluator, output_dir):
    """Generate a dedicated plot showing filtering statistics and results."""
    results = evaluator.results
    if results is None or 'filtering_info' not in results:
        print("No filtering information available for plotting.")
        return
    
    filter_info = results['filtering_info']
    original_test_losses = filter_info['original_test_losses']
    original_train_losses = filter_info['original_train_losses']
    test_mask = filter_info['test_mask']
    train_mask = filter_info['train_mask']
    filter_mode = filter_info['filter_mode']
    filter_params = filter_info['filter_params']
    
    # Calculate bounds for visualization
    if filter_mode == 'median':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        q1 = np.percentile(original_test_losses, 25)
        q3 = np.percentile(original_test_losses, 75)
        iqr = q3 - q1
        iqr_lower = q1 - k * iqr
        iqr_upper = q3 + k * iqr
        
        final_lower = max(iqr_lower, hard_low_cut) if hard_low_cut is not None else iqr_lower
        final_upper = min(iqr_upper, hard_high_cut) if hard_high_cut is not None else iqr_upper
        
    elif filter_mode == 'mean':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        mean_loss = np.mean(original_test_losses)
        std_loss = np.std(original_test_losses)
        std_lower = mean_loss - k * std_loss
        std_upper = mean_loss + k * std_loss
        
        final_lower = max(std_lower, hard_low_cut) if hard_low_cut is not None else std_lower
        final_upper = min(std_upper, hard_high_cut) if hard_high_cut is not None else std_upper
        
    elif filter_mode == 'percentile':
        lower_p = filter_params.get('lower_percentile', 0.05)
        upper_p = filter_params.get('upper_percentile', 0.95)
        final_lower = np.percentile(original_test_losses, lower_p * 100)
        final_upper = np.percentile(original_test_losses, upper_p * 100)
        
    elif filter_mode == 'hardcode':
        final_lower = filter_params.get('lower_bound', 1.5)
        final_upper = filter_params.get('upper_bound', 7.0)
    
    # Create comprehensive filtering statistics plot
    plt.figure(figsize=(20, 12))
    
    # Panel 1: Test data filtering visualization
    plt.subplot(2, 3, 1)
    
    # Plot original test data
    plt.hist(original_test_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Test Data')
    
    # Highlight the kept region
    kept_test_losses = np.array(original_test_losses)[test_mask]
    plt.hist(kept_test_losses, bins=50, alpha=0.8, color='green', edgecolor='darkgreen', label=f'Kept Data ({len(kept_test_losses)} files)')
    
    # Add vertical lines for bounds
    plt.axvline(final_lower, color='red', linestyle='-', linewidth=2, label=f'Lower Bound: {final_lower:.3f}')
    plt.axvline(final_upper, color='red', linestyle='-', linewidth=2, label=f'Upper Bound: {final_upper:.3f}')
    
    # Add hard cutoff labels if they exist
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low Cut: {hard_low_cut:.3f}')
        if hard_high_cut is not None:
            plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High Cut: {hard_high_cut:.3f}')
    
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title(f'Test Data Filtering\n{filter_mode.upper()} Mode - Kept: {len(kept_test_losses)}/{len(original_test_losses)} files')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Training data filtering visualization
    plt.subplot(2, 3, 2)
    
    # Plot original training data
    plt.hist(original_train_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Training Data')
    
    # Highlight the kept region
    kept_train_losses = np.array(original_train_losses)[train_mask]
    plt.hist(kept_train_losses, bins=50, alpha=0.8, color='green', edgecolor='darkgreen', label=f'Kept Data ({len(kept_train_losses)} files)')
    
    # Add vertical lines for bounds
    plt.axvline(final_lower, color='red', linestyle='-', linewidth=2, label=f'Lower Bound: {final_lower:.3f}')
    plt.axvline(final_upper, color='red', linestyle='-', linewidth=2, label=f'Upper Bound: {final_upper:.3f}')
    
    # Add hard cutoff labels if they exist
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low Cut: {hard_low_cut:.3f}')
        if hard_high_cut is not None:
            plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High Cut: {hard_high_cut:.3f}')
    
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title(f'Training Data Filtering\n{filter_mode.upper()} Mode - Kept: {len(kept_train_losses)}/{len(original_train_losses)} files')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Box plot comparison
    plt.subplot(2, 3, 3)
    plt.boxplot([original_test_losses, kept_test_losses, original_train_losses, kept_train_losses], 
               labels=['Original\nTest', 'Filtered\nTest', 'Original\nTrain', 'Filtered\nTrain'])
    plt.ylabel('Reconstruction Loss')
    plt.title('Original vs Filtered Data Comparison')
    plt.grid(True, alpha=0.3)
    
    # Panel 4: Filtering bounds with highlighted kept region
    plt.subplot(2, 3, 4)
    
    # Plot histogram of test data
    plt.hist(original_test_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Data')
    
    # Highlight the kept region with a filled area
    plt.axvspan(final_lower, final_upper, alpha=0.4, color='green', label=f'KEPT REGION\n({len(kept_test_losses)} files)')
    
    # Add bound lines with clear labels
    plt.axvline(final_lower, color='red', linestyle='-', linewidth=3, label=f'LOWER BOUND\n{final_lower:.3f}')
    plt.axvline(final_upper, color='red', linestyle='-', linewidth=3, label=f'UPPER BOUND\n{final_upper:.3f}')
    
    # Add hard cutoff lines if they exist
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=3, label=f'HARD LOW CUT\n{hard_low_cut:.3f}')
        if hard_high_cut is not None:
            plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=3, label=f'HARD HIGH CUT\n{hard_high_cut:.3f}')
    
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title(f'Filtering Bounds - {filter_mode.upper()} Mode\nRed = bounds, Orange = hard cuts, Green = kept region')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Panel 5: Statistics summary
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    # Create text summary
    summary_text = f"""
FILTERING STATISTICS SUMMARY

Filter Mode: {filter_mode.upper()}
Filter Parameters: {filter_params}

TEST DATA:
• Original files: {len(original_test_losses)}
• Kept files: {len(kept_test_losses)}
• Filtered out: {len(original_test_losses) - len(kept_test_losses)}
• Kept percentage: {100*len(kept_test_losses)/len(original_test_losses):.1f}%

TRAINING DATA:
• Original files: {len(original_train_losses)}
• Kept files: {len(kept_train_losses)}
• Filtered out: {len(original_train_losses) - len(kept_train_losses)}
• Kept percentage: {100*len(kept_train_losses)/len(original_train_losses):.1f}%

FILTERING BOUNDS:
• Lower bound: {final_lower:.3f}
• Upper bound: {final_upper:.3f}
• Range: {final_upper - final_lower:.3f}
"""
    
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            summary_text += f"• Hard low cutoff: {hard_low_cut:.3f}\n"
        if hard_high_cut is not None:
            summary_text += f"• Hard high cutoff: {hard_high_cut:.3f}\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Panel 6: Loss distribution comparison
    plt.subplot(2, 3, 6)
    
    # Plot both original and filtered distributions
    plt.hist(original_test_losses, bins=30, alpha=0.5, label='Original Test', color='blue', density=True)
    plt.hist(kept_test_losses, bins=30, alpha=0.7, label='Filtered Test', color='green', density=True)
    plt.hist(original_train_losses, bins=30, alpha=0.5, label='Original Train', color='red', density=True)
    plt.hist(kept_train_losses, bins=30, alpha=0.7, label='Filtered Train', color='orange', density=True)
    
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Density')
    plt.title('Loss Distribution Comparison\n(Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtering_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Filtering statistics plot saved to {output_dir}/filtering_statistics.png")

def _plot_filtering_comparison(evaluator, output_dir):
    """Create a dedicated plot showing before vs after filtering comparison."""
    results = evaluator.results
    if results is None or 'filtering_info' not in results:
        print("No filtering information available for comparison plot.")
        return
    
    filter_info = results['filtering_info']
    original_test_losses = filter_info['original_test_losses']
    original_train_losses = filter_info['original_train_losses']
    test_mask = filter_info['test_mask']
    train_mask = filter_info['train_mask']
    filter_mode = filter_info['filter_mode']
    filter_params = filter_info['filter_params']
    
    # Calculate bounds
    if filter_mode == 'median':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        q1 = np.percentile(original_test_losses, 25)
        q3 = np.percentile(original_test_losses, 75)
        iqr = q3 - q1
        iqr_lower = q1 - k * iqr
        iqr_upper = q3 + k * iqr
        
        final_lower = max(iqr_lower, hard_low_cut) if hard_low_cut is not None else iqr_lower
        final_upper = min(iqr_upper, hard_high_cut) if hard_high_cut is not None else iqr_upper
        
    elif filter_mode == 'mean':
        k = filter_params.get('k', 1.5)
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        
        mean_loss = np.mean(original_test_losses)
        std_loss = np.std(original_test_losses)
        std_lower = mean_loss - k * std_loss
        std_upper = mean_loss + k * std_loss
        
        final_lower = max(std_lower, hard_low_cut) if hard_low_cut is not None else std_lower
        final_upper = min(std_upper, hard_high_cut) if hard_high_cut is not None else std_upper
        
    elif filter_mode == 'percentile':
        lower_p = filter_params.get('lower_percentile', 0.05)
        upper_p = filter_params.get('upper_percentile', 0.95)
        final_lower = np.percentile(original_test_losses, lower_p * 100)
        final_upper = np.percentile(original_test_losses, upper_p * 100)
        
    elif filter_mode == 'hardcode':
        final_lower = filter_params.get('lower_bound', 1.5)
        final_upper = filter_params.get('upper_bound', 7.0)
    
    # Get filtered data
    kept_test_losses = np.array(original_test_losses)[test_mask]
    kept_train_losses = np.array(original_train_losses)[train_mask]
    
    # Create the comparison plot
    plt.figure(figsize=(20, 12))
    
    # Test Data: Before vs After
    plt.subplot(2, 3, 1)
    plt.hist(original_test_losses, bins=50, alpha=0.7, color='red', edgecolor='black', label=f'Original ({len(original_test_losses)} files)')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Test Data - BEFORE Filtering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.hist(kept_test_losses, bins=50, alpha=0.7, color='green', edgecolor='black', label=f'Filtered ({len(kept_test_losses)} files)')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Test Data - AFTER Filtering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    # Show both overlaid
    plt.hist(original_test_losses, bins=50, alpha=0.5, color='red', edgecolor='black', label=f'Original ({len(original_test_losses)} files)')
    plt.hist(kept_test_losses, bins=50, alpha=0.7, color='green', edgecolor='black', label=f'Kept ({len(kept_test_losses)} files)')
    plt.axvline(final_lower, color='blue', linestyle='--', linewidth=2, label=f'Lower: {final_lower:.3f}')
    plt.axvline(final_upper, color='blue', linestyle='--', linewidth=2, label=f'Upper: {final_upper:.3f}')
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low: {hard_low_cut:.3f}')
        if hard_high_cut is not None:
            plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High: {hard_high_cut:.3f}')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Test Data - COMPARISON')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training Data: Before vs After
    plt.subplot(2, 3, 4)
    plt.hist(original_train_losses, bins=50, alpha=0.7, color='red', edgecolor='black', label=f'Original ({len(original_train_losses)} files)')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Training Data - BEFORE Filtering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.hist(kept_train_losses, bins=50, alpha=0.7, color='green', edgecolor='black', label=f'Filtered ({len(kept_train_losses)} files)')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Training Data - AFTER Filtering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # Show both overlaid
    plt.hist(original_train_losses, bins=50, alpha=0.5, color='red', edgecolor='black', label=f'Original ({len(original_train_losses)} files)')
    plt.hist(kept_train_losses, bins=50, alpha=0.7, color='green', edgecolor='black', label=f'Kept ({len(kept_train_losses)} files)')
    plt.axvline(final_lower, color='blue', linestyle='--', linewidth=2, label=f'Lower: {final_lower:.3f}')
    plt.axvline(final_upper, color='blue', linestyle='--', linewidth=2, label=f'Upper: {final_upper:.3f}')
    if filter_mode in ['median', 'mean']:
        hard_low_cut = filter_params.get('hard_low_cut', None)
        hard_high_cut = filter_params.get('hard_high_cut', None)
        if hard_low_cut is not None:
            plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low: {hard_low_cut:.3f}')
        if hard_high_cut is not None:
            plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High: {hard_high_cut:.3f}')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Training Data - COMPARISON')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtering_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plot comparison
    plt.figure(figsize=(12, 8))
    
    # Box plot showing the impact on outliers
    plt.subplot(1, 2, 1)
    plt.boxplot([original_test_losses, kept_test_losses], labels=['Original Test', 'Filtered Test'])
    plt.ylabel('Reconstruction Loss')
    plt.title('Test Data: Impact on Outliers')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([original_train_losses, kept_train_losses], labels=['Original Train', 'Filtered Train'])
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Data: Impact on Outliers')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Filtering comparison plots saved to {output_dir}")

def _plot_autoencoder_results(evaluator, output_dir):
    """Generate plots for autoencoder results."""
    results = evaluator.results
    if results is None:
        print("No autoencoder results available for plotting.")
        return
    
    # Loss distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['all_losses'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Losses')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # If training reconstruction data is available, create comparison plot
    if 'train_mean_recon_loss' in results:
        plt.figure(figsize=(15, 10))
        
        # Check if filtering was applied
        filtering_applied = 'filtering_info' in results
        
        if filtering_applied:
            # Get filtering information
            filter_info = results['filtering_info']
            original_test_losses = filter_info['original_test_losses']
            original_train_losses = filter_info['original_train_losses']
            test_mask = filter_info['test_mask']
            train_mask = filter_info['train_mask']
            filter_mode = filter_info['filter_mode']
            filter_params = filter_info['filter_params']
            
            # Calculate bounds for visualization
            if filter_mode == 'median':
                k = filter_params.get('k', 1.5)
                hard_low_cut = filter_params.get('hard_low_cut', None)
                hard_high_cut = filter_params.get('hard_high_cut', None)
                
                q1 = np.percentile(original_test_losses, 25)
                q3 = np.percentile(original_test_losses, 75)
                iqr = q3 - q1
                iqr_lower = q1 - k * iqr
                iqr_upper = q3 + k * iqr
                
                final_lower = max(iqr_lower, hard_low_cut) if hard_low_cut is not None else iqr_lower
                final_upper = min(iqr_upper, hard_high_cut) if hard_high_cut is not None else iqr_upper
                
            elif filter_mode == 'mean':
                k = filter_params.get('k', 1.5)
                hard_low_cut = filter_params.get('hard_low_cut', None)
                hard_high_cut = filter_params.get('hard_high_cut', None)
                
                mean_loss = np.mean(original_test_losses)
                std_loss = np.std(original_test_losses)
                std_lower = mean_loss - k * std_loss
                std_upper = mean_loss + k * std_loss
                
                final_lower = max(std_lower, hard_low_cut) if hard_low_cut is not None else std_lower
                final_upper = min(std_upper, hard_high_cut) if hard_high_cut is not None else std_upper
                
            elif filter_mode == 'percentile':
                lower_p = filter_params.get('lower_percentile', 0.05)
                upper_p = filter_params.get('upper_percentile', 0.95)
                final_lower = np.percentile(original_test_losses, lower_p * 100)
                final_upper = np.percentile(original_test_losses, upper_p * 100)
                
            elif filter_mode == 'hardcode':
                final_lower = filter_params.get('lower_bound', 1.5)
                final_upper = filter_params.get('upper_bound', 7.0)
            
            # Create a single comprehensive plot
            plt.figure(figsize=(16, 10))
            
            # Main plot: Original test data with highlighted kept region
            plt.subplot(2, 2, 1)
            
            # Plot original test data
            plt.hist(original_test_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Test Data')
            
            # Highlight the kept region
            kept_test_losses = np.array(original_test_losses)[test_mask]
            plt.hist(kept_test_losses, bins=50, alpha=0.8, color='green', edgecolor='darkgreen', label=f'Kept Data ({len(kept_test_losses)} files)')
            
            # Add vertical lines for bounds
            plt.axvline(final_lower, color='red', linestyle='--', linewidth=2, label=f'Lower Bound: {final_lower:.3f}')
            plt.axvline(final_upper, color='red', linestyle='--', linewidth=2, label=f'Upper Bound: {final_upper:.3f}')
            
            # Add hard cutoff labels if they exist
            if filter_mode in ['median', 'mean']:
                hard_low_cut = filter_params.get('hard_low_cut', None)
                hard_high_cut = filter_params.get('hard_high_cut', None)
                if hard_low_cut is not None:
                    plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low Cut: {hard_low_cut:.3f}')
                if hard_high_cut is not None:
                    plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High Cut: {hard_high_cut:.3f}')
            
            plt.xlabel('Reconstruction Loss')
            plt.ylabel('Frequency')
            plt.title(f'Test Data Filtering - {filter_mode.upper()} Mode\n(Kept: {len(kept_test_losses)}/{len(original_test_losses)} files)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Training data plot
            plt.subplot(2, 2, 2)
            
            # Plot original training data
            plt.hist(original_train_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Training Data')
            
            # Highlight the kept region
            kept_train_losses = np.array(original_train_losses)[train_mask]
            plt.hist(kept_train_losses, bins=50, alpha=0.8, color='green', edgecolor='darkgreen', label=f'Kept Data ({len(kept_train_losses)} files)')
            
            # Add vertical lines for bounds
            plt.axvline(final_lower, color='red', linestyle='--', linewidth=2, label=f'Lower Bound: {final_lower:.3f}')
            plt.axvline(final_upper, color='red', linestyle='--', linewidth=2, label=f'Upper Bound: {final_upper:.3f}')
            
            # Add hard cutoff labels if they exist
            if filter_mode in ['median', 'mean']:
                hard_low_cut = filter_params.get('hard_low_cut', None)
                hard_high_cut = filter_params.get('hard_high_cut', None)
                if hard_low_cut is not None:
                    plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard Low Cut: {hard_low_cut:.3f}')
                if hard_high_cut is not None:
                    plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=2, label=f'Hard High Cut: {hard_high_cut:.3f}')
            
            plt.xlabel('Reconstruction Loss')
            plt.ylabel('Frequency')
            plt.title(f'Training Data Filtering - {filter_mode.upper()} Mode\n(Kept: {len(kept_train_losses)}/{len(original_train_losses)} files)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot comparison
            plt.subplot(2, 2, 3)
            plt.boxplot([original_test_losses, kept_test_losses, original_train_losses, kept_train_losses], 
                       labels=['Original\nTest', 'Filtered\nTest', 'Original\nTrain', 'Filtered\nTrain'])
            plt.ylabel('Reconstruction Loss')
            plt.title('Original vs Filtered Data Comparison')
            plt.grid(True, alpha=0.3)
            
            # Filtering bounds visualization with clear highlighting
            plt.subplot(2, 2, 4)
            
            # Plot histogram of test data
            plt.hist(original_test_losses, bins=50, alpha=0.7, color='lightgray', edgecolor='black', label='Original Data')
            
            # Highlight the kept region with a filled area
            plt.axvspan(final_lower, final_upper, alpha=0.4, color='green', label=f'KEPT REGION\n({len(kept_test_losses)} files)')
            
            # Add bound lines with clear labels
            plt.axvline(final_lower, color='red', linestyle='-', linewidth=3, label=f'LOWER BOUND\n{final_lower:.3f}')
            plt.axvline(final_upper, color='red', linestyle='-', linewidth=3, label=f'UPPER BOUND\n{final_upper:.3f}')
            
            # Add hard cutoff lines if they exist
            if filter_mode in ['median', 'mean']:
                hard_low_cut = filter_params.get('hard_low_cut', None)
                hard_high_cut = filter_params.get('hard_high_cut', None)
                if hard_low_cut is not None:
                    plt.axvline(hard_low_cut, color='orange', linestyle=':', linewidth=3, label=f'HARD LOW CUT\n{hard_low_cut:.3f}')
                if hard_high_cut is not None:
                    plt.axvline(hard_high_cut, color='orange', linestyle=':', linewidth=3, label=f'HARD HIGH CUT\n{hard_high_cut:.3f}')
            
            plt.xlabel('Reconstruction Loss')
            plt.ylabel('Frequency')
            plt.title(f'Filtering Bounds - {filter_mode.upper()} Mode\nRed lines = bounds, Orange lines = hard cuts, Green = kept region')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'test_vs_training_reconstruction.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        else:
            # Original plotting without filtering
            plt.subplot(1, 2, 1)
            plt.hist(results['all_losses'], bins=30, alpha=0.7, label='Test Data', color='blue', edgecolor='black')
            plt.hist(results['train_corresponding_recon_loss'], bins=30, alpha=0.7, label='Training Data', color='red', edgecolor='black')
            plt.xlabel('Reconstruction Loss')
            plt.ylabel('Frequency')
            plt.title('Test vs Training Reconstruction Losses')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot box plot comparison
            plt.subplot(1, 2, 2)
            plt.boxplot([results['all_losses'], results['train_corresponding_recon_loss']], 
                       labels=['Test Data', 'Training Data'])
            plt.ylabel('Reconstruction Loss')
            plt.title('Test vs Training Loss Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_vs_training_reconstruction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save loss statistics
    stats_file = os.path.join(output_dir, 'loss_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Reconstruction Loss Statistics:\n")
        f.write(f"Mean loss: {results['mean_loss']:.4f}\n")
        f.write(f"Std loss: {results['std_loss']:.4f}\n")
        f.write(f"Min loss: {results['min_loss']:.4f}\n")
        f.write(f"Max loss: {results['max_loss']:.4f}\n")
        f.write(f"Number of samples: {len(results['all_losses'])}\n")
        
        # Add training reconstruction statistics if available
        if 'train_mean_recon_loss' in results:
            f.write(f"\nTraining Reconstruction Loss Statistics:\n")
            f.write(f"Mean training loss: {results['train_mean_recon_loss']:.4f}\n")
            f.write(f"Std training loss: {results['train_std_recon_loss']:.4f}\n")
            f.write(f"Number of training samples: {len(results['train_corresponding_recon_loss'])}\n")
        
        # Add filtering statistics if available
        if 'filtering_info' in results:
            filter_info = results['filtering_info']
            f.write(f"\nFiltering Statistics:\n")
            f.write(f"Filter mode: {filter_info['filter_mode']}\n")
            f.write(f"Filter parameters: {filter_info['filter_params']}\n")
            f.write(f"Test data: {len(results['test_corresponding_recon_loss'])} kept out of {len(filter_info['original_test_losses'])} total\n")
            f.write(f"Training data: {len(results['train_corresponding_recon_loss'])} kept out of {len(filter_info['original_train_losses'])} total\n")
    
    # Create training reconstruction CSV if available
    if 'train_mean_recon_loss' in results:
        create_recon_csv(results, output_dir, mode = 'train')
    
    if 'test_mean_recon_loss' in results:
        create_recon_csv(results, output_dir, mode = 'test')
    
    # Create validation reconstruction CSV if available
    if 'val_mean_recon_loss' in results:
        create_recon_csv(results, output_dir, mode = 'val')
    
    # Create filtering comparison plots if filtering was applied
    if 'filtering_info' in results:
        _plot_filtering_comparison(evaluator, output_dir)
    
    print(f"Autoencoder plots saved to {output_dir}")

def _plot_patient_results(evaluator, output_dir):
    """Generate plots for patient-level results."""
    results = evaluator.results
    if results is None:
        print("No patient results available for plotting.")
        return
    
    # Patient-level Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = results['patient_confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    plt.title('Patient-Level Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient-level ROC Curve
    plt.figure(figsize=(8, 6))
    roc_data = results['patient_roc_curve']
    plt.plot(roc_data['fpr'], roc_data['tpr'], 
            label=f'Patient ROC Curve (AUC = {roc_data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient-level Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    pr_data = results['patient_pr_curve']
    plt.plot(pr_data['recall'], pr_data['precision'], 
            label=f'Patient PR Curve (AP = {pr_data["avg_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patient-Level Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient-level Key Metrics Summary Plot
    plt.figure(figsize=(10, 6))
    metrics = ['Patient\nAccuracy', 'Patient\nSensitivity', 'Patient\nSpecificity', 'Patient\nROC AUC', 'Patient\nAvg\nPrecision']
    values = [
        results['total_patient_accuracy'],
        results['patient_sensitivity'],
        results['patient_specificity'],
        results['patient_roc_curve']['auc'],
        results['patient_pr_curve']['avg_precision']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.xlabel('Patient-Level Metrics')
    plt.ylabel('Score')
    plt.title('Patient-Level Classification Metrics Summary')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_key_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patient probability distribution
    plt.figure(figsize=(10, 6))
    patient_probs = results['patient_probabilities']
    patient_labels = results['patient_labels']
    
    # Separate probabilities by true label
    bad_probs = patient_probs[patient_labels == 0]
    good_probs = patient_probs[patient_labels == 1]
    
    plt.hist(bad_probs, bins=20, alpha=0.7, label='Bad Patients', color='red', edgecolor='black')
    plt.hist(good_probs, bins=20, alpha=0.7, label='Good Patients', color='green', edgecolor='black')
    plt.xlabel('Patient Probability Score')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient Probability Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patient_probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save patient-level classification report as text
    report_file = os.path.join(output_dir, 'patient_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("=== PATIENT-LEVEL CLASSIFICATION EVALUATION RESULTS ===\n\n")
        f.write(f"Patient Accuracy: {results['total_patient_accuracy']:.4f}\n")
        f.write(f"Patient Sensitivity (True Positive Rate): {results['patient_sensitivity']:.4f}\n")
        f.write(f"Patient Specificity (True Negative Rate): {results['patient_specificity']:.4f}\n")
        f.write(f"Patient ROC AUC: {results['patient_roc_curve']['auc']:.4f}\n")
        f.write(f"Patient Average Precision: {results['patient_pr_curve']['avg_precision']:.4f}\n\n")
        
        f.write("Patient-Level Confusion Matrix:\n")
        f.write(str(results['patient_confusion_matrix']))
        f.write("\n\n")
        
        f.write("Detailed Patient-Level Classification Report:\n")
        f.write(str(classification_report(results['patient_labels'], results['patient_predictions'], target_names=['Bad', 'Good'], zero_division='warn')))
        
        f.write(f"\nNumber of Patients: {len(results['unique_patients'])}\n")
        f.write(f"Number of Bad Patients: {np.sum(results['patient_labels'] == 0)}\n")
        f.write(f"Number of Good Patients: {np.sum(results['patient_labels'] == 1)}\n")
    
    print(f"Patient-level plots saved to {output_dir}")

def convert_numpy_types(obj):
    """Recursively convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj