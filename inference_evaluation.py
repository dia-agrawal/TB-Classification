#!/usr/bin/env python3
"""
Inference and Evaluation Script for Audio Classification Model
=============================================================

This script provides functionality to:
1. Load trained models (autoencoder, classifier, triplet loss)
2. Perform inference on new audio data
3. Evaluate model performance
4. Generate detailed evaluation reports
5. Visualize results

Usage:
    python inference_evaluation.py --mode classifier --checkpoint_path path/to/checkpoint.ckpt --data_path path/to/test/data
"""

import argparse
import os, random
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import yaml
from tqdm import tqdm
import json
from pathlib import Path

# Add the Audio_mae path
sys.path.append('Audio_mae')

from lightning_combined import UnifiedLightningModel, AudioDataset, load_config
from torch.utils.data import DataLoader

# Set seed for reproducibility
seed = 0   
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AudioInferenceEngine:
    """Engine for performing inference on audio data using trained models."""
    
    def __init__(self, checkpoint_path, mode='classifier', device='auto', model_params=None):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            mode (str): Model mode ('auto', 'classifier', 'tripletloss')
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.device = self._setup_device(device)
        self.model_params = model_params
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Mode: {mode}, Device: {self.device}")
    
    def _setup_device(self, device):
        """Setup the device for inference."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            model_params = hparams.get('model_params', {})
            mode = hparams.get('mode', self.mode)
            triplet_recon_loss = hparams.get('triplet_recon_loss', None)
            triplet_recon_weight = hparams.get('triplet_recon_weight', 1.0)
        else:
            # Fallback to default parameters
            model_params = self.model_params
            mode = self.mode
            triplet_recon_loss = None
            triplet_recon_weight = 1.0
        
        # Create model
        model = UnifiedLightningModel(
            model_params=model_params,
            mode=mode,
            triplet_recon_loss=triplet_recon_loss,
            triplet_recon_weight=triplet_recon_weight,
            class_weight=[4.0, 1.0]  # Default class weights for inference
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        # Ensure model is in float32 precision
        model = model.float()
        
        return model
    
    def predict_single(self, audio_path):
        """
        Perform inference on a single audio file.
        
        Args:
            audio_path (str): Path to the audio file (.pt format)
            
        Returns:
            dict: Prediction results
        """
        try:
            # Load audio data
            mel = torch.load(audio_path)
            mel = mel[:, 1:]  # Remove first column
            mel = F.pad(input=mel, pad=(0, 0, 0, 59), mode='constant', value=0)
            mel = mel.unsqueeze(0).to(self.device)  # Add batch dimension
            # Ensure float32 precision to match model parameters
            mel = mel.float()
            
            with torch.no_grad():
                if self.mode == 'classifier':
                    # Get logits and probabilities
                    logits = self.model.model.forward_encoder_nomasking_classification(mel)
                    probabilities = F.softmax(logits, dim=1)
                    predicted_class = torch.argmax(logits, dim=1).item()
                    confidence = torch.max(probabilities, dim=1)[0].item()
                                        
                    
                    
                    return {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities.cpu().numpy()[0],
                        'logits': logits.cpu().numpy()[0]
                    }
                
                elif self.mode == 'tripletloss':
                    # Get embeddings
                    embeddings = self.model.model.forward_encoder_nomasking(mel)
                                        
                    return {
                        'embeddings': embeddings.cpu().numpy()[0],
                        'embedding_norm': torch.norm(embeddings, dim=1).item()
                    }
                
                elif self.mode == 'auto':
                    # Get reconstruction
                    loss, pred, mask = self.model.model(mel, mask_ratio=0.8)
                    return {
                        'reconstruction_loss': loss.item(),
                        'prediction': pred.cpu().numpy(),
                        'mask': mask.cpu().numpy()
                    }
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def predict_batch(self, audio_paths, batch_size=32):
        """
        Perform batch inference on multiple audio files.
        
        Args:
            audio_paths (list): List of audio file paths
            batch_size (int): Batch size for inference
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Processing batches"):
            batch_paths = audio_paths[i:i + batch_size]
            batch_results = []
            
            for path in batch_paths:
                result = self.predict_single(path)
                if result is not None:
                    result['file_path'] = path
                    batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


class AudioEvaluator:
    """Evaluator for assessing model performance."""
    
    def __init__(self, model_engine, test_data_path, mode='classifier'):
        """
        Initialize the evaluator.
        
        Args:
            model_engine (AudioInferenceEngine): Loaded inference engine
            test_data_path (str): Path to test data directory
            mode (str): Model mode
        """
        self.model_engine = model_engine
        self.test_data_path = test_data_path
        self.mode = mode
        self.results = None
    
    def load_test_data(self):
        """Load test data for evaluation."""
        if self.mode == 'classifier':
            # Load classifier test data
            good_dir = os.path.join(self.test_data_path, 'good')  #.
            bad_dir = os.path.join(self.test_data_path, 'bad')    #.
            
            test_ds = AudioDataset(                                #.
                mode='classifier',
                data_dir_good=[good_dir],
                data_dir_bad=[bad_dir],
                apply_specaugment=False,
                shuffle=False
            )

        elif self.mode == 'tripletloss':
            test_ds = AudioDataset(
                mode='triplet',
                data_dir_good=[os.path.join(self.test_data_path, 'good')],
                data_dir_bad=[os.path.join(self.test_data_path, 'bad')]
            )
        else:
            test_ds = AudioDataset(
                mode='auto',
                data_dir_good=[os.path.join(self.test_data_path, 'good')]
            )
        return test_ds
    
    def evaluate_knn(self, test_ds, X_ref, Y_ref, K=5):
        """Evaluate triplet loss performance using KNN"""
        print("Evaluating triplet loss performance using KNN...")

        knn_predictions = []
        knn_ground_truth = []
        
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        self.model_engine.model.eval()
        
        with torch.no_grad():
            for BatchOfMels, BatchOfLabels, _ in test_loader: 
                BatchOfMels = BatchOfMels.to(self.model_engine.device).float()         # [1,1,256,256]
                BatchOfEmb = self.model_engine.model.model.forward_encoder_nomasking(BatchOfMels)  # [1,D]
                BatchOfEmb = BatchOfEmb.cpu().numpy()
                BatchOfLabels = BatchOfLabels.cpu().numpy()
                
                for query, true_label in zip(BatchOfEmb, BatchOfLabels):
                    # print(X_ref.shape)
                    # print(query.shape)
                    euc_distance = np.linalg.norm(X_ref - query, axis=1) 
                #[[D_mel1] - [d], [D_mel2] - [d], ...]
                
                #finding k nearest neighbor: 
                    knn_idx = np.argsort(euc_distance)
                    knn_idx = knn_idx[:K] #using only the fist K values (first K=5 smallest values)
                    knn_idx_labels = Y_ref[knn_idx]
                
                    pred = int(knn_idx_labels.sum() > (K/2)) 
                #if more than half of the labels are 1, then the prediction is 1, otherwise 0
                    knn_predictions.append(pred)
                #append prediction to knn_predictions
                    #print("size of me",true_label.size)
                    knn_ground_truth.append(int(true_label))
                #append ground truth to knn_ground_truth

            acc = np.mean(np.array(knn_predictions) == np.array(knn_ground_truth))
            
            return {"KNN_accuracy_majority_rule": acc}

    def run_knn(self, train_ds, test_ds, K=5):
        X_ref, Y_ref = [], []
        ref_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

        with torch.no_grad():
            for anchor, positive, negative in ref_loader:
                # anchor comes from "good" dir → label = 1
                emb = self.model_engine.model.model.forward_encoder_nomasking(
                            anchor.to(self.model_engine.device).float()
                    )
                X_ref.append(emb.squeeze(0).cpu().numpy())
                Y_ref.append(1)

                # positive is also "good" → label = 1
                emb_p = self.model_engine.model.model.forward_encoder_nomasking(
                            positive.to(self.model_engine.device).float()
                    )
                X_ref.append(emb_p.squeeze(0).cpu().numpy())
                Y_ref.append(1)

                # negative comes from "bad" → label = 0
                emb_n = self.model_engine.model.model.forward_encoder_nomasking(
                            negative.to(self.model_engine.device).float()
                    )
                X_ref.append(emb_n.squeeze(0).cpu().numpy())
                Y_ref.append(0)

        X_ref = np.stack(X_ref)     # shape: (3 * N_ref, 256)
        Y_ref = np.array(Y_ref)     # shape: (3 * N_ref,)
        return self.evaluate_knn(test_ds, X_ref, Y_ref, K)

        
    def evaluate_classifier(self, test_ds, patient_result=False, threshold=0.55):
        """Evaluate classifier performance."""
        print("Evaluating classifier performance...")
        
        # Create dataloader
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        per_patient_predictions = dict()
        per_patient_labels = dict()
        per_patient_probabilities = dict()
        
        self.model_engine.model.eval()
        
            
        all_file_paths = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):                
                if self.mode == 'classifier':
                    mel, labels, file_path = batch
                    mel = mel.to(self.model_engine.device).float()  # Ensure float32 precision
                    labels = labels.to(self.model_engine.device)
                    
                    logits = self.model_engine.model.model.forward_encoder_nomasking_classification(mel)
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    
                    labels = labels.cpu().tolist()
                    predictions = predictions.cpu().tolist()
                    probabilities = probabilities.cpu().tolist()

                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    all_probabilities.extend(probabilities)
                    all_file_paths.extend(file_path)
                    
                    if patient_result:
                        for f, p, l, pr in zip(file_path, predictions, labels, probabilities):
                            # Extract patient ID from file path
                            # Assuming file path structure: .../patient_id_.../filename.pt
                            # or filename contains patient_id
                            try:
                                # Try to extract from filename first
                                filename = Path(f).stem
                                if '_' in filename:
                                    patient_id = filename.split('_')[1]
                                else:
                                    # Try to extract from path
                                    path_parts = Path(f).parts
                                    for part in path_parts:
                                        if part.startswith('R2D2') or (len(part) > 8 and part.isalnum()):
                                            patient_id = part
                                            break
                                    else:
                                        # Fallback: use filename as patient ID
                                        patient_id = filename
                                
                                # print(f"File: {f} -> Patient ID: {patient_id} -> Label: {l} -> Prediction: {p}")
                            except Exception as e:
                                print(f"Error extracting patient ID from {f}: {e}")
                                patient_id = Path(f).stem
                            if patient_id not in per_patient_predictions:
                                per_patient_predictions[patient_id] = [p]
                                per_patient_labels[patient_id] = [l]
                                per_patient_probabilities[patient_id] = [pr]
                            else:
                                if per_patient_labels[patient_id][-1] != l:
                                    print (f"patient_id: {patient_id} \n predictions: {predictions} \n labels: {labels} \n probabilities: {probabilities}")
                                    assert False, "predictions and labels are not the same"
                                
                                # print (f"p: {p} {p.cpu().numpy()}") # {p.cpu().numpy()[0]}")
                                per_patient_predictions[patient_id].append(p)
                                per_patient_labels[patient_id].append(l)
                                per_patient_probabilities[patient_id].append(pr)
                                
                            
        # if patient_result:
        #     patient_ids = [Path(f).stem.split('_')[1] for f in all_file_paths]

                    
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # print(f"all_file_paths: {all_file_paths}")
        
        if patient_result:
            respective_patient_accuracy = []
            # per_patient_labels = []
            # per_patient_predictions = []
            
            # unique_patients = np.unique(patient_ids)
            # # print(f"unique_patients: {unique_patients}")
            
            # patient_files = {pid: [] for pid in unique_patients}
            # # print(f"patient_files: {patient_files}")
            # for patient in unique_patients:
            #     tmp = []
            #     for idx, id in enumerate(patient_ids):
            #         if id == patient:
            #             tmp.append(idx)
            #     patient_files[patient] = tmp
                
            # # print(f"patient_files: {patient_files}")
                
            # for patient, idxs in patient_files.items():
            p_labels = []
            p_predictions = []
            p_probabilities = []

            # print (f"per_patient_predictions: {per_patient_predictions.keys()}")
            # print (f"per_patient_labels: {per_patient_labels.keys()}")
            # print (f"per_patient_probabilities: {per_patient_probabilities.keys()}")

            
            for patient_id in per_patient_predictions.keys():
                patient_predictions = per_patient_predictions[patient_id]
                patient_labels = per_patient_labels[patient_id]
                
                if_all_labels_zero = np.all(np.array(patient_labels) == 0)
                if_all_labels_one = np.all(np.array(patient_labels) == 1)
                
                assert if_all_labels_zero or if_all_labels_one, f"patient_id: {patient_id} {if_all_labels_zero} {if_all_labels_one} \n patient_labels should be either all 0 or all 1 {patient_labels}"
                    
                patient_probabilities = per_patient_probabilities[patient_id]
                
                patient_accuracy = 1 if np.mean(patient_predictions == patient_labels) > 0.5 else 0
                respective_patient_accuracy.append(patient_accuracy)
                    
                if np.mean(patient_predictions) > threshold:
                    p_predictions.append(1)
                    
                else:
                    p_predictions.append(0)
                
                # Average only the positive class probability (index 1) for ROC curve
                # Validate probability structure
                if len(patient_probabilities[0]) != 2:
                    print(f"Warning: Patient {patient_id} has unexpected probability structure: {patient_probabilities[0]}")
                
                positive_probs = [prob[1] for prob in patient_probabilities]
                p_probabilities.append(np.mean(positive_probs))
                p_labels.append(patient_labels[0])
                    

            total_patient_accuracy = np.mean(respective_patient_accuracy)
            #print(f"total_patient_accuracy: {total_patient_accuracy}")
            
        accuracy = np.mean(all_predictions == all_labels)
        if patient_result:
            # print (f"File wise accuracy: {accuracy} \n Patient wise accuracy: {total_patient_accuracy} \n all_predictions: {per_patient_predictions}  \n all_labels: {per_patient_labels}")
            print (f"File wise accuracy: {accuracy} \n Patient wise accuracy: {total_patient_accuracy} (threshold: {threshold})")
        else:
            print (f"File wise accuracy: {accuracy}")
        
        # Per-class accuracy
        unique_labels = np.unique(all_labels)
        class_names = ['Bad', 'Good']
        per_class_accuracy = {}
        per_class_counts = {}
        
        for i, label in enumerate(unique_labels):
            mask = all_labels == label # [True, False, True, False, False]
            class_count = np.sum(mask) #2
            class_correct = np.sum((all_labels == label) & (all_predictions == label)) 
            #[TRue, False, True, False, False] & [True, True, True, False, False] => 2
            per_class_accuracy[class_names[i]] = class_correct / class_count if class_count > 0 else 0.0 #class[bad] = 2/2 = 1.0 
            per_class_counts[class_names[i]] = int(class_count) #class count [bad] = 2
        
        # Balanced accuracy (average of per-class accuracy)
        balanced_accuracy = np.mean(list(per_class_accuracy.values())) #overall accuracy 
        
        # Calculate sensitivity and specificity
        # Assuming: 0 = Bad (Negative), 1 = Good (Positive)
        # Sensitivity = TP / (TP + FN) = Recall for positive class
        # Specificity = TN / (TN + FP) = Recall for negative class
        
        # Confusion matrix values
        cm = confusion_matrix(all_labels, all_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Handle edge cases where confusion matrix might not be 2x2
            sensitivity = per_class_accuracy.get('Good', 0.0)
            specificity = per_class_accuracy.get('Bad', 0.0)
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # ROC curve
        all_probabilities = np.array(all_probabilities)
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])
        avg_precision = average_precision_score(all_labels, all_probabilities[:, 1])
        
        if patient_result:
            # Calculate patient-level confusion matrix and metrics
            # patient_predictions = []
            # patient_labels = []
            
            # for patient, idxs in patient_files.items():
            #     patient_label = all_labels[idxs[0]]  # All samples for a patient have same label
            #     # Use probabilities instead of binary predictions for patient-level decision
            #     patient_prob = np.mean(all_probabilities[idxs, 1])  # Average probability for positive class
            #     patient_prediction = 1 if patient_prob > 0.5 else 0
            #     patient_predictions.append(patient_prediction)
            #     patient_labels.append(patient_label)
            
            p_predictions = np.array(p_predictions)
            p_labels = np.array(p_labels)
            
            # Patient-level confusion matrix
            patient_cm = confusion_matrix(p_labels, p_predictions)
            
            # Patient-level metrics
            if patient_cm.shape == (2, 2):
                tn, fp, fn, tp = patient_cm.ravel()
                patient_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                patient_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                patient_sensitivity = 0.0
                patient_specificity = 0.0
            
            # Patient-level classification report
            patient_report = classification_report(
                p_labels, p_predictions, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Patient-level ROC curve
            # patient_probabilities = []
            # for patient, idxs in patient_files.items():
            #     patient_prob = np.mean(all_probabilities[idxs, 1])  # Average probability for positive class
            #     patient_probabilities.append(patient_prob)
            
            p_probabilities = np.array(p_probabilities)
            
            # Debug information
            # print(f"\nPatient-level debugging info:")
            # print(f"Number of patients: {len(p_labels)}")
            # print(f"Patient labels: {p_labels}")
            # print(f"Patient predictions: {p_predictions}")
            # print(f"Patient probabilities: {p_probabilities}")
            # print(f"Confusion matrix:\n{patient_cm}")
            
            # Additional debugging for label assignment
            # print(f"\nDetailed patient analysis:")
            # print(f"Total patients: {len(per_patient_predictions.keys())}")
            # print(f"Patient IDs: {list(per_patient_predictions.keys())}")
            
            # Summary statistics
            # total_bad_patients = sum(1 for lab in p_labels if lab == 0)
            # total_good_patients = sum(1 for lab in p_labels if lab == 1)
            # print(f"Total bad patients: {total_bad_patients}")
            # print(f"Total good patients: {total_good_patients}")
            
            # for i, patient_id in enumerate(per_patient_predictions.keys()):
            #     patient_preds = per_patient_predictions[patient_id]
            #     patient_labs = per_patient_labels[patient_id]
            #     patient_probs = per_patient_probabilities[patient_id]
            #     print(f"Patient {patient_id}:")
            #     print(f"  Individual predictions: {patient_preds}")
            #     print(f"  Individual labels: {patient_labs}")
            #     print(f"  Individual probabilities: {[prob[1] for prob in patient_probs]}")
            #     print(f"  Patient-level prediction: {p_predictions[i]}")
            #     print(f"  Patient-level label: {p_labels[i]}")
            #     print(f"  Patient-level probability: {p_probabilities[i]:.4f}")
            #     print(f"  Patient accuracy: {respective_patient_accuracy[i]}")
            #     print()
            
            p_fpr, p_tpr, _ = roc_curve(p_labels, p_probabilities)
            p_roc_auc = auc(p_fpr, p_tpr)
            
            # Patient-level Precision-Recall curve
            p_precision, p_recall, _ = precision_recall_curve(p_labels, p_probabilities)
            p_avg_precision = average_precision_score(p_labels, p_probabilities)
            
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'per_class_accuracy': per_class_accuracy,
                'per_class_counts': per_class_counts,
                'classification_report': report,
                'confusion_matrix': cm,
                'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
                'pr_curve': {'precision': precision, 'recall': recall, 'avg_precision': avg_precision},
                'predictions': all_predictions,
                'labels': all_labels,
                'probabilities': all_probabilities,
                'total_patient_accuracy': total_patient_accuracy,
                # Patient-level results
                'patient_predictions': p_predictions,
                'patient_labels': p_labels,
                'patient_confusion_matrix': patient_cm,
                'patient_sensitivity': patient_sensitivity,
                'patient_specificity': patient_specificity,
                'patient_classification_report': patient_report,
                'patient_roc_curve': {'fpr': p_fpr, 'tpr': p_tpr, 'auc': p_roc_auc},
                'patient_pr_curve': {'precision': p_precision, 'recall': p_recall, 'avg_precision': p_avg_precision},
                'patient_probabilities': p_probabilities,
                'patient_files': all_file_paths,
                'unique_patients': list(per_patient_predictions.keys()),
            }
        else: 
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'per_class_accuracy': per_class_accuracy,
                'per_class_counts': per_class_counts,
                'classification_report': report,
                'confusion_matrix': cm,
                'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
                'pr_curve': {'precision': precision, 'recall': recall, 'avg_precision': avg_precision},
                'predictions': all_predictions,
                'labels': all_labels,
                'probabilities': all_probabilities,
            }
    
    def evaluate_triplet(self, test_ds, patient_result=False):
        """Evaluate triplet loss model performance."""
        print("Evaluating triplet loss model performance...")
        
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        
        all_embeddings = []
        all_labels = []
        all_file_paths = []
        
        per_patient_embeddings = dict()
        per_patient_labels = dict()
        
        self.model_engine.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting embeddings"):
                if len(batch) == 3:  # anchor, positive, negative
                    anchor, positive, negative = batch
                    anchor = anchor.to(self.model_engine.device).float()
                    
                    embeddings = self.model_engine.model.model.forward_encoder_nomasking(anchor)
                    all_embeddings.extend(embeddings.cpu().numpy())
                    all_labels.extend([1] * len(embeddings))  # Assuming anchor samples are positive
                else:  # single samples
                    mel, labels, file_path = batch
                    mel = mel.to(self.model_engine.device).float()
                    
                    embeddings = self.model_engine.model.model.forward_encoder_nomasking(mel)
                    all_embeddings.extend(embeddings.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_file_paths.extend(file_path)
                    
                    if patient_result:
                        for f, emb, l in zip(file_path, embeddings.cpu().numpy(), labels.cpu().numpy()):
                            patient_id = Path(f).stem.split('_')[1]
                            if patient_id not in per_patient_embeddings:
                                per_patient_embeddings[patient_id] = [emb]
                                per_patient_labels[patient_id] = [l]
                            else:
                                if per_patient_labels[patient_id][-1] != l:
                                    print(f"patient_id: {patient_id} \n embeddings shape: {embeddings.shape} \n labels: {labels.cpu().numpy()}")
                                    assert False, "labels are not the same for patient"
                                
                                per_patient_embeddings[patient_id].append(emb)
                                per_patient_labels[patient_id].append(l)
        
        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)
        
        # Basic embedding statistics
        embedding_stats = {
            'mean_norm': np.mean(np.linalg.norm(all_embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(all_embeddings, axis=1)),
            'mean_embedding': np.mean(all_embeddings, axis=0),
            'std_embedding': np.std(all_embeddings, axis=0)
        }
        
        result = {
            'embeddings': all_embeddings,
            'labels': all_labels,
            'embedding_stats': embedding_stats
        }
        
        # Add patient-level analysis if requested
        if patient_result and len(all_file_paths) > 0:
            p_embeddings = []
            p_labels = []
            
            for patient_id in per_patient_embeddings.keys():
                patient_embeddings = per_patient_embeddings[patient_id]
                patient_labels = per_patient_labels[patient_id]
                
                if_all_labels_zero = np.all(np.array(patient_labels) == 0)
                if_all_labels_one = np.all(np.array(patient_labels) == 1)
                
                assert if_all_labels_zero or if_all_labels_one, f"patient_id: {patient_id} {if_all_labels_zero} {if_all_labels_one} \n patient_labels should be either all 0 or all 1 {patient_labels}"
                
                # Average embeddings for this patient
                avg_patient_emb = np.mean(patient_embeddings, axis=0)
                p_embeddings.append(avg_patient_emb)
                p_labels.append(patient_labels[0])
            
            p_embeddings = np.array(p_embeddings)
            p_labels = np.array(p_labels)
            
            # Patient-level embedding statistics
            patient_level_stats = {
                'mean_norm': np.mean(np.linalg.norm(p_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(p_embeddings, axis=1)),
                'mean_embedding': np.mean(p_embeddings, axis=0),
                'std_embedding': np.std(p_embeddings, axis=0)
            }
            
            # Calculate patient-level statistics
            patient_embedding_stats = {}
            for patient_id in per_patient_embeddings.keys():
                patient_emb = per_patient_embeddings[patient_id]
                patient_embedding_stats[patient_id] = {
                    'mean_norm': np.mean(np.linalg.norm(patient_emb, axis=1)),
                    'std_norm': np.std(np.linalg.norm(patient_emb, axis=1)),
                    'mean_embedding': np.mean(patient_emb, axis=0),
                    'std_embedding': np.std(patient_emb, axis=0),
                    'num_samples': len(patient_emb)
                }
            
            result.update({
                'patient_embeddings': per_patient_embeddings,
                'patient_labels': per_patient_labels,
                'patient_embedding_stats': patient_embedding_stats,
                'avg_patient_embeddings': p_embeddings,
                'avg_patient_labels': p_labels,
                'patient_level_stats': patient_level_stats,
                'unique_patients': list(per_patient_embeddings.keys()),
                'patient_files': all_file_paths,
            })
            
            # Debug information
            # print(f"\nTriplet Patient-level debugging info:")
            # print(f"Number of patients: {len(per_patient_embeddings.keys())}")
            # print(f"Patient labels: {p_labels}")
            # print(f"Bad patients: {np.sum(p_labels == 0)}")
            # print(f"Good patients: {np.sum(p_labels == 1)}")
        
        return result
    
    def evaluate_autoencoder(self, test_ds):
        """Evaluate autoencoder performance."""
        print("Evaluating autoencoder performance...")
        
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
        
        all_losses = []
        
        self.model_engine.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating reconstruction"):
                mel = batch[0] if isinstance(batch, (list, tuple)) else batch
                mel = mel.to(self.model_engine.device).float()  # Ensure float32 precision
                
                loss, _, _ = self.model_engine.model.model(mel, mask_ratio=0.8)
                all_losses.append(loss.item())
        
        return {
            'mean_loss': np.mean(all_losses),
            'std_loss': np.std(all_losses),
            'min_loss': np.min(all_losses),
            'max_loss': np.max(all_losses),
            'all_losses': all_losses
        }
    
    def run_evaluation(self, patient_result=False, threshold=0.55):
        """Run complete evaluation."""
        print(f"Starting evaluation for {self.mode} mode...")
        
        # Load test data
        test_ds = self.load_test_data()
        print(f"Loaded {len(test_ds)} test samples")
        
        # Run evaluation based on mode
        if self.mode == 'classifier':
            self.results = self.evaluate_classifier(test_ds, patient_result, threshold)
        elif self.mode == 'tripletloss':
            self.results = self.evaluate_triplet(test_ds, patient_result)
        elif self.mode == 'auto':
            self.results = self.evaluate_autoencoder(test_ds)
        
        return self.results
    
    def generate_report(self, output_dir='evaluation_results', threshold=0.55):
        """Generate comprehensive evaluation report."""
        if self.results is None:
            print("No evaluation results available. Run evaluation first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, f'{self.mode}_evaluation_results.json')
        
        # Convert all numpy types to JSON-serializable
        # cm = results['confusion_matrix']
        json_results = convert_numpy_types(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Generate visualizations
        if self.mode == 'classifier':
            self._plot_classifier_results(output_dir, threshold)
            # If patient results are available, also plot patient-level results
            if 'patient_confusion_matrix' in self.results:
                self._plot_patient_results(output_dir)
        elif self.mode == 'tripletloss':
            self._plot_triplet_results(output_dir)
        elif self.mode == 'auto':
            self._plot_autoencoder_results(output_dir)
    
    def _plot_classifier_results(self, output_dir, threshold=0.55):
        """Generate plots for classifier results."""
        results = self.results
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
                                        target_names=['Bad', 'Good'])))
            
            f.write(f"\nROC AUC: {results['roc_curve']['auc']:.4f}\n")
            f.write(f"Average Precision: {results['pr_curve']['avg_precision']:.4f}\n")
        
        print(f"Classifier plots saved to {output_dir}")
    
    def _plot_triplet_results(self, output_dir):
        """Generate plots for triplet loss results."""
        results = self.results
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
        
        # If patient-level results are available, also plot patient-level results
        if 'avg_patient_embeddings' in results:
            self._plot_triplet_patient_results(output_dir)
        
        print(f"Triplet plots saved to {output_dir}")
    
    def _plot_triplet_patient_results(self, output_dir):
        """Generate plots for triplet loss patient-level results."""
        results = self.results
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
    
    def _plot_autoencoder_results(self, output_dir):
        """Generate plots for autoencoder results."""
        results = self.results
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
        
        # Save loss statistics
        stats_file = os.path.join(output_dir, 'loss_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Reconstruction Loss Statistics:\n")
            f.write(f"Mean loss: {results['mean_loss']:.4f}\n")
            f.write(f"Std loss: {results['std_loss']:.4f}\n")
            f.write(f"Min loss: {results['min_loss']:.4f}\n")
            f.write(f"Max loss: {results['max_loss']:.4f}\n")
            f.write(f"Number of samples: {len(results['all_losses'])}\n")
        
        print(f"Autoencoder plots saved to {output_dir}")
    
    def _plot_patient_results(self, output_dir):
        """Generate plots for patient-level results."""
        results = self.results
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
            f.write(str(classification_report(results['patient_labels'], results['patient_predictions'], target_names=['Bad', 'Good'])))
            
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

def main():
    """Main function for running inference and evaluation."""
    parser = argparse.ArgumentParser(description="Audio Model Inference and Evaluation")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['auto', 'classifier', 'tripletloss'],
                       help='Model mode')
    parser.add_argument('--config', type=str, default="default_config.yaml", help='Path to YAML config file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.55,
                       help='Threshold for patient-level prediction (default: 0.55)')
    parser.add_argument('--data_path', type=str, required=True,
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
    parser.add_argument('--run_knn', action='store_true', help='Run KNN evaluation')  #. added
    parser.add_argument('--K', type=int, default=5, help='Number of nearest neighbors for KNN')  #. added    
    args = parser.parse_args()
    
    config = load_config(args.config, args)

    # paths = get_output_paths(config.get('prefix', ''), config.get('base_path', '.'))
    model_params = config.get('model', {})
    
    # Initialize inference engine
    print("Loading model...")
    engine = AudioInferenceEngine(
        checkpoint_path=args.checkpoint_path,
        mode=args.mode,
        model_params=model_params,
        device=args.device
    )
    
    if args.inference_only:
        # Run inference on single file or directory
        if os.path.isfile(args.data_path):
            # Single file inference
            result = engine.predict_single(args.data_path)
            if result:
                print(f"Prediction for {args.data_path}:")
                print(json.dumps(convert_numpy_types(result), indent=2))
        else:
            # Directory inference
            audio_files = []
            for ext in ['*.pt']:
                audio_files.extend(Path(args.data_path).rglob(ext))
            
            if audio_files:
                results = engine.predict_batch([str(f) for f in audio_files], args.batch_size)
                
                # Save results
                os.makedirs(args.output_dir, exist_ok=True)
                results_file = os.path.join(args.output_dir, 'inference_results.json')
                
                with open(results_file, 'w') as f:
                    json.dump(convert_numpy_types(results), f, indent=2)
                
                print(f"Inference results saved to {results_file}")
            else:
                print(f"No .pt files found in {args.data_path}")
    else:
        # Run full evaluation
        # Run full evaluation
        print("Running evaluation...")
        evaluator = AudioEvaluator(engine, args.data_path, args.mode)

        # If requested, run KNN before the standard evaluation
        if args.run_knn:
            # build a classifier‐style reference set
            train_ds = AudioDataset(
                mode='classifier',
                data_dir_good=[os.path.join(args.train_data_path, 'good')],
                data_dir_bad =[os.path.join(args.train_data_path, 'bad')],
                apply_specaugment=False,
                shuffle=False
            )
            test_ds = evaluator.load_test_data()
            knn_res = evaluator.run_knn(train_ds, test_ds, K=args.K)
            print(f"KNN evaluation result: {knn_res}")


        results = evaluator.run_evaluation(args.patient_result, args.threshold)
        
        # Generate detailed report and plots
        print(f"\nGenerating detailed evaluation report...")
        evaluator.generate_report(args.output_dir, args.threshold)
        
        # Print summary
        if results is not None and args.mode == 'classifier':
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
                                          target_names=['Bad', 'Good']))
                
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
        elif results is not None and args.mode == 'tripletloss':
            print(f"\nTriplet Loss Evaluation Summary:")
            print(f"Mean embedding norm: {results['embedding_stats']['mean_norm']:.4f}")
            print(f"Std embedding norm: {results['embedding_stats']['std_norm']:.4f}")
            print(f"Number of embeddings: {len(results['embeddings'])}")
            
            if args.patient_result and results is not None and 'avg_patient_embeddings' in results:
                print(f"\n=== TRIPLET LOSS PATIENT-LEVEL EVALUATION RESULTS ===")
                print(f"Number of patients: {len(results['unique_patients'])}")
                print(f"Patient-level mean embedding norm: {results['patient_level_stats']['mean_norm']:.4f}")
                print(f"Patient-level std embedding norm: {results['patient_level_stats']['std_norm']:.4f}")
                print(f"Number of bad patients: {np.sum(results['avg_patient_labels'] == 0)}")
                print(f"Number of good patients: {np.sum(results['avg_patient_labels'] == 1)}")
                
                # Calculate separation between bad and good patients
                bad_patient_norms = np.linalg.norm(results['avg_patient_embeddings'][results['avg_patient_labels'] == 0], axis=1)
                good_patient_norms = np.linalg.norm(results['avg_patient_embeddings'][results['avg_patient_labels'] == 1], axis=1)
                
                if len(bad_patient_norms) > 0 and len(good_patient_norms) > 0:
                    print(f"Bad patients - mean norm: {np.mean(bad_patient_norms):.4f}, std: {np.std(bad_patient_norms):.4f}")
                    print(f"Good patients - mean norm: {np.mean(good_patient_norms):.4f}, std: {np.std(good_patient_norms):.4f}")
                    
                    # Calculate separation metric
                    separation = abs(np.mean(good_patient_norms) - np.mean(bad_patient_norms)) / (np.std(good_patient_norms) + np.std(bad_patient_norms))
                    print(f"Separation between patient groups: {separation:.4f}")
    

        elif results is not None and args.mode == 'auto':
            print(f"\nAutoencoder Evaluation Summary:")
            print(f"Mean reconstruction loss: {results['mean_loss']:.4f}")
            print(f"Std reconstruction loss: {results['std_loss']:.4f}")
    

if __name__ == "__main__":
    main()
