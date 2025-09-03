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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


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
    
    
    def predict_ifiles(self, audio_paths, size=32):
        """
        Perform batch inference on multiple audio files.
        
        Args:
            audio_paths (list): List of audio file paths
            batch_size (int): Batch size for inference
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(audio_paths), size), desc="Processing batches"):
            batch_paths = audio_paths[i:i + size]
            batch_results = []
            
            for path in batch_paths:
                try:
                    mel = self._preprocess(path)
                    
                    with torch.no_grad():
                        if self.mode in 'classifier':
                            # Get logits and probabilities
                            logits = self.model.model.forward_encoder_nomasking_classification(mel)
                            probabilities = F.softmax(logits, dim=1)
                            predicted_class = torch.argmax(logits, dim=1).item()
                            confidence = torch.max(probabilities, dim=1)[0].item()
                            
                            result = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'probabilities': probabilities.cpu().numpy()[0],
                                'logits': logits.cpu().numpy()[0], 
                                'file_path': path
                            }
                                                
                        elif self.mode == 'tripletloss': 
                            emb = self.model.model.forward_encoder_nomasking(mel).cpu().numpy()[0]
                            norm = float(np.linalg.norm(emb))
                            
                            result = {
                                'embedding': emb,
                                'file_path': path, 
                                'embedding_norm': norm, 
                            
                            } 
                        else: 
                            loss, pred, mask = self.model.model(mel, mask_ratio=0.8)
                            result = {
                                'file_path':path, 
                                'reconstruction_loss': loss.item(),
                                'prediction': pred.cpu().numpy(),
                                'mask': mask.cpu().numpy()
                            }
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
                
                batch_results.append(result)
            results.extend(batch_results)
            
        return results

    def _preprocess(self, path): 
        mel = torch.load(path)[:,1:]
        return F.pad(mel,(0,0,0,59)).unsqueeze(0).float().to(self.device)

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
        if self.mode in ('classifier' ,'tripletloss'):
            # Load classifier test data
            # Input data, labels, file_path
            # classify and evaluate on patient level
            # ROC curve on patient level and collect multiple statistics
            good_dir = os.path.join(self.test_data_path, 'good')  #.
            bad_dir = os.path.join(self.test_data_path, 'bad')    #.
            
            test_ds = AudioDataset(                                #.
                mode='classifier',
                data_dir_good=[good_dir],
                data_dir_bad=[bad_dir],
                apply_specaugment=False,
                extend_bad=False,
                shuffle=False
            )

        else:
            test_ds = AudioDataset(
                mode='auto',
                data_dir_good=[os.path.join(self.test_data_path, 'good')]
            )
        return test_ds
    
    def evaluate_kMeansCluster(self,
                           test_ds,
                           X_ref,           # shape: (N_ref, D)
                           Y_ref,           # shape: (N_ref,)
                           n_clusters=10,
                           threshold=0.5,
                           patient_result=False,
                           run_mj=False,    # toggle KNN majority-rule
                           K=5):
        """K-Means clustering + optional KNN majority-rule baseline."""
        print("Evaluating triplet loss performance using kMeansCluster...")

        # Prepare
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        self.model_engine.model.eval()

        # Fit k-means on reference embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_ref)
        centroids = kmeans.cluster_centers_      # (n_clusters, D)
        ref_labels = kmeans.labels_

        # Map each cluster → class by majority vote over Y_ref
        cluster2class = {}
        clusterProb  = {}
        for c in range(n_clusters):
            idxs = np.where(ref_labels == c)[0]
            p = Y_ref[idxs].mean()
            cluster2class[c] = int(p > threshold)
            clusterProb[c]    = p

        # Accumulators
        all_preds, all_labels, all_probs = [], [], []
        mj_preds, mj_truth = [], []
        all_embs, all_paths = [], []

        per_patient_preds = {} 
        per_patient_labels = {}  
        per_patient_probs  = {}  
        
        with torch.no_grad():
            for mels, labels, paths in tqdm(test_loader, desc="Evaluating kMeans"):
                # 1) get embeddings
                embs = self.model_engine.model.model.forward_encoder_nomasking(
                            mels.to(self.model_engine.device).float())
                embs_np = embs.cpu().numpy()
                labels   = labels.cpu().tolist()

                # store
                all_embs.extend(embs_np.tolist())
                all_labels.extend(labels)
                all_paths.extend(paths)

                # 2) k-means assignment
                dists = np.linalg.norm(
                            centroids[None, :, :] - embs_np[:, None, :],
                            axis=2
                        )               # shape (batch, n_clusters)
                cluster_ids = np.argmin(dists, axis=1)
                preds = [cluster2class[c] for c in cluster_ids]
                probs = [[1 - clusterProb[c], clusterProb[c]] for c in cluster_ids]

                all_preds.extend(preds)
                all_probs.extend(probs)

                # 3) optional KNN majority-rule (inline, no external fn)
                if run_mj:
                    # pairwise to all X_ref
                    d_knn = np.linalg.norm(
                                X_ref[None, :, :] - embs_np[:, None, :],
                                axis=2
                            )               # shape (batch, N_ref)
                    # take K nearest per sample
                    idxs = np.argsort(d_knn, axis=1)[:, :K]     # (batch, K)
                    lbls = Y_ref[idxs]                           # (batch, K)
                    mj_batch = (lbls.sum(axis=1) > (K/2)).astype(int).tolist()
                    mj_preds.extend(mj_batch)
                    mj_truth.extend(labels)
                
                if patient_result:
                    self.extract_patient_predictions(
                        paths, preds, labels, probs,
                        per_patient_preds,
                        per_patient_labels,
                        per_patient_probs
                    )


        # 4) Compute final metrics
        results = {}
        if run_mj:
            mj_acc = np.mean(np.array(mj_preds) == np.array(mj_truth))
            results['KNN_accuracy_majority_rule'] = mj_acc

        km_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        results['Kmeans_cluster_accuracy'] = km_acc

        emb_array = np.vstack(all_embs)
        
        # (You can still compute embedding_stats, pos/neg distances, etc.)
        results['embeddings'] = emb_array
        results['embedding_stats'] = {
            'mean_norm': np.mean(np.linalg.norm(emb_array, axis=1)),
            'std_norm':  np.std( np.linalg.norm(emb_array, axis=1) )
        }
        classifier_metrics = self.compute_classifier_statistics(
                all_preds,
                all_labels,
                all_probs,
                per_patient_preds,
                per_patient_labels,
                per_patient_probs,
                all_paths,
                patient_result,
                threshold
            )
        
        results.update(classifier_metrics)
        
        return results


    def _build_reference_set(self, train_ds, max_ref_samples=None):
        #EDITED triplet mode to classifer mode 
        """Build reference set from training data for KNN"""
        print("Building reference set from training data...")
        
        X_ref, Y_ref = [], []
        loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        self.model_engine.model.eval()
        with torch.no_grad():
            for (mels, labels, _) in loader:
                # preprocess & forward through encoder
                
                embs = self.model_engine.model.model.forward_encoder_nomasking(mels.to(self.model_engine.device).float())
                X_ref.extend(embs.cpu().numpy())
                Y_ref.extend(labels.cpu().numpy())
                if max_ref_samples and len(X_ref) >= max_ref_samples:
                    break

        X_ref = np.stack(X_ref)[:max_ref_samples]
        Y_ref = np.array(Y_ref)[:max_ref_samples]
        return X_ref, Y_ref

    
    def evaluate_reference_thresholds(self, train_ds, max_ref_samples=None):
        # TODO: may be in step 2. If we want to use classifier mode, we get good embedding and bad embeddings
        # And shuffle good embedding to get anchor embeddings
        ref_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)
        all_labels = []
        all_embs = []
        
        with torch.no_grad():
            for batch in tqdm(ref_loader, desc="Building Average Thresholds"):
                mels, labels, _ = batch
                Embs = self.model_engine.model.model.forward_encoder_nomasking(mels.to(self.model_engine.device).float())
                all_labels.extend(labels.cpu().numpy())
                all_embs.extend(Embs.cpu().numpy() )
                
                if max_ref_samples and len(all_embs) >= max_ref_samples:
                    break

            
            all_embs = np.vstack(all_embs) #(N,D) 
            all_labels = np.array(all_labels) #(N,)
            
            emb_p = all_embs[all_labels == 1]
            emb_n = all_embs[all_labels == 0]
            emb_a = emb_p.copy()
            np.random.shuffle(emb_a)
            
            emb_a_truncated = emb_a[:len(emb_n)]
                
            d_pos = np.linalg.norm(emb_a - emb_p, axis=1)
            d_neg = np.linalg.norm(emb_a_truncated - emb_n, axis=1)
            
            Mean_Pos_th = d_pos.mean()
            Std_Pos_th = d_pos.std()
            Mean_Neg_th = d_neg.mean()
            Std_Neg_th = d_neg.std()
            
        return Mean_Pos_th, Std_Pos_th, Mean_Neg_th, Std_Neg_th, all_embs, all_labels 
        
    def compute_classifier_statistics(self, all_predictions, all_labels, all_probabilities,
                                     per_patient_predictions, per_patient_labels, per_patient_probabilities,
                                     all_file_paths, patient_result, threshold):
        """
        Computes and aggregates statistics for classifier evaluation.
        """
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # print(f"all_file_paths: {all_file_paths}")
        
        if patient_result:
            respective_patient_accuracy = []

            p_labels = []
            p_predictions = []
            p_probabilities = []

            
            for patient_id in per_patient_predictions.keys():
                patient_predictions = per_patient_predictions[patient_id]
                patient_labels = per_patient_labels[patient_id]
                
                if_all_labels_zero = np.all(np.array(patient_labels) == 0)
                if_all_labels_one = np.all(np.array(patient_labels) == 1)
                
                assert if_all_labels_zero or if_all_labels_one, f"patient_id: {patient_id} {if_all_labels_zero} {if_all_labels_one} \n patient_labels should be either all 0 or all 1 {patient_labels}"
                    
                patient_probabilities = per_patient_probabilities[patient_id]
                
                # patient_accuracy = 1 if np.mean(patient_predictions == patient_labels) > threshold else 0
                    
                if np.mean(patient_predictions) > threshold:
                    pred = 1
                else:
                    pred = 0
                p_predictions.append(pred)
                
                # Average only the positive class probability (index 1) for ROC curve
                # Validate probability structure
                if len(patient_probabilities[0]) != 2:
                    print(f"Warning: Patient {patient_id} has unexpected probability structure: {patient_probabilities[0]}")
                
                positive_probs = [prob[1] for prob in patient_probabilities]
                p_probabilities.append(np.mean(positive_probs))
                p_labels.append(patient_labels[0])
                respective_patient_accuracy.append(pred == patient_labels[0])

                    

            total_patient_accuracy = np.mean(respective_patient_accuracy)
            # print(f"total_patient_accuracy: {total_patient_accuracy}, {respective_patient_accuracy} {p_labels == p_predictions}")
            
        accuracy = np.mean(all_predictions == all_labels)
        if patient_result:
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
        
        result = {
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

        if patient_result:
            # idxs = []
            # for idx in p_predictions: 
            #     if idx >= 0.65 or idx <= 0.45: 
            #         idxs.append(idx)
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
           
            p_probabilities = np.array(p_probabilities)
                          
            p_fpr, p_tpr, _ = roc_curve(p_labels, p_probabilities)
            p_roc_auc = auc(p_fpr, p_tpr)
            
            # Patient-level Precision-Recall curve
            p_precision, p_recall, _ = precision_recall_curve(p_labels, p_probabilities)
            p_avg_precision = average_precision_score(p_labels, p_probabilities)
            
            result.update({
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
            })
        return result 
    
    def evaluate_triplet(self, test_ds, patient_result=False, train_ds=None, threshold=0.5, run_mj=False, k = 5):
        """Evaluate triplet loss model performance."""
        print("Evaluating triplet loss model performance...")
        
        # test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
                
        self.model_engine.model.eval()
        
        Pdist_m, Pdist_std, Ndist_m, Ndist_std, X_ref, Y_ref = self.evaluate_reference_thresholds(train_ds=train_ds)
        result = {
            'positive_distance_mean':Pdist_m, 
            'positive_distance_std':Pdist_std, 
            'negative_distance_mean':Ndist_m,
            'negative_distance_std':Ndist_std 
        }
        
        # # TODO: Run above for test_ds and return test embeddings
        
        # with torch.no_grad():
        #     for batch in tqdm(test_loader, desc="Extracting positive and negative embeddings"):
        #         if len(batch) == 3:  # anchor, positive, negative
        #             mels, labels, _ = batch
        #             embs = self.model_engine.model.model.forward_encoder_nomasking(mels.to(self.model_engine.device).float())
        #             embs = embs.cpu().numpy() 
        #             labels = labels.cpu().numpy()


        #             if False: 
        #                 d_pos = (embeddings_a - embeddings_p).norm(dim=1)
        #                 d_neg = (embeddings_a - embeddings_n).norm(dim=1)    
                    
        
        # # Calculate embedding statistics
        # embedding_stats = {
        #     'mean_norm': np.mean(np.linalg.norm(embs, axis=1)),
        #     'std_norm': np.std(np.linalg.norm(embs, axis=1)),
        #     'mean_embedding': np.mean(embs, axis=0),
        #     'std_embedding': np.std(embs, axis=0)
        # }
        
        # result.update({
        #     'embedding_stats': embedding_stats,
        #     'embeddings':embs,
        #     'labels':labels,
        #     'file_paths': len(_)
        #     })
        
        # X_ref, Y_ref = self._build_reference_set(train_ds)
        # VA: Changed
        kmeans_res = self.evaluate_kMeansCluster(test_ds, X_ref, Y_ref, n_clusters=10, threshold=threshold, patient_result=patient_result, run_mj=run_mj, K=k)
        result.update(kmeans_res)
                
        return result
    
    def evaluate_autoencoder(self, test_ds):
        """Evaluate autoencoder performance."""
        print("Evaluating autoencoder performance...")
        
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        
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
    
    def extract_patient_predictions(
    self,
    file_path, predictions, labels, probabilities,
    per_patient_predictions, per_patient_labels, per_patient_probabilities):
        """Extracts patient-level predictions, but only if confidence ∉ [0.3, 0.7]."""
        for f, p, l, pr in zip(file_path, predictions, labels, probabilities):
            # 1) only keep confident predictions
            conf = pr[p]  # confidence of the predicted class

            # 2) derive patient_id as before
            try:
                filename = Path(f).stem
                if '_' in filename:
                    patient_id = filename.split('_')[1]
                else:
                    patient_id = next(
                        (part for part in Path(f).parts
                        if part.startswith('R2D2') or (len(part) > 8 and part.isalnum())),
                        filename
                    )
            except Exception:
                patient_id = Path(f).stem

            # 3) append into your per-patient dicts
            if patient_id not in per_patient_predictions:
                per_patient_predictions[patient_id]   = [p]
                per_patient_labels[patient_id]         = [l]
                per_patient_probabilities[patient_id] = [pr]
            else:
                # sanity check that labels stay consistent:
                assert per_patient_labels[patient_id][-1] == l, \
                    f"Mismatched label for {patient_id}: {l} vs {per_patient_labels[patient_id][-1]}"
                per_patient_predictions[patient_id].append(p)
                per_patient_labels[patient_id].append(l)
                per_patient_probabilities[patient_id].append(pr)


    def evaluate_classifier(self, test_ds, patient_result=False, threshold=0.55):
        """Evaluate classifier performance."""
        print("Evaluating classifier performance...")
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
                        self.extract_patient_predictions(
                            file_path, predictions, labels, probabilities,
                            per_patient_predictions, per_patient_labels, per_patient_probabilities
                        )

        return self.compute_classifier_statistics(
            all_predictions, all_labels, all_probabilities,
            per_patient_predictions, per_patient_labels, per_patient_probabilities,
            all_file_paths, patient_result, threshold
        )
    
    def run_evaluation(self, patient_result=False, train_dataset = None, threshold=0.55, run_mj = False, K=5):
        """Run complete evaluation."""
        print(f"Starting evaluation for {self.mode} mode...")
        
        # Load test data
        test_ds = self.load_test_data()
        print(f"Loaded {len(test_ds)} test samples")
        
        # Run evaluation based on mode
        if self.mode == 'classifier':
            self.results = self.evaluate_classifier(test_ds, patient_result, threshold)
        elif self.mode == 'tripletloss':
            self.results = self.evaluate_triplet(test_ds, patient_result, train_ds=train_dataset, threshold=threshold, run_mj=run_mj, k=K)
        elif self.mode == 'auto':
            self.results = self.evaluate_autoencoder(test_ds)
        
        return self.results
    
    def generate_report(self, output_dir='evaluation_results',file_prefix = "default", threshold=0.55):
        """Generate comprehensive evaluation report."""
        if self.results is None:
            print("No evaluation results available. Run evaluation first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, f'{self.mode}_{file_prefix}_evaluation_results.json')
        
        # Convert all numpy types to JSON-serializable
        # cm = results['confusion_matrix']
        
        filtered = dict(self.results)
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
        if self.mode == 'classifier':
            self._plot_classifier_results(output_dir, threshold)
            # If patient results are available, also plot patient-level results
            if 'patient_confusion_matrix' in self.results:
                self._plot_patient_results(output_dir)
        elif self.mode == 'tripletloss':
            self._plot_triplet_results(output_dir)
            self._plot_classifier_results(output_dir, threshold)
            # If patient results are available, also plot patient-level results
            if 'patient_confusion_matrix' in self.results:
                self._plot_patient_results(output_dir)
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
        
        # t-SNE visualization
        print(f"t-SNE debug: embeddings shape: {results['embeddings'].shape}")
        print(f"t-SNE debug: embeddings sample: {results['embeddings'][:2]}")
        print(f"t-SNE debug: embeddings min/max: {np.min(results['embeddings']):.6f}, {np.max(results['embeddings']):.6f}")
        print(f"t-SNE debug: embeddings has NaN: {np.isnan(results['embeddings']).any()}")
        print(f"t-SNE debug: number of embeddings: {len(results['embeddings'])}")
        
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
                import traceback
                print(f"Warning: Could not create t-SNE visualization: {e}")
                traceback.print_exc()
        
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
        
        # Patient-level t-SNE visualization
        print(f"Patient t-SNE debug: embeddings shape: {results['avg_patient_embeddings'].shape}")
        print(f"Patient t-SNE debug: embeddings sample: {results['avg_patient_embeddings'][:2]}")
        print(f"Patient t-SNE debug: embeddings min/max: {np.min(results['avg_patient_embeddings']):.6f}, {np.max(results['avg_patient_embeddings']):.6f}")
        print(f"Patient t-SNE debug: embeddings has NaN: {np.isnan(results['avg_patient_embeddings']).any()}")
        print(f"Patient t-SNE debug: number of embeddings: {len(results['avg_patient_embeddings'])}")
        
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
                import traceback
                print(f"Warning: Could not create patient-level t-SNE visualization: {e}")
                traceback.print_exc()
        
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
    args = parser.parse_args()

    config = load_config(args.config, args)

    # paths = get_output_paths(config.get('prefix', ''), config.get('base_path', '.'))
    model_params = config.get('model', {})
    
    if args.evaluate_using_dir: 
        current_max_val = 0
        current_max_epoch = None
        for ckpt in os.listdir(os.path.join(args.evaluate_using_dir, 'checkpoints')): 
            txt = ckpt.split('acc=')
            val_acc = (txt[2]).split('.ckpt')
            if float(val_acc[0]) >= current_max_val: 
                current_max_val = float(val_acc[0])
                current_max_epoch = os.path.join(args.evaluate_using_dir, 'checkpoints',ckpt)
        yaml_file = os.path.join(args.evaluate_using_dir, 'logs', 'config_used.yaml')
        with open(yaml_file, 'r') as file: 
            data = yaml.safe_load(file)
        
        base_dir = None
        base_path = None
        batch_size = None 
        mode_type = None 
        train_data_path = None
        for key, value in data.items(): 
            base_dir = os.path.join(value,'test') if key == 'base_dir' else base_dir
            train_data_path = os.path.join(value,'train') if key == 'base_dir' else train_data_path
            base_path = value if key == 'base_path' else base_path
            batch_size = value if key == 'batch_size' else batch_size
            mode_type = value if key == 'mode_type' else mode_type
        checkpoint = current_max_epoch
        assert base_dir is not None, "config_used.yaml doesn't contain a data path (base_dir)"
        assert mode_type is not None, "config_used.yaml doesn't contain a mode type (mode_type)"
    else: 
        try: 
            checkpoint=args.checkpoint_path
            mode_type = args.mode
            base_dir = args.data_path  
            train_data_path = None
            if args.mode == 'tripletloss': 
                if args.train_data_path is None:
                    raise ValueError('need train data path for triplet mode')
                train_data_path = args.train_data_path
        except Exception as e: 
            print('must use --checkpoint_path , --mode, --data_path')
            raise e
    
    # Initialize inference engine
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
        else:
            results = evaluator.run_evaluation(
                patient_result=args.patient_result,
                train_dataset=None,
                threshold=args.threshold
            )      
              
        # Generate detailed report and plots
        print(f"\nGenerating detailed evaluation report...")
        evaluator.generate_report(args.output_dir, "default", args.threshold)
        
        # Print summary
        # FIXME: Print this 
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
        
        # CHANGED: print this for tripletloss mode anyway                
        if results is not None and mode_type == 'tripletloss':
            print(f"\nTriplet Loss Evaluation Summary:")
            print(f"Mean embedding norm: {results['embedding_stats']['mean_norm']:.4f}")
            print(f"Std embedding norm: {results['embedding_stats']['std_norm']:.4f}")
                
            print("=============Stats that matter=============")
            print(f"Positive mean distances from anchor {results['positive_distance_mean']}")
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
    

if __name__ == "__main__":
    main()
