import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.append('Audio_mae')

from lightning_combined import AudioDataset
from torch.utils.data import DataLoader

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
            
            # Extract patient IDs from current file paths
            original_patients = set()
            for file_path in all_file_paths:
                try:
                    filename = Path(file_path).stem
                    if '_' in filename:
                        patient_id = filename.split('_')[1]
                    else:
                        patient_id = next(
                            (part for part in Path(file_path).parts
                            if part.startswith('R2D2') or (len(part) > 8 and part.isalnum())),
                            filename
                        )
                    original_patients.add(patient_id)
                except Exception:
                    patient_id = Path(file_path).stem
                    original_patients.add(patient_id)

            
            for patient_id in per_patient_predictions.keys():
                patient_predictions = per_patient_predictions[patient_id]
                patient_labels = per_patient_labels[patient_id]
                
                if_all_labels_zero = np.all(np.array(patient_labels) == 0)
                if_all_labels_one = np.all(np.array(patient_labels) == 1)
                
                assert if_all_labels_zero or if_all_labels_one, f"patient_id: {patient_id} {if_all_labels_zero} {if_all_labels_one} \n patient_labels should be either all 0 or all 1 {patient_labels}"
                    
                patient_probabilities = per_patient_probabilities[patient_id]
                
                # NOTE: This uses mean of binary predictions (0s and 1s) for patient prediction
                # but mean of probabilities for ROC curve. This might cause inconsistency.
                # Alternative: use mean of probabilities for both prediction and ROC curve
                # Alternative approach (uncomment to use):
                # mean_positive_prob = np.mean([prob[1] for prob in patient_probabilities])
                # pred = 1 if mean_positive_prob > threshold else 0
                # p_probabilities.append(mean_positive_prob)
                
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
                mean_positive_prob = np.mean(positive_probs)
                p_probabilities.append(mean_positive_prob)
                p_labels.append(patient_labels[0])
                respective_patient_accuracy.append(pred == patient_labels[0])
                
                # Debug: Print patient-level info for first few patients
                if len(p_predictions) <= 3:
                    print(f"Patient {patient_id}: label={patient_labels[0]}, pred={pred}, mean_pred={np.mean(patient_predictions):.3f}, mean_prob={mean_positive_prob:.3f}")



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
            output_dict=True,
            zero_division=0
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
                output_dict=True,
                zero_division=0
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
    
    def evaluate_autoencoder(self, test_ds, train_ds, val_ds=None, generate_ignored_recon = False):
        """Evaluate autoencoder performance."""
        print("Evaluating autoencoder performance...")
        
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        all_losses = []
        all_energies = []
        all_file_paths = []
        self.model_engine.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating test reconstruction"):
                mel, batch_paths = batch
                mel = mel.to(self.model_engine.device).float()  # Ensure float32 precision
                
                # Get model output for entire batch (efficient)
                loss, pred, mask = self.model_engine.model.model(mel, mask_ratio=0.0)
                
                # Calculate individual losses and energies from batch output
                batch_losses = []
                batch_energies = []
                for i in range(mel.size(0)):  # For each sample in the batch
                    # Calculate MSE loss for each sample individually
                    # Ensure same shape: pred[i] is [256, 256], mel[i] is [1, 256, 256]
                    sample_loss = F.mse_loss(pred[i], mel[i].squeeze(0), reduction='mean')
                    batch_losses.append(sample_loss.item())
                    
                    # Calculate signal energy (mean square energy) for each sample
                    # mel[i] is [1, 256, 256], we calculate energy across all dimensions
                    sample_energy = torch.mean(mel[i] ** 2).item()
                    batch_energies.append(sample_energy)
                
                all_losses.extend(batch_losses)  # Add all individual losses
                all_energies.extend(batch_energies)  # Add all individual energies
                all_file_paths.extend(batch_paths)
                
        
        test_mean_loss = np.mean(all_losses)
        test_std_loss = np.std(all_losses)
        test_mean_energy = np.mean(all_energies)
        test_std_energy = np.std(all_energies)
        
        result = {
            'mean_loss': np.mean(all_losses),
            'std_loss': np.std(all_losses),
            'min_loss': np.min(all_losses),
            'max_loss': np.max(all_losses),
            'all_losses': all_losses,
            'mean_energy': np.mean(all_energies),
            'std_energy': np.std(all_energies),
            'min_energy': np.min(all_energies),
            'max_energy': np.max(all_energies),
            'all_energies': all_energies
        }
        if generate_ignored_recon:
            result.update({
                'test_mean_recon_loss':test_mean_loss, 
                'test_std_recon_loss':test_std_loss, 
                'test_corresponding_recon_loss': all_losses,
                'test_corresponding_path': all_file_paths,
                'test_mean_energy': test_mean_energy,
                'test_std_energy': test_std_energy,
                'test_corresponding_energy': all_energies})
    
                
        if generate_ignored_recon: 
            print (f"Generating ignored reconstruction for training data")
            all_train_losses = []
            all_train_energies = []
            corresponding_path = []
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)
            
            assert train_ds.mode == 'auto', "train_ds mode should be auto"
            
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(train_loader, desc="Evaluating training reconstruction")):
                    # print(f"[DEBUG] batch: {batch}")
                    
                    # mel = batch[0] if isinstance(batch, (list, tuple)) else batch
                    mel, batch_paths = batch
                    mel = mel.to(self.model_engine.device).float()  # Ensure float32 precision
                    
                    # Get model output for entire batch (efficient)
                    loss, pred, mask = self.model_engine.model.model(mel, mask_ratio=0.0)
                    
                    # Calculate individual losses and energies from batch output
                    batch_losses = []
                    batch_energies = []
                    for i in range(mel.size(0)):  # For each sample in the batch
                        # Calculate MSE loss for each sample individually
                        # Ensure same shape: pred[i] is [256, 256], mel[i] is [1, 256, 256]
                        sample_loss = F.mse_loss(pred[i], mel[i].squeeze(0), reduction='mean')
                        batch_losses.append(sample_loss.item())
                        
                        # Calculate signal energy (mean square energy) for each sample
                        sample_energy = torch.mean(mel[i] ** 2).item()
                        batch_energies.append(sample_energy)
                    
                    all_train_losses.extend(batch_losses)  # Add all individual losses 
                    all_train_energies.extend(batch_energies)  # Add all individual energies
                    corresponding_path.extend(batch_paths)
                    
            all_train_losses = np.array(all_train_losses)
            all_train_energies = np.array(all_train_energies)
            # corresponding_path is now a list of file path strings, no need for complex processing
            corresponding_path = np.array(corresponding_path, dtype=object)
        
            train_mean_loss = np.mean(all_train_losses)
            train_std_loss = np.std(all_train_losses)
            train_mean_energy = np.mean(all_train_energies)
            train_std_energy = np.std(all_train_energies)
            
            result.update({ 
                           'train_mean_recon_loss':train_mean_loss, 
                           'train_std_recon_loss':train_std_loss, 
                           'train_corresponding_recon_loss': all_train_losses.tolist(),
                           'train_corresponding_path': corresponding_path.tolist(),
                           'train_mean_energy': train_mean_energy,
                           'train_std_energy': train_std_energy,
                           'train_corresponding_energy': all_train_energies.tolist()})

        # Process validation data if available
        if generate_ignored_recon and val_ds is not None:
            print (f"Generating ignored reconstruction for validation data")
            all_val_losses = []
            all_val_energies = []
            val_corresponding_path = []
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
            
            assert val_ds.mode == 'auto', "val_ds mode should be auto"
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating validation reconstruction")):
                    mel, batch_paths = batch
                    mel = mel.to(self.model_engine.device).float()  # Ensure float32 precision
                    
                    # Get model output for entire batch (efficient)
                    loss, pred, mask = self.model_engine.model.model(mel, mask_ratio=0.0)
                    
                    # Calculate individual losses and energies from batch output
                    batch_losses = []
                    batch_energies = []
                    for i in range(mel.size(0)):  # For each sample in the batch
                        # Calculate MSE loss for each sample individually
                        # Ensure same shape: pred[i] is [256, 256], mel[i] is [1, 256, 256]
                        sample_loss = F.mse_loss(pred[i], mel[i].squeeze(0), reduction='mean')
                        batch_losses.append(sample_loss.item())
                        
                        # Calculate signal energy (mean square energy) for each sample
                        sample_energy = torch.mean(mel[i] ** 2).item()
                        batch_energies.append(sample_energy)
                    
                    all_val_losses.extend(batch_losses)  # Add all individual losses 
                    all_val_energies.extend(batch_energies)  # Add all individual energies
                    val_corresponding_path.extend(batch_paths)
                    
            all_val_losses = np.array(all_val_losses)
            all_val_energies = np.array(all_val_energies)
            val_corresponding_path = np.array(val_corresponding_path, dtype=object)
        
            val_mean_loss = np.mean(all_val_losses)
            val_std_loss = np.std(all_val_losses)
            val_mean_energy = np.mean(all_val_energies)
            val_std_energy = np.std(all_val_energies)
            
            result.update({ 
                           'val_mean_recon_loss':val_mean_loss, 
                           'val_std_recon_loss':val_std_loss, 
                           'val_corresponding_recon_loss': all_val_losses.tolist(),
                           'val_corresponding_path': val_corresponding_path.tolist(),
                           'val_mean_energy': val_mean_energy,
                           'val_std_energy': val_std_energy,
                           'val_corresponding_energy': all_val_energies.tolist()})
        
        return result 
    
    
    def extract_patient_predictions(
    self,
    file_path, predictions, labels, probabilities,
    per_patient_predictions, per_patient_labels, per_patient_probabilities):
        skipped_low_confidence = 0
        for f, p, l, pr in zip(file_path, predictions, labels, probabilities):
            # 1) only keep confident predictions
            conf = pr[p]
            if conf < 0.2:
                skipped_low_confidence += 1
                continue  # skip low confidence prediction

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
        
        if skipped_low_confidence > 0:
            print(f"Warning: Skipped {skipped_low_confidence} predictions due to low confidence (< 0.2)")

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
    
    def run_evaluation(self, patient_result=False, train_dataset = None, val_dataset = None, threshold=0.55, run_mj = False, K=5, generate_ignored_recon = False):
        """Run complete evaluation."""
        print(f"Starting evaluation for {self.mode} mode...")
        
        # Load test data
        test_ds = self.load_test_data()
        print(f"Loaded {len(test_ds)} test samples")
        
        print(f"Loaded {len(test_ds)} test samples")
        
        # Run evaluation based on mode
        if self.mode == 'classifier':
            self.results = self.evaluate_classifier(test_ds, patient_result, threshold)
        elif self.mode == 'tripletloss':
            self.results = self.evaluate_triplet(test_ds, patient_result, train_ds=train_dataset, threshold=threshold, run_mj=run_mj, k=K)
        elif self.mode == 'auto':
            self.results = self.evaluate_autoencoder(test_ds, train_ds = train_dataset, val_ds = val_dataset, generate_ignored_recon = generate_ignored_recon)
        
        return self.results
    

