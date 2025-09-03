import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append('Audio_mae')

from lightning_combined import UnifiedLightningModel

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
            class_weight=[4.0, 1.0],  # Default class weights for inference
            init_margin=0.2 if mode == 'tripletloss' else None  # Only use init_margin for tripletloss mode
        )
        
        # Load state dict with strict=False to handle missing keys
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
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
                        if self.mode == 'classifier':
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
