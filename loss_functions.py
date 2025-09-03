import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedLoss(nn.Module):
    def __init__(self, mode="who", sens_weight=3.0, spec_weight=1.0,
                 sens_thresh=0.9, spec_thresh=0.7, alpha = 0.25, gamma = 2.0, fn_weight=4.0, fp_weight=1.0):
       
        super().__init__()
        self.mode = mode
        self.sens_weight = sens_weight
        self.spec_weight = spec_weight
        self.sens_thresh = sens_thresh
        self.spec_thresh = spec_thresh
        self.alpha = alpha 
        self.gamma = gamma 
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight 
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        if self.mode == "who":
            return self._who_compliant_loss(y_true, y_pred)
        elif self.mode == "focal": 
            return self._focal_loss_tb(y_true, y_pred)
        elif self.mode == "asymmetric": 
            return self.asymmetric_loss(y_true, y_pred)
        elif self.mode == "auc": 
            return self.auc_maximizing_loss(y_true, y_pred)
        elif self.mode == "bce":
            return F.binary_cross_entropy(y_pred, y_true)
        else:
            raise ValueError(f"Unsupported loss mode: {self.mode}")

    def _who_compliant_loss(self, y_true, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y_true)
        y_pred_binary = (y_pred > 0.5).float()

        tp = torch.sum(y_true * y_pred_binary)
        fn = torch.sum(y_true * (1 - y_pred_binary))
        tn = torch.sum((1 - y_true) * (1 - y_pred_binary))
        fp = torch.sum((1 - y_true) * y_pred_binary)

        sensitivity = tp / (tp + fn + self.eps)
        specificity = tn / (tn + fp + self.eps)

        sens_penalty = torch.relu(self.sens_thresh - sensitivity) * self.sens_weight
        spec_penalty = torch.relu(self.spec_thresh - specificity) * self.spec_weight

        return bce_loss + sens_penalty + spec_penalty

    def _focal_loss_tb(self, y_true, y_pred):
        """Focal Loss adapted for TB detection"""
        ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Extra weight for TB cases to boost sensitivity
        tb_weight = torch.where(y_true == 1, 2.0, 1.0)
        return torch.mean(focal_loss * tb_weight)
    
    def asymmetric_loss(self, y_true, y_pred):
        """Heavily penalize false negatives (missed TB cases)"""
        pos_loss = y_true * torch.pow(1 - y_pred, 2) * torch.log(y_pred + 1e-8)
        neg_loss = (1 - y_true) * torch.pow(y_pred, 2) * torch.log(1 - y_pred + 1e-8)
        
        weighted_loss = -(self.fn_weight * pos_loss + self.fp_weight * neg_loss)
        return torch.mean(weighted_loss)
    
    def auc_maximizing_loss(self, y_true, y_pred):
        """Directly optimize ROC-AUC using pairwise ranking"""
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        pos_scores = y_pred[pos_mask].unsqueeze(1)
        neg_scores = y_pred[neg_mask].unsqueeze(0)
        
        # All positive vs negative pairs
        score_diff = pos_scores - neg_scores
        ranking_loss = torch.sigmoid(-score_diff)
        
        return torch.mean(ranking_loss)
