"""
Hierarchical Consistency Loss
==============================

Based on HiPART (CVPR 2025) multi-scale token structure:
- Enforce consistency between hierarchical levels
- Parent predictions should match aggregated child predictions
- Helps learn structural priors and handle occlusion

For SDPose:
- Coarse (6 parts) ← Mid (11 groups) ← Fine (17 keypoints)
- Bi-directional consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class HierarchicalConsistencyLoss(nn.Module):
    """Hierarchical Consistency for multi-level tokens/predictions.
    
    Enforces parent-child aggregation consistency:
    - Fine → Mid aggregation should match Mid prediction
    - Mid → Coarse aggregation should match Coarse prediction
    
    Args:
        agg_mode (str): Aggregation method. 'mean' or 'weighted'. Default: 'mean'
        loss_weight (float): Overall weight. Default: 1e-5
        use_l1 (bool): Use L1 loss (True) or L2/MSE (False). Default: True
        apply_to_heatmap (bool): Apply to heatmap predictions. Default: True
        apply_to_tokens (bool): Apply to token embeddings. Default: True
    """
    
    def __init__(self,
                 agg_mode='mean',
                 loss_weight=1e-5,
                 use_l1=True,
                 apply_to_heatmap=True,
                 apply_to_tokens=True):
        super().__init__()
        self.agg_mode = agg_mode
        self.loss_weight = loss_weight
        self.use_l1 = use_l1
        self.apply_to_heatmap = apply_to_heatmap
        self.apply_to_tokens = apply_to_tokens
        
        # Load aggregation matrices (will be set later)
        self.agg_f2m = None  # Fine → Mid [NUM_MID, NUM_FINE]
        self.agg_m2c = None  # Mid → Coarse [NUM_COARSE, NUM_MID]
        
        print(f"[HierarchicalConsistencyLoss] Initialized:")
        print(f"  agg_mode={agg_mode}, L1={use_l1}")
        print(f"  apply_to: heatmap={apply_to_heatmap}, tokens={apply_to_tokens}")
    
    def set_aggregation_matrices(self, agg_f2m, agg_m2c):
        """
        Set aggregation matrices.
        
        Args:
            agg_f2m: [NUM_MID, NUM_FINE] - Fine to Mid aggregation
            agg_m2c: [NUM_COARSE, NUM_MID] - Mid to Coarse aggregation
        """
        self.agg_f2m = agg_f2m
        self.agg_m2c = agg_m2c
    
    def aggregate_predictions(self, fine_pred, agg_matrix):
        """
        Aggregate predictions using aggregation matrix.
        
        Args:
            fine_pred: [B, K_fine, ...] - Fine-level predictions
            agg_matrix: [K_coarse, K_fine] - Aggregation weights
        
        Returns:
            coarse_pred: [B, K_coarse, ...] - Aggregated predictions
        """
        if agg_matrix is None:
            raise ValueError("Aggregation matrix not set!")
        
        # Move agg_matrix to same device
        agg_matrix = agg_matrix.to(fine_pred.device)
        
        if fine_pred.dim() == 4:  # Heatmap [B, K, H, W]
            B, K_fine, H, W = fine_pred.shape
            K_coarse = agg_matrix.shape[0]
            
            # Reshape for matrix multiplication
            fine_flat = fine_pred.view(B, K_fine, H * W)  # [B, K_fine, H*W]
            
            # Aggregate: [K_coarse, K_fine] @ [B, K_fine, H*W]
            # Use einsum for cleaner implementation
            coarse_flat = torch.einsum('ck,bkp->bcp', agg_matrix, fine_flat)  # [B, K_coarse, H*W]
            
            coarse_pred = coarse_flat.view(B, K_coarse, H, W)  # [B, K_coarse, H, W]
            
        elif fine_pred.dim() == 3:  # Tokens [B, K, D]
            # Aggregate tokens: [K_coarse, K_fine] @ [B, K_fine, D]
            coarse_pred = torch.einsum('ck,bkd->bcd', agg_matrix, fine_pred)  # [B, K_coarse, D]
            
        else:
            raise ValueError(f"Unsupported dimension: {fine_pred.dim()}")
        
        return coarse_pred
    
    def consistency_loss(self, pred_parent, pred_child_aggregated):
        """
        Compute consistency loss between parent and aggregated children.
        
        Args:
            pred_parent: [B, K_parent, ...] - Parent predictions
            pred_child_aggregated: [B, K_parent, ...] - Aggregated child predictions
        
        Returns:
            loss: Scalar
        """
        if self.use_l1:
            diff = torch.abs(pred_parent - pred_child_aggregated)
        else:
            diff = (pred_parent - pred_child_aggregated) ** 2
        
        return diff.mean()
    
    def forward(self, 
                fine_heatmap=None, mid_heatmap=None, coarse_heatmap=None,
                fine_tokens=None, mid_tokens=None, coarse_tokens=None,
                target_weight=None):
        """
        Hierarchical consistency loss.
        
        Args:
            fine_heatmap: [B, 17, H, W] - Fine predictions
            mid_heatmap: [B, 11, H, W] - Mid predictions
            coarse_heatmap: [B, 6, H, W] - Coarse predictions
            fine_tokens: [B, 17, D] - Fine tokens
            mid_tokens: [B, 11, D] - Mid tokens
            coarse_tokens: [B, 6, D] - Coarse tokens
            target_weight: [B, 17, 1] - Visibility (only for fine)
        
        Returns:
            loss: Scalar
        """
        total_loss = 0.0
        device = fine_heatmap.device if fine_heatmap is not None else fine_tokens.device
        
        # (1) Heatmap consistency
        if self.apply_to_heatmap and fine_heatmap is not None:
            # Fine → Mid
            if mid_heatmap is not None and self.agg_f2m is not None:
                fine_to_mid_agg = self.aggregate_predictions(fine_heatmap, self.agg_f2m)  # [B, 11, H, W]
                loss_f2m = self.consistency_loss(mid_heatmap, fine_to_mid_agg)
                total_loss += loss_f2m
            
            # Mid → Coarse
            if mid_heatmap is not None and coarse_heatmap is not None and self.agg_m2c is not None:
                mid_to_coarse_agg = self.aggregate_predictions(mid_heatmap, self.agg_m2c)  # [B, 6, H, W]
                loss_m2c = self.consistency_loss(coarse_heatmap, mid_to_coarse_agg)
                total_loss += loss_m2c
        
        # (2) Token consistency
        if self.apply_to_tokens and fine_tokens is not None:
            # Fine → Mid
            if mid_tokens is not None and self.agg_f2m is not None:
                fine_to_mid_agg = self.aggregate_predictions(fine_tokens, self.agg_f2m)  # [B, 11, D]
                loss_f2m_tok = self.consistency_loss(mid_tokens, fine_to_mid_agg)
                total_loss += loss_f2m_tok
            
            # Mid → Coarse
            if mid_tokens is not None and coarse_tokens is not None and self.agg_m2c is not None:
                mid_to_coarse_agg = self.aggregate_predictions(mid_tokens, self.agg_m2c)  # [B, 6, D]
                loss_m2c_tok = self.consistency_loss(coarse_tokens, mid_to_coarse_agg)
                total_loss += loss_m2c_tok
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss * self.loss_weight


# Test
if __name__ == '__main__':
    print("Testing HierarchicalConsistencyLoss...")
    
    # Import config
    import sys
    sys.path.insert(0, '.')
    from hierarchical_token_config import (
        get_aggregation_matrix_fine_to_mid,
        get_aggregation_matrix_mid_to_coarse
    )
    
    B, H, W, D = 4, 64, 48, 192
    
    # Dummy data
    fine_hm = torch.randn(B, 17, H, W)
    mid_hm = torch.randn(B, 11, H, W)
    coarse_hm = torch.randn(B, 6, H, W)
    
    fine_tok = torch.randn(B, 17, D)
    mid_tok = torch.randn(B, 11, D)
    coarse_tok = torch.randn(B, 6, D)
    
    # Create loss
    loss_fn = HierarchicalConsistencyLoss(loss_weight=1e-5)
    
    # Set aggregation matrices
    agg_f2m = get_aggregation_matrix_fine_to_mid()
    agg_m2c = get_aggregation_matrix_mid_to_coarse()
    loss_fn.set_aggregation_matrices(agg_f2m, agg_m2c)
    
    # Test
    loss = loss_fn(
        fine_heatmap=fine_hm, mid_heatmap=mid_hm, coarse_heatmap=coarse_hm,
        fine_tokens=fine_tok, mid_tokens=mid_tok, coarse_tokens=coarse_tok
    )
    
    print(f"Hierarchical Consistency Loss: {loss.item():.6f}")
    print("✅ HierarchicalConsistencyLoss test passed!")

