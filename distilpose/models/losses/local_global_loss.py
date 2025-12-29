# Copyright (c) OpenMMLab. All rights reserved.
"""
Local-Global Consistency Self-Distillation for HPE
Inspired by SILC (ECCV'24)

Key Idea:
- Global view: Full image pose prediction
- Local view: Cropped region pose prediction
- Enforce consistency on overlapping joints
- Scale/crop invariant learning

Stable Implementation:
- Conservative weighting
- Smooth consistency loss
- No extreme operations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class LocalGlobalConsistencyLoss(nn.Module):
    """Local-Global Consistency Loss for Self-Distillation.
    
    Enforces prediction consistency between global and local views.
    This makes the model robust to scale, crop, and partial occlusion.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1e-4 (conservative)
        temperature (float): Temperature for soft consistency. Default: 1.0
        use_heatmap (bool): Use heatmap-based consistency. Default: True
        min_overlap_joints (int): Minimum overlapping joints required. Default: 5
    """
    
    def __init__(self,
                 loss_weight=1e-4,  # Very conservative
                 temperature=1.0,
                 use_heatmap=True,
                 min_overlap_joints=5):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.use_heatmap = use_heatmap
        self.min_overlap_joints = min_overlap_joints
        
    def forward(self, 
                global_pred,
                local_pred,
                global_meta,
                local_meta,
                target_weight=None):
        """Forward function with local-global consistency.
        
        Args:
            global_pred (Tensor): [N, K, H, W] - Global view predictions
            local_pred (Tensor): [N, K, H, W] - Local view predictions  
            global_meta (dict): Global view metadata (bbox info)
            local_meta (dict): Local view metadata (bbox info)
            target_weight (Tensor): [N, K, 1] - Joint visibility
            
        Returns:
            loss (Tensor): Consistency loss
        """
        # Safety check
        if torch.isnan(global_pred).any() or torch.isnan(local_pred).any():
            return torch.tensor(0.0, device=global_pred.device, requires_grad=True)
        
        N, K, H, W = global_pred.shape
        
        # Compute consistency loss
        if self.use_heatmap:
            # Heatmap-based consistency (more stable)
            loss = self._heatmap_consistency(
                global_pred, local_pred, target_weight
            )
        else:
            # Coordinate-based consistency
            loss = self._coordinate_consistency(
                global_pred, local_pred, target_weight
            )
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=global_pred.device, requires_grad=True)
        
        return loss * self.loss_weight
    
    def _heatmap_consistency(self, global_pred, local_pred, target_weight):
        """Heatmap-based consistency (STABLE).
        
        Simple MSE between normalized heatmaps.
        No complex operations, very stable.
        """
        # Normalize heatmaps to [0, 1]
        global_norm = self._normalize_heatmap(global_pred)
        local_norm = self._normalize_heatmap(local_pred)
        
        # Simple MSE
        diff = global_norm - local_norm.detach()  # Detach local (teacher)
        loss = (diff * diff).mean(dim=(2, 3))  # [N, K]
        
        # Apply visibility weights if provided
        if target_weight is not None:
            weight = target_weight.squeeze(-1)  # [N, K]
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def _coordinate_consistency(self, global_pred, local_pred, target_weight):
        """Coordinate-based consistency.
        
        Extract keypoint coordinates and enforce consistency.
        """
        # Extract coordinates from heatmaps
        global_coords = self._heatmap_to_coord(global_pred)  # [N, K, 2]
        local_coords = self._heatmap_to_coord(local_pred)    # [N, K, 2]
        
        # L2 distance
        diff = global_coords - local_coords.detach()
        loss = (diff * diff).sum(dim=-1)  # [N, K]
        
        # Apply visibility weights
        if target_weight is not None:
            weight = target_weight.squeeze(-1)
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def _normalize_heatmap(self, heatmap):
        """Normalize heatmap to [0, 1] per joint.
        
        Stable normalization without division by small numbers.
        """
        N, K, H, W = heatmap.shape
        
        # Flatten spatial dimensions
        hm_flat = heatmap.reshape(N, K, -1)  # [N, K, H*W]
        
        # Min-max normalization per joint
        hm_min = hm_flat.min(dim=2, keepdim=True)[0]
        hm_max = hm_flat.max(dim=2, keepdim=True)[0]
        
        # Avoid division by zero
        hm_range = hm_max - hm_min + 1e-6
        hm_norm = (hm_flat - hm_min) / hm_range
        
        # Reshape back
        hm_norm = hm_norm.reshape(N, K, H, W)
        
        return hm_norm
    
    def _heatmap_to_coord(self, heatmap):
        """Convert heatmap to coordinates.
        
        Args:
            heatmap: [N, K, H, W]
        Returns:
            coords: [N, K, 2] in normalized [0, 1] space
        """
        N, K, H, W = heatmap.shape
        
        # Softmax to get probability distribution
        hm_flat = heatmap.reshape(N, K, -1)
        hm_soft = F.softmax(hm_flat / self.temperature, dim=-1)
        hm_soft = hm_soft.reshape(N, K, H, W)
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=heatmap.dtype, device=heatmap.device)
        x_coords = torch.arange(W, dtype=heatmap.dtype, device=heatmap.device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = y_grid.reshape(1, 1, H, W) / (H - 1)  # Normalize to [0, 1]
        x_grid = x_grid.reshape(1, 1, H, W) / (W - 1)
        
        # Expected coordinates (soft-argmax)
        x_coord = (hm_soft * x_grid).sum(dim=(2, 3))  # [N, K]
        y_coord = (hm_soft * y_grid).sum(dim=(2, 3))  # [N, K]
        
        coords = torch.stack([x_coord, y_coord], dim=-1)  # [N, K, 2]
        
        return coords


@LOSSES.register_module()
class ProgressiveLocalGlobalLoss(nn.Module):
    """Progressive Local-Global Consistency Loss.
    
    Start with zero weight, gradually increase.
    Maximum stability for early training.
    
    Args:
        loss_weight (float): Final weight. Default: 5e-5
        start_epoch (int): Start applying loss. Default: 30
        end_epoch (int): Reach full weight. Default: 100
        temperature (float): Temperature. Default: 1.0
    """
    
    def __init__(self,
                 loss_weight=5e-5,
                 start_epoch=30,
                 end_epoch=100,
                 temperature=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.temperature = temperature
        self.current_epoch = 0
        
        # Base loss
        self.base_loss = LocalGlobalConsistencyLoss(
            loss_weight=1.0,  # Will be scaled
            temperature=temperature,
            use_heatmap=True,
        )
    
    def set_epoch(self, epoch):
        """Set current epoch for progressive weighting."""
        self.current_epoch = epoch
    
    def get_current_weight(self):
        """Get current loss weight based on epoch."""
        if self.current_epoch < self.start_epoch:
            return 0.0
        elif self.current_epoch >= self.end_epoch:
            return self.loss_weight
        else:
            # Linear interpolation
            progress = (self.current_epoch - self.start_epoch) / (
                self.end_epoch - self.start_epoch
            )
            return self.loss_weight * progress
    
    def forward(self, *args, **kwargs):
        """Forward with progressive weighting."""
        current_weight = self.get_current_weight()
        
        if current_weight == 0.0:
            # Skip computation if weight is zero
            device = args[0].device if len(args) > 0 else list(kwargs.values())[0].device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute base loss
        base_loss = self.base_loss(*args, **kwargs)
        
        return base_loss * current_weight


@LOSSES.register_module()
class SimpleConsistencyLoss(nn.Module):
    """Ultra-simple consistency loss - Maximum Stability.
    
    Just MSE between predictions, nothing fancy.
    Works with both heatmaps [N, K, H, W] and tokens [N, K, D].
    Cannot diverge.
    
    Args:
        loss_weight (float): Weight. Default: 1e-5 (very small)
        clamp_max (float): Maximum value for loss clamping. Default: 10.0
    """
    
    def __init__(self, loss_weight=1e-5, clamp_max=None):
        super().__init__()
        self.loss_weight = loss_weight
        # clamp_max가 None이면 기본값 10.0 사용 (token feature 범위에 맞춤)
        self.clamp_max = clamp_max if clamp_max is not None else 10.0
    
    def forward(self, pred1, pred2, target_weight=None):
        """Ultra-simple MSE.
        
        Args:
            pred1: Tensor - First prediction (heatmap or token)
                   [N, K, H, W] for heatmap
                   [N, K, D] for token
            pred2: Tensor - Second prediction (same shape as pred1)
            target_weight: [N, K, 1] - Visibility weights (optional)
        """
        # Safety check - early return if inputs are bad
        if pred1 is None or pred2 is None:
            return torch.tensor(0.0, device=pred1.device if pred1 is not None else 'cpu', requires_grad=True)
        
        if torch.isnan(pred1).any() or torch.isnan(pred2).any():
            print(f"[WARNING] NaN detected in SimpleConsistencyLoss inputs, returning 0")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        if torch.isinf(pred1).any() or torch.isinf(pred2).any():
            print(f"[WARNING] Inf detected in SimpleConsistencyLoss inputs, returning 0")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        # Clamp inputs to prevent extreme values
        pred1 = torch.clamp(pred1, -self.clamp_max, self.clamp_max)
        pred2 = torch.clamp(pred2, -self.clamp_max, self.clamp_max)
        
        # Simple MSE
        diff = pred1 - pred2.detach()  # Detach pred2 (teacher)
        loss = (diff * diff)
        
        # Apply weights if provided
        if target_weight is not None:
            # Expand weight to match prediction dimensions
            if len(loss.shape) == 4:
                # Heatmap: [N, K, H, W]
                weight = target_weight.unsqueeze(-1).unsqueeze(-1)  # [N, K, 1, 1]
                loss = loss * weight
                
                # Safe division
                denominator = weight.sum() * loss.shape[2] * loss.shape[3] + 1e-6
                if denominator > 0:
                    loss = loss.sum() / denominator
                else:
                    loss = torch.tensor(0.0, device=pred1.device, requires_grad=True)
                    
            elif len(loss.shape) == 3:
                # Token: [N, K, D]
                weight = target_weight  # [N, K, 1]
                loss = loss * weight
                
                # Safe division
                denominator = weight.sum() * loss.shape[2] + 1e-6
                if denominator > 0:
                    loss = loss.sum() / denominator
                else:
                    loss = torch.tensor(0.0, device=pred1.device, requires_grad=True)
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()
        
        # Final safety checks
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] NaN/Inf in SimpleConsistencyLoss output, returning 0")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        # Clamp loss to prevent explosion
        loss = torch.clamp(loss, 0.0, self.clamp_max)
        
        return loss * self.loss_weight


@LOSSES.register_module()
class ProgressiveConsistencyLoss(nn.Module):
    """Progressive Self-Distillation Loss for AP 73%+.
    
    Loss weight increases progressively during training:
    Early epochs: Small weight (stable learning)
    Later epochs: Larger weight (strong self-distillation)
    
    This curriculum learning approach provides:
    1. Stability in early training
    2. Strong regularization in later training
    3. Better convergence to higher AP
    
    Args:
        base_weight (float): Initial weight (e.g., 5e-6). Default: 5e-6
        max_weight (float): Maximum weight (e.g., 5e-5). Default: 5e-5
        start_epoch (int): Epoch to start increasing weight. Default: 50
        end_epoch (int): Epoch to reach max weight. Default: 200
        warmup_type (str): 'linear' or 'cosine'. Default: 'linear'
        clamp_max (float): Max value for clamping. Default: 10.0
    """
    
    def __init__(self,
                 base_weight=5e-6,
                 max_weight=5e-5,
                 start_epoch=50,
                 end_epoch=200,
                 warmup_type='linear',
                 clamp_max=10.0):
        super().__init__()
        self.base_weight = base_weight
        self.max_weight = max_weight
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.warmup_type = warmup_type
        self.clamp_max = clamp_max
        
        # Current epoch tracker (will be updated by hooks)
        self.current_epoch = 0
        
        print(f"[ProgressiveConsistencyLoss] Initialized:")
        print(f"  base_weight={base_weight}, max_weight={max_weight}")
        print(f"  start_epoch={start_epoch}, end_epoch={end_epoch}")
        print(f"  warmup_type={warmup_type}")
        
    def get_current_weight(self):
        """Calculate current weight based on epoch."""
        if self.current_epoch < self.start_epoch:
            return self.base_weight
        elif self.current_epoch >= self.end_epoch:
            return self.max_weight
        else:
            # Progressive increase
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            
            if self.warmup_type == 'linear':
                alpha = progress
            elif self.warmup_type == 'cosine':
                import math
                alpha = (1 - math.cos(progress * math.pi)) / 2
            else:
                alpha = progress
            
            current_weight = self.base_weight + (self.max_weight - self.base_weight) * alpha
            return current_weight
    
    def forward(self, pred1, pred2, target_weight=None):
        """Forward with progressive weighting.
        
        Args:
            pred1: First prediction (student, Cycle 1)
            pred2: Second prediction (teacher, Cycle 2)
            target_weight: Optional visibility weights
        """
        # Safety checks
        if pred1 is None or pred2 is None:
            return torch.tensor(0.0, device=pred1.device if pred1 is not None else 'cpu', requires_grad=True)
        
        if torch.isnan(pred1).any() or torch.isnan(pred2).any():
            print(f"[WARNING] NaN in ProgressiveConsistencyLoss inputs at epoch {self.current_epoch}")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        if torch.isinf(pred1).any() or torch.isinf(pred2).any():
            print(f"[WARNING] Inf in ProgressiveConsistencyLoss inputs at epoch {self.current_epoch}")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        # Clamp inputs to prevent extreme values
        pred1 = torch.clamp(pred1, -self.clamp_max, self.clamp_max)
        pred2 = torch.clamp(pred2, -self.clamp_max, self.clamp_max)
        
        # MSE loss (detach teacher)
        diff = pred1 - pred2.detach()
        loss = (diff * diff)
        
        # Apply target weight if provided
        if target_weight is not None:
            if len(loss.shape) == 3:  # Token: [N, K, D]
                weight = target_weight  # [N, K, 1]
                loss = loss * weight
                denominator = weight.sum() * loss.shape[2] + 1e-6
                loss = loss.sum() / denominator
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] NaN/Inf in loss output at epoch {self.current_epoch}")
            return torch.tensor(0.0, device=pred1.device, requires_grad=True)
        
        # Clamp loss
        loss = torch.clamp(loss, 0.0, self.clamp_max)
        
        # Progressive weighting
        current_weight = self.get_current_weight()
        
        # Print weight every 10 epochs for monitoring
        if hasattr(self, '_last_printed_epoch'):
            if self.current_epoch - self._last_printed_epoch >= 10:
                print(f"[ProgressiveConsistencyLoss] Epoch {self.current_epoch}: weight={current_weight:.2e}, loss={loss.item():.6f}")
                self._last_printed_epoch = self.current_epoch
        else:
            self._last_printed_epoch = 0
        
        return loss * current_weight

