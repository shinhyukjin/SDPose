# Copyright (c) OpenMMLab. All rights reserved.
"""
Foreground Self-Distillation for Human Pose Estimation
Inspired by FSD-BEV (ECCV'24)

Key Idea:
- Weight distillation loss by foreground importance
- Foreground = regions near keypoints (high heatmap values)
- Background = empty regions (low heatmap values)
- Focus self-distillation on important regions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class ForegroundTokenDistilLoss(nn.Module):
    """Foreground-weighted Token Distillation Loss.
    
    Applies spatial weighting based on heatmap activations to focus
    distillation on foreground (person/joint) regions.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        foreground_weight (float): Weight multiplier for foreground regions. Default: 2.0
        background_weight (float): Weight multiplier for background regions. Default: 0.5
        threshold (float): Heatmap threshold to determine foreground. Default: 0.1
        temperature (float): Temperature for soft weighting. Default: 1.0
        use_spatial_weight (bool): Use spatial weighting for visual tokens. Default: True
    """
    
    def __init__(self, 
                 loss_weight=1.0,
                 foreground_weight=2.0,
                 background_weight=0.5,
                 threshold=0.1,
                 temperature=1.0,
                 use_spatial_weight=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.threshold = threshold
        self.temperature = temperature
        self.use_spatial_weight = use_spatial_weight
        
    def compute_foreground_mask(self, heatmap):
        """Compute foreground mask from heatmap.
        
        Args:
            heatmap (Tensor): [N, K, H, W] - Predicted heatmaps
            
        Returns:
            mask (Tensor): [N, H, W] - Foreground importance mask
        """
        # Max pooling across keypoints to get overall person mask
        # [N, K, H, W] -> [N, H, W]
        person_mask, _ = heatmap.max(dim=1)
        
        # Soft weighting with temperature
        # Higher temperature = smoother transition
        person_mask = torch.sigmoid(person_mask / self.temperature)
        
        # Apply foreground/background weights
        # High activation -> foreground_weight
        # Low activation -> background_weight
        weight_mask = (person_mask * (self.foreground_weight - self.background_weight) + 
                      self.background_weight)
        
        return weight_mask
    
    def forward(self, token_s, token_t, heatmap=None):
        """Forward function with foreground weighting.
        
        Args:
            token_s (Tensor): [N, K, D] - Student tokens (Cycle 2)
            token_t (Tensor): [N, K, D] - Teacher tokens (Cycle 1)
            heatmap (Tensor): [N, J, H, W] - Heatmap for foreground mask (optional)
            
        Returns:
            loss (Tensor): Weighted distillation loss
        """
        # Basic MSE loss
        diff = token_s - token_t.detach()
        loss = diff * diff  # [N, K, D]
        
        # If no spatial weighting or no heatmap, return standard loss
        if not self.use_spatial_weight or heatmap is None:
            return loss.mean() * self.loss_weight
        
        N, K, D = token_s.shape
        
        # Compute foreground mask from heatmap
        # [N, H, W]
        foreground_mask = self.compute_foreground_mask(heatmap)
        H, W = foreground_mask.shape[1:]
        
        # Assume tokens are arranged as [keypoint_tokens, visual_tokens]
        # We need to know how many are visual tokens
        # For SDPose: visual_tokens = H//patch_h * W//patch_w
        # Typically patch_size = [4, 3], so visual_tokens = (H//4) * (W//3)
        
        # Infer number of visual tokens from spatial dimensions
        # Assume heatmap size = feature size (64x48 typically)
        patch_h, patch_w = 4, 3  # Default patch size for SDPose
        num_patches = (H // patch_h) * (W // patch_w)
        
        # If K matches num_patches, all are visual tokens
        # Otherwise, assume first tokens are keypoint, rest are visual
        if K == num_patches:
            # All visual tokens
            visual_loss = loss  # [N, K, D]
            
            # Downsample mask to patch resolution
            # [N, H, W] -> [N, H//patch_h, W//patch_w]
            mask_patches = F.avg_pool2d(
                foreground_mask.unsqueeze(1),
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w)
            ).squeeze(1)  # [N, H//patch_h, W//patch_w]
            
            # Reshape to match tokens: [N, num_patches]
            mask_flat = mask_patches.reshape(N, -1)  # [N, K]
            
            # Apply spatial weighting: [N, K, D] * [N, K, 1]
            weighted_loss = visual_loss * mask_flat.unsqueeze(-1)
            
            # Normalize by total weight to maintain loss scale
            total_weight = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0)  # [N, 1]
            normalized_loss = weighted_loss.sum(dim=(1, 2)) / total_weight.squeeze()  # [N]
            
            return normalized_loss.mean() * self.loss_weight
        else:
            # Mixed: keypoint + visual tokens
            # Heuristic: if K > num_patches, likely has keypoint tokens at start
            # Apply uniform weight to keypoint tokens, spatial weight to visual tokens
            
            # Simple approach: apply mean loss (no spatial weighting for now)
            # In practice, need to know exact token structure
            return loss.mean() * self.loss_weight


@LOSSES.register_module()
class AdaptiveForegroundDistilLoss(nn.Module):
    """Adaptive Foreground Distillation with learnable weighting.
    
    Instead of fixed foreground/background weights, learn to weight
    different spatial regions based on their importance for pose estimation.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        num_keypoints (int): Number of keypoints. Default: 17
        use_keypoint_guidance (bool): Use keypoint-specific weighting. Default: True
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 num_keypoints=17,
                 use_keypoint_guidance=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_keypoints = num_keypoints
        self.use_keypoint_guidance = use_keypoint_guidance
        
        # Learnable per-keypoint importance weights
        if use_keypoint_guidance:
            self.keypoint_weights = nn.Parameter(torch.ones(num_keypoints))
        
    def forward(self, token_s, token_t, heatmap=None, target_weight=None):
        """Forward with adaptive weighting.
        
        Args:
            token_s (Tensor): [N, K, D] - Student tokens
            token_t (Tensor): [N, K, D] - Teacher tokens
            heatmap (Tensor): [N, J, H, W] - Heatmap for guidance
            target_weight (Tensor): [N, J, 1] - Visibility weights
            
        Returns:
            loss (Tensor): Adaptive weighted loss
        """
        # Basic MSE
        diff = token_s - token_t.detach()
        loss = (diff * diff).mean(dim=-1)  # [N, K]
        
        # If using keypoint guidance and heatmap is provided
        if self.use_keypoint_guidance and heatmap is not None:
            N, J, H, W = heatmap.shape
            
            # Compute per-keypoint importance from heatmap activation
            # [N, J, H, W] -> [N, J]
            keypoint_activation = heatmap.flatten(2).max(dim=2)[0]  # Max activation per keypoint
            
            # Apply learnable weights: [J] * [N, J] -> [N, J]
            importance = torch.sigmoid(self.keypoint_weights) * keypoint_activation
            
            # Normalize
            importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-6)
            
            # If target_weight is provided, mask invisible keypoints
            if target_weight is not None:
                importance = importance * target_weight.squeeze(-1)
            
            # Apply to loss (assume first J tokens are keypoint tokens)
            if loss.shape[1] >= J:
                # Weight keypoint token losses (avoid in-place operation)
                weighted_kpt_loss = loss[:, :J] * importance
                # Concatenate weighted keypoint tokens with rest
                if loss.shape[1] > J:
                    loss = torch.cat([weighted_kpt_loss, loss[:, J:]], dim=1)
                else:
                    loss = weighted_kpt_loss
        
        return loss.mean() * self.loss_weight


@LOSSES.register_module()
class DynamicForegroundDistilLoss(nn.Module):
    """Dynamic Foreground Distillation with progressive weighting.
    
    Gradually increase foreground emphasis during training:
    - Early epochs: uniform weighting (learn global structure)
    - Later epochs: strong foreground emphasis (refine details)
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        start_epoch (int): Epoch to start foreground weighting. Default: 50
        end_epoch (int): Epoch to reach full foreground weight. Default: 150
        max_fg_weight (float): Maximum foreground weight. Default: 3.0
        min_bg_weight (float): Minimum background weight. Default: 0.3
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 start_epoch=50,
                 end_epoch=150,
                 max_fg_weight=3.0,
                 min_bg_weight=0.3):
        super().__init__()
        self.loss_weight = loss_weight
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.max_fg_weight = max_fg_weight
        self.min_bg_weight = min_bg_weight
        self.current_epoch = 0
        
        # Base distillation
        self.base_criterion = nn.MSELoss(reduction='none')
        
    def set_epoch(self, epoch):
        """Set current epoch for progressive weighting."""
        self.current_epoch = epoch
        
    def get_progressive_weights(self):
        """Compute current foreground/background weights."""
        if self.current_epoch < self.start_epoch:
            # Uniform weighting
            return 1.0, 1.0
        elif self.current_epoch > self.end_epoch:
            # Full foreground emphasis
            return self.max_fg_weight, self.min_bg_weight
        else:
            # Linear interpolation
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            fg_weight = 1.0 + progress * (self.max_fg_weight - 1.0)
            bg_weight = 1.0 - progress * (1.0 - self.min_bg_weight)
            return fg_weight, bg_weight
    
    def forward(self, token_s, token_t, heatmap=None):
        """Forward with progressive foreground weighting.
        
        Args:
            token_s (Tensor): Student tokens
            token_t (Tensor): Teacher tokens
            heatmap (Tensor): Heatmap for foreground mask
            
        Returns:
            loss (Tensor): Progressive weighted loss
        """
        # Get current weights
        fg_weight, bg_weight = self.get_progressive_weights()
        
        # Compute loss
        loss = self.base_criterion(token_s, token_t.detach())
        
        # If uniform or no heatmap, return simple loss
        if fg_weight == bg_weight or heatmap is None:
            return loss.mean() * self.loss_weight
        
        # Compute foreground mask
        person_mask, _ = heatmap.max(dim=1)  # [N, H, W]
        person_mask = torch.sigmoid(person_mask)
        
        # Create weight mask
        weight_mask = person_mask * (fg_weight - bg_weight) + bg_weight
        
        # Apply spatial weighting (simplified - assume tokens are spatial)
        # In practice, need proper spatial alignment
        weighted_loss = loss.mean()  # Placeholder
        
        return weighted_loss * self.loss_weight

