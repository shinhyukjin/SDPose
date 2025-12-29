# Copyright (c) OpenMMLab. All rights reserved.
"""
HD-Distill: Hard-aware Dynamic Distillation Loss

Key Idea:
- 어려운 관절/샘플에 더 높은 distillation weight 부여
- Teacher (마지막 cycle)의 정보를 어려운 부분에 더 강하게 전달
- 쉬운 관절은 덜 강하게 학습 → capacity 효율 향상

Implementation:
- Per-joint difficulty: heatmap L2 error 기반
- Weighted distillation: difficulty에 비례한 weight
- Instance-level difficulty: joint 평균으로 계산 (AC-Depth용)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class HardAwareDistillLoss(nn.Module):
    """Hard-aware Dynamic Distillation Loss.
    
    어려운 관절/샘플에 더 높은 weight를 부여하여
    teacher의 정보를 더 강하게 전달.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        diff_normalize_k (float): Normalization constant for difficulty. Default: 0.5
        use_min_weight (float): Minimum weight for easy joints. Default: 0.1
    """
    
    def __init__(self, 
                 loss_weight=1.0, 
                 diff_normalize_k=0.5,
                 use_min_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.diff_normalize_k = diff_normalize_k
        self.use_min_weight = use_min_weight
    
    def compute_joint_difficulty(self, heatmaps, target_heatmaps):
        """Compute per-joint difficulty based on L2 error.
        
        Args:
            heatmaps: [B, J, H, W] student heatmaps
            target_heatmaps: [B, J, H, W] GT heatmaps
            
        Returns:
            diff: [B, J] difficulty scores (0~1)
        """
        with torch.no_grad():
            # Per-joint L2 error
            diff = (heatmaps - target_heatmaps) ** 2  # [B, J, H, W]
            diff = diff.mean(dim=(2, 3))  # [B, J]
            # Normalize & clamp to [0, 1]
            diff = torch.clamp(diff / self.diff_normalize_k, 0.0, 1.0)
        return diff
    
    def forward(self, student_heatmaps, teacher_heatmaps, target_heatmaps):
        """Forward function.
        
        Args:
            student_heatmaps: [B, J, H, W] cycle c heatmaps
            teacher_heatmaps: [B, J, H, W] final cycle heatmaps (detached)
            target_heatmaps: [B, J, H, W] GT heatmaps
            
        Returns:
            loss: weighted distillation loss
        """
        # Safety check
        if torch.isnan(student_heatmaps).any() or torch.isnan(teacher_heatmaps).any():
            device = student_heatmaps.device
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        # Compute joint difficulty
        joint_diff = self.compute_joint_difficulty(
            student_heatmaps, target_heatmaps)  # [B, J]
        
        # Add minimum weight for easy joints (avoid zero weight)
        if self.use_min_weight > 0:
            joint_diff = torch.clamp(
                joint_diff, min=self.use_min_weight, max=1.0)
        
        # Reshape to [B, J, 1, 1] for broadcasting
        w = joint_diff.unsqueeze(-1).unsqueeze(-1)  # [B, J, 1, 1]
        
        # Weighted MSE
        diff = (student_heatmaps - teacher_heatmaps.detach()) ** 2  # [B, J, H, W]
        loss = (w * diff).mean()
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            device = student_heatmaps.device
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        return loss * self.loss_weight


@LOSSES.register_module()
class DifficultyRegressionLoss(nn.Module):
    """Difficulty Regression Loss for training difficulty head.
    
    AC-Depth에서 difficulty head를 학습하기 위한 loss.
    GT difficulty (heatmap error 기반)와 predicted difficulty 간의 MSE.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        diff_normalize_k (float): Normalization constant for GT difficulty. Default: 0.5
    """
    
    def __init__(self, 
                 loss_weight=1.0,
                 diff_normalize_k=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.diff_normalize_k = diff_normalize_k
    
    def compute_instance_difficulty(self, heatmaps, target_heatmaps):
        """Compute instance-level difficulty (average over joints).
        
        Args:
            heatmaps: [B, J, H, W] student heatmaps
            target_heatmaps: [B, J, H, W] GT heatmaps
            
        Returns:
            diff: [B, 1] instance difficulty scores (0~1)
        """
        with torch.no_grad():
            # Per-joint L2 error
            diff = (heatmaps - target_heatmaps) ** 2  # [B, J, H, W]
            diff = diff.mean(dim=(2, 3))  # [B, J]
            # Normalize & clamp
            diff = torch.clamp(diff / self.diff_normalize_k, 0.0, 1.0)
            # Average over joints -> instance level
            diff = diff.mean(dim=1, keepdim=True)  # [B, 1]
        return diff
    
    def forward(self, pred_difficulty, heatmaps, target_heatmaps):
        """Forward function.
        
        Args:
            pred_difficulty: [B, 1] predicted difficulty from difficulty head
            heatmaps: [B, J, H, W] student heatmaps
            target_heatmaps: [B, J, H, W] GT heatmaps
            
        Returns:
            loss: MSE loss between predicted and GT difficulty
        """
        # Compute GT instance difficulty
        target_diff = self.compute_instance_difficulty(
            heatmaps, target_heatmaps)  # [B, 1]
        
        # MSE loss
        loss = F.mse_loss(pred_difficulty, target_diff)
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            device = pred_difficulty.device
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        return loss * self.loss_weight





