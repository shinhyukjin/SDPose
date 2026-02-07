"""
Entropy-Weighted Knowledge Distillation Loss
=============================================

Based on EA-KD (ICCV 2025) idea:
- Calculate teacher's uncertainty (entropy) per pixel
- Adaptive temperature and weighting based on entropy
- Strong soft guidance on uncertain joints
- Foreground mask to suppress background noise

For SDPose:
- Teacher: Cycle 2 output (more refined)
- Student: Cycle 1 output (coarse)
- Apply to both heatmaps and tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class EntropyWeightedKDLoss(nn.Module):
    """Entropy-Weighted Knowledge Distillation for SDPose.
    
    Key ideas:
    1. Compute entropy from teacher predictions
    2. Adaptive temperature T(x) = T0 + β·norm(H(x))
    3. Adaptive weight w(x) = clip(norm(H(x)), a, b)
    4. Foreground mask to focus on keypoint regions
    
    Args:
        base_temperature (float): Base temperature T0. Default: 2.0
        temp_beta (float): Temperature scaling β. Default: 1.0
        weight_min (float): Minimum weight a. Default: 0.5
        weight_max (float): Maximum weight b. Default: 2.0
        loss_weight (float): Overall loss weight. Default: 1e-5
        use_foreground_mask (bool): Use GT heatmap as foreground. Default: True
        kl_mode (bool): Use KL divergence (True) or MSE (False). Default: True
    """
    
    def __init__(self,
                 base_temperature=2.0,
                 temp_beta=1.0,
                 temperature_min=0.5,
                 temperature_max=5.0,
                 weight_min=0.5,
                 weight_max=2.0,
                 loss_weight=1e-5,
                 use_foreground_mask=True,
                 mask_blur_kernel=3,
                 mask_blur_iters=1,
                 visibility_floor=0.0,
                 kl_mode=True,
                 normalize_by_area=True,
                 scale_by_temperature=True,
                 loss_scale=1.0,
                 warn_threshold=1e-12,
                 use_weighted_distribution=True):
        super().__init__()
        self.base_temperature = base_temperature
        self.temp_beta = temp_beta
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.loss_weight = loss_weight
        self.use_foreground_mask = use_foreground_mask
        self.mask_blur_kernel = mask_blur_kernel
        self.mask_blur_iters = mask_blur_iters
        self.visibility_floor = visibility_floor
        self.kl_mode = kl_mode
        self.normalize_by_area = normalize_by_area
        self.scale_by_temperature = scale_by_temperature
        self.loss_scale = loss_scale
        self.warn_threshold = warn_threshold
        self.use_weighted_distribution = use_weighted_distribution
        
        # For temperature annealing and adaptive weight scaling
        self.current_epoch = 0
        self.total_epochs = 330  # Default, can be updated
        self.use_temperature_annealing = True  # Enable temperature annealing
        self.temp_annealing_start = base_temperature * 1.5  # Start 50% higher
        self.temp_annealing_end = base_temperature  # End at base temperature
        
        # For enhanced entropy normalization
        self.use_robust_normalization = True  # Enable percentile-based normalization
        self.entropy_percentile_low = 0.05  # 5th percentile
        self.entropy_percentile_high = 0.95  # 95th percentile
        
        print(f"[EntropyWeightedKDLoss] Initialized:")
        print(f"  T0={base_temperature}, β={temp_beta}")
        print(f"  weight_range=[{weight_min}, {weight_max}]")
        print(f"  foreground_mask={use_foreground_mask}, KL={kl_mode}")
        print(f"  normalize_by_area={normalize_by_area}, scale_by_temperature={scale_by_temperature}")
        print(f"  loss_scale={loss_scale}, warn_threshold={warn_threshold:.1e}")
        print(f"  use_weighted_distribution={use_weighted_distribution}")
        print(f"  Temperature annealing={self.use_temperature_annealing}")
        print(f"  Robust normalization={self.use_robust_normalization}")
    
    def compute_entropy(self, logits, temperature=1.0):
        """
        Compute entropy from logits.
        
        Args:
            logits: [B, K, H, W] - teacher logits (heatmap)
            temperature: float - temperature for softmax
        
        Returns:
            entropy: [B, K, H, W] - entropy map
        """
        # Softmax over spatial dimensions (H, W)
        B, K, H, W = logits.shape
        logits_flat = logits.view(B, K, -1) / temperature  # [B, K, H*W]
        probs = F.softmax(logits_flat, dim=-1)  # [B, K, H*W]
        
        # Entropy: H(x) = -Σ p_c log(p_c)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, K]
        entropy = entropy.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        entropy = entropy.expand(B, K, H, W)  # [B, K, H, W]
        
        return entropy
    
    def normalize_entropy(self, entropy):
        """
        Normalize entropy to [0, 1] range.
        
        Enhanced version with percentile-based robust normalization (optional).
        
        Args:
            entropy: [B, K, H, W]
        
        Returns:
            normalized: [B, K, H, W]
        """
        B, K, H, W = entropy.shape
        entropy_flat = entropy.view(B, K, -1)  # [B, K, H*W]
        
        if self.use_robust_normalization:
            # Percentile-based robust normalization (outlier 제거)
            # Clamp to percentile range to remove extreme outliers
            lower_percentile = torch.quantile(entropy_flat, self.entropy_percentile_low, dim=-1, keepdim=True)  # [B, K, 1]
            upper_percentile = torch.quantile(entropy_flat, self.entropy_percentile_high, dim=-1, keepdim=True)  # [B, K, 1]
            
            # Clamp outliers (use torch.minimum/torch.maximum for Tensor broadcasting)
            entropy_clipped = torch.maximum(torch.minimum(entropy_flat, upper_percentile), lower_percentile)
            
            # Normalize using clipped values
            min_val = entropy_clipped.min(dim=-1, keepdim=True)[0]  # [B, K, 1]
            max_val = entropy_clipped.max(dim=-1, keepdim=True)[0]  # [B, K, 1]
        else:
            # Original normalization
            min_val = entropy_flat.min(dim=-1, keepdim=True)[0]  # [B, K, 1]
            max_val = entropy_flat.max(dim=-1, keepdim=True)[0]  # [B, K, 1]
            entropy_clipped = entropy_flat
        
        # Avoid division by zero
        range_val = (max_val - min_val) + 1e-10
        normalized = (entropy_clipped - min_val) / range_val  # [B, K, H*W]
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized.view(B, K, H, W)
    
    def get_current_temperature(self):
        """
        Get current base temperature with annealing.
        
        Returns:
            current_temp: float - annealed base temperature
        """
        if not self.use_temperature_annealing:
            return self.base_temperature
        
        if self.total_epochs <= 0:
            return self.base_temperature
        
        # Linear annealing from high to low temperature
        progress = min(1.0, max(0.0, self.current_epoch / self.total_epochs))
        current_temp = self.temp_annealing_start * (1 - progress) + self.temp_annealing_end * progress
        
        return current_temp
    
    def get_foreground_mask(self, gt_heatmap, threshold=0.01):
        """
        Create foreground mask from GT heatmap.
        
        Args:
            gt_heatmap: [B, K, H, W] - ground truth heatmap
            threshold: float - minimum value to consider foreground
        
        Returns:
            mask: [B, K, H, W] - foreground weight map [0, 1]
        """
        # Gaussian sum across keypoints
        foreground = gt_heatmap.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        foreground = foreground.expand_as(gt_heatmap)  # [B, K, H, W]
        
        # Normalize to [0, 1]
        fg_max = foreground.max()
        if fg_max > 0:
            foreground = foreground / fg_max
        
        # Optional smoothing via average pooling (soft mask)
        B, K, H, W = gt_heatmap.shape
        if self.mask_blur_kernel > 1:
            kernel = self.mask_blur_kernel
            padding = kernel // 2
            for _ in range(max(1, self.mask_blur_iters)):
                foreground = F.avg_pool2d(
                    foreground.view(-1, 1, H, W),
                    kernel_size=kernel,
                    stride=1,
                    padding=padding)
                foreground = foreground.view(B, K, H, W)
        
        # Apply threshold
        foreground = torch.clamp(foreground, min=threshold, max=1.0)
        
        return foreground
    
    def forward(self, student_logits, teacher_logits, gt_heatmap=None, target_weight=None):
        """
        Entropy-weighted KD loss.
        
        Args:
            student_logits: [B, K, H, W] - Cycle 1 predictions
            teacher_logits: [B, K, H, W] - Cycle 2 predictions (teacher)
            gt_heatmap: [B, K, H, W] - Ground truth for foreground mask
            target_weight: [B, K, 1] - Keypoint visibility weights
        
        Returns:
            loss: Scalar tensor
        """
        # Enhanced safety checks with detailed logging
        if student_logits is None or teacher_logits is None:
            import warnings
            warnings.warn("[EntropyWeightedKDLoss] Input is None! Returning zero loss.")
            return torch.tensor(0.0, device=student_logits.device if student_logits is not None else 'cuda', requires_grad=True)
        
        # Shape validation
        if student_logits.shape != teacher_logits.shape:
            import warnings
            warnings.warn(f"[EntropyWeightedKDLoss] Shape mismatch: student={student_logits.shape}, teacher={teacher_logits.shape}")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # NaN/Inf checks
        if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
            import warnings
            warnings.warn("[EntropyWeightedKDLoss] NaN detected in inputs! Returning zero loss.")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        if torch.isinf(student_logits).any() or torch.isinf(teacher_logits).any():
            import warnings
            warnings.warn("[EntropyWeightedKDLoss] Inf detected in inputs! Returning zero loss.")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        B, K, H, W = teacher_logits.shape
        
        # 1. Get current temperature (with annealing)
        current_base_temp = self.get_current_temperature()
        
        # 2. Compute entropy from teacher (with annealed temperature)
        entropy = self.compute_entropy(teacher_logits, current_base_temp)  # [B, K, H, W]
        entropy_norm = self.normalize_entropy(entropy)  # [B, K, H, W], range [0,1] (with robust normalization)
        
        # 3. Adaptive temperature per pixel (using annealed base temperature)
        T_adaptive = current_base_temp + self.temp_beta * entropy_norm  # [B, K, H, W]
        T_adaptive = torch.clamp(T_adaptive, min=self.temperature_min, max=self.temperature_max)
        
        # 4. Adaptive weight per pixel (lerp to preserve ordering)
        w_adaptive = self.weight_min + entropy_norm * (self.weight_max - self.weight_min)
        w_adaptive = torch.clamp(w_adaptive, min=self.weight_min, max=self.weight_max)
        
        # 5. Foreground mask (optional)
        if self.use_foreground_mask and gt_heatmap is not None:
            fg_mask = self.get_foreground_mask(gt_heatmap)  # [B, K, H, W]
            w_adaptive = w_adaptive * fg_mask  # Element-wise multiplication
        
        # 6. Compute distillation loss
        if self.kl_mode:
            # KL divergence with adaptive temperature
            # Teacher: high temp (softer), Student: adaptive temp
            temp_flat = T_adaptive.view(B, K, -1)
            teacher_soft = F.softmax(teacher_logits.view(B, K, -1) / temp_flat, dim=-1)
            student_log_soft = F.log_softmax(student_logits.view(B, K, -1) / temp_flat, dim=-1)

            if self.use_weighted_distribution:
                weight_map = w_adaptive
                if target_weight is not None:
                    vis_weight = target_weight.unsqueeze(-1)  # [B, K, 1, 1]
                    if self.visibility_floor > 0:
                        vis_weight = torch.clamp(vis_weight, min=self.visibility_floor)
                    weight_map = weight_map * vis_weight

                weight_flat = weight_map.view(B, K, -1)
                weight_flat = weight_flat / (weight_flat.sum(dim=-1, keepdim=True) + 1e-12)

                teacher_prob_w = teacher_soft * weight_flat
                teacher_prob_w = teacher_prob_w / (teacher_prob_w.sum(dim=-1, keepdim=True) + 1e-12)

                student_prob = torch.exp(student_log_soft)
                student_prob_w = student_prob * weight_flat
                student_prob_w = student_prob_w / (student_prob_w.sum(dim=-1, keepdim=True) + 1e-12)
                student_log_prob_w = torch.log(student_prob_w + 1e-12)

                kl_loss = teacher_prob_w * (torch.log(teacher_prob_w + 1e-12) - student_log_prob_w)
                kl_loss = kl_loss.sum(dim=-1)  # [B, K]
                if self.scale_by_temperature:
                    kl_loss = kl_loss * (temp_flat.mean(dim=-1) ** 2)

                weighted_loss = kl_loss
            else:
                # KL(teacher || student)
                kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='none')  # [B, K, H*W]
                kl_loss = kl_loss.view(B, K, H, W)  # [B, K, H, W]
                if self.scale_by_temperature:
                    kl_loss = kl_loss * (T_adaptive ** 2)
                weighted_loss = kl_loss * w_adaptive  # [B, K, H, W]
            
        else:
            # MSE with adaptive weights
            diff = (student_logits - teacher_logits.detach()) ** 2  # [B, K, H, W]
            weighted_loss = diff * w_adaptive
        
        # 7. Apply target weight (visibility)
        if target_weight is not None and not (self.kl_mode and self.use_weighted_distribution):
            # target_weight: [B, K, 1] → [B, K, 1, 1]
            vis_weight = target_weight.unsqueeze(-1)  # [B, K, 1, 1]
            if self.visibility_floor > 0:
                vis_weight = torch.clamp(vis_weight, min=self.visibility_floor)
            weighted_loss = weighted_loss * vis_weight
            
            # Normalize by visible keypoints
            denominator = vis_weight.sum() + 1e-6
            if self.normalize_by_area:
                denominator = denominator * H * W
            loss = weighted_loss.sum() / denominator
        else:
            loss = weighted_loss.mean()
        
        # Apply loss scale before final checks
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        # Enhanced final safety check with detailed logging
        if torch.isnan(loss) or torch.isinf(loss):
            import warnings
            warnings.warn(f"[EntropyWeightedKDLoss] Invalid loss value (NaN/Inf): {loss.item()}. Returning zero loss.")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Check if loss is suspiciously small (possible calculation error)
        loss_value = loss.item()
        if abs(loss_value) < self.warn_threshold:
            import warnings
            warnings.warn(f"[EntropyWeightedKDLoss] Loss is extremely small ({loss_value:.10e}). This might indicate a calculation error.")
        
        final_loss = loss * self.loss_weight
        return final_loss


@LOSSES.register_module()
class EntropyWeightedTokenKDLoss(nn.Module):
    """Entropy-Weighted Token Distillation.
    
    For token-level distillation with entropy weighting.
    Each token gets weight based on average entropy of its spatial region.
    
    Args:
        loss_weight (float): Overall weight. Default: 1e-5
        weight_min (float): Min weight. Default: 0.5
        weight_max (float): Max weight. Default: 2.0
    """
    
    def __init__(self,
                 loss_weight=1e-5,
                 weight_min=0.5,
                 weight_max=2.0,
                 eps=1e-6,
                 visibility_floor=0.0,
                 detach_teacher=True,
                 normalize_per_instance=True,
                 loss_scale=1.0,
                 warn_threshold=1e-12):
        super().__init__()
        self.loss_weight = loss_weight
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.eps = eps
        self.visibility_floor = visibility_floor
        self.detach_teacher = detach_teacher
        self.normalize_per_instance = normalize_per_instance
        self.loss_scale = loss_scale
        self.warn_threshold = warn_threshold
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
        # For enhanced entropy normalization (token-level)
        self.use_robust_normalization = True
        self.entropy_percentile_low = 0.05
        self.entropy_percentile_high = 0.95
        
        # For epoch tracking (for future use)
        self.current_epoch = 0
        self.total_epochs = 330
    
    def compute_token_weights(self, teacher_heatmap):
        """
        Compute per-token weights from teacher heatmap entropy.
        
        Enhanced with robust normalization (percentile-based).
        
        Args:
            teacher_heatmap: [B, K, H, W] - teacher predictions
        
        Returns:
            weights: [B, K, 1] - per-keypoint weights
        """
        B, K, H, W = teacher_heatmap.shape
        
        # Compute entropy per keypoint
        hm_flat = teacher_heatmap.view(B, K, -1)  # [B, K, H*W]
        probs = F.softmax(hm_flat, dim=-1)  # [B, K, H*W]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, K]
        
        if self.use_robust_normalization:
            # Percentile-based robust normalization
            if self.normalize_per_instance:
                lower_percentile = torch.quantile(entropy, self.entropy_percentile_low, dim=-1, keepdim=True)  # [B, 1]
                upper_percentile = torch.quantile(entropy, self.entropy_percentile_high, dim=-1, keepdim=True)  # [B, 1]
                # Use torch.minimum/torch.maximum for Tensor broadcasting
                entropy_clipped = torch.maximum(torch.minimum(entropy, upper_percentile), lower_percentile)
                min_val, _ = entropy_clipped.min(dim=-1, keepdim=True)
                max_val, _ = entropy_clipped.max(dim=-1, keepdim=True)
            else:
                lower_percentile = torch.quantile(entropy, self.entropy_percentile_low)
                upper_percentile = torch.quantile(entropy, self.entropy_percentile_high)
                # For scalar values, use item() or direct comparison
                if lower_percentile.numel() == 1 and upper_percentile.numel() == 1:
                    entropy_clipped = torch.clamp(entropy, min=lower_percentile.item(), max=upper_percentile.item())
                else:
                    entropy_clipped = torch.maximum(torch.minimum(entropy, upper_percentile), lower_percentile)
                min_val = entropy_clipped.min()
                max_val = entropy_clipped.max()
        else:
            # Original normalization
            if self.normalize_per_instance:
                min_val, _ = entropy.min(dim=-1, keepdim=True)
                max_val, _ = entropy.max(dim=-1, keepdim=True)
            else:
                min_val = entropy.min()
                max_val = entropy.max()
            entropy_clipped = entropy
        
        entropy_norm = (entropy_clipped - min_val) / (max_val - min_val + 1e-10)
        entropy_norm = torch.nan_to_num(entropy_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Adaptive weights via linear interpolation
        weights = self.weight_min + entropy_norm * (self.weight_max - self.weight_min)
        weights = torch.clamp(weights, min=self.weight_min, max=self.weight_max)
        
        return weights.unsqueeze(-1)  # [B, K, 1]
    
    def forward(self, student_tokens, teacher_tokens, teacher_heatmap=None, target_weight=None):
        """
        Entropy-weighted token distillation.
        
        Args:
            student_tokens: [B, K, D] - Cycle 1 tokens
            teacher_tokens: [B, K, D] - Cycle 2 tokens
            teacher_heatmap: [B, K, H, W] - For entropy computation
            target_weight: [B, K, 1] - Visibility weights
        
        Returns:
            loss: Scalar
        """
        # Safety checks
        if torch.isnan(student_tokens).any() or torch.isnan(teacher_tokens).any():
            return torch.tensor(0.0, device=student_tokens.device, requires_grad=True)
        
        # Compute token weights from entropy
        if teacher_heatmap is not None:
            entropy_weights = self.compute_token_weights(teacher_heatmap)  # [B, K, 1]
        else:
            entropy_weights = torch.ones(student_tokens.shape[0], 
                                         student_tokens.shape[1], 
                                         1, device=student_tokens.device)

        token_len = student_tokens.shape[1]
        weight_len = entropy_weights.shape[1]
        if weight_len != token_len:
            # Fallback: broadcast mean entropy weight across tokens when counts mismatch
            mean_weights = entropy_weights.mean(dim=1, keepdim=True)
            entropy_weights = mean_weights.expand(-1, token_len, -1)
        
        # Cosine similarity loss (better than MSE for tokens)
        teacher_tokens_eval = teacher_tokens.detach() if self.detach_teacher else teacher_tokens
        diff = 1 - self.cosine_similarity(student_tokens, teacher_tokens_eval)  # [B, K]
        diff = diff.unsqueeze(-1)  # [B, K, 1]
        
        # Apply entropy weights
        weighted_loss = diff * entropy_weights  # [B, K, 1]
        
        # Apply visibility weights
        if target_weight is not None:
            vis = torch.clamp(target_weight, min=self.visibility_floor) if self.visibility_floor > 0 else target_weight
            if vis.shape[1] != token_len:
                vis = vis.mean(dim=1, keepdim=True).expand(-1, token_len, -1)
            weighted_loss = weighted_loss * vis
            loss = weighted_loss.sum() / (vis.sum() + self.eps)
        else:
            loss = weighted_loss.mean()
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=student_tokens.device, requires_grad=True)

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        if abs(loss.item()) < self.warn_threshold:
            import warnings
            warnings.warn(f"[EntropyWeightedTokenKDLoss] Loss is extremely small ({loss.item():.10e}).")

        return loss * self.loss_weight


# Test
if __name__ == '__main__':
    print("Testing EntropyWeightedKDLoss...")
    
    B, K, H, W = 4, 17, 64, 48
    D = 192
    
    # Dummy data
    student_hm = torch.randn(B, K, H, W)
    teacher_hm = torch.randn(B, K, H, W)
    gt_hm = torch.randn(B, K, H, W).abs()
    target_weight = torch.ones(B, K, 1)
    
    student_tokens = torch.randn(B, K, D)
    teacher_tokens = torch.randn(B, K, D)
    
    # Test heatmap loss
    loss_fn = EntropyWeightedKDLoss(loss_weight=1e-5)
    loss = loss_fn(student_hm, teacher_hm, gt_hm, target_weight)
    print(f"Heatmap KD Loss: {loss.item():.6f}")
    
    # Test token loss
    token_loss_fn = EntropyWeightedTokenKDLoss(loss_weight=1e-5)
    token_loss = token_loss_fn(student_tokens, teacher_tokens, teacher_hm, target_weight)
    print(f"Token KD Loss: {token_loss.item():.6f}")
    
    print("✅ EntropyWeightedKDLoss test passed!")


