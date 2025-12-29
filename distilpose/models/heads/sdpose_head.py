import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import HEADS, build_loss

from .utils.tokenbase import SDPose

BN_MOMENTUM = 0.1

@HEADS.register_module()
class SDPoseHead(nn.Module):
    """
    "TokenPose: Learning Keypoint Tokens for Human Pose Estimation".

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        tokenpose_cfg (dict): Config for tokenpose.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 loss_vis_token_dist=None,
                 loss_kpt_token_dist=None,
                 loss_ew_heatmap=None,
                 loss_ew_token_vis=None,
                 loss_ew_token_kpt=None,
                 loss_hd_distill=None,
                 tokenpose_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        
        super().__init__()
        self.epoch = 0  # which would be update in SetEpochInfoHook!

        self.in_channels = in_channels
        self.num_joints = num_joints
        
        self.keypoint_loss = build_loss(loss_keypoint)
        if loss_vis_token_dist is not None:
            self.vis_token_dist_loss = build_loss(loss_vis_token_dist)
        else:
            self.vis_token_dist_loss = None
        if loss_kpt_token_dist is not None:
            self.kpt_token_dist_loss = build_loss(loss_kpt_token_dist)
        else:
            self.kpt_token_dist_loss = None

        if loss_ew_heatmap is not None:
            self.ew_heatmap_loss = build_loss(loss_ew_heatmap)
        else:
            self.ew_heatmap_loss = None

        if loss_ew_token_vis is not None:
            self.ew_token_vis_loss = build_loss(loss_ew_token_vis)
        else:
            self.ew_token_vis_loss = None

        if loss_ew_token_kpt is not None:
            self.ew_token_kpt_loss = build_loss(loss_ew_token_kpt)
        else:
            self.ew_token_kpt_loss = None

        if loss_hd_distill is not None:
            self.hd_distill_loss = build_loss(loss_hd_distill)
        else:
            self.hd_distill_loss = None

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        
        # For adaptive loss weight scaling
        self.total_epochs = 330  # Default, will be updated from config
        self.use_adaptive_ew_weight = True  # Enable adaptive EW loss weight scaling

        self.tokenpose_cfg = {} if tokenpose_cfg is None else tokenpose_cfg

        self.tokenpose = SDPose(feature_size=tokenpose_cfg.feature_size, 
                                           patch_size=tokenpose_cfg.patch_size, 
                                           num_keypoints=self.num_joints, 
                                           dim=tokenpose_cfg.dim, 
                                           depth=tokenpose_cfg.depth, 
                                           heads=tokenpose_cfg.heads,
                                           mlp_ratio=tokenpose_cfg.mlp_ratio, 
                                           heatmap_size=tokenpose_cfg.heatmap_size,
                                           channels=in_channels,
                                           pos_embedding_type=tokenpose_cfg.pos_embedding_type,
                                           apply_init=tokenpose_cfg.apply_init,
                                           cycle_num=tokenpose_cfg.cycle_num)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        x = self.tokenpose(x)
        return x

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        losses = dict()
        output_len = len(output)
        
        # Get device from first output
        device = output[0].pred.device

        # Initialize losses as torch tensors (not Python int!)
        heatmap_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        for i in range(output_len):
            current_loss = self.keypoint_loss(output[i].pred, target, target_weight)
            # NaN safety check
            if not torch.isnan(current_loss) and not torch.isinf(current_loss):
                heatmap_loss = heatmap_loss + current_loss
            else:
                print(f"[WARNING] NaN/Inf detected in heatmap_loss at cycle {i}, skipping...")
        
        losses['heatmap_loss'] = heatmap_loss
        
        # Token distillation with optional heatmap guidance
        if self.vis_token_dist_loss is not None:
            vis_dist_losses = []
            for i in range(output_len-1):
                # Check if loss supports heatmap-based foreground weighting
                if hasattr(self.vis_token_dist_loss, 'use_spatial_weight'):
                    # Pass heatmap for foreground weighting
                    current_loss = self.vis_token_dist_loss(
                        output[i].vis_token, 
                        output[i+1].vis_token,
                        heatmap=output[i].pred.detach()  # Use teacher (Cycle 1) heatmap
                    )
                else:
                    # Standard distillation without spatial weighting
                    current_loss = self.vis_token_dist_loss(
                        output[i].vis_token, 
                        output[i+1].vis_token
                    )
                
                # NaN safety check and collect valid losses
                if not torch.isnan(current_loss) and not torch.isinf(current_loss):
                    vis_dist_losses.append(current_loss)
            
            # Use mean instead of sum to prevent accumulation explosion
            if len(vis_dist_losses) > 0:
                vis_dist_loss = torch.stack(vis_dist_losses).mean()
            else:
                vis_dist_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            
            losses["vis_dist_loss"] = vis_dist_loss
        
        if self.kpt_token_dist_loss is not None:
            kpt_dist_losses = []
            for i in range(output_len-1):
                # Keypoint tokens already focus on keypoints, less need for spatial weighting
                # But can still benefit from visibility weighting
                if hasattr(self.kpt_token_dist_loss, 'use_keypoint_guidance'):
                    current_loss = self.kpt_token_dist_loss(
                        output[i].kpt_token, 
                        output[i+1].kpt_token,
                        heatmap=output[i].pred.detach(),
                        target_weight=target_weight
                    )
                else:
                    current_loss = self.kpt_token_dist_loss(
                        output[i].kpt_token, 
                        output[i+1].kpt_token
                    )
                
                # NaN safety check and collect valid losses
                if not torch.isnan(current_loss) and not torch.isinf(current_loss):
                    kpt_dist_losses.append(current_loss)
            
            # Use mean instead of sum to prevent accumulation explosion
            if len(kpt_dist_losses) > 0:
                kpt_dist_loss = torch.stack(kpt_dist_losses).mean()
            else:
                kpt_dist_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            
            losses["kpt_dist_loss"] = kpt_dist_loss

        # Entropy-weighted distillation (requires at least Cycle-2 output)
        if output_len >= 2:
            teacher_idx = 1  # Cycle-2 supervises Cycle-1

            if self.ew_heatmap_loss is not None:
                # Enhanced debugging: Check inputs before loss calculation
                import logging
                logger = logging.getLogger(__name__)
                
                # Update epoch info in loss function for temperature annealing
                if hasattr(self.ew_heatmap_loss, 'current_epoch'):
                    self.ew_heatmap_loss.current_epoch = self.epoch
                if hasattr(self.ew_heatmap_loss, 'total_epochs'):
                    self.ew_heatmap_loss.total_epochs = self.total_epochs
                
                # Input validation
                student_pred = output[0].pred
                teacher_pred = output[teacher_idx].pred.detach()
                
                # Debug: Check for invalid inputs (first iteration of each epoch)
                if not hasattr(self, '_ew_debug_logged') or self._ew_debug_logged != self.epoch:
                    if torch.isnan(student_pred).any() or torch.isnan(teacher_pred).any():
                        logger.warning(f"[EW_DEBUG] Epoch {self.epoch}: NaN detected in inputs!")
                    if torch.isinf(student_pred).any() or torch.isinf(teacher_pred).any():
                        logger.warning(f"[EW_DEBUG] Epoch {self.epoch}: Inf detected in inputs!")
                    logger.info(f"[EW_DEBUG] Epoch {self.epoch}: student_pred shape={student_pred.shape}, "
                              f"teacher_pred shape={teacher_pred.shape}, target shape={target.shape}")
                    self._ew_debug_logged = self.epoch
                
                ew_hm_loss_raw = self.ew_heatmap_loss(
                    student_pred,
                    teacher_pred,
                    target,
                    target_weight)
                
                # Adaptive loss weight scaling
                weight_factor = 1.0
                if self.use_adaptive_ew_weight:
                    # Gradually increase EW Loss weight during training
                    # Epoch 0-50: 0.5x (약한 영향으로 안정적 학습)
                    # Epoch 50-150: 0.5x → 1.0x (선형 증가)
                    # Epoch 150+: 1.0x (원래 weight)
                    if self.epoch < 50:
                        weight_factor = 0.5
                    elif self.epoch < 150:
                        progress = (self.epoch - 50) / 100.0
                        weight_factor = 0.5 + 0.5 * progress
                    else:
                        weight_factor = 1.0
                
                # Enhanced debugging: Log raw loss value periodically
                losses['ew_heatmap_loss'] = ew_hm_loss_raw * weight_factor
                loss_val = ew_hm_loss_raw.item()
                
                # Log at epoch start and periodically
                if (self.epoch == 1 and not hasattr(self, '_ew_first_logged')) or \
                   (self.epoch % 10 == 0 and not hasattr(self, f'_ew_epoch_{self.epoch}_logged')):
                    logger.info(f"[EW_DEBUG] Epoch {self.epoch}: EW Heatmap Loss (raw)={loss_val:.10f}, "
                              f"output_len={output_len}, teacher_idx={teacher_idx}")
                    if self.epoch == 1:
                        self._ew_first_logged = True
                    else:
                        setattr(self, f'_ew_epoch_{self.epoch}_logged', True)
                
                # Warning if loss is suspiciously small or zero
                if abs(loss_val) < 1e-8:
                    logger.warning(f"[EW_DEBUG] Epoch {self.epoch}: EW Heatmap Loss is very small ({loss_val:.10f})!")

            if self.ew_token_vis_loss is not None:
                # Update epoch info in loss function
                if hasattr(self.ew_token_vis_loss, 'current_epoch'):
                    self.ew_token_vis_loss.current_epoch = self.epoch
                if hasattr(self.ew_token_vis_loss, 'total_epochs'):
                    self.ew_token_vis_loss.total_epochs = self.total_epochs
                
                ew_tok_vis_loss_raw = self.ew_token_vis_loss(
                    output[0].vis_token,
                    output[teacher_idx].vis_token,
                    output[teacher_idx].pred.detach(),
                    target_weight)
                
                # Adaptive loss weight scaling
                weight_factor = 1.0
                if self.use_adaptive_ew_weight:
                    if self.epoch < 50:
                        weight_factor = 0.5
                    elif self.epoch < 150:
                        progress = (self.epoch - 50) / 100.0
                        weight_factor = 0.5 + 0.5 * progress
                    else:
                        weight_factor = 1.0
                
                losses['ew_token_vis_loss'] = ew_tok_vis_loss_raw * weight_factor
                # Enhanced debugging: Periodic logging
                import logging
                logger = logging.getLogger(__name__)
                loss_val = ew_tok_vis_loss_raw.item()
                if (self.epoch == 1 and not hasattr(self, '_ew_token_vis_first_logged')) or \
                   (self.epoch % 10 == 0 and not hasattr(self, f'_ew_token_vis_epoch_{self.epoch}_logged')):
                    logger.info(f"[EW_DEBUG] Epoch {self.epoch}: EW Token Vis Loss (raw)={loss_val:.10f}")
                    if self.epoch == 1:
                        self._ew_token_vis_first_logged = True
                    else:
                        setattr(self, f'_ew_token_vis_epoch_{self.epoch}_logged', True)
                if abs(loss_val) < 1e-8:
                    logger.warning(f"[EW_DEBUG] Epoch {self.epoch}: EW Token Vis Loss is very small ({loss_val:.10f})!")

            if self.ew_token_kpt_loss is not None:
                # Update epoch info in loss function
                if hasattr(self.ew_token_kpt_loss, 'current_epoch'):
                    self.ew_token_kpt_loss.current_epoch = self.epoch
                if hasattr(self.ew_token_kpt_loss, 'total_epochs'):
                    self.ew_token_kpt_loss.total_epochs = self.total_epochs
                
                ew_tok_kpt_loss_raw = self.ew_token_kpt_loss(
                    output[0].kpt_token,
                    output[teacher_idx].kpt_token,
                    output[teacher_idx].pred.detach(),
                    target_weight)
                
                # Adaptive loss weight scaling
                weight_factor = 1.0
                if self.use_adaptive_ew_weight:
                    if self.epoch < 50:
                        weight_factor = 0.5
                    elif self.epoch < 150:
                        progress = (self.epoch - 50) / 100.0
                        weight_factor = 0.5 + 0.5 * progress
                    else:
                        weight_factor = 1.0
                
                losses['ew_token_kpt_loss'] = ew_tok_kpt_loss_raw * weight_factor
                # Enhanced debugging: Periodic logging
                import logging
                logger = logging.getLogger(__name__)
                loss_val = ew_tok_kpt_loss_raw.item()
                if (self.epoch == 1 and not hasattr(self, '_ew_token_kpt_first_logged')) or \
                   (self.epoch % 10 == 0 and not hasattr(self, f'_ew_token_kpt_epoch_{self.epoch}_logged')):
                    logger.info(f"[EW_DEBUG] Epoch {self.epoch}: EW Token Kpt Loss (raw)={loss_val:.10f}")
                    if self.epoch == 1:
                        self._ew_token_kpt_first_logged = True
                    else:
                        setattr(self, f'_ew_token_kpt_epoch_{self.epoch}_logged', True)
                if abs(loss_val) < 1e-8:
                    logger.warning(f"[EW_DEBUG] Epoch {self.epoch}: EW Token Kpt Loss is very small ({loss_val:.10f})!")

        # HD-Distill: Hard-aware Dynamic Distillation
        # 마지막 cycle (teacher)와 이전 cycles (student) 간 weighted distillation
        if self.hd_distill_loss is not None and output_len >= 2:
            hd_distill_losses = []
            final_cycle_idx = output_len - 1  # 마지막 cycle이 teacher
            
            for i in range(final_cycle_idx):
                # 각 cycle i의 heatmap과 마지막 cycle의 heatmap 간 HD-Distill
                current_loss = self.hd_distill_loss(
                    output[i].pred,                    # student heatmaps [B, J, H, W]
                    output[final_cycle_idx].pred.detach(),  # teacher (final cycle) [B, J, H, W]
                    target                              # GT heatmaps [B, J, H, W]
                )
                
                # NaN safety check
                if not torch.isnan(current_loss) and not torch.isinf(current_loss):
                    hd_distill_losses.append(current_loss)
            
            # Average over all cycles
            if len(hd_distill_losses) > 0:
                hd_distill_loss = torch.stack(hd_distill_losses).mean()
            else:
                hd_distill_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            
            losses['hd_distill_loss'] = hd_distill_loss

        return losses

    def get_accuracy(self, outputs, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                outputs[-2].pred.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            
        accuracy['acc_pose'] = float(avg_acc)


        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        output = output[0].pred

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
