"""
SDPose with Foreground Self-Distillation - STABLE High Performance
===================================================================
발산 문제를 해결하면서 최대 성능을 끌어내는 설정

Key Changes from FSD Best:
1. Reduced loss weight (8e-6 → 3e-6) - 안정성
2. Moderate foreground emphasis (3.0 → 2.5) - 균형
3. Stronger gradient clipping (1.0 → 0.5) - 폭발 방지
4. Standard keypoint loss (Adaptive → Standard) - 안정성
5. Smooth temperature (0.8 → 1.0) - 부드러운 mask

Expected Performance:
- AP: 73.4~73.6 (안정적으로 달성)
- No divergence (loss 안정적)
- Smooth training curve
"""

_base_ = ['../../../../_base_/datasets/coco.py']

# ============================================================================
# Experiment Configuration
# ============================================================================
date = '1027'
exp_description = 'fsd_stable'  # Stable high performance
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

load_from = None
resume_from = None

# ============================================================================
# Training Configuration - Stabilized
# ============================================================================
log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=5,
    save_last=True,
)

evaluation = dict(
    interval=10,  # 10 epoch마다 (5→10, 안정성)
    metric='mAP',
    save_best='AP',
    rule='greater'
)

optimizer = dict(
    type='Adam',
    lr=1e-3,
)

# 🛡️ Strong gradient clipping for stability
optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))  # 1.0 → 0.5

# Learning rate schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[200, 260])

total_epochs = 300

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# ============================================================================
# Model with STABLE FSD Configuration
# ============================================================================
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        
        # Main heatmap loss
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        
        # 🌟 Visual Token: Moderate Foreground Emphasis (STABLE)
        loss_vis_token_dist=dict(
            type='ForegroundTokenDistilLoss',
            loss_weight=3e-6,              # 8e-6 → 3e-6 (62% 감소, 안정성)
            foreground_weight=2.5,         # 3.0 → 2.5 (약간 완화)
            background_weight=0.4,         # 0.3 → 0.4 (약간 증가)
            threshold=0.1,                 # 0.08 → 0.1 (원래대로)
            temperature=1.0,               # 0.8 → 1.0 (부드러운 mask)
            use_spatial_weight=True,
        ),
        
        # 🌟 Keypoint Token: Standard Loss (STABLE)
        loss_kpt_token_dist=dict(
            type='TokenDistilLoss',        # Adaptive → Standard (안정성)
            loss_weight=5e-6,              # 원래 수준
        ),
        
        tokenpose_cfg=dict(
            feature_size=[64, 48],
            patch_size=[4, 3],
            dim=192,
            depth=12,
            heads=8,
            mlp_ratio=3,
            heatmap_size=[64, 48],
            pos_embedding_type='sine-full',
            apply_init=True,
            cycle_num=2
        )),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=11))

# ============================================================================
# Dataset Configuration
# ============================================================================
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='/dockerdata/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=45, scale_factor=0.35),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, unbiased_encoding=True),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = '/dockerdata/coco/'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)















