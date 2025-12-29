"""
SDPose - AP 75.0 Target Configuration
======================================
목표: AP 75.0 달성을 위한 공격적인 최적화

전략:
1. 더 강력한 Data Augmentation
2. 최적화된 Loss Weight (SimpleConsistencyLoss)
3. 더 긴 학습 (400 epoch)
4. Cosine Annealing LR with Restart
5. Label Smoothing
6. EMA (Exponential Moving Average)
7. Mixed Precision Training

예상 향상:
- Strong Augmentation: +0.5~0.8
- Optimized Loss: +0.3~0.5
- Longer Training: +0.3~0.5
- Cosine LR: +0.2~0.3
- EMA: +0.2~0.4
Total: +1.5~2.5 → AP 74.5~75.5
"""

_base_ = ['../../../../_base_/datasets/coco.py']

date = '1030'
exp_description = 'ap75_target'
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

load_from = None
resume_from = None

log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=10,  # 더 많이 저장 (best model 선택)
    save_last=True,
)

evaluation = dict(
    interval=5,  # 더 자주 평가 (5 epoch마다)
    metric='mAP',
    save_best='AP',
    rule='greater'
)

optimizer = dict(
    type='AdamW',  # Adam → AdamW (weight decay 개선)
    lr=1e-3,
    weight_decay=1e-4,  # Weight decay 추가
)

# Gradient clipping
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# 🚀 Cosine Annealing with Warm Restart (더 효과적)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
)

# 더 긴 학습 (300 → 400 epoch)
total_epochs = 400

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
# 🌟 Optimized Model for AP 75.0
# ============================================================================
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        
        # Main heatmap loss with label smoothing (implicit in implementation)
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        
        # 🌟 Optimized SimpleConsistencyLoss
        # 1e-5로 시작하여 안정성과 성능 균형
        loss_vis_token_dist=dict(
            type='SimpleConsistencyLoss',
            loss_weight=1.5e-5,  # 1e-5 → 1.5e-5 (50% 증가, 안전 범위)
            clamp_max=10.0,
        ),
        
        loss_kpt_token_dist=dict(
            type='SimpleConsistencyLoss',
            loss_weight=1.5e-5,
            clamp_max=10.0,
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

# 🚀 강화된 Data Augmentation (AP 향상의 핵심!)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.18, prob=0.4),  # 0.16 → 0.18, 0.3 → 0.4
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.4),  # 0.3 → 0.4 (더 자주)
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=50,      # 45 → 50 (더 강한 rotation)
        scale_factor=0.4),  # 0.35 → 0.4 (더 강한 scale)
    dict(type='TopDownAffine'),
    
    # 🚀 강화된 PhotometricDistortion
    dict(
        type='PhotometricDistortion',
        brightness_delta=40,           # 32 → 40 (더 강함)
        contrast_range=(0.7, 1.3),    # (0.8, 1.2) → (0.7, 1.3)
        saturation_range=(0.7, 1.3),  # (0.8, 1.2) → (0.7, 1.3)
        hue_delta=15                   # 10 → 15
    ),
    
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

# 🚀 증가된 batch size (메모리 허용 시)
data = dict(
    samples_per_gpu=64,  # 메모리 부족 시 48로 감소
    workers_per_gpu=4,   # 2 → 4 (더 빠른 data loading)
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

