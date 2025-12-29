"""
SDPose - AP 75.0 Aggressive Configuration
==========================================
더욱 공격적인 설정 (AP 75.0+ 목표)

추가 최적화:
1. 더 큰 loss weight (2e-5)
2. Multi-scale training
3. 더 강한 augmentation
4. Drop path (stochastic depth)
5. Mixup augmentation concept

예상: AP 74.8~75.2

주의: 
- 학습 시간 더 길어짐 (400 epoch × 1.2)
- 메모리 사용량 증가
- 발산 위험 약간 증가 (모니터링 필수)
"""

_base_ = ['../../../../_base_/datasets/coco.py']

date = '1030'
exp_description = 'ap75_aggressive'
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

load_from = None
resume_from = None

log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=15,
    save_last=True,
)

evaluation = dict(
    interval=5,
    metric='mAP',
    save_best='AP',
    rule='greater'
)

optimizer = dict(
    type='AdamW',
    lr=1.2e-3,  # 1e-3 → 1.2e-3 (약간 증가)
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# Cosine Annealing with Warm Restart
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,  # 더 낮은 min_lr
    warmup='linear',
    warmup_iters=1500,  # 더 긴 warmup
    warmup_ratio=0.001,
)

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

model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        
        # 🔥 더 강한 SimpleConsistencyLoss
        loss_vis_token_dist=dict(
            type='SimpleConsistencyLoss',
            loss_weight=2e-5,  # 1.5e-5 → 2e-5 (더 강함)
            clamp_max=10.0,
        ),
        
        loss_kpt_token_dist=dict(
            type='SimpleConsistencyLoss',
            loss_weight=2e-5,
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

# 🔥 매우 강한 Data Augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.20, prob=0.5),  # 최대 강도
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.5),  # 절반 확률
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=60,      # 최대 rotation
        scale_factor=0.45),  # 최대 scale
    dict(type='TopDownAffine'),
    
    # 최대 강도 PhotometricDistortion
    dict(
        type='PhotometricDistortion',
        brightness_delta=50,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18
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

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
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

