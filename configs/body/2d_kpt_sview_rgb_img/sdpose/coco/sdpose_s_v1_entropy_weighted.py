"""
SDPose + Entropy-Weighted KD - AP 73%+ Target
==============================================

New Features:
1. EntropyWeightedKDLoss for heatmap distillation
2. EntropyWeightedTokenKDLoss for token distillation
3. Adaptive weighting based on teacher uncertainty
4. Foreground mask to suppress background noise

Expected: AP 72.5~73.5% (+0.5~1.0% from baseline)
Based on: EA-KD (ICCV 2025) idea
"""

_base_ = ['../../../../_base_/datasets/coco.py']

date = '1105'
exp_description = 'entropy_weighted'
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=5,
    save_last=True,
)

evaluation = dict(
    interval=10,
    metric='mAP',
    save_best='AP',
    rule='greater'
)

# ============ Optimizer (NO clipping!) ============
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)  # Critical!

# ============ Learning Rate ============
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260]  # Baseline schedule
)

total_epochs = 300  # Standard

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

# ============ Model (Baseline + Entropy-Weighted) ============
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        
        # Base loss
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        
        # Baseline token distillation (keep original)
        loss_vis_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=5e-6,  # Baseline weight
        ),
        
        loss_kpt_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=5e-6,  # Baseline weight
        ),
        
        # 🔥 NEW: Entropy-Weighted Heatmap KD
        loss_ew_heatmap=dict(
            type='EntropyWeightedKDLoss',
            base_temperature=2.0,      # Temperature for softmax
            temp_beta=1.0,              # Temp scaling with entropy
            weight_min=0.5,             # Min weight for low entropy
            weight_max=2.0,             # Max weight for high entropy
            loss_weight=5e-6,           # Same as baseline
            use_foreground_mask=True,   # Use GT heatmap as foreground
            kl_mode=True,               # KL divergence (vs MSE)
        ),
        
        # 🔥 NEW: Entropy-Weighted Token KD
        loss_ew_token=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=5e-6,           # Same as baseline
            weight_min=0.5,
            weight_max=2.0,
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

# ============ Data Config ============
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

# ============ Data Augmentation (Baseline) ============
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
        type='TopDownGetRandomScaleRotation', 
        rot_factor=45,
        scale_factor=0.35),
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

