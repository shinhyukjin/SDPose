"""
SDPose + Entropy-Weighted KD (Tuned v3.5) - EW Loss Weight 증가
================================================================

v3.4 기반 + EW Loss Weight 증가:
- v3.4의 모든 최적화 유지 (Temperature Annealing, Enhanced Normalization, Adaptive Scaling)
- EW Loss Weight 증가: 8.0e-6/8.5e-6 → 1.0e-5

핵심 변경사항:
1. loss_ew_heatmap: 8.0e-6 → 1.0e-5 (25% 증가)
2. loss_ew_token_vis: 8.5e-6 → 1.0e-5 (약 18% 증가)
3. loss_ew_token_kpt: 8.5e-6 → 1.0e-5 (약 18% 증가)

예상 효과: +0.3~0.5%p AP 향상
목표: v3.4 (72.93%) → v3.5 (73.2~73.4% AP)
"""

_base_ = ['../../../../_base_/datasets/coco.py']

date = '1212'
exp_description = 'entropy_weighted_tuned_v3.6'
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=6,
    save_last=True,
)

evaluation = dict(
    interval=10,
    metric='mAP',
    save_best='AP',
    rule='greater'
)

# ============ Optimizer (v3.4와 동일) ============
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)  # v3.4와 동일

# ============ Learning Rate (v3.4와 동일) ============
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=900,
    warmup_ratio=0.001,
    step=[220, 290, 310]
)

total_epochs = 330  # v3.4와 동일

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

# ============ Model (v3.4 + EW Loss Weight 증가) ============
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],

        # Base loss
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),

        # TokenDistilLoss (v3.4와 동일)
        loss_vis_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=1.5e-6,
        ),

        loss_kpt_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=1.5e-6,
        ),

        # 🔥 Entropy-Weighted Heatmap KD (EW Loss Weight 증가)
        loss_ew_heatmap=dict(
            type='EntropyWeightedKDLoss',
            base_temperature=2.5,
            temp_beta=1.25,
            temperature_min=1.2,
            temperature_max=5.0,
            weight_min=0.25,
            weight_max=3.2,
            loss_weight=1.0e-5,  # 8.0e-6 → 1.0e-5 (25% 증가)
            use_foreground_mask=True,
            mask_blur_kernel=5,
            mask_blur_iters=2,
            visibility_floor=0.1,
            kl_mode=True,
            # Temperature annealing & robust normalization은 코드에서 자동 활성화됨
        ),

        # 🔥 Entropy-Weighted Token KD (vis tokens, EW Loss Weight 증가)
        loss_ew_token_vis=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=1.0e-5,  # 8.5e-6 → 1.0e-5 (약 18% 증가)
            weight_min=0.2,
            weight_max=2.3,
            eps=1e-6,
            visibility_floor=0.05,
            detach_teacher=True,
            normalize_per_instance=True,
            # Robust normalization은 코드에서 자동 활성화됨
        ),

        # 🔥 Entropy-Weighted Token KD (keypoint tokens, EW Loss Weight 증가)
        loss_ew_token_kpt=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=1.0e-5,  # 8.5e-6 → 1.0e-5 (약 18% 증가)
            weight_min=0.4,
            weight_max=2.8,
            eps=1e-6,
            visibility_floor=0.1,
            detach_teacher=True,
            normalize_per_instance=True,
            # Robust normalization은 코드에서 자동 활성화됨
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

# ============ Data Config (v3.4와 동일) ============
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
    use_gt_bbox=True,  # Use GT bbox since detection file is not available
    det_bbox_thr=0.0,
    bbox_file='',  # Empty string to use GT bbox
)

# ============ Data Augmentation (v3.4와 동일) ============
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



