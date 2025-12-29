"""
SDPose + Entropy-Weighted KD (Tuned v4) - 300 Epoch Optimized
=============================================================

Optimization based on Tuned v3 (AP 72.81%):
- Epoch: 330 → 300 (Baseline과 동일)
- LR Schedule: Baseline과 동일 [200, 260]
- Loss weight balancing for better convergence
- Fine-tuned entropy parameters for 300 epoch training
- Expected: AP 73.0~73.2% (+0.2~0.4% from v3)

Key improvements:
1. Adjusted loss weights for better balance
2. Optimized temperature scaling for 300 epoch
3. Refined weight ranges based on v3 results
4. Baseline-like LR schedule for fair comparison
"""

_base_ = ['../../../../_base_/datasets/coco.py']

date = '1121'
exp_description = 'entropy_weighted_tuned_v4'
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

# ============ Optimizer (NO clipping! Baseline과 동일) ============
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)  # Baseline과 동일 (critical!)

# ============ Learning Rate (Baseline과 동일 스케줄) ============
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,      # Baseline과 동일
    warmup_ratio=0.001,
    step=[200, 260]        # Baseline과 동일 (300 epoch용)
)

total_epochs = 300  # Baseline과 동일

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

# ============ Model (EW-KD v4: Optimized for 300 epochs) ============
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],

        # Base loss
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),

        # Baseline token distillation (균형 조정)
        loss_vis_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=2e-6,  # v3: 1.5e-6 → 2e-6 (약간 증가)
        ),

        loss_kpt_token_dist=dict(
            type='TokenDistilLoss',
            loss_weight=2e-6,  # v3: 1.5e-6 → 2e-6 (약간 증가)
        ),

        # 🔥 Entropy-Weighted Heatmap KD (최적화)
        loss_ew_heatmap=dict(
            type='EntropyWeightedKDLoss',
            base_temperature=2.5,      # v3과 동일
            temp_beta=1.3,             # v3: 1.25 → 1.3 (더 강한 엔트로피 강조)
            temperature_min=1.2,
            temperature_max=5.0,
            weight_min=0.3,            # v3: 0.25 → 0.3 (안정성)
            weight_max=3.0,            # v3: 3.2 → 3.0 (균형)
            loss_weight=8.5e-6,        # v3: 8.0e-6 → 8.5e-6 (약간 증가)
            use_foreground_mask=True,
            mask_blur_kernel=5,
            mask_blur_iters=2,
            visibility_floor=0.1,
            kl_mode=True,
        ),

        # 🔥 Entropy-Weighted Token KD (vis tokens) - 최적화
        loss_ew_token_vis=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=9.0e-6,        # v3: 8.5e-6 → 9.0e-6 (약간 증가)
            weight_min=0.3,            # v3: 0.2 → 0.3 (안정성)
            weight_max=2.5,            # v3: 2.3 → 2.5 (범위 확대)
            eps=1e-6,
            visibility_floor=0.05,
            detach_teacher=True,
            normalize_per_instance=True,
        ),

        # 🔥 Entropy-Weighted Token KD (keypoint tokens) - 최적화
        loss_ew_token_kpt=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=9.0e-6,        # v3: 8.5e-6 → 9.0e-6 (약간 증가)
            weight_min=0.5,            # v3: 0.4 → 0.5 (안정성)
            weight_max=2.8,            # v3과 동일
            eps=1e-6,
            visibility_floor=0.1,
            detach_teacher=True,
            normalize_per_instance=True,
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

# ============ Data Augmentation (Baseline과 동일) ============
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















