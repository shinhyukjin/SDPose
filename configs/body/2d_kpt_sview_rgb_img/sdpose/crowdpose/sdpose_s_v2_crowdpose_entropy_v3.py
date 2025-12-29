"""
SDPose-S-V2 + Entropy-Weighted KD (CrowdPose)
============================================

- StemNet backbone + TokenPose S-V2 head
- CrowdPose 14 keypoints, Entropy-Weighted KD idea 적용
- 비교 대상: 논문 Table4 의 SDPose-S-V2 (AP 64.5 / AR 73.7)
"""

_base_ = ['../../../../../mmpose/configs/_base_/datasets/crowdpose.py']

date = '1118'
exp_description = 'crowdpose_s_v2_entropy_v3'
exp_name = f'sdpose_s_v2_{date}_{exp_description}'
work_dir = f'./work_dirs/{exp_name}'

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

checkpoint_config = dict(interval=10, max_keep_ckpts=6, save_last=True)
evaluation = dict(interval=10, metric='mAP', save_best='AP', rule='greater')

# Optimizer / LR
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=900,
    warmup_ratio=0.001,
    step=[220, 290, 310],
)
total_epochs = 330

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# CrowdPose: 14 joints
channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[list(range(14))],
    inference_channel=list(range(14)),
)

# Model (TokenPose S-V2 head + EW-KD idea)
model = dict(
    type='TopDown',
    backbone=dict(type='StemNet'),
    keypoint_head=dict(
        type='SDPoseHead',
        in_channels=256,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        loss_vis_token_dist=dict(type='TokenDistilLoss', loss_weight=5e-6),
        loss_kpt_token_dist=dict(type='TokenDistilLoss', loss_weight=5e-6),
        loss_ew_heatmap=dict(
            type='EntropyWeightedKDLoss',
            base_temperature=2.5,
            temp_beta=1.25,
            temperature_min=1.2,
            temperature_max=5.0,
            weight_min=0.25,
            weight_max=3.2,
            loss_weight=8.0e-6,
            use_foreground_mask=True,
            mask_blur_kernel=5,
            mask_blur_iters=2,
            visibility_floor=0.1,
            kl_mode=True,
        ),
        loss_ew_token_vis=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=8.5e-6,
            weight_min=0.2,
            weight_max=2.3,
            eps=1e-6,
            visibility_floor=0.05,
            detach_teacher=True,
            normalize_per_instance=True,
        ),
        loss_ew_token_kpt=dict(
            type='EntropyWeightedTokenKDLoss',
            loss_weight=8.5e-6,
            weight_min=0.4,
            weight_max=2.8,
            eps=1e-6,
            visibility_floor=0.1,
            detach_teacher=True,
            normalize_per_instance=True,
        ),
        tokenpose_cfg=dict(
            feature_size=[64, 48],
            patch_size=[2, 2],  # S-V2
            dim=256,
            depth=12,
            heads=8,
            mlp_ratio=3,
            heatmap_size=[64, 48],
            pos_embedding_type='sine-full',
            apply_init=True,
            cycle_num=2,
        )),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=11))

# Data cfg
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
    bbox_file='/dockerdata/crowdpose/crowdpose_annotations/det_for_crowd_test_0.1_0.5.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
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
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
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

data_root = '/dockerdata/crowdpose'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/crowdpose_annotations/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/crowdpose_annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCrowdPoseDataset',
        ann_file=f'{data_root}/crowdpose_annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

