# SDPose 설정 파일 가이드

## 📁 실험 관리

### 1️⃣ 새로운 실험 시작하기

**파일**: `sdpose_s_v1_stemnet_coco_256x192.py`

```python
# ============================================================================
# Experiment Configuration - MODIFY HERE
# ============================================================================

# 실험 이름 변경 (work_dirs 폴더 이름이 됩니다)
exp_name = 'sdpose_s_v1_stemnet_coco_256x192_maskedkd_v2'

# 작업 디렉토리 (자동으로 생성됨)
work_dir = f'./work_dirs/{exp_name}'
```

**결과**: 
- 체크포인트: `./work_dirs/sdpose_s_v1_stemnet_coco_256x192_maskedkd_v2/*.pth`
- 로그: `./work_dirs/sdpose_s_v1_stemnet_coco_256x192_maskedkd_v2/*.log`

---

### 2️⃣ 이전 체크포인트에서 재개하기

```python
# 학습 재개할 체크포인트 경로 지정
resume_from = './work_dirs/sdpose_s_v1_stemnet_coco_256x192/epoch_100.pth'

# 또는 latest 체크포인트 사용
resume_from = './work_dirs/sdpose_s_v1_stemnet_coco_256x192/latest.pth'
```

---

### 3️⃣ Pre-trained 모델 사용하기

```python
# Pre-trained 모델 로드 (새로운 학습 시작)
load_from = './pretrained_model/sdpose_s_v1.pth'

# resume_from과의 차이점:
# - load_from: 모델 가중치만 로드 (epoch, optimizer state 초기화)
# - resume_from: 모든 상태 로드 (epoch, optimizer state 포함)
```

---

## 💾 체크포인트 저장 설정

```python
checkpoint_config = dict(
    interval=10,           # 10 epoch마다 저장
    max_keep_ckpts=3,      # 최근 3개 체크포인트만 유지 (디스크 절약)
    save_last=True,        # latest.pth 항상 저장
    out_dir=None           # work_dir 사용 (기본값)
)
```

**저장되는 파일**:
- `epoch_10.pth`, `epoch_20.pth`, `epoch_30.pth`, ...
- `latest.pth` (가장 최근 체크포인트)
- `best_AP_epoch_XX.pth` (최고 성능 모델)

---

## 📊 데이터셋 경로 설정

```python
# COCO 데이터셋 경로
data_root = '/dockerdata/coco/'

# Windows 예시
# data_root = 'D:/datasets/coco/'

# Linux 예시
# data_root = '/home/user/datasets/coco/'

# 필요한 폴더 구조:
# {data_root}/
# ├── annotations/
# │   ├── person_keypoints_train2017.json
# │   └── person_keypoints_val2017.json
# ├── train2017/
# └── val2017/
```

---

## 🎯 학습 하이퍼파라미터

```python
# Batch size
data = dict(
    samples_per_gpu=64,    # GPU당 batch size
    workers_per_gpu=2,     # 데이터 로딩 워커 수
)

# Optimizer
optimizer = dict(
    type='Adam',
    lr=1e-3,              # Learning rate
)

# Learning rate schedule
lr_config = dict(
    policy='step',
    step=[200, 260],      # LR 감소 시점
)

# Total epochs
total_epochs = 300
```

---

## 📈 로깅 및 평가

### 텍스트 로그
```python
log_config = dict(
    interval=50,          # 50 iteration마다 로그 출력
    hooks=[
        dict(type='TextLoggerHook'),
    ])
```

### TensorBoard 활성화
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # 주석 제거
    ])
```

TensorBoard 실행:
```bash
tensorboard --logdir=./work_dirs/your_exp_name
```

### 평가 설정
```python
evaluation = dict(
    interval=10,          # 10 epoch마다 평가
    metric='mAP',         # 평가 지표
    save_best='AP',       # AP 기준으로 best 모델 저장
)
```

---

## 🔬 MaskedKD 설정

```python
loss_vis_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,     # Loss weight (1e-6 ~ 1e-4 권장)
    mask_ratio=0.3,       # 마스킹 비율 (0.2 ~ 0.5 권장)
    mask_strategy='random'  # 'random' or 'importance'
),
```

**실험 추천**:
- `mask_ratio`: 0.2, 0.3, 0.5
- `loss_weight`: 1e-6, 5e-6, 1e-5
- `mask_strategy`: 'random' (안정적), 'importance' (성능 향상 가능)

---

## 🚀 실험 예시

### 실험 1: MaskedKD 비율 테스트
```python
exp_name = 'sdpose_s_v1_maskedkd_ratio_0.5'
mask_ratio = 0.5  # 50% 마스킹
```

### 실험 2: Loss weight 조정
```python
exp_name = 'sdpose_s_v1_maskedkd_weight_1e4'
loss_weight = 1e-4  # 더 강한 distillation
```

### 실험 3: Importance masking
```python
exp_name = 'sdpose_s_v1_maskedkd_importance'
mask_strategy = 'importance'  # 중요도 기반 마스킹
```

---

## 🛠️ 학습 명령어

### 단일 GPU
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

### 멀티 GPU (예: 8 GPUs)
```bash
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py 8
```

### 학습 재개
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py \
    --resume-from work_dirs/sdpose_s_v1_stemnet_coco_256x192/latest.pth
```

### 평가만 실행
```bash
python tools/test.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py \
    work_dirs/sdpose_s_v1_stemnet_coco_256x192/best_AP_epoch_250.pth \
    --eval mAP
```

---

## 📋 체크리스트

실험 시작 전 확인사항:

- [ ] `exp_name` 수정 (실험 구분용)
- [ ] `data_root` 경로 확인 (COCO 데이터셋)
- [ ] `samples_per_gpu` 조정 (GPU 메모리에 맞게)
- [ ] `total_epochs` 설정
- [ ] `resume_from` 설정 (재개 시)
- [ ] 디스크 용량 확인 (체크포인트 저장 공간)

---

## 💡 팁

1. **디스크 절약**: `max_keep_ckpts=3` (최근 3개만 유지)
2. **빠른 실험**: `total_epochs=100`, `evaluation.interval=5`
3. **안정성**: `grad_clip` 활성화됨 (기본값: max_norm=1.0)
4. **모니터링**: TensorBoard 활성화 권장
5. **재현성**: 실험마다 고유한 `exp_name` 사용

---

## 🐛 문제 해결

### Loss 발산
```python
# Gradient clipping 확인
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# Loss weight 감소
loss_weight=1e-6  # 더 작게
```

### OOM (Out of Memory)
```python
# Batch size 감소
samples_per_gpu=32  # 64 → 32

# Workers 감소
workers_per_gpu=1   # 2 → 1
```

### 느린 학습
```python
# Workers 증가 (데이터 로딩 병렬화)
workers_per_gpu=4   # 2 → 4
```


