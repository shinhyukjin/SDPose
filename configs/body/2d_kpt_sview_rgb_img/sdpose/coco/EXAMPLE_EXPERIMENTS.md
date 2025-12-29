# 실험 예제 모음

다양한 실험 시나리오별 설정 예제입니다. 필요한 부분을 복사해서 config 파일에 붙여넣으세요.

---

## 🎯 실험 1: 기본 학습 (Baseline)

```python
# 실험 이름
exp_name = 'sdpose_s_v1_baseline_1024'

# 데이터셋
data_root = '/dockerdata/coco/'

# 하이퍼파라미터
samples_per_gpu = 64
total_epochs = 300

# MaskedKD 비활성화 (기본 SDPose만 사용)
loss_vis_token_dist = None
loss_kpt_token_dist = None
```

**용도**: 기본 성능 측정, 비교 baseline

---

## 🔬 실험 2: MaskedKD with 30% Masking

```python
# 실험 이름
exp_name = 'sdpose_s_v1_maskedkd_30_1024'

# MaskedKD 설정
loss_vis_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,
    mask_ratio=0.3,  # 30% 마스킹
    mask_strategy='random'
),
loss_kpt_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,
    mask_ratio=0.3,
    mask_strategy='random'
),
```

**용도**: 논문 제안 방법, 기본 MaskedKD

---

## 🎲 실험 3: 다양한 Masking 비율 테스트

### 3-1: 20% Masking (Light)
```python
exp_name = 'sdpose_s_v1_maskedkd_20_1024'
mask_ratio = 0.2  # 가벼운 마스킹
loss_weight = 1e-5
```

### 3-2: 50% Masking (Heavy)
```python
exp_name = 'sdpose_s_v1_maskedkd_50_1024'
mask_ratio = 0.5  # 강한 마스킹
loss_weight = 5e-6  # weight 감소 권장
```

### 3-3: 70% Masking (Extreme)
```python
exp_name = 'sdpose_s_v1_maskedkd_70_1024'
mask_ratio = 0.7  # 극단적 마스킹
loss_weight = 1e-6  # weight 크게 감소
```

**분석**: 어떤 마스킹 비율이 최적인지 비교

---

## 🧠 실험 4: Importance-based Masking

```python
exp_name = 'sdpose_s_v1_maskedkd_importance_1024'

loss_vis_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,
    mask_ratio=0.3,
    mask_strategy='importance'  # 중요도 기반
),
loss_kpt_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,
    mask_ratio=0.3,
    mask_strategy='importance'
),
```

**가설**: 덜 중요한 토큰을 마스킹하면 더 효과적

---

## ⚖️ 실험 5: Loss Weight 조정

### 5-1: Strong Distillation
```python
exp_name = 'sdpose_s_v1_maskedkd_strong_1024'
loss_weight = 1e-4  # 10배 증가
mask_ratio = 0.3
```

### 5-2: Weak Distillation
```python
exp_name = 'sdpose_s_v1_maskedkd_weak_1024'
loss_weight = 1e-6  # 10배 감소
mask_ratio = 0.3
```

### 5-3: Asymmetric Weights
```python
exp_name = 'sdpose_s_v1_maskedkd_asym_1024'

loss_vis_token_dist=dict(
    loss_weight=1e-5,  # Visual token
    # ...
),
loss_kpt_token_dist=dict(
    loss_weight=5e-5,  # Keypoint token (5배 강함)
    # ...
),
```

**분석**: Distillation 강도가 성능에 미치는 영향

---

## 🔄 실험 6: Cycle 수 조정

```python
exp_name = 'sdpose_s_v1_maskedkd_cycle3_1024'

tokenpose_cfg=dict(
    # ... 기존 설정 ...
    cycle_num=3  # 기본 2 → 3
)
```

**Trade-off**: 
- Cycle 증가 → 성능 향상 가능, 학습 시간 증가
- Cycle 감소 → 빠른 학습, 성능 저하 가능

---

## 📏 실험 7: 다양한 Batch Size

### 7-1: Large Batch (더 안정적)
```python
exp_name = 'sdpose_s_v1_maskedkd_bs128_1024'
samples_per_gpu = 128
lr = 2e-3  # Batch size 2배 → LR도 2배
```

### 7-2: Small Batch (GPU 메모리 부족 시)
```python
exp_name = 'sdpose_s_v1_maskedkd_bs32_1024'
samples_per_gpu = 32
lr = 5e-4  # Batch size 1/2 → LR도 1/2
```

---

## ⏱️ 실험 8: 빠른 프로토타입 (Short Training)

```python
exp_name = 'sdpose_s_v1_maskedkd_quick_test_1024'

# 짧은 학습
total_epochs = 100  # 300 → 100
lr_config = dict(
    step=[70, 90]  # 조기 LR 감소
)

# 빈번한 평가
evaluation = dict(interval=5)  # 5 epoch마다
checkpoint_config = dict(interval=5)

# 적은 체크포인트 유지
max_keep_ckpts = 2
```

**용도**: 아이디어 빠른 검증, 디버깅

---

## 🔥 실험 9: Fine-tuning from Pre-trained

```python
exp_name = 'sdpose_s_v1_maskedkd_finetune_1024'

# Pre-trained 모델 로드
load_from = './work_dirs/sdpose_s_v1_baseline_1024/best_AP_epoch_250.pth'

# Fine-tuning 설정
total_epochs = 50  # 짧은 학습
optimizer = dict(
    type='Adam',
    lr=1e-4,  # 낮은 learning rate
)
lr_config = dict(
    step=[30, 45]
)
```

**용도**: Baseline에서 시작하여 MaskedKD 추가 효과 측정

---

## 📊 실험 10: 앙상블을 위한 Multiple Runs

```python
# Run 1
exp_name = 'sdpose_s_v1_maskedkd_seed1_1024'
# random seed 설정 (mmcv runner 옵션)

# Run 2
exp_name = 'sdpose_s_v1_maskedkd_seed2_1024'

# Run 3
exp_name = 'sdpose_s_v1_maskedkd_seed3_1024'
```

**분석**: 
- 평균 성능 및 표준편차 계산
- 앙상블 모델 구성

---

## 🎨 실험 11: 혼합 전략

```python
exp_name = 'sdpose_s_v1_maskedkd_mixed_1024'

# Visual token: Random masking
loss_vis_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=1e-5,
    mask_ratio=0.3,
    mask_strategy='random'
),

# Keypoint token: Importance masking
loss_kpt_token_dist=dict(
    type='MaskedTokenDistilLoss', 
    loss_weight=2e-5,  # 더 강하게
    mask_ratio=0.3,
    mask_strategy='importance'  # 다른 전략
),
```

**가설**: 다른 토큰 타입에 다른 전략이 효과적

---

## 📈 실험 결과 기록 템플릿

```markdown
## 실험 결과

| Exp Name | Mask Ratio | Loss Weight | Strategy | AP | AP50 | AP75 | Notes |
|----------|------------|-------------|----------|-----|------|------|-------|
| baseline | - | - | - | 72.3 | 90.5 | 80.1 | Baseline |
| masked_30 | 0.3 | 1e-5 | random | 72.8 | 91.0 | 80.5 | +0.5 AP |
| masked_50 | 0.5 | 1e-5 | random | 72.6 | 90.8 | 80.3 | 마스킹 과다 |
| importance | 0.3 | 1e-5 | importance | 73.1 | 91.2 | 80.8 | Best! |
```

---

## 💡 실험 팁

1. **한 번에 하나씩**: 여러 변수를 동시에 바꾸지 말 것
2. **Baseline 먼저**: 항상 baseline과 비교
3. **여러 번 실행**: 중요한 실험은 3회 이상 반복
4. **로그 저장**: 모든 실험의 config와 결과를 기록
5. **디스크 관리**: 불필요한 체크포인트 정리

---

## 🚀 배치 실험 스크립트 예제

```bash
#!/bin/bash
# run_experiments.sh

# 실험 1: Baseline
python tools/train.py configs/.../sdpose_baseline.py

# 실험 2: MaskedKD 20%
python tools/train.py configs/.../sdpose_masked_20.py

# 실험 3: MaskedKD 30%
python tools/train.py configs/.../sdpose_masked_30.py

# 실험 4: MaskedKD 50%
python tools/train.py configs/.../sdpose_masked_50.py
```

---

**행운을 빕니다! 🎉**


