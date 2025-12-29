# 🎯 실험 설정 치트시트

> 빠르게 참고할 수 있는 한 페이지 가이드

---

## 🚀 가장 자주 수정하는 3가지

```python
# 1. 날짜 + 실험 내용
date = '1024'                    # ← 오늘 날짜
exp_description = 'maskedkd_30'  # ← 실험 내용

# 2. 데이터셋 경로
data_root = '/dockerdata/coco/'  # ← 데이터 위치

# 3. 배치 사이즈
samples_per_gpu = 64             # ← GPU 메모리에 맞게
```

**이것만 바꾸면 끝!**

---

## 📝 실험 이름 작성 템플릿

```python
# 패턴 1: 날짜 + 방법
date = '1024'
exp_description = 'baseline'
# → sdpose_s_v1_1024_baseline

# 패턴 2: 날짜 + 파라미터
date = '1024'
exp_description = 'mask30_lr1e4'
# → sdpose_s_v1_1024_mask30_lr1e4

# 패턴 3: 날짜 + 버전
date = '1024'
exp_description = 'v1'
# → sdpose_s_v1_1024_v1
```

---

## 🔄 학습 재개 (Resume)

```python
date = '1024'
exp_description = 'maskedkd_30'
resume_from = f'./work_dirs/sdpose_s_v1_{date}_{exp_description}/latest.pth'
```

---

## ⚙️ 주요 하이퍼파라미터

```python
# Batch size & Workers
samples_per_gpu = 64    # 32, 64, 128
workers_per_gpu = 2     # 1, 2, 4

# Learning rate
lr = 1e-3               # 1e-4, 5e-4, 1e-3, 2e-3

# Training epochs
total_epochs = 300      # 100, 200, 300

# Checkpoint interval
checkpoint_config = dict(interval=10)  # 5, 10, 20
```

---

## 🎭 MaskedKD 설정

```python
# Masking 비율
mask_ratio = 0.3        # 0.2, 0.3, 0.5, 0.7

# Loss weight
loss_weight = 1e-5      # 1e-6, 5e-6, 1e-5, 1e-4

# Masking 전략
mask_strategy = 'random'           # or 'importance'
```

---

## 📂 결과 위치

```bash
./work_dirs/{exp_name}/
├── latest.pth              # ← 재개용
├── best_AP_epoch_XXX.pth   # ← 최고 성능
├── epoch_XXX.pth           # ← 주기적 저장
└── YYYYMMDD_HHMMSS.log     # ← 로그
```

---

## 🛠️ 자주 쓰는 명령어

```bash
# 학습 시작
python tools/train.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py

# 멀티 GPU (8개)
./tools/dist_train.sh configs/.../sdpose_s_v1_stemnet_coco_256x192.py 8

# 평가
python tools/test.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py \
    work_dirs/sdpose_s_v1_1024_baseline/best_AP_epoch_XXX.pth

# 로그 실시간 확인
tail -f work_dirs/sdpose_s_v1_1024_baseline/*.log

# 실험 폴더 찾기
ls work_dirs/ | grep 1024
```

---

## 💡 빠른 실험 설정

### Quick Test (빠르게 확인)
```python
date = '1024'
exp_description = 'quick_test'
total_epochs = 50
evaluation = dict(interval=5)
```

### Full Training (정식 학습)
```python
date = '1024'
exp_description = 'final'
total_epochs = 300
evaluation = dict(interval=10)
```

### Fine-tuning (추가 학습)
```python
date = '1024'
exp_description = 'finetune'
load_from = './work_dirs/.../best_AP_epoch_XXX.pth'
lr = 1e-4
total_epochs = 50
```

---

## 🐛 문제 해결 빠른 참조

### OOM (메모리 부족)
```python
samples_per_gpu = 32    # 64 → 32
workers_per_gpu = 1     # 2 → 1
```

### Loss 발산
```python
# Gradient clipping 확인 (기본 활성화)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# Loss weight 감소
loss_weight = 1e-6      # 1e-5 → 1e-6
```

### 느린 학습
```python
workers_per_gpu = 4     # 2 → 4
```

---

## 📊 실험 결과 확인

```bash
# 최종 AP 확인
grep "best AP" work_dirs/sdpose_s_v1_1024_baseline/*.log

# 특정 epoch 결과
grep "Epoch \[300\]" work_dirs/sdpose_s_v1_1024_baseline/*.log

# Best 모델 찾기
find work_dirs/sdpose_s_v1_1024_baseline -name "best_AP*"
```

---

## ✅ 실험 전 체크리스트

```python
# [ ] 1. 날짜 설정
date = '____'  # 오늘 날짜!

# [ ] 2. 실험 이름
exp_description = '_____'  # 의미있게!

# [ ] 3. 데이터 경로
data_root = '/____/coco/'  # 확인!

# [ ] 4. Resume 설정
resume_from = None  # or 'path/to/checkpoint.pth'

# [ ] 5. GPU 설정
samples_per_gpu = 64  # GPU 메모리 확인!
```

---

## 🎯 실험별 추천 설정

| 목적 | Epochs | Batch Size | LR | Eval Interval |
|------|--------|------------|-----|---------------|
| Quick Test | 50 | 64 | 1e-3 | 5 |
| Baseline | 300 | 64 | 1e-3 | 10 |
| MaskedKD | 300 | 64 | 1e-3 | 10 |
| Fine-tune | 50 | 32 | 1e-4 | 5 |
| Debug | 10 | 32 | 1e-3 | 1 |

---

## 🔗 더 자세한 문서

- **QUICK_START.md** - 빠른 시작 가이드
- **CONFIG_GUIDE.md** - 전체 설정 가이드  
- **EXAMPLE_EXPERIMENTS.md** - 실험 예제 모음

---

**이 페이지를 북마크하세요!** 📌


