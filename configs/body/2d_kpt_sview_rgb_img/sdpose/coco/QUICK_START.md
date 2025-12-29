# 🚀 빠른 시작 가이드

## 1️⃣ 새로운 실험 시작하기

### Step 1: Config 파일 열기
```bash
configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

### Step 2: 날짜와 실험 내용만 수정!
```python
# 날짜 (MMDD 형식)
date = '1024'  # 10월 24일

# 실험 내용 (간단히)
exp_description = 'maskedkd_30'

# 자동 생성: sdpose_s_v1_1024_maskedkd_30
```

### Step 3: 학습 시작
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

**결과 저장 위치**:
```
./work_dirs/sdpose_s_v1_1024_maskedkd_30/
```

---

## 📝 실험 이름 예시

### 날짜별 실험
```python
date = '1024'
exp_description = 'baseline'
# → sdpose_s_v1_1024_baseline

date = '1024'
exp_description = 'maskedkd_test'
# → sdpose_s_v1_1024_maskedkd_test

date = '1025'
exp_description = 'importance_mask'
# → sdpose_s_v1_1025_importance_mask
```

### 실험 타입별
```python
# Baseline
exp_description = 'baseline'

# MaskedKD 변형
exp_description = 'maskedkd_30'      # 30% masking
exp_description = 'maskedkd_50'      # 50% masking
exp_description = 'importance_mask'  # Importance masking

# Loss weight 조정
exp_description = 'strong_distil'    # Strong distillation
exp_description = 'weak_distil'      # Weak distillation

# Fine-tuning
exp_description = 'finetune_v2'      # Fine-tuning

# 테스트
exp_description = 'quick_test'       # Quick test
exp_description = 'debug'            # Debug run
```

---

## 🔄 학습 재개하기

### Step 1: Config 수정
```python
date = '1024'
exp_description = 'maskedkd_30'

# 재개할 체크포인트 지정
resume_from = './work_dirs/sdpose_s_v1_1024_maskedkd_30/latest.pth'
```

### Step 2: 학습 재개
```bash
python tools/train.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py
```

---

## 📊 여러 실험 동시 진행

### 실험 1: Baseline
```python
date = '1024'
exp_description = 'baseline'
# 다른 설정들...
```
**저장**: `./work_dirs/sdpose_s_v1_1024_baseline/`

### 실험 2: MaskedKD 30%
```python
date = '1024'
exp_description = 'maskedkd_30'
mask_ratio = 0.3
# 다른 설정들...
```
**저장**: `./work_dirs/sdpose_s_v1_1024_maskedkd_30/`

### 실험 3: MaskedKD 50%
```python
date = '1024'
exp_description = 'maskedkd_50'
mask_ratio = 0.5
# 다른 설정들...
```
**저장**: `./work_dirs/sdpose_s_v1_1024_maskedkd_50/`

**Tip**: Config 파일을 복사해서 여러 개 만들어도 됩니다!
```bash
cp sdpose_s_v1_stemnet_coco_256x192.py sdpose_baseline.py
cp sdpose_s_v1_stemnet_coco_256x192.py sdpose_maskedkd_30.py
cp sdpose_s_v1_stemnet_coco_256x192.py sdpose_maskedkd_50.py
```

---

## 🎯 일반적인 실험 패턴

### 패턴 1: 날짜 + 설명
```python
date = '1024'
exp_description = 'test_idea_1'
# → sdpose_s_v1_1024_test_idea_1
```

### 패턴 2: 날짜 + 버전
```python
date = '1024'
exp_description = 'v1'
# → sdpose_s_v1_1024_v1

date = '1024'
exp_description = 'v2'
# → sdpose_s_v1_1024_v2
```

### 패턴 3: 날짜 + 파라미터
```python
date = '1024'
exp_description = 'lr1e3_bs64'  # lr=1e-3, batch_size=64
# → sdpose_s_v1_1024_lr1e3_bs64

date = '1024'
exp_description = 'mask30_weight1e5'  # mask_ratio=0.3, weight=1e-5
# → sdpose_s_v1_1024_mask30_weight1e5
```

---

## 💡 네이밍 컨벤션 권장사항

### ✅ 좋은 이름
```python
exp_description = 'maskedkd_30'         # 명확함
exp_description = 'baseline_v2'         # 버전 표시
exp_description = 'finetune_lr1e4'      # 주요 파라미터 표시
exp_description = 'importance_mask'     # 방법 명시
```

### ❌ 피해야 할 이름
```python
exp_description = 'test'                # 너무 일반적
exp_description = 'exp1'                # 내용 불명확
exp_description = 'asdfasdf'            # 의미 없음
exp_description = 'this_is_a_very_long_experiment_name_with_too_many_details'  # 너무 김
```

### 📏 적당한 길이
- **권장**: 10-20자
- **간결하지만 의미 있게**
- **특수문자 피하기** (언더스코어 `_` 는 OK)

---

## 📁 결과 폴더 구조

```
work_dirs/
├── sdpose_s_v1_1024_baseline/
│   ├── 20251024_153000.log
│   ├── latest.pth
│   ├── best_AP_epoch_250.pth
│   ├── epoch_290.pth
│   └── epoch_300.pth
│
├── sdpose_s_v1_1024_maskedkd_30/
│   ├── 20251024_180000.log
│   ├── latest.pth
│   └── ...
│
└── sdpose_s_v1_1025_finetune/
    ├── 20251025_090000.log
    ├── latest.pth
    └── ...
```

---

## 🔍 실험 결과 찾기

### 최근 실험 찾기
```bash
# 최근 수정된 폴더 확인
ls -lt work_dirs/

# 특정 날짜 실험 찾기
ls work_dirs/ | grep 1024
```

### 최고 성능 모델 찾기
```bash
# best 모델 찾기
find work_dirs/ -name "best_AP*"

# 특정 실험의 best 모델
ls work_dirs/sdpose_s_v1_1024_maskedkd_30/best_AP*
```

### 로그 확인
```bash
# 최근 로그 보기
tail -f work_dirs/sdpose_s_v1_1024_maskedkd_30/*.log

# 최종 결과 확인
grep "Epoch \[300\]" work_dirs/sdpose_s_v1_1024_maskedkd_30/*.log
```

---

## 🎓 실전 사용 예시

### 시나리오 1: 오늘 여러 실험 돌리기

#### 실험 1 (오전)
```python
date = '1024'
exp_description = 'baseline'
```
```bash
python tools/train.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py
```

#### 실험 2 (오후)
```python
date = '1024'
exp_description = 'maskedkd_30'
mask_ratio = 0.3
```
```bash
python tools/train.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py
```

#### 실험 3 (저녁)
```python
date = '1024'
exp_description = 'maskedkd_50'
mask_ratio = 0.5
```
```bash
python tools/train.py configs/.../sdpose_s_v1_stemnet_coco_256x192.py
```

---

### 시나리오 2: 지난 실험 재개

```python
# Config 파일
date = '1023'  # 어제 날짜
exp_description = 'maskedkd_30'
resume_from = './work_dirs/sdpose_s_v1_1023_maskedkd_30/epoch_200.pth'

# 추가 학습
total_epochs = 350  # 300 → 350 (50 epoch 더)
```

---

### 시나리오 3: Fine-tuning

```python
# Config 파일
date = '1024'
exp_description = 'finetune'

# Baseline 모델 로드
load_from = './work_dirs/sdpose_s_v1_1023_baseline/best_AP_epoch_280.pth'

# Fine-tuning 설정
total_epochs = 50
lr = 1e-4  # 낮은 learning rate
```

---

## ✅ 체크리스트

실험 시작 전 확인:

- [ ] `date` 설정 (오늘 날짜)
- [ ] `exp_description` 작성 (의미있게)
- [ ] `data_root` 경로 확인
- [ ] `samples_per_gpu` 조정 (GPU 메모리)
- [ ] 디스크 공간 확인
- [ ] 이전 실험과 이름 겹치지 않는지 확인

---

## 🚨 자주 하는 실수

### ❌ 실수 1: 같은 이름으로 여러 실험
```python
# 실험 1
date = '1024'
exp_description = 'test'  # ❌

# 실험 2 (덮어씌워짐!)
date = '1024'
exp_description = 'test'  # ❌
```

**해결**: 구체적인 이름 사용
```python
date = '1024'
exp_description = 'test_baseline'  # ✅

date = '1024'
exp_description = 'test_maskedkd'  # ✅
```

---

### ❌ 실수 2: 날짜 안 바꿈
```python
# 매일 날짜 업데이트!
date = '1024'  # 오늘 날짜로!
```

---

### ❌ 실수 3: resume_from 경로 틀림
```python
# 경로 확인!
resume_from = './work_dirs/sdpose_s_v1_1024_maskedkd_30/latest.pth'  # ✅
resume_from = './work_dirs/wrong_name/latest.pth'  # ❌ (파일 없음)
```

---

## 💪 이제 시작하세요!

1. Config 파일 열기
2. `date`와 `exp_description` 수정
3. `python tools/train.py ...` 실행
4. 끝! 🎉

**더 자세한 내용**: `CONFIG_GUIDE.md`, `EXAMPLE_EXPERIMENTS.md` 참고


