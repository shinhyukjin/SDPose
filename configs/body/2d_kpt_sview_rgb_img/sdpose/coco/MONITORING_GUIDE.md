# 학습 모니터링 가이드

## 🎯 주요 변경사항

### 1. **원본 SDPose로 복원**
- ❌ 제거: Token/Heatmap Clamping (학습 방해)
- ❌ 제거: MaskedTokenDistilLoss (self-distillation과 부적합)
- ✅ 복원: 원본 TokenDistilLoss (loss_weight=5e-6)

### 2. **모니터링 강화**
- ✅ Evaluation interval: 10 → 5 epoch (더 자주 체크)
- ✅ TrainingMonitorHook: Loss spike, 성능 하락 자동 감지
- ✅ DetailedLossLogHook: 모든 loss 항목 상세 로깅

---

## 🚀 빠른 시작

### 1. **학습 전 체크 (권장)**
```bash
# 몇 iteration만 실행하여 정상 작동 확인
python tools/quick_check.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py --iterations 10
```

**확인사항:**
- ✅ Forward pass 정상 작동
- ✅ Loss 값이 NaN/Inf가 아님
- ✅ 모든 loss 항목이 계산됨

### 2. **정식 학습 시작**
```bash
# 단일 GPU
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py

# 멀티 GPU (예: 4 GPUs)
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py 4
```

---

## 📊 로그 모니터링

### **자동 감지 기능**

#### 1. **Loss Spike 감지**
```
⚠️  WARNING: Loss spike detected!
   Current: 5.2341
   Recent avg: 2.1234
   Ratio: 2.47x
```
**의미:** Loss가 갑자기 2배 이상 증가
**조치:** 
- Learning rate가 너무 높은지 확인
- Gradient clipping 확인 (현재: max_norm=5.0)
- 데이터 이상 확인

#### 2. **성능 하락 감지**
```
⚠️  WARNING: Performance drop detected!
   Current AP: 0.650
   Best AP: 0.720
   Drop: 0.070
```
**의미:** AP가 이전 최고치보다 0.05 이상 하락
**조치:**
- Overfitting 가능성
- Learning rate schedule 확인
- 이전 checkpoint로 복원 고려

#### 3. **상세 Loss 로깅**
```
Iter [50] loss: 2.345678, heatmap_loss: 2.100000, vis_dist_loss: 0.000123, kpt_dist_loss: 0.000234
Iter [100] loss: 2.123456, heatmap_loss: 1.950000, vis_dist_loss: 0.000087, kpt_dist_loss: 0.000165
```
**확인사항:**
- `heatmap_loss`: 메인 loss, 점진적으로 감소해야 함
- `vis_dist_loss`: ~1e-4 수준 (5e-6 weight)
- `kpt_dist_loss`: ~1e-4 수준 (5e-6 weight)

---

## 📈 정상 학습 패턴

### **Epoch별 예상 성능**

| Epoch | Loss | AP | 비고 |
|-------|------|-----|------|
| 1-10 | 3.0 → 1.5 | 0.1 → 0.3 | 초기 학습 |
| 10-50 | 1.5 → 0.8 | 0.3 → 0.55 | 빠른 개선 |
| 50-150 | 0.8 → 0.5 | 0.55 → 0.68 | 안정적 개선 |
| 150-250 | 0.5 → 0.4 | 0.68 → 0.72 | 느린 개선 |
| 250-300 | 0.4 → 0.38 | 0.72 → 0.73 | 수렴 |

### **예상 최종 성능**
- **AP**: ~0.730 (COCO validation)
- **Params**: ~9M
- **GFLOPs**: ~4.5

---

## 🔍 문제 진단

### **Case 1: Loss가 감소하지 않음**
```
Epoch 10: Loss = 3.2
Epoch 20: Loss = 3.1
Epoch 30: Loss = 3.0
```
**가능한 원인:**
1. Learning rate가 너무 작음 → config에서 증가 시도 (1e-3 → 3e-3)
2. Batch size가 너무 작음 → 최소 32 이상 권장
3. 데이터 augmentation이 너무 강함 → 줄여보기

### **Case 2: AP가 0.0000**
```
Epoch 100: AP = 0.0000
```
**가능한 원인:**
1. ❌ Heatmap이 제대로 생성되지 않음
2. ❌ 모델 출력이 이상함 (NaN, Inf, 극단값)

**즉시 조치:**
```bash
# Quick check 실행
python tools/quick_check.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py

# 로그에서 NaN/Inf 확인
grep -i "nan\|inf" work_dirs/*/latest.log
```

### **Case 3: Loss가 급증 후 발산**
```
Epoch 45: Loss = 0.8
Epoch 46: Loss = 2.5
Epoch 47: Loss = NaN
```
**가능한 원인:**
1. Gradient explosion → Gradient clipping 강화 (5.0 → 3.0)
2. Learning rate가 너무 높음 → 감소 또는 warmup 추가
3. Batch size 변경 시 발생 → Learning rate도 함께 조정

---

## 💾 체크포인트 관리

### **현재 설정**
```python
checkpoint_config = dict(
    interval=10,           # 10 epoch마다 저장
    max_keep_ckpts=3,     # 최근 3개만 유지
    save_last=True,       # latest.pth는 항상 유지
)
```

### **복원 방법**
```bash
# 특정 epoch부터 재시작
python tools/train.py \
    configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py \
    --resume-from work_dirs/sdpose_s_v1_1024_baseline/epoch_100.pth

# Best model로 재시작
python tools/train.py \
    configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py \
    --resume-from work_dirs/sdpose_s_v1_1024_baseline/best_AP_epoch_150.pth
```

---

## 📝 통계 파일

학습 완료 후 자동 생성되는 파일:
```
work_dirs/sdpose_s_v1_1024_baseline/
├── training_stats.txt       # 전체 학습 통계
├── *.log                     # 학습 로그
└── *.log.json               # JSON 형식 로그
```

### **training_stats.txt 예시**
```
Training Statistics
================================================================================

Loss Statistics:
  - Mean: 0.8543
  - Std: 0.2341
  - Min: 0.3821
  - Max: 3.2145
  - Final: 0.3912

Best AP: 0.7234
```

---

## 🎓 팁

### **1. 학습 초반 (Epoch 1-50)**
- Loss가 빠르게 감소해야 함
- AP가 0.5 이상 도달해야 함
- 이 시기에 문제가 있으면 config 점검 필요

### **2. 학습 중반 (Epoch 50-150)**
- 안정적인 개선 기대
- Loss spike가 발생하면 learning rate schedule 확인
- AP 정체 시 augmentation 강화 고려

### **3. 학습 후반 (Epoch 150-300)**
- 느리지만 꾸준한 개선
- Overfitting 주의 (validation AP 하락 시)
- Early stopping 고려 가능

### **4. 실험 관리**
```python
# Config에서 실험 이름 변경
date = '1027'
exp_description = 'original_sdpose'  # 실험 내용
exp_name = f'sdpose_s_v1_{date}_{exp_description}'
# → work_dirs/sdpose_s_v1_1027_original_sdpose/
```

---

## 🆘 트러블슈팅

### **문제: CUDA out of memory**
```python
# Config 수정
data = dict(
    samples_per_gpu=64,  # → 32로 줄이기
    workers_per_gpu=2,
)
```

### **문제: 학습이 너무 느림**
```python
# 데이터 로더 워커 증가
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,  # → 4 또는 8로 증가
)
```

### **문제: Hook import 에러**
```bash
# Hook 파일 위치 확인
ls distilpose/models/detectors/training_monitor_hook.py

# mmcv 버전 확인
python -c "import mmcv; print(mmcv.__version__)"
# 최소: 1.3.0 필요
```

---

## 📞 도움이 필요하면

1. **로그 확인**: `work_dirs/*/latest.log`
2. **Quick check 실행**: 모델이 정상 작동하는지 확인
3. **통계 확인**: `training_stats.txt`에서 이상 패턴 찾기
4. **이전 checkpoint로 복원**: 문제가 생긴 시점 이전으로

