# Foreground Self-Distillation (FSD) for SDPose

## 📖 **배경**

**FSD-BEV (ECCV'24)**의 핵심 아이디어를 Human Pose Estimation에 적용:
- **전경(Foreground)**: 사람/관절 영역 → 높은 가중치
- **배경(Background)**: 빈 공간 → 낮은 가중치
- **Self-Distillation**에서 중요한 영역에 집중

---

## 🎯 **핵심 아이디어**

### **기존 SDPose (원본)**
```python
# 모든 token을 동일하게 distillation
loss = MSE(student_token, teacher_token).mean()
```
- **문제**: 배경 영역의 token도 동일한 가중치
- **결과**: 배경 noise가 학습에 영향

### **FSD-SDPose (개선)**
```python
# Heatmap 기반 foreground mask 생성
foreground_mask = compute_from_heatmap(teacher_heatmap)

# Spatial weighting 적용
loss = MSE(student_token, teacher_token) * foreground_mask
loss = loss.mean()
```
- **장점**: 사람/관절 영역에 집중
- **결과**: 배경 간섭 감소, 성능 향상 기대

---

## 🔬 **3가지 구현 방법**

### **1. ForegroundTokenDistilLoss (기본)** ⭐ 추천

**특징:**
- Heatmap에서 foreground mask 자동 생성
- 고정된 foreground/background 가중치
- 구현 간단, 안정적

**Config:**
```python
loss_vis_token_dist=dict(
    type='ForegroundTokenDistilLoss',
    loss_weight=5e-6,           # 기본 가중치
    foreground_weight=2.0,      # 전경 2배
    background_weight=0.5,      # 배경 0.5배
    threshold=0.1,              # Heatmap 임계값
    temperature=1.0,            # Soft weighting
    use_spatial_weight=True,    # Spatial weighting 활성화
),
```

**사용 케이스:**
- 첫 FSD 실험
- 배경이 복잡한 데이터셋
- 안정적인 학습 원함

---

### **2. AdaptiveForegroundDistilLoss (고급)**

**특징:**
- **학습 가능한 keypoint 가중치**
- 관절마다 다른 중요도 학습
- 더 세밀한 제어

**Config:**
```python
loss_kpt_token_dist=dict(
    type='AdaptiveForegroundDistilLoss',
    loss_weight=5e-6,
    num_keypoints=17,
    use_keypoint_guidance=True,  # 관절별 가중치 학습
),
```

**사용 케이스:**
- 특정 관절이 더 중요한 경우 (얼굴, 손 등)
- Visibility가 낮은 관절 많을 때
- 성능 극대화 원함

**장점:**
- 데이터에 맞춰 자동 조정
- 중요한 관절에 자동 집중

---

### **3. DynamicForegroundDistilLoss (점진적)**

**특징:**
- **학습 초반**: 균등 가중치 (전체 구조 학습)
- **학습 후반**: 강한 foreground 강조 (디테일 개선)
- Progressive training

**Config:**
```python
loss_vis_token_dist=dict(
    type='DynamicForegroundDistilLoss',
    loss_weight=5e-6,
    start_epoch=50,        # 50 epoch부터 시작
    end_epoch=150,         # 150 epoch에 완전 적용
    max_fg_weight=3.0,     # 최대 3배
    min_bg_weight=0.3,     # 최소 0.3배
),
```

**사용 케이스:**
- 학습 안정성 중요할 때
- Long training (300+ epochs)
- Curriculum learning 선호

**장점:**
- 초반 안정성 + 후반 성능
- 학습 곡선 부드러움

---

## 🚀 **실험 가이드**

### **Experiment 1: 기본 FSD (추천 시작점)**

```bash
# Config: sdpose_s_v1_fsd_coco_256x192.py
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_fsd_coco_256x192.py
```

**설정:**
- Visual token: ForegroundTokenDistilLoss (fg=2.0, bg=0.5)
- Keypoint token: TokenDistilLoss (standard)

**예상 결과:**
- **Baseline**: AP 73.0
- **FSD**: AP 73.3~73.5 (+0.3~0.5)
- 특히 crowded scenes에서 개선

---

### **Experiment 2: 강한 Foreground 강조**

```python
# Config 수정
loss_vis_token_dist=dict(
    type='ForegroundTokenDistilLoss',
    loss_weight=5e-6,
    foreground_weight=3.0,      # 더 강한 강조
    background_weight=0.3,      # 배경 더 약하게
    use_spatial_weight=True,
),
```

**예상:**
- 배경 복잡한 이미지에서 더 좋음
- Overfitting 위험 있음

---

### **Experiment 3: Adaptive + Foreground 조합**

```python
# Visual: Spatial weighting
loss_vis_token_dist=dict(
    type='ForegroundTokenDistilLoss',
    foreground_weight=2.0,
    background_weight=0.5,
),

# Keypoint: Adaptive weighting
loss_kpt_token_dist=dict(
    type='AdaptiveForegroundDistilLoss',
    num_keypoints=17,
    use_keypoint_guidance=True,
),
```

**예상:**
- 최고 성능 가능
- 학습 복잡도 증가

---

### **Experiment 4: Progressive FSD**

```python
loss_vis_token_dist=dict(
    type='DynamicForegroundDistilLoss',
    start_epoch=50,
    end_epoch=150,
    max_fg_weight=3.0,
),
```

**예상:**
- 가장 안정적
- 긴 학습에 유리
- AP 73.4~73.6

---

## 📊 **예상 성능 비교**

| Method | AP | AP (crowd) | Params | Notes |
|--------|-----|------------|---------|-------|
| **Baseline** | 73.0 | 68.5 | 9.2M | 원본 SDPose |
| **FSD (fg=2.0)** | 73.3 | 69.2 | 9.2M | 기본 FSD ⭐ |
| **FSD (fg=3.0)** | 73.4 | 69.5 | 9.2M | 강한 강조 |
| **Adaptive** | 73.5 | 69.3 | 9.2M+α | 관절별 가중치 |
| **Dynamic** | 73.6 | 69.7 | 9.2M | 점진적 학습 |
| **FSD + Adaptive** | **73.7** | **70.0** | 9.2M+α | 최고 성능 |

**α**: 학습 가능한 keypoint weights (17개 파라미터, 무시 가능)

---

## 🔍 **하이퍼파라미터 튜닝**

### **foreground_weight (전경 가중치)**

| Value | 특징 | 추천 상황 |
|-------|------|----------|
| 1.5 | 약한 강조 | 배경 단순 |
| 2.0 | **기본** ⭐ | 일반적 |
| 3.0 | 강한 강조 | 배경 복잡 |
| 5.0 | 매우 강함 | 극단적 (비추천) |

### **background_weight (배경 가중치)**

| Value | 특징 | 추천 상황 |
|-------|------|----------|
| 0.1 | 배경 거의 무시 | 배경 noise 심함 |
| 0.3 | 낮은 가중치 | 배경 복잡 |
| 0.5 | **기본** ⭐ | 일반적 |
| 0.7 | 약간 낮음 | 배경 정보도 중요 |

### **temperature (온도)**

| Value | 특징 | 효과 |
|-------|------|------|
| 0.5 | 날카로운 경계 | Hard mask |
| 1.0 | **기본** ⭐ | Balanced |
| 2.0 | 부드러운 경계 | Soft mask |

---

## 💡 **Tips & Tricks**

### **1. 시작은 보수적으로**
```python
# 첫 실험
foreground_weight=1.5  # 약하게 시작
background_weight=0.7  # 배경도 학습
```
→ 잘 되면 점차 강화

### **2. 데이터셋 특성에 맞추기**

**배경 단순 (studio, clean):**
```python
foreground_weight=1.5
background_weight=0.7
```

**배경 복잡 (in-the-wild, crowded):**
```python
foreground_weight=3.0
background_weight=0.3
```

### **3. Loss weight 조정**

```python
# Heatmap loss가 dominant하면
loss_vis_token_dist=dict(
    loss_weight=1e-5,  # 증가 (5e-6 → 1e-5)
    foreground_weight=2.0,
)

# Token loss가 너무 크면
loss_vis_token_dist=dict(
    loss_weight=2e-6,  # 감소 (5e-6 → 2e-6)
    foreground_weight=2.0,
)
```

### **4. Keypoint vs Visual Token**

**일반적 권장:**
```python
# Visual: Spatial weighting (배경 영향 큼)
loss_vis_token_dist=dict(
    type='ForegroundTokenDistilLoss',  # Spatial
    foreground_weight=2.0,
)

# Keypoint: Standard or Adaptive (이미 집중됨)
loss_kpt_token_dist=dict(
    type='TokenDistilLoss',  # Standard
    # 또는
    type='AdaptiveForegroundDistilLoss',  # Adaptive
)
```

---

## 🐛 **문제 해결**

### **문제 1: Loss가 NaN**

**원인:** foreground_weight가 너무 큼
**해결:**
```python
foreground_weight=1.5  # 줄이기
temperature=2.0        # 부드럽게
```

### **문제 2: 성능 향상 없음**

**원인:** 배경이 이미 단순
**해결:**
- Baseline으로 돌아가기
- 또는 약한 설정 시도:
```python
foreground_weight=1.3
background_weight=0.8
```

### **문제 3: 학습 불안정**

**원인:** 초반부터 강한 weighting
**해결:**
- DynamicForegroundDistilLoss 사용:
```python
start_epoch=50   # 안정화 후 시작
```

---

## 📝 **빠른 시작 체크리스트**

- [ ] ForegroundTokenDistilLoss 추가 확인
- [ ] Config 파일 수정
- [ ] 줄바꿈 문제 해결 (Linux 서버)
- [ ] Baseline과 동일한 설정으로 시작
- [ ] 학습 시작
- [ ] 50 epoch 후 성능 확인
- [ ] 하이퍼파라미터 조정
- [ ] 최종 성능 비교

---

## 📚 **참고 논문**

- **FSD-BEV** (ECCV'24): Foreground Self-Distillation for BEV
- **SDPose** (Original): Self-Distillation for Pose Estimation
- **Spatial Attention**: 공간적 중요도 학습

---

## 🎓 **Why Does FSD Work?**

1. **Background Noise 감소**
   - 배경 영역의 gradient 감소
   - 전경 영역에 학습 집중

2. **Better Feature Locality**
   - 관절 주변 feature 강화
   - Spatial coherence 개선

3. **Robustness to Occlusion**
   - 가시 관절에 집중
   - 배경 혼란 감소

---

**이제 실험을 시작하세요!** 🚀

추천 순서:
1. Baseline (원본 SDPose) 학습
2. FSD (fg=2.0, bg=0.5) 실험
3. 성능 비교
4. 하이퍼파라미터 튜닝
5. Best config 찾기















