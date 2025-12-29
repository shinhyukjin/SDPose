# SDPose Configuration & Documentation

> 📚 SDPose 실험 설정 및 사용 가이드 모음

---

## 📖 문서 목록

### 🚀 [QUICK_START.md](QUICK_START.md) - **여기서 시작하세요!**
- 5분 안에 실험 시작하기
- 날짜 + 실험 내용으로 간단히 설정
- 실제 사용 예시와 시나리오

**추천 대상**: 처음 사용하는 사람, 빠르게 실험 시작하고 싶은 사람

---

### 🎯 [CHEAT_SHEET.md](CHEAT_SHEET.md) - **자주 참고하세요!**
- 한 페이지 빠른 참조 가이드
- 가장 자주 수정하는 설정들
- 명령어 모음
- 문제 해결 빠른 참조

**추천 대상**: 이미 사용해본 사람, 빠른 참조가 필요한 사람

---

### 📘 [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - **상세한 설명**
- 전체 설정 항목 상세 설명
- 체크포인트 관리
- 데이터셋 설정
- 하이퍼파라미터 튜닝
- 로깅 및 평가
- 문제 해결

**추천 대상**: 설정을 깊이 이해하고 싶은 사람, 고급 튜닝

---

### 🔬 [EXAMPLE_EXPERIMENTS.md](EXAMPLE_EXPERIMENTS.md) - **실험 아이디어**
- 11가지 실험 시나리오
- Baseline, MaskedKD 변형
- Masking 비율 테스트
- Loss weight 조정
- Fine-tuning 예제

**추천 대상**: 다양한 실험을 시도하고 싶은 사람, 연구자

---

## ⚡ 5분 빠른 시작

### 1. Config 파일 열기
```bash
vi configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

### 2. 두 줄만 수정!
```python
date = '1024'              # ← 오늘 날짜 (MMDD)
exp_description = 'test'   # ← 실험 내용 간단히
```

### 3. 학습 시작
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

**끝!** 🎉

결과는 `./work_dirs/sdpose_s_v1_1024_test/`에 저장됩니다.

---

## 📂 파일 구조

```
configs/body/2d_kpt_sview_rgb_img/sdpose/coco/
├── sdpose_s_v1_stemnet_coco_256x192.py  ← 메인 설정 파일
│
├── README.md                  ← 이 파일 (문서 인덱스)
├── QUICK_START.md             ← 빠른 시작 가이드
├── CHEAT_SHEET.md             ← 한 페이지 참조
├── CONFIG_GUIDE.md            ← 상세 설정 가이드
└── EXAMPLE_EXPERIMENTS.md     ← 실험 예제 모음
```

---

## 🎯 상황별 추천 문서

### "처음 사용해요"
1. [QUICK_START.md](QUICK_START.md) 읽기
2. Config 파일에서 `date`와 `exp_description` 수정
3. 학습 시작!

### "빠르게 실험하고 싶어요"
1. [CHEAT_SHEET.md](CHEAT_SHEET.md) 열어두기
2. 필요한 부분만 복사/붙여넣기
3. 실행!

### "설정을 자세히 알고 싶어요"
1. [CONFIG_GUIDE.md](CONFIG_GUIDE.md) 정독
2. 필요한 설정 조정
3. 실험!

### "어떤 실험을 해야 할지 모르겠어요"
1. [EXAMPLE_EXPERIMENTS.md](EXAMPLE_EXPERIMENTS.md) 참고
2. 마음에 드는 실험 선택
3. 설정 복사해서 사용!

### "문제가 생겼어요"
1. [CHEAT_SHEET.md](CHEAT_SHEET.md) → 문제 해결 섹션
2. [CONFIG_GUIDE.md](CONFIG_GUIDE.md) → 문제 해결 가이드

---

## 💡 주요 기능

### ✨ 간단한 실험 관리
```python
# 날짜와 내용만 입력!
date = '1024'
exp_description = 'maskedkd_30'

# 자동 생성: sdpose_s_v1_1024_maskedkd_30
```

### 💾 자동 체크포인트 관리
- 최근 3개 체크포인트 자동 유지
- `latest.pth` 항상 저장 (재개용)
- `best_AP_epoch_XXX.pth` 최고 성능 저장

### 🔒 안전장치
- Gradient clipping (발산 방지)
- 안정적인 초기 하이퍼파라미터
- MaskedKD 버그 수정 완료

### 📊 유연한 설정
- Batch size, Learning rate 쉽게 조정
- MaskedKD 파라미터 실험 가능
- Fine-tuning, Resume 지원

---

## 🔧 Config 파일 주요 섹션

### 1. 실험 설정 (가장 자주 수정)
```python
date = '1024'
exp_description = 'baseline'
data_root = '/dockerdata/coco/'
```

### 2. 학습 설정
```python
samples_per_gpu = 64
total_epochs = 300
lr = 1e-3
```

### 3. MaskedKD 설정
```python
mask_ratio = 0.3
loss_weight = 1e-5
mask_strategy = 'random'
```

### 4. 체크포인트 설정
```python
checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=3,
)
```

---

## 📈 실험 워크플로우

```
1. Config 수정
   ↓
2. 학습 시작
   ↓
3. 로그 확인
   ↓
4. 결과 분석
   ↓
5. 다음 실험 계획
   ↓
반복!
```

---

## 🎓 학습 명령어

### 단일 GPU
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
```

### 멀티 GPU (8 GPUs)
```bash
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py 8
```

### 학습 재개
```bash
python tools/train.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py
# Config에서 resume_from 설정 필요
```

### 평가
```bash
python tools/test.py configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v1_stemnet_coco_256x192.py \
    work_dirs/sdpose_s_v1_1024_baseline/best_AP_epoch_250.pth \
    --eval mAP
```

---

## 🆘 도움이 필요하신가요?

### 빠른 참조
- [CHEAT_SHEET.md](CHEAT_SHEET.md)에서 자주 쓰는 명령어와 설정 확인

### 문제 해결
1. [CHEAT_SHEET.md](CHEAT_SHEET.md) - 문제 해결 빠른 참조
2. [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - 상세 문제 해결 가이드

### 실험 아이디어
- [EXAMPLE_EXPERIMENTS.md](EXAMPLE_EXPERIMENTS.md)에서 11가지 실험 예제 참고

---

## 📝 실험 기록 템플릿

```markdown
## 실험: sdpose_s_v1_1024_maskedkd_30

- **날짜**: 2024-10-24
- **목적**: MaskedKD with 30% masking 효과 검증
- **설정**:
  - mask_ratio: 0.3
  - loss_weight: 1e-5
  - mask_strategy: random
- **결과**:
  - AP: 72.8 (+0.5 from baseline)
  - AP50: 91.0
  - AP75: 80.5
- **비고**: Baseline 대비 성능 향상 확인
```

---

## 🎉 시작하세요!

1. **빠르게 시작**: [QUICK_START.md](QUICK_START.md) 읽기
2. **Config 수정**: `date`와 `exp_description` 수정
3. **학습 시작**: 명령어 실행
4. **결과 확인**: `work_dirs/` 폴더 확인

**행운을 빕니다!** 🚀

---

## 📚 추가 리소스

- **SDPose 논문**: [arXiv 2404.03518](https://arxiv.org/abs/2404.03518)
- **MMPose 문서**: [mmpose.readthedocs.io](https://mmpose.readthedocs.io/)
- **COCO 데이터셋**: [cocodataset.org](https://cocodataset.org/)

---

**마지막 업데이트**: 2024-10-24


















