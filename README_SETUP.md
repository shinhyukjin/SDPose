# SDPose 환경 설정 가이드

이 문서는 SDPose 실험 환경을 설정하는 방법을 설명합니다.

## 빠른 시작

전체 설정을 한 번에 수행하려면:

```bash
bash setup_complete.sh
```

## 단계별 설정

### 1. 환경 설정만 수행

```bash
bash setup_environment.sh
```

이 스크립트는 다음 작업을 수행합니다:
- Miniconda 설치 (이미 설치되어 있으면 스킵)
- Python 3.10 conda 환경 생성 (`sdpose`)
- PyTorch 2.5.1 (CUDA 12.1) 설치
- 필수 패키지 설치 (numpy, opencv-python, mmcv-full, 등)
- SDPose 프로젝트 설정

### 2. COCO 데이터셋 다운로드

```bash
bash download_coco_dataset.sh
```

이 스크립트는 다음 작업을 수행합니다:
- COCO annotations 다운로드
- COCO training images 다운로드 (~118k images, ~19GB)
- COCO validation images 다운로드 (~5k images)
- 디렉토리 구조 정리
- 데이터셋 검증

**주의사항:**
- 데이터셋 다운로드는 시간이 오래 걸릴 수 있습니다 (네트워크 속도에 따라)
- 디스크 공간이 충분한지 확인하세요 (최소 30GB 이상)

## 수동 설정

스크립트를 사용하지 않고 수동으로 설정하려면:

### 1. Conda 환경 생성

```bash
conda create -n sdpose python=3.10 -y
conda activate sdpose
```

### 2. PyTorch 설치

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 기본 패키지 설치

```bash
pip install numpy opencv-python matplotlib scipy scikit-image Pillow tqdm pyyaml easydict einops timm
pip install numpy==1.23.5
pip install xtcocotools
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0.0/index.html
pip install yapf
```

### 4. COCO 데이터셋 다운로드

```bash
DATA_ROOT="/dockerdata/coco"
mkdir -p ${DATA_ROOT}/{annotations,train2017,val2017,person_detection_results}

# Annotations
cd ${DATA_ROOT}/annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
cp annotations/person_keypoints_*.json .
rm -rf annotations annotations_trainval2017.zip

# Training images
cd ${DATA_ROOT}
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

# Validation images
cd ${DATA_ROOT}
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# Detection results (선택사항)
# 실제 파일은 Google Drive나 OneDrive에서 다운로드 필요
cd ${DATA_ROOT}/person_detection_results
echo "[]" > COCO_val2017_detections_AP_H_56_person.json
```

## 환경 활성화

학습을 시작하기 전에 환경을 활성화하세요:

```bash
source ~/miniconda3/bin/activate
conda activate sdpose
cd /root/SDPose
export PYTHONPATH=/root/SDPose:$PYTHONPATH
```

## 학습 시작

분산 학습 예제:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    --use-env \
    tools/train.py \
    configs/body/2d_kpt_sview_rgb_img/sdpose/coco/your_config.py \
    --work-dir work_dirs/your_experiment \
    --launcher pytorch
```

## 문제 해결

### 1. ModuleNotFoundError

Python 캐시 문제일 수 있습니다:

```bash
cd /root/SDPose
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

### 2. CUDA out of memory

`configs` 파일에서 `samples_per_gpu`를 줄이거나 GPU 수를 늘리세요.

### 3. 데이터셋 경로 오류

설정 파일에서 `data_root` 경로를 확인하세요. 기본값은 `/dockerdata/coco/`입니다.

### 4. Detection results 파일 없음

`use_gt_bbox=True`로 설정하여 GT bbox를 사용할 수 있습니다:

```python
data_cfg = dict(
    ...
    use_gt_bbox=True,
    bbox_file='',
)
```

## 필요한 디스크 공간

- COCO 데이터셋: ~30GB
- 학습 체크포인트: 실험마다 ~2-5GB
- Python 패키지: ~5GB

**총 권장 디스크 공간: 50GB 이상**






