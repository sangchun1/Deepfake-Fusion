# Spatial-Frequency Fusion for Deepfake Image Detection

CIFAKE와 고해상도 얼굴 이미지 데이터셋에서 공간 정보와 주파수 정보를 함께 활용하는 fusion 기반 딥페이크 이미지 탐지 프로젝트입니다.  
이 프로젝트는 단순한 백본 성능 비교를 넘어서, spatial-only, frequency-only, fusion 구조를 비교하여 주파수 정보가 실제로 탐지에 얼마나 기여하는지를 분석하는 것을 목표로 합니다.

---

## 1. Overview

최근 생성형 AI의 발전으로 실제 이미지와 AI 생성 이미지의 구분이 점점 어려워지고 있습니다.  
본 프로젝트는 이미지 한 장을 입력으로 받아 해당 이미지가 real인지 fake인지 판별하는 이진 분류 문제를 다룹니다.

핵심 아이디어는 다음과 같습니다.

- 공간 도메인 정보: 원본 이미지의 픽셀 기반 시각 특징 사용
- 주파수 도메인 정보: FFT magnitude 또는 DCT 기반 스펙트럼 특징 사용
- fusion 구조: 공간 특징과 주파수 특징을 결합하여 최종 분류

즉, 같은 이미지를 두 가지 관점에서 보고 판단하는 탐지기를 설계하고, 그 효과를 정량적 성능과 시각화 결과를 통해 검증합니다.

---

## 2. Objectives

이 프로젝트의 주요 목표는 다음과 같습니다.

1. spatial-only, frequency-only, fusion 모델의 성능 비교
2. 저해상도 데이터셋과 고해상도 데이터셋에서 주파수 정보의 효과 검증
3. Grad-CAM 및 attention 기반 시각화를 통한 설명가능성 분석
4. 주파수 정보가 fake 탐지에 독립적으로 유효한지 확인

---

## 3. Datasets

### 3.1 CIFAKE
- 설명: CIFAR-10 기반 real 이미지와 Stable Diffusion 1.4 기반 fake 이미지로 구성된 이진 분류 데이터셋
- 규모: real 60,000장 / fake 60,000장
- 해상도: 32×32
- 용도: 기본 실험 및 저해상도 환경에서의 spatial-frequency fusion 효과 검증

링크:  
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

### 3.2 130k Real vs Fake Face
- 설명: real 얼굴 이미지와 생성형 모델로 만든 fake 얼굴 이미지로 구성된 고해상도 얼굴 데이터셋
- 규모: real 70,000장 / fake 60,000장
- 생성 모델: FLUX1.DEV, FLUX1.PRO, SDXL 등
- 용도: 고해상도 얼굴 이미지에서 주파수 특징의 일반화 가능성 검증

링크:  
https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face

---

## 4. Method

### 4.1 Spatial branch
원본 이미지를 입력으로 사용하는 공간 도메인 모델입니다.  
이미지의 질감, 경계, 색 분포, 구조적 특징 등을 학습하여 real/fake를 구분합니다.

사용 모델:
- ResNet-18
- EfficientNet-V2-S(선택)
- ViT

### 4.2 Frequency branch
이미지를 FFT magnitude 또는 DCT 기반 스펙트럼 표현으로 변환한 뒤, 이를 입력으로 사용하는 주파수 도메인 모델입니다.  
생성 이미지가 픽셀 공간에서는 자연스러워 보여도 주파수 공간에서는 비정상적인 패턴을 남길 수 있다는 점에 주목합니다.

사용 기준 모델:
- SPAI 기반 주파수 표현 활용
- frequency-only 실험 포함

### 4.3 Fusion model
공간 도메인 특징과 주파수 도메인 특징을 각각 추출한 뒤, 이를 하나의 표현으로 결합하여 최종적으로 real/fake를 분류합니다.

비교 실험은 아래 세 가지 설정으로 수행합니다.

- spatial-only
- frequency-only
- fusion

---

## 5. Model Configuration

### Baseline
- ResNet-18

### Additional spatial models
- EfficientNet-V2-S(선택)
- ViT

### Frequency-based model
- SPAI-inspired frequency encoder

### Final proposed model
- Spatial-Frequency Fusion Detector

---

## 6. Experiments

### 6.1 Main comparison
다음 세 가지 실험을 수행합니다.

- spatial-only: 원본 이미지 기반 분류
- frequency-only: FFT magnitude 이미지 기반 분류
- fusion: 공간 특징과 주파수 특징을 결합한 분류

### 6.2 Cross-dataset validation
두 데이터셋에서 공통적으로 실험을 수행하여,
- 저해상도 데이터에서 주파수 정보가 유효한지
- 고해상도 데이터에서도 동일한 경향이 나타나는지
를 비교합니다.

### 6.3 Explainability analysis
- CNN 계열: Grad-CAM
- ViT 계열: attention map / rollout 기반 시각화
- fusion 모델: 공간 브랜치와 주파수 브랜치의 반응 비교

이 분석을 통해 모델이 실제로 어떤 단서를 보고 fake를 판단하는지 확인합니다.

---

## 7. Evaluation Metrics

본 프로젝트에서는 다음 지표를 사용합니다.

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

또한 정량 지표와 함께 다음 분석을 포함합니다.

- 오탐(False Positive) 사례 분석
- 미탐(False Negative) 사례 분석
- Grad-CAM / attention 시각화 결과 비교

---

## 8. Expected Contributions

이 프로젝트의 기대 기여는 다음과 같습니다.

1. 공간 정보와 주파수 정보를 결합한 fusion 탐지기 설계
2. spatial-only, frequency-only, fusion의 체계적 비교
3. 저해상도와 고해상도 데이터셋에서 주파수 정보의 유효성 검증
4. 딥페이크 탐지에서 주파수 정보가 실제로 어떤 역할을 하는지 해석 가능한 형태로 제시

---

## 9. Installation

### 9.1 Clone repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 9.2 Install PyTorch
> [!NOTE]
> PyTorch is **not** installed through `pyproject.toml`.
> Please install a CUDA-enabled PyTorch build that matches your system first.
> For CUDA 12.8, for example:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

> If your CUDA version or platform is different, use the official PyTorch install guide:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 9.3 Install dependencies
```bash
pip install -U pip
pip install -e .
```

또는 개발용 의존성까지 함께 설치하려면:
```bash
pip install -e ".[dev]"
```

---

## 10. Running Experiments

### Train spatial-only model
```bash
python scripts/train.py --config configs/train/spatial_resnet18.yaml
```

### Train frequency-only model
```bash
python scripts/train.py --config configs/train/frequency_spai.yaml
```

### Train fusion model
```bash
python scripts/train.py --config configs/train/fusion_resnet18_spai.yaml
```

### Evaluate model
```bash
python scripts/evaluate.py --config configs/train/fusion_resnet18_spai.yaml
```

### Generate explainability results
```bash
python scripts/visualize.py --config configs/train/fusion_resnet18_spai.yaml
```

---

## 11. Planned Results Table

| Dataset   | Model     | Input Type          | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --------- | --------- | ------------------- | -------- | --------- | ------ | -- | ------- |
| CIFAKE    | ResNet-18 | Spatial             | -        | -         | -      | -  | -       |
| CIFAKE    | ViT       | Spatial             | -        | -         | -      | -  | -       |
| CIFAKE    | SPAI      | Frequency           | -        | -         | -      | -  | -       |
| CIFAKE    | Fusion    | Spatial + Frequency | -        | -         | -      | -  | -       |
| 130k Face | ResNet-18 | Spatial             | -        | -         | -      | -  | -       |
| 130k Face | ViT       | Spatial             | -        | -         | -      | -  | -       |
| 130k Face | SPAI      | Frequency           | -        | -         | -      | -  | -       |
| 130k Face | Fusion    | Spatial + Frequency | -        | -         | -      | -  | -       |

---

## 12. Roadmap

* [x] Project topic selection
* [x] Initial outline design
* [ ] Dataset download and preprocessing
* [ ] Frequency transform pipeline implementation
* [ ] Spatial-only baseline training
* [ ] Frequency-only baseline training
* [ ] Fusion architecture implementation
* [ ] Explainability analysis
* [ ] Final evaluation and report writing

---

## 13. References

* CIFAKE: [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
* 130k Real vs Fake Face: [https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face](https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face)

---

## 14. Summary

이 프로젝트는 CIFAKE와 고해상도 얼굴 이미지 데이터셋을 이용하여, 공간 정보와 주파수 정보를 함께 활용하는 fusion 기반 딥페이크 탐지기를 설계하고 검증하는 것을 목표로 합니다.

핵심 질문은 다음과 같습니다.

* 주파수 정보는 fake 탐지에 독립적으로 유효한가?
* 공간 정보와 결합했을 때 실제로 성능이 좋아지는가?
* 저해상도와 고해상도 데이터셋에서 같은 경향이 나타나는가?

최종적으로는 단순히 어떤 모델의 정확도가 높은지를 보여주는 것이 아니라, 딥페이크 탐지에서 주파수 정보가 어떤 방식으로 기여하는지를 설명하는 프로젝트를 지향합니다.
