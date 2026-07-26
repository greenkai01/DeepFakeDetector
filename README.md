# Explainable DeepFake Detector

> EfficientNet 기반 딥페이크 탐지 모델과 Grad-CAM을 활용한 설명 가능한 AI(XAI) 프로젝트

---

## 📌 Project Overview

최근 생성형 AI 기술의 발전으로 누구나 손쉽게 딥페이크 영상을 제작할 수 있게 되었다. 실제로 학교에서도 친구들의 얼굴을 이용해 간단한 딥페이크 이미지를 만드는 모습을 보며, 딥페이크 제작의 접근성이 매우 높아졌음을 체감하였다. 이러한 경험을 계기로 딥페이크가 사회적으로 심각한 문제로 이어질 수 있다고 판단하였고, AI를 활용하여 이를 탐지할 수 있는 모델을 직접 개발하였다.

본 프로젝트는 단순히 기존 딥페이크 탐지 모델을 사용하는 것이 아니라, **EfficientNet-B0를 특징 추출기(Feature Extractor)로 활용하고, 딥페이크 탐지에 적합한 분류기(Classifier)를 직접 설계**하였다. 또한 Accuracy뿐 아니라 Precision, Recall, F1-score, ROC-AUC, Confusion Matrix를 활용하여 모델의 성능을 다각도로 분석하였으며, Grad-CAM을 적용하여 AI가 어떤 얼굴 영역을 근거로 판단하는지 시각화하였다.

---

# Features

- EfficientNet-B0 기반 Transfer Learning
- Custom Binary Classifier 설계
- BCEWithLogitsLoss 기반 학습
- Adam Optimizer
- Learning Rate Scheduler
- Early Stopping
- Accuracy / Precision / Recall / F1-score 계산
- ROC Curve 및 ROC-AUC 분석
- Confusion Matrix 생성
- Grad-CAM을 이용한 Explainable AI
- Prediction Confidence 출력

---

# Project Structure

```
DeepFakeDetector/

│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── metrics.py
├── visualization.py
├── gradcam.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# Model Architecture

```
Input Image (224×224)

↓

EfficientNet-B0

↓

Global Average Pooling

↓

Custom Classifier

↓

Fake Probability
```

---

# Training

```
Image

↓

Forward

↓

Loss

↓

Backpropagation

↓

Weight Update

↓

Validation

↓

Save Best Model
```

---

# Evaluation Metrics

본 프로젝트에서는 단순 Accuracy만 사용하는 것이 아니라 다양한 성능 지표를 함께 분석하였다.

| Metric | Description |
|---------|-------------|
| Accuracy | 전체 정확도 |
| Precision | 가짜라고 예측한 것 중 실제 가짜의 비율 |
| Recall | 실제 가짜를 얼마나 찾아냈는지 |
| F1-score | Precision과 Recall의 조화평균 |
| ROC-AUC | 모델의 전체 분류 성능 |
| Confusion Matrix | 오탐(False Positive)과 미탐(False Negative) 분석 |

---

# Explainable AI

Grad-CAM을 적용하여 모델이 얼굴의 어느 영역을 근거로 딥페이크 여부를 판단하는지 시각화하였다.

이를 통해

- 눈 주변
- 입 주변
- 얼굴 윤곽
- 피부 경계

등 모델이 주목하는 영역을 분석하였다.

---

# Training Result

학습 후 자동 생성

```
results/

loss_curve.png

accuracy_curve.png

roc_curve.png

confusion_matrix.png

metrics.txt

gradcam_result.png
```

---

# Installation

```bash
git clone https://github.com/your-id/DeepFakeDetector.git

cd DeepFakeDetector

pip install -r requirements.txt
```

---

# Train

```bash
python train.py
```

---

# Evaluate

```bash
python evaluate.py
```

---

# Predict

```bash
python predict.py
```

---

# Technologies

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

# What I Learned

본 프로젝트를 수행하며 단순히 딥러닝 모델을 구현하는 것에서 그치지 않고, 전이학습(Transfer Learning)의 원리와 오류역전파(Backpropagation)를 통한 학습 과정을 이해하였다. 또한 Accuracy만으로는 모델의 성능을 충분히 평가할 수 없음을 확인하고 Precision, Recall, F1-score, ROC-AUC 등 다양한 지표를 비교하여 모델을 분석하였다. 특히 Grad-CAM을 활용해 AI의 판단 근거를 시각화함으로써 설명 가능한 AI(XAI)의 중요성을 이해할 수 있었다.

---

# Future Work

- Video DeepFake Detection
- Face Landmark 기반 특징 추가
- Vision Transformer(ViT) 적용
- 실시간 Webcam 탐지
- Web 서비스 배포 (Flask/FastAPI)
- ONNX 모델 변환 및 경량화

---

# License

This project is created for educational and research purposes.