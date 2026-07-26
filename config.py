"""
============================================================
DeepFake Detector

config.py

프로젝트 전체에서 사용하는 설정 파일
============================================================


from pathlib import Path


print(
    "real 개수:",
    len(list(Path("dataset/train/real").glob("*")))
)

print(
    "fake 개수:",
    len(list(Path("dataset/train/fake").glob("*")))
)
"""

from pathlib import Path
import torch

# ============================================================
# Project
# ============================================================

PROJECT_NAME = "Explainable DeepFake Detector"

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Dataset
# ============================================================

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

# ============================================================
# Save Directory
# ============================================================

WEIGHT_DIR = BASE_DIR / "weights"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"
LOG_DIR = RESULT_DIR / "logs"

for directory in [
    WEIGHT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# Image
# ============================================================

IMAGE_SIZE = 224

IMAGE_CHANNEL = 3

NUM_CLASSES = 1

# ============================================================
# Dataset
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 4

PIN_MEMORY = True

# ============================================================
# Training
# ============================================================

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# ============================================================
# Early Stopping
# ============================================================

PATIENCE = 5

MIN_DELTA = 0.001

# ============================================================
# Scheduler
# ============================================================

LR_FACTOR = 0.5

LR_PATIENCE = 2

# ============================================================
# Threshold
# ============================================================

THRESHOLD = 0.5

# ============================================================
# Random Seed
# ============================================================

SEED = 42

# ============================================================
# Model
# ============================================================

MODEL_NAME = "EfficientNet-B0"

PRETRAINED = True

FREEZE_BACKBONE = True  # 백본 특징 추출기는 고정하고 분류기(Classifier)만 학습
LEARNING_RATE = 1e-3  # 백본을 고정했으므로 Learning Rate를 약간 높여도 안전함

DROPOUT = 0.4

# ============================================================
# Save Model
# ============================================================

BEST_MODEL = WEIGHT_DIR / "best_model.pth"

LAST_MODEL = WEIGHT_DIR / "last_model.pth"

# ============================================================
# Evaluation Metrics
# ============================================================

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

# ============================================================
# Grad-CAM
# ============================================================

USE_GRADCAM = True

# EfficientNet의 마지막 Convolution Layer
TARGET_LAYER = "features.8"

# ============================================================
# Visualization
# ============================================================

SAVE_LOSS_CURVE = True

SAVE_ACCURACY_CURVE = True

SAVE_CONFUSION_MATRIX = True

SAVE_ROC_CURVE = True

# ============================================================
# Print Configuration
# ============================================================

def print_config():

    print("=" * 60)
    print(PROJECT_NAME)
    print("=" * 60)

    print(f"Device          : {DEVICE}")
    print(f"Image Size      : {IMAGE_SIZE}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Learning Rate   : {LEARNING_RATE}")
    print(f"Weight Decay    : {WEIGHT_DECAY}")
    print(f"Optimizer       : Adam")
    print(f"Loss Function   : BCEWithLogitsLoss")
    print(f"Model           : {MODEL_NAME}")
    print(f"Pretrained      : {PRETRAINED}")
    print(f"Metrics         : {', '.join(METRICS)}")

    print("=" * 60)