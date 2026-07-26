"""
============================================================
metrics.py

Evaluation Metrics for DeepFake Detector

Author : YeonU Seo
============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import config


# ==========================================================
# Calculate Metrics
# ==========================================================

def calculate_metrics(labels, probs, threshold=config.THRESHOLD):
    """
    labels : 실제 정답 (0,1)
    probs  : 모델이 출력한 확률
    """

    labels = np.array(labels)
    probs = np.array(probs)

    preds = (probs >= threshold).astype(int)

    metrics = {

        "Accuracy":
            accuracy_score(labels, preds),

        "Precision":
            precision_score(labels, preds),

        "Recall":
            recall_score(labels, preds),

        "F1-score":
            f1_score(labels, preds),

        "ROC-AUC":
            roc_auc_score(labels, probs)

    }

    return metrics, preds

# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=["Real", "Fake"]

    )

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    plt.savefig(

        config.FIGURE_DIR /

        "confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    return cm

# ==========================================================
# ROC Curve
# ==========================================================

def plot_roc_curve(labels, probs):

    fpr, tpr, _ = roc_curve(labels, probs)

    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6,6))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {auc:.4f}"

    )

    plt.plot(

        [0,1],

        [0,1],

        "--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(

        config.FIGURE_DIR /

        "roc_curve.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(metrics):

    print()

    print("="*60)

    print("Evaluation Result")

    print("="*60)

    for key, value in metrics.items():

        print(f"{key:<12}: {value:.4f}")

    print("="*60)

# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(metrics):

    save_path = (

        config.RESULT_DIR /

        "metrics.txt"

    )

    with open(

        save_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write("="*50 + "\n")

        f.write("Evaluation Result\n")

        f.write("="*50 + "\n\n")

        for key, value in metrics.items():

            f.write(

                f"{key:<12}: {value:.4f}\n"

            )

# ==========================================================
# Classification Report
# ==========================================================

def print_classification_report(labels, preds):

    print()

    print(

        classification_report(

            labels,

            preds,

            target_names=[

                "Real",

                "Fake"

            ]

        )

    )