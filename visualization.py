"""
============================================================
visualization.py

Visualization Functions

Author : YeonU Seo
============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config


# ==========================================================
# Loss Curve
# ==========================================================

def plot_loss_curve(train_losses, valid_losses):

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses,
             label="Train Loss",
             linewidth=2)

    plt.plot(valid_losses,
             label="Validation Loss",
             linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        config.FIGURE_DIR / "loss_curve.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Accuracy Curve
# ==========================================================

def plot_accuracy_curve(train_accs, valid_accs):

    plt.figure(figsize=(8,5))

    plt.plot(train_accs,
             label="Train Accuracy",
             linewidth=2)

    plt.plot(valid_accs,
             label="Validation Accuracy",
             linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        config.FIGURE_DIR /

        "accuracy_curve.png",

        dpi=300

    )

    plt.close()

# ==========================================================
# Learning Rate
# ==========================================================

def plot_learning_rate(lr_history):

    plt.figure(figsize=(8,5))

    plt.plot(
        lr_history,
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    plt.title("Learning Rate Scheduler")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        config.FIGURE_DIR /

        "learning_rate.png",

        dpi=300

    )

    plt.close()

# ==========================================================
# Probability Distribution
# ==========================================================

def plot_probability_distribution(probs):

    plt.figure(figsize=(7,5))

    plt.hist(

        probs,

        bins=20

    )

    plt.xlabel("Fake Probability")

    plt.ylabel("Count")

    plt.title("Prediction Probability")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        config.FIGURE_DIR /

        "probability_distribution.png",

        dpi=300

    )

    plt.close()

# ==========================================================
# Compare
# ==========================================================

def compare_train_valid(

        train_losses,

        valid_losses,

        train_accs,

        valid_accs

):

    print()

    print("="*60)

    print("Training Summary")

    print("="*60)

    print(

        f"Best Train Accuracy : "

        f"{max(train_accs):.4f}"

    )

    print(

        f"Best Valid Accuracy : "

        f"{max(valid_accs):.4f}"

    )

    print(

        f"Lowest Train Loss   : "

        f"{min(train_losses):.4f}"

    )

    print(

        f"Lowest Valid Loss   : "

        f"{min(valid_losses):.4f}"

    )

    print("="*60)