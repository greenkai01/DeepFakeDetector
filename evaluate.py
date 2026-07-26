"""
============================================================
evaluate.py

Evaluate DeepFake Detector

Author : YeonU Seo
============================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DeepFakeDataset, valid_transform
from model import DeepFakeDetector

from metrics import (
    calculate_metrics,
    print_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report
)


# ==========================================================
# Test DataLoader
# ==========================================================

def build_test_loader():

    test_dataset = DeepFakeDataset(
        config.TEST_DIR,
        transform=valid_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return test_loader


# ==========================================================
# Evaluate
# ==========================================================

@torch.no_grad()
def evaluate(model, dataloader, criterion):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_probs = []

    progress = tqdm(
        dataloader,
        desc="Evaluating"
    )

    for images, labels in progress:

        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)

        preds = (
            probs >= config.THRESHOLD
        ).float()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        running_correct += (
            preds == labels
        ).sum().item()

        total += batch_size

        all_labels.extend(
            labels.cpu().numpy().flatten()
        )

        all_probs.extend(
            probs.cpu().numpy().flatten()
        )

        progress.set_postfix({

            "Loss":
            f"{running_loss/total:.4f}",

            "Acc":
            f"{100*running_correct/total:.2f}%"

        })

    loss = running_loss / total

    acc = running_correct / total

    return loss, acc, all_labels, all_probs

# ==========================================================
# Main
# ==========================================================

def main():

    print("="*60)
    print("DeepFake Detector Evaluation")
    print("="*60)

    test_loader = build_test_loader()

    model = DeepFakeDetector().to(config.DEVICE)

    model.load_state_dict(

        torch.load(

            config.BEST_MODEL,

            map_location=config.DEVICE

        )

    )

    criterion = nn.BCEWithLogitsLoss()

    loss, acc, labels, probs = evaluate(

        model,

        test_loader,

        criterion

    )

    print()

    print(f"Test Loss : {loss:.4f}")

    print(f"Test Acc  : {acc:.4f}")

    metrics, preds = calculate_metrics(

        labels,

        probs

    )

    print_metrics(metrics)

    save_metrics(metrics)

    plot_confusion_matrix(

        labels,

        preds

    )

    plot_roc_curve(

        labels,

        probs

    )

    print_classification_report(

        labels,

        preds

    )

    print()

    print("Evaluation Complete")


if __name__ == "__main__":

    main()