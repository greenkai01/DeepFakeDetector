"""
============================================================
train.py

Train DeepFake Detector

Author : YeonU Seo
============================================================
"""

import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DeepFakeDataset, train_transform, valid_transform
from model import DeepFakeDetector


# ==========================================================
# Random Seed
# ==========================================================

def set_seed(seed=config.SEED):
    """
    결과 재현성을 위해 모든 난수를 고정한다.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# DataLoader
# ==========================================================

def build_dataloader():

    train_dataset = DeepFakeDataset(
        config.TRAIN_DIR,
        transform=train_transform
    )

    valid_dataset = DeepFakeDataset(
        config.VALID_DIR,
        transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, valid_loader


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # -----------------------------
        # Gradient 초기화
        # -----------------------------

        optimizer.zero_grad()

        # -----------------------------
        # Forward
        # -----------------------------

        outputs = model(images)
        labels = labels.view_as(outputs).float()
        loss = criterion(outputs, labels)

        # -----------------------------
        # Backpropagation
        # -----------------------------

        loss.backward()

        optimizer.step()

        # -----------------------------
        # Accuracy 계산
        # -----------------------------

        probs = torch.sigmoid(outputs)

        preds = (probs >= config.THRESHOLD).float()

        correct = (preds == labels).sum().item()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        running_correct += correct
        total += batch_size

        progress_bar.set_postfix({
            "Loss": f"{running_loss / total:.4f}",
            "Acc": f"{100 * running_correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc

# ==========================================================
# Validation
# ==========================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validation 1 Epoch
    """

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_probs = []

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # -----------------------------
        # Forward
        # -----------------------------

        outputs = model(images)
        labels = labels.view_as(outputs).float()
        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)

        preds = (probs >= config.THRESHOLD).float()

        correct = (preds == labels).sum().item()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        running_correct += correct
        total += batch_size

        # Metrics 계산용 저장
        all_labels.extend(
            labels.cpu().numpy().flatten()
        )

        all_probs.extend(
            probs.cpu().numpy().flatten()
        )

        progress_bar.set_postfix({

            "Loss":
            f"{running_loss / total:.4f}",

            "Acc":
            f"{100 * running_correct / total:.2f}%"

        })

    epoch_loss = running_loss / total

    epoch_acc = running_correct / total

    return (
        epoch_loss,
        epoch_acc,
        all_labels,
        all_probs
    )


# ==========================================================
# Early Stopping
# ==========================================================

class EarlyStopping:

    def __init__(

        self,

        patience=config.PATIENCE,

        min_delta=config.MIN_DELTA

    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.stop = False


    def __call__(self, val_loss):

        # 개선됨

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            print(

                f"EarlyStopping : "

                f"{self.counter}/{self.patience}"

            )

            if self.counter >= self.patience:

                self.stop = True


# ==========================================================
# Optimizer
# ==========================================================

def build_optimizer(model):

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY

    )

    return optimizer


# ==========================================================
# Scheduler
# ==========================================================

def build_scheduler(optimizer):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.LR_FACTOR,
    patience=config.LR_PATIENCE
    )
    return scheduler

# ==========================================================
# Save Model
# ==========================================================

def save_checkpoint(model, save_path):

    torch.save(model.state_dict(), save_path)

    print(f"Model saved -> {save_path}")


# ==========================================================
# Plot History
# ==========================================================

def plot_history(train_losses,
                 valid_losses,
                 train_accs,
                 valid_accs):

    import matplotlib.pyplot as plt

    # Loss
    plt.figure(figsize=(8,5))

    plt.plot(train_losses,label="Train")
    plt.plot(valid_losses,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        config.FIGURE_DIR / "loss_curve.png",
        dpi=300
    )

    plt.close()


    # Accuracy

    plt.figure(figsize=(8,5))

    plt.plot(train_accs,label="Train")
    plt.plot(valid_accs,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        config.FIGURE_DIR / "accuracy_curve.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Main
# ==========================================================

def main():

    print("="*60)
    print("Explainable DeepFake Detector")
    print("="*60)

    set_seed()

    train_loader, valid_loader = build_dataloader()

    model = DeepFakeDetector().to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(model)

    scheduler = build_scheduler(optimizer)

    early_stopping = EarlyStopping()

    best_acc = 0

    train_losses = []
    valid_losses = []

    train_accs = []
    valid_accs = []

    for epoch in range(config.EPOCHS):

        print()

        print(
            f"Epoch "
            f"{epoch+1}/{config.EPOCHS}"
        )

        train_loss, train_acc = train_one_epoch(

            model,
            train_loader,
            criterion,
            optimizer,
            config.DEVICE

        )

        valid_loss, valid_acc, labels, probs = validate(

            model,
            valid_loader,
            criterion,
            config.DEVICE

        )

        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_acc:.4f}"
        )

        print(
            f"Valid Loss : {valid_loss:.4f}"
        )

        print(
            f"Valid Acc  : {valid_acc:.4f}"
        )

        # Best Model

        if valid_acc > best_acc:

            best_acc = valid_acc

            save_checkpoint(
                model,
                config.BEST_MODEL
            )

        # EarlyStopping

        early_stopping(valid_loss)

        if early_stopping.stop:

            print()

            print("Early Stopping")

            break

    save_checkpoint(

        model,

        config.LAST_MODEL

    )

    plot_history(

        train_losses,

        valid_losses,

        train_accs,

        valid_accs

    )

    print()

    print("="*60)

    print("Training Finished")

    print(f"Best Accuracy : {best_acc:.4f}")

    print("="*60)


if __name__ == "__main__":

    main()