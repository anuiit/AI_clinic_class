# train.py
import time
import json
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)  # [B, num_labels]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"[Epoch {epoch:02d} | Batch {batch_idx+1:04d}] "
                  f"Train loss: {avg_loss:.4f}")

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate on the validation set.
    Returns:
        - val_loss
        - label_accuracy
        - precision_micro
        - recall_micro
        - f1_micro
    """
    model.eval()
    running_loss = 0.0

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += (preds == targets).sum().item()
        total += targets.numel()

        tp += ((preds == 1) & (targets == 1)).sum().item()
        fp += ((preds == 1) & (targets == 0)).sum().item()
        fn += ((preds == 0) & (targets == 1)).sum().item()

    avg_loss = running_loss / len(dataloader)
    label_accuracy = correct / total if total > 0 else 0.0

    eps = 1e-8
    precision_micro = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    if precision_micro + recall_micro > 0:
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    else:
        f1_micro = 0.0

    return {
        "val_loss": avg_loss,
        "label_accuracy": label_accuracy,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

