import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import time
from typing import Dict
from torch.utils.data import DataLoader, random_split
from data import (
    load_dataframe_with_labels,
    GlyphDataset,
    build_default_transforms,
)
from model import build_resnet18_multilabel
from sklearn.metrics import precision_recall_fscore_support

from train import train_one_epoch, evaluate
import json

# -------------------------
# CONFIG
# -------------------------

GLYPH_PATH = "glyph_images/"
CSV_PATH = "training_data_up.csv"
BATCH_SIZE = 16
NUM_EPOCHS = 30
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # keep 0 on Windows, can raise on Linux
DROPOUT = 0.3
HIDDEN = 512

SEED = 42
CHECKPOINT_PATH = "best_resnet18_codex.pt"
HISTORY_PATH = "training_history.json"

# -------------------------


torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

# --- Load data ---
print(f"Loading dataset from: {CSV_PATH}")
df, label_columns = load_dataframe_with_labels(CSV_PATH)

print(f"Found {len(label_columns)} label columns.")
print("Example label columns:", label_columns[:10])

transform = build_default_transforms()
full_dataset = GlyphDataset(df, label_columns=label_columns, transform=transform)

num_samples = len(full_dataset)
num_val = int(VAL_SPLIT * num_samples)
num_train = num_samples - num_val

train_dataset, val_dataset = random_split(
    full_dataset,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(SEED),
)

print(f"Dataset size: {num_samples} samples")
print(f"Train: {num_train} | Val: {num_val}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# --- Model / loss / optim ---
num_labels = len(label_columns)
model = build_resnet18_multilabel(num_labels=num_labels, dropout=DROPOUT, hidden=HIDDEN).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
history = []

# --- Training loop ---
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )

    val_metrics = evaluate(
        model, val_loader, criterion, device
    )

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_metrics['val_loss']:.4f} | "
        f"Val label acc: {val_metrics['label_accuracy']:.4f} | "
        f"Val P_micro: {val_metrics['precision_micro']:.4f} | "
        f"Val R_micro: {val_metrics['recall_micro']:.4f} | "
        f"Val F1_micro: {val_metrics['f1_micro']:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    # Log epoch history
    history.append(
        {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["val_loss"]),
            "val_label_accuracy": float(val_metrics["label_accuracy"]),
            "val_precision_micro": float(val_metrics["precision_micro"]),
            "val_recall_micro": float(val_metrics["recall_micro"]),
            "val_f1_micro": float(val_metrics["f1_micro"]),
            "time_sec": float(elapsed),
        }
    )

    # Save best checkpoint
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "label_columns": label_columns,
            },
            CHECKPOINT_PATH,
        )
        print(f"  -> New best model saved to {CHECKPOINT_PATH}")

# Save training history
with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print(f"Training finished. History saved to {HISTORY_PATH}")