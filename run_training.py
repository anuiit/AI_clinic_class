import torch
import torch.nn as nn

import numpy as np
import time
from torch.utils.data import DataLoader, random_split
from torch.amp.grad_scaler import GradScaler

from data import (
    load_dataframe_with_labels,
    GlyphDataset,
    build_default_transforms,
)
from model import MultiLabelModel
from sklearn.metrics import precision_recall_fscore_support
from skmultilearn.model_selection import iterative_train_test_split

from train import train_one_epoch, evaluate, compute_per_label_metrics, get_lr_for_epoch
from run_manager import RunManager
from visualize import (
    plot_training_history,
    plot_per_label_analysis,
    plot_confusion_matrix_summary,
    visualize_predictions,
    plot_tla_comparison,
    plot_learning_rate_schedule,
    plot_per_label_thresholds,
    plot_label_difficulty_analysis,
    plot_threshold_f1_improvement,
    plot_comprehensive_metrics_correlation,
)
from gradcam import visualize_gradcam_for_predictions
from AsymmetricLossOptimized import AsymmetricLossOptimized, AsymmetricLossOptimizedM
import json

# Architecture-specific configs (auto-selected)
BOTTLENECK_CONFIGS = {
    "resnet18": 256,
    "resnet50": 768,
    "vgg16": 1536,
    "vit_b16": 512,
    "efficientnet_b0": 640,
    "efficientnet_b3": 960,
    "efficientnet_b4": 1024,
    "efficientnet_b7": 1536,
}

IMG_SIZE_CONFIGS = {
    "resnet18": 224,
    "resnet50": 224,
    "vgg16": 224,
    "vit_b16": 384,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b7": 600,
}

BATCH_SIZE_CONFIGS = {
    "resnet18": 32,
    "resnet50": 32,
    "vgg16": 24,
    "vit_b16": 16,
    "efficientnet_b0": 32,
    "efficientnet_b3": 24,
    "efficientnet_b4": 20,
    "efficientnet_b7": 8,
}

# -------------------------
# CONFIG
# -------------------------

time.sleep(12500)  # ensure different timestamps for multiple runs

CSV_PATH = "data_csv/training_data_all_images.csv" # most recent: training_data_all_images.csv
MIN_NUMBER_OF_SAMPLES = 50  # only keep samples with at least this many labels

BACKBONE_MODEL = "efficientnet_b3" # options: resnet18, resnet50, vgg16, vit_b16, efficientnet_b7
SEED = 42
IMG_SIZE = IMG_SIZE_CONFIGS.get(BACKBONE_MODEL, 224)
LOSS = "asymmetric"  # options: "bce", "asymmetric"

BATCH_SIZE = BATCH_SIZE_CONFIGS.get(BACKBONE_MODEL, 32)
NUM_EPOCHS = 75
FREEZING_BACKBONE_EPOCHS = 10
VAL_SPLIT = 0.2
FROZEN_LR = 1e-3
UNFREEZE_LR = 1e-4

DROPOUT = 0.3
HIDDEN = [512, 256]
CUSTOM_WEIGHTS_LOSS = False # Use custom pos_weight for BCEWithLogitsLoss
NORMALIZATION = nn.BatchNorm1d  # Normalization layer class (nn.BatchNorm1d, nn.LayerNorm, or None)
ACTIVATION = nn.ReLU(inplace=True)
CUSTOM_HEAD = False

# -------------------------

USE_BOTTLENECK = True  # Add bottleneck for dimensionality reduction
BOTTLENECK_DIM = BOTTLENECK_CONFIGS.get(BACKBONE_MODEL, 1024)  # Bottleneck size (512 for small models, 1024-1536 for large)

EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0  # keep 0 on Windows, can raise on Linux

# -------------------------
# ADRW / TLA CONFIG
# -------------------------
USE_ADRW_TLA = True
ADRW_START_EPOCH = 21   # phase 3 starts here
NU = 0.0                # exponent for ADRW: alpha_y = freq^(-NU)
TAU = 0.5               # strength for TLA: delta_y = TAU * log(freq)

# -------------------------

# After config setup, add validation
if USE_ADRW_TLA:
    if ADRW_START_EPOCH <= FREEZING_BACKBONE_EPOCHS:
        raise ValueError(
            f"ADRW_START_EPOCH ({ADRW_START_EPOCH}) must be > "
            f"FREEZING_BACKBONE_EPOCHS ({FREEZING_BACKBONE_EPOCHS})"
        )
    if ADRW_START_EPOCH > NUM_EPOCHS:
        print(f"⚠️  Warning: ADRW_START_EPOCH ({ADRW_START_EPOCH}) > NUM_EPOCHS ({NUM_EPOCHS})")
        print("    ADRW/TLA will never activate!")

if LOSS == "bce" and CUSTOM_WEIGHTS_LOSS and USE_ADRW_TLA:
    print("⚠️  Warning: Using BCE with custom pos_weights AND ADRW/TLA")
    print("    This creates double-weighting. Consider disabling one of them.")

print(f"🎯 Using {BACKBONE_MODEL}")
print(f"   Image Size: {IMG_SIZE}×{IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Bottleneck: {BOTTLENECK_DIM}")

# Initialize run manager
run_manager = RunManager(base_dir="runs")

# Update checkpoint and history paths to use run directory
CHECKPOINT_PATH = run_manager.get_path(f"best_model.pt")
HISTORY_PATH = run_manager.get_path(f"training_history.json")

# Save configuration
config = {
    "csv_path": CSV_PATH,
    "MIN_NUMBER_OF_SAMPLES": MIN_NUMBER_OF_SAMPLES,
    "model": BACKBONE_MODEL,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "val_split": VAL_SPLIT,
    "frozen_lr": FROZEN_LR,
    "unfreeze_lr": UNFREEZE_LR,
    "dropout": DROPOUT,
    "activation": str(ACTIVATION),
    "normalization": str(NORMALIZATION) if NORMALIZATION else None,
    "custom_head": CUSTOM_HEAD,
    "use_bottleneck": USE_BOTTLENECK,
    "bottleneck_dim": BOTTLENECK_DIM,
    "hidden": HIDDEN,
    "seed": SEED,
    "custom_weights": CUSTOM_WEIGHTS_LOSS,
    "loss_function": LOSS,
    "use_adrw_tla": USE_ADRW_TLA,
    "adrw_start_epoch": ADRW_START_EPOCH,
    "nu": NU,
    "tau": TAU,
}
run_manager.save_config(config)

# -------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

# --- Load data ---
print(f"Loading dataset from: {CSV_PATH}")
# Load raw dataframe first
df_raw, all_label_columns = load_dataframe_with_labels(CSV_PATH)

# Extract X (indices or paths) and Y (labels) for splitting
# We use indices to map back to the dataframe later
X_indices = df_raw.index.to_numpy().reshape(-1, 1)
Y_all = df_raw[all_label_columns].to_numpy()

print(f"Starting Iterative Stratification (Split: {1-VAL_SPLIT:.2f}/{VAL_SPLIT:.2f})...")
# This ensures every label is represented proportionally in train and val
X_train_idx, Y_train_strat, X_val_idx, Y_val_strat = iterative_train_test_split(
    X_indices, Y_all, test_size=VAL_SPLIT
)

# Convert back to 1D arrays
train_indices = X_train_idx.flatten()
val_indices = X_val_idx.flatten()

# Create the initial DataFrames
df_train = df_raw.iloc[train_indices].reset_index(drop=True)
df_val = df_raw.iloc[val_indices].reset_index(drop=True)

print(f"Split complete. Train: {len(df_train)} | Val: {len(df_val)}")

# -------------------------
# PREVENT LEAKAGE: FILTER LABELS BASED ON TRAIN SET ONLY
# -------------------------

if MIN_NUMBER_OF_SAMPLES > 0:
    print(f"Filtering labels with < {MIN_NUMBER_OF_SAMPLES} occurrences (calculated on TRAIN set)...")
    
    # 1. Calculate counts ONLY on training data
    train_label_counts = df_train[all_label_columns].sum(axis=0)
    
    # 2. Identify valid labels
    valid_labels = train_label_counts[train_label_counts >= MIN_NUMBER_OF_SAMPLES].index.tolist()
    
    # 3. Filter both DataFrames to keep only valid labels
    # We drop the old label columns and keep only the valid ones (plus image paths/metadata)
    non_label_cols = [c for c in df_train.columns if c not in all_label_columns]
    
    df_train = df_train[non_label_cols + valid_labels]
    df_val = df_val[non_label_cols + valid_labels]
    
    # Update our reference list of labels
    label_columns = valid_labels
    
    print(f"Labels retained: {len(label_columns)} / {len(all_label_columns)}")
else:
    label_columns = all_label_columns

# -------------------------
# ADRW / TLA WEIGHTS
# -------------------------
if USE_ADRW_TLA:
    # frequency per label on TRAIN set
    num_pos = df_train[label_columns].sum().values.astype(np.float32)
    freq = num_pos / len(df_train)
    freq = np.clip(freq, 1e-8, 1.0)
    freq_t = torch.tensor(freq, dtype=torch.float32, device=device)

    # TLA: additive logit adjustment
    if TAU > 0:
        delta_y = -TAU * torch.log(freq_t)
    else:
        delta_y = None

    # ADRW: class re-weighting
    if NU > 0:
        alpha_y = (freq_t ** (-NU))
    else:
        alpha_y = None  # ← Explicitly None when disabled

    print("\n" + "="*60)
    print("ADRW/TLA CONFIGURATION")
    print("="*60)
    print(f"NU: {NU} (ADRW {'ENABLED' if NU > 0 else 'DISABLED'})")
    print(f"TAU: {TAU} (TLA {'ENABLED' if TAU > 0 else 'DISABLED'})")
    
    if delta_y is not None:
        print(f"\nTLA adjustments:")
        print(f"  delta_y range: [{delta_y.min():.4f}, {delta_y.max():.4f}]")
        print(f"  delta_y[0:5]: {delta_y[:5].tolist()}")
    
    if alpha_y is not None:
        print(f"\nADRW weights:")
        print(f"  alpha_y range: [{alpha_y.min():.4f}, {alpha_y.max():.4f}]")
        print(f"  alpha_y[0:5]: {alpha_y[:5].tolist()}")
    
    print("="*60 + "\n")
else:
    delta_y = None
    alpha_y = None

# -------------------------
# CREATE DATASETS
# -------------------------

transform = build_default_transforms(size=IMG_SIZE)

# Note: We now pass the specific pre-split DataFrames
train_dataset = GlyphDataset(df_train, label_columns=label_columns, transform=transform)
val_dataset = GlyphDataset(df_val, label_columns=label_columns, transform=transform)

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

# --- Model Initialization  ---
num_labels = len(label_columns)
model = MultiLabelModel(
    base_model=BACKBONE_MODEL,
    num_labels=num_labels,
    dropout=DROPOUT,
    hidden=HIDDEN,
    batch_size=BATCH_SIZE,
    activation=ACTIVATION,
    normalization=NORMALIZATION,
    custom_head=CUSTOM_HEAD,
    use_bottleneck=USE_BOTTLENECK,
    bottleneck_dim=BOTTLENECK_DIM,
).to(device)

# torch.compile() requires Triton which is not available on Windows
# Skip compilation on Windows or if Triton is not available
import platform
if torch.cuda.is_available() and torch.__version__ >= "2.0" and platform.system() != "Windows":
    try:
        model = torch.compile(model, mode="default")  # type: ignore[assignment]
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available: {e}")
        print("Continuing with eager mode (no performance impact expected)")

print("Model used: ", BACKBONE_MODEL)

if LOSS == "asymmetric":
    criterion = AsymmetricLossOptimizedM(gamma_neg=4, gamma_pos=1, clip=0.05)
elif LOSS == "bce":
    if CUSTOM_WEIGHTS_LOSS:
        num_pos = df_train[label_columns].sum().values
        num_neg = np.array(len(df_train)) - num_pos

    # (num_pos + 1e-8) to avoid div by zero
    raw = torch.tensor(num_neg / (np.array(num_pos) + 1e-8), dtype=torch.float32).to(device)
    pos_weight = torch.sqrt(1.0 + torch.log(raw)).to(device)

    print("="*60)
    print("CUSTOM POS WEIGHTS FOR BCEWITHLOGITSLOSS")
    print("Number of labels:", num_labels, "/ ", len(label_columns))
    print("Number of positives per label:\n", num_pos.tolist())
    print("Number of negatives per label:\n", num_neg.tolist())
    print("Pos weights for BCEWithLogitsLoss:\n", pos_weight.tolist())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.AdamW(model.parameters(), lr=FROZEN_LR)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=FROZEN_LR * 3, steps_per_epoch=len(train_loader), epochs=FREEZING_BACKBONE_EPOCHS
# )
# scaler = GradScaler() if torch.cuda.is_available() else None

optimizer = torch.optim.AdamW(model.parameters(), lr=FROZEN_LR)
scaler = GradScaler() if torch.cuda.is_available() else None

epochs_without_improvement = 0
best_val_loss = float("inf")
best_val_f1 = 0.0  # Changed from inf - we want to MAXIMIZE F1
best_val_recall = float("inf")
history = []
unfrozen = False  # Track if we've already unfrozen


# --- Training loop ---
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    if FREEZING_BACKBONE_EPOCHS > 0:
        if epoch <= FREEZING_BACKBONE_EPOCHS:
            model.freeze_backbone()
            if epoch == 1:
                model.print_trainable_status()
        elif not unfrozen:  # Only recreate optimizer once
            model.unfreeze_backbone()
            model.print_trainable_status()
            optimizer = torch.optim.AdamW(model.parameters(), lr=UNFREEZE_LR)
            unfrozen = True

    # --- Set LR for this epoch (3-phase schedule) ---
    lr_this_epoch = get_lr_for_epoch(
        epoch,
        frozen_lr=FROZEN_LR,
        unfrozen_lr=UNFREEZE_LR,
        freeze_epochs=FREEZING_BACKBONE_EPOCHS,
        total_epochs=NUM_EPOCHS,
        adrw_start_epoch=ADRW_START_EPOCH,
    )
    for pg in optimizer.param_groups:
        pg["lr"] = lr_this_epoch

    current_lr = lr_this_epoch

    if USE_ADRW_TLA and epoch >= ADRW_START_EPOCH and alpha_y is not None:
        alpha_y_train = alpha_y
    else:
        alpha_y_train = None

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch,
        scaler,
        delta_y=delta_y,          # TLA always applied
        alpha_y_train=alpha_y_train,  # ADRW only in phase 3
    )

    # Check for NaN loss and stop training
    if np.isnan(train_loss):
        print(f"\n⚠️  NaN detected in training loss at epoch {epoch}. Stopping training.")
        print("This usually indicates gradient explosion. Check learning rate and model stability.")
        break

    # 1. Evaluate WITHOUT TLA (raw model performance)
    val_metrics_raw = evaluate(
        model, val_loader, criterion, device, delta_y=None
    )

    # 2. Evaluate WITH TLA (deployment performance)
    if USE_ADRW_TLA:
        val_metrics_tla = evaluate(
            model, val_loader, criterion, device, 
            delta_y=delta_y
        )

    # OneCycleLR steps after each batch in train_one_epoch, not here
    # For ReduceLROnPlateau or similar, uncomment below:
    # scheduler.step(val_metrics["val_loss"])

    # current_lr = optimizer.param_groups[0]['lr']

    elapsed = time.time() - start_time

    status_line = (
        f"Loss: {train_loss:.4f} → {val_metrics_raw['val_loss']:.4f} │ "
        f"mF1: {val_metrics_raw['f1_micro']:.4f} @ {val_metrics_raw['best_threshold']:.2f} │ "
        f"MF1: {val_metrics_raw['f1_macro']:.4f} │ Recall: {val_metrics_raw['recall_micro']:.4f} │ "
        f"LR: {current_lr:.2e} │ {elapsed:.0f}s"
    )
    print(status_line)

    # TLA supplement (when active)
    if USE_ADRW_TLA and epoch >= ADRW_START_EPOCH:
        tla_boost = val_metrics_tla['f1_micro'] - val_metrics_raw['f1_micro']
        boost_indicator = "↑" if tla_boost > 0.01 else ("↓" if tla_boost < -0.01 else "→")
        print(
            f"  TLA {boost_indicator} F1: {val_metrics_tla['f1_micro']:.4f}@{val_metrics_tla['best_threshold']:.2f} │ "
            f"({tla_boost:+.4f})  "
            f"Best: {val_metrics_tla['f1_macro']:.4f}"
        )

    history_entry = {
        "epoch": epoch,
        "learning_rate": current_lr,
        "train_loss": float(train_loss),
        "val_loss": val_metrics_raw["val_loss"],
        "val_f1_micro": val_metrics_raw["f1_micro"],
        "val_f1_macro": val_metrics_raw["f1_macro"],
        "val_precision_micro": val_metrics_raw["precision_micro"],
        "val_recall_micro": val_metrics_raw["recall_micro"],
        "val_label_accuracy": val_metrics_raw["label_accuracy"],
        "best_threshold": val_metrics_raw["best_threshold"],
        "time_sec": float(elapsed),
    }
    
    # Add TLA metrics only if TLA is enabled
    if USE_ADRW_TLA and epoch >= ADRW_START_EPOCH:
        history_entry.update({
            "val_tla_loss": val_metrics_tla["val_loss"],
            "val_tla_f1_micro": val_metrics_tla["f1_micro"],
            "val_tla_f1_macro": val_metrics_tla["f1_macro"],
            "val_tla_recall_micro": val_metrics_tla["recall_micro"],
            "best_tla_threshold": val_metrics_tla["best_threshold"],
        })
    
    history.append(history_entry)

    # SAVE CONDITION: Prioritize Micro F1 (better for multi-label imbalanced datasets)
    if val_metrics_raw["f1_micro"] > best_val_f1:  # MAXIMIZE F1, not minimize loss
        best_val_loss = val_metrics_raw["val_loss"]
        best_val_f1 = val_metrics_raw["f1_micro"]
        epochs_without_improvement = 0
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics_raw["val_loss"],
            "val_f1": val_metrics_raw["f1_macro"],
            "val_recall": val_metrics_raw["recall_micro"],
            "best_raw_threshold": val_metrics_raw["best_threshold"],  # Fixed key name
            "label_columns": label_columns,
        }
        
        # Add TLA metrics only if TLA is enabled
        if USE_ADRW_TLA:
            checkpoint_data.update({
                "val_tla_loss": val_metrics_tla["val_loss"],
                "val_tla_f1": val_metrics_tla["f1_macro"],
                "val_tla_recall": val_metrics_tla["recall_micro"],
                "best_tla_threshold": val_metrics_tla["best_threshold"],
            })
        
        torch.save(checkpoint_data, str(CHECKPOINT_PATH))
        print(f"  ✓ Best model saved (F1: {best_val_f1:.4f}, loss: {best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  ⏳ No improvement in F1 for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")
    
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping: no F1 improvement for {EARLY_STOPPING_PATIENCE} epochs")
        break

    print("-" * 60)

# Save training history
with open(str(HISTORY_PATH), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print(f"Training finished. History saved to {HISTORY_PATH}")
print(f"\nBest val loss: {best_val_loss:.4f} | Best F1: {best_val_f1:.4f}")

# -------------------------
# POST-TRAINING ANALYSIS
# -------------------------

print("\n" + "="*60)
print("GENERATING POST-TRAINING ANALYSIS")
print("="*60)

# Load best model
print("\nLoading best model checkpoint...")
checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use the correct threshold key
threshold_raw = checkpoint.get("best_raw_threshold", 0.5)
threshold_tla = checkpoint.get("best_tla_threshold", 0.5) if USE_ADRW_TLA else None

# Compute per-label metrics (RAW - no TLA)
print("Computing per-label metrics (raw model)...")
df_label_metrics_raw = compute_per_label_metrics(
    model, val_loader, label_columns, device, 
    threshold=threshold_raw, 
    delta_y=None  # No TLA adjustment
)

# Save raw metrics
label_metrics_path = run_manager.get_path("per_label_metrics_raw.csv")
df_label_metrics_raw.to_csv(label_metrics_path, index=False)
print(f"Raw per-label metrics saved to: {label_metrics_path}")

# Optionally compute TLA-adjusted metrics
df_label_metrics_tla = None  # Initialize to None
if USE_ADRW_TLA and threshold_tla is not None:
    print("Computing per-label metrics (with TLA)...")
    df_label_metrics_tla = compute_per_label_metrics(
        model, val_loader, label_columns, device, 
        threshold=threshold_tla, 
        delta_y=delta_y
    )
    
    label_metrics_tla_path = run_manager.get_path("per_label_metrics_tla.csv")
    df_label_metrics_tla.to_csv(label_metrics_tla_path, index=False)
    print(f"TLA per-label metrics saved to: {label_metrics_tla_path}")

# Use raw metrics for text report and visualizations
df_label_metrics = df_label_metrics_raw

# Save label metrics DataFrame
label_metrics_path = run_manager.get_path("per_label_metrics.csv")
df_label_metrics.to_csv(label_metrics_path, index=False)
print(f"Per-label metrics saved to: {label_metrics_path}")

# Save formatted text report
label_metrics_txt_path = run_manager.get_path("per_label_metrics.txt")
with open(label_metrics_txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("PER-LABEL METRICS REPORT\n")
    f.write("=" * 100 + "\n\n")
    
    # Summary statistics
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Total Labels: {len(df_label_metrics)}\n")
    f.write(f"Labels with F1 > 0: {(df_label_metrics['f1'] > 0).sum()}\n")
    f.write(f"Labels with F1 = 0: {(df_label_metrics['f1'] == 0).sum()}\n")
    f.write(f"Average F1 Score: {df_label_metrics['f1'].mean():.4f}\n")
    f.write(f"Average Precision: {df_label_metrics['precision'].mean():.4f}\n")
    f.write(f"Average Recall: {df_label_metrics['recall'].mean():.4f}\n")
    f.write(f"Total True Positives: {df_label_metrics['true_positives'].sum()}\n")
    f.write(f"Total False Positives: {df_label_metrics['false_positives'].sum()}\n")
    f.write(f"Total False Negatives: {df_label_metrics['false_negatives'].sum()}\n\n")
    
    # Full table
    f.write("DETAILED METRICS BY LABEL\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Label':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
    f.write("-" * 100 + "\n")
    
    # Sort by F1 score descending
    df_sorted = df_label_metrics.sort_values('f1', ascending=False)
    for _, row in df_sorted.iterrows():
        f.write(f"{row['label']:<25} {int(row['true_positives']):>6} {int(row['false_positives']):>6} "
                f"{int(row['false_negatives']):>6} {row['precision']:>10.4f} {row['recall']:>10.4f} "
                f"{row['f1']:>10.4f}\n")
    
    f.write("\n" + "=" * 100 + "\n")
    f.write("TOP 10 LABELS BY F1 SCORE\n")
    f.write("=" * 100 + "\n")
    top_10 = df_sorted.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        f.write(f"{i:2}. {row['label']:<25} F1: {row['f1']:.4f}  "
                f"Precision: {row['precision']:.4f}  Recall: {row['recall']:.4f}\n")
    
    f.write("\n" + "=" * 100 + "\n")
    f.write("WORST 10 LABELS BY F1 SCORE\n")
    f.write("=" * 100 + "\n")
    worst_10 = df_sorted.tail(10)
    for i, (_, row) in enumerate(worst_10.iterrows(), 1):
        f.write(f"{i:2}. {row['label']:<25} F1: {row['f1']:.4f}  "
                f"Precision: {row['precision']:.4f}  Recall: {row['recall']:.4f}\n")
    
    # Labels with F1 < 0.3
    low_f1 = df_label_metrics[df_label_metrics['f1'] < 0.3]
    if len(low_f1) > 0:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"LABELS WITH F1 < 0.3 ({len(low_f1)} labels)\n")
        f.write("=" * 100 + "\n")
        for _, row in low_f1.sort_values('f1').iterrows():
            f.write(f"  {row['label']:<25} F1: {row['f1']:.4f}  "
                    f"TP: {int(row['true_positives']):>3}  FP: {int(row['false_positives']):>3}  "
                    f"FN: {int(row['false_negatives']):>3}\n")

print(f"Per-label metrics text report saved to: {label_metrics_txt_path}")

# Generate visualizations
print("\nGenerating visualizations...")

# 1. Training history plots
print("  - Training history plots...")
plot_training_history(history, run_manager.run_dir)

# 2. Learning rate schedule
print("  - Learning rate schedule...")
plot_learning_rate_schedule(
    history, 
    run_manager.run_dir, 
    freeze_epochs=FREEZING_BACKBONE_EPOCHS,
    adrw_start_epoch=ADRW_START_EPOCH if USE_ADRW_TLA else None
)

# 3. TLA comparison (if applicable)
if USE_ADRW_TLA:
    print("  - TLA vs Raw performance comparison...")
    plot_tla_comparison(
        history, 
        run_manager.run_dir, 
        use_adrw_tla=USE_ADRW_TLA,
        adrw_start_epoch=ADRW_START_EPOCH
    )

# 4. Per-label analysis
print("  - Per-label analysis plots...")
plot_per_label_analysis(df_label_metrics, run_manager.run_dir)

# 5. Per-label threshold analysis
print("  - Per-label threshold analysis...")
plot_per_label_thresholds(df_label_metrics, run_manager.run_dir)

# 6. Label difficulty analysis
print("  - Label difficulty analysis...")
plot_label_difficulty_analysis(df_label_metrics, run_manager.run_dir)

# 7. Threshold optimization impact
print("  - Threshold optimization impact...")
plot_threshold_f1_improvement(df_label_metrics, run_manager.run_dir, df_label_metrics_tla)

# 8. Comprehensive metrics correlation
print("  - Metrics correlation analysis...")
plot_comprehensive_metrics_correlation(df_label_metrics, run_manager.run_dir)

# 9. Confusion matrix summary
print("  - Confusion matrix summary...")
plot_confusion_matrix_summary(df_label_metrics, run_manager.run_dir)

# 4. Prediction samples
print("  - Prediction samples...")
visualize_predictions(
    model, 
    val_dataset, 
    label_columns, 
    device, 
    run_manager.run_dir / "prediction_samples",
    num_samples=10,
    threshold=threshold_raw,
    prob_threshold=0.1
)

# 5. Grad-CAM visualizations
print("  - Grad-CAM visualizations...")
visualize_gradcam_for_predictions(
    model,
    val_dataset,
    label_columns,
    device,
    run_manager.run_dir / "gradcam_samples",
    num_samples=10,
    threshold=threshold_raw
    # target_layer_name auto-detected based on architecture
)

print("\n" + "="*60)
print(f"ALL OUTPUTS SAVED TO: {run_manager.run_dir}")
print("="*60)
print("\nGenerated files:")
print(f"  Configuration:")
print(f"    - config.txt / config.json")
print(f"  Model:")
print(f"    - {CHECKPOINT_PATH.name}")
print(f"    - {HISTORY_PATH.name}")
print(f"  Metrics:")
print(f"    - per_label_metrics.csv")
print(f"    - per_label_metrics.txt")
print(f"  Training Analysis:")
print(f"    - training_history.png")
print(f"    - learning_rate_schedule.png")
if USE_ADRW_TLA:
    print(f"    - tla_comparison.png")
print(f"  Label Performance:")
print(f"    - top_10_labels.png")
print(f"    - worst_10_labels.png")
print(f"    - low_f1_labels.png (if applicable)")
print(f"    - label_support.png")
print(f"    - label_difficulty_analysis.png")
print(f"  Threshold Analysis:")
print(f"    - per_label_thresholds.png")
print(f"    - threshold_optimization_impact.png")
print(f"  Advanced Analysis:")
print(f"    - metrics_correlation_analysis.png")
print(f"    - confusion_summary.png")
print(f"  Predictions:")
print(f"    - prediction_sample_*.png (10 samples)")
print(f"    - gradcam_sample_*.png (10 samples)")
