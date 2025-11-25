from typing import Dict, List, Optional
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

from torch.utils.data import DataLoader

from model import MultiLabelModel

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    delta_y: Optional[torch.Tensor] = None,
    alpha_y_train: Optional[torch.Tensor] = None,
    # scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(images)  # [B, num_labels]

            # --- TLA: additive logit adjustment (all phases) ---
            if delta_y is not None:
                logits = logits + delta_y  # broadcast over batch

            # --- raw per-element loss (we need reduction='none') ---
            loss_raw = criterion(logits, targets, reduction="none")  # [B, C]

            # --- ADRW: class re-weighting only in phase 3 (alpha_y_train set in caller) ---
            if alpha_y_train is not None:
                # alpha_y_train: [C], broadcast to [B, C]
                loss_raw = loss_raw * alpha_y_train

            # Final scalar loss
            loss = loss_raw.mean()
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  NaN/Inf detected in loss at batch {batch_idx+1}")
            print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            # Skip this batch
            continue

        # Backward pass with optional scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping (unscales gradients first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Step scheduler if provided (e.g., OneCycleLR steps per batch)
        # if scheduler is not None:
        #     scheduler.step()

        running_loss += loss.item()

        avg_loss = running_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'batch': f'{batch_idx+1}/{len(dataloader)}'
        })

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: MultiLabelModel,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    delta_y: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0

    all_targets = []
    all_probs = []

    # Collect predictions
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if delta_y is not None:
            logits = logits + delta_y
        
        loss = criterion(logits, targets)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    avg_loss = running_loss / len(dataloader)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Find best threshold maximizing MICRO F1 (global TP/FP/FN)
    best_threshold = 0.5
    best_micro_f1 = 0.0
    
    # We'll also track macro F1 at the best micro threshold
    final_macro_f1 = 0.0
    final_accuracy = 0.0
    final_recall = 0.0
    final_precision = 0.0

    for threshold in torch.arange(0.1, 0.95, 0.05):
        preds = (all_probs >= threshold).float()

        # MICRO: Global TP/FP/FN across ALL predictions
        tp_micro = ((preds == 1) & (all_targets == 1)).sum().item()
        fp_micro = ((preds == 1) & (all_targets == 0)).sum().item()
        fn_micro = ((preds == 0) & (all_targets == 1)).sum().item()
        
        p_micro = tp_micro / (tp_micro + fp_micro + 1e-8)
        r_micro = tp_micro / (tp_micro + fn_micro + 1e-8)
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-8)

        if f1_micro > best_micro_f1:
            best_micro_f1 = f1_micro
            best_threshold = threshold.item()
            final_recall = r_micro
            final_precision = p_micro
            
            # Calculate macro F1 at this threshold (for reporting)
            tp_class = ((preds == 1) & (all_targets == 1)).sum(dim=0).float()
            fp_class = ((preds == 1) & (all_targets == 0)).sum(dim=0).float()
            fn_class = ((preds == 0) & (all_targets == 1)).sum(dim=0).float()
            
            p_class = tp_class / (tp_class + fp_class + 1e-8)
            r_class = tp_class / (tp_class + fn_class + 1e-8)
            f1_class = 2 * p_class * r_class / (p_class + r_class + 1e-8)
            final_macro_f1 = f1_class.mean().item()
            
            # Label accuracy
            correct_preds = (preds == all_targets).float().sum().item()
            final_accuracy = correct_preds / all_targets.numel()

    return {
        "val_loss": avg_loss,
        "label_accuracy": final_accuracy,
        "precision_micro": final_precision,
        "recall_micro": final_recall,
        "f1_micro": best_micro_f1,           # Micro F1 (optimized)
        "best_threshold": best_threshold,
        "f1_macro": final_macro_f1,           # Macro F1 (just for reporting)
    }

@torch.no_grad()
def find_best_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    delta_y: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            logits = logits + delta_y if delta_y is not None else logits
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in torch.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).float()

        tp = ((preds == 1) & (all_targets == 1)).sum().item()
        fp = ((preds == 1) & (all_targets == 0)).sum().item()
        fn = ((preds == 0) & (all_targets == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold.item()

    return best_threshold

def get_lr_for_epoch(epoch: int,
                     frozen_lr: float,
                     unfrozen_lr: float,
                     freeze_epochs: int,
                     total_epochs: int,
                     adrw_start_epoch: int) -> float:
    """
    3-phase schedule:
      Phase 1 (1..freeze_epochs): linear warmup from 0.1*frozen_lr to frozen_lr
      Phase 2 (freeze_epochs+1..20): cosine from unfrozen_lr to 0.3*unfrozen_lr
      Phase 3 (21..total_epochs): cosine from 0.3*unfrozen_lr to 0.03*unfrozen_lr
    """
    # --- Phase 1: warmup (frozen backbone) ---
    if epoch <= freeze_epochs:
        start_lr = frozen_lr * 0.1
        end_lr = frozen_lr
        if freeze_epochs == 1:
            return end_lr
        t = (epoch - 1) / (freeze_epochs - 1)
        return start_lr + t * (end_lr - start_lr)

    # we fix the phase boundaries explicitly
    phase2_start = freeze_epochs + 1   # 6
    phase2_end = adrw_start_epoch - 1  # 20
    phase3_start = adrw_start_epoch      # 21

    # --- Phase 2: cosine decay (unfrozen backbone, main training) ---
    if epoch <= phase2_end:
        # cosine from UNFREEZE_LR → 0.3 * UNFREEZE_LR
        eta_max = unfrozen_lr
        eta_min = unfrozen_lr * 0.3
        T_max = phase2_end - phase2_start + 1    # e.g. 20 - 6 + 1 = 15
        # local step index in this phase: 0 .. T_max-1
        step = epoch - phase2_start
        t = step / max(T_max - 1, 1)
        # CosineAnnealingLR formula
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t))

    # --- Phase 3: cosine decay (fine-tuning + ADRW) ---
    # epochs phase3_start .. total_epochs
    eta_max = unfrozen_lr * 0.3
    eta_min = unfrozen_lr * 0.03
    T_max = total_epochs - phase3_start + 1
    step = epoch - phase3_start
    t = step / max(T_max - 1, 1)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def compute_per_label_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_columns: List[str],
    device: torch.device,
    threshold: float = 0.5,
    delta_y: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
    Compute per-label metrics: TP, FP, FN, precision, recall, F1.
    Also finds the best threshold per label and computes best metrics and mAP.
    Returns a DataFrame with one row per label.
    """
    model.eval()
    num_labels = len(label_columns)
    
    # Collect all predictions and targets
    all_probs = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        logits = model(images)
        if delta_y is not None:
            logits = logits + delta_y
        probs = torch.sigmoid(logits)
        
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [N, num_labels]
    all_targets = torch.cat(all_targets, dim=0)  # [N, num_labels]
    
    # Compute metrics at the given threshold
    preds = (all_probs >= threshold).float()
    tp_per_label = ((preds == 1) & (all_targets == 1)).sum(dim=0).long()
    fp_per_label = ((preds == 1) & (all_targets == 0)).sum(dim=0).long()
    fn_per_label = ((preds == 0) & (all_targets == 1)).sum(dim=0).long()
    tn_per_label = ((preds == 0) & (all_targets == 0)).sum(dim=0).long()
    
    # Convert to numpy for DataFrame
    tp_arr = tp_per_label.numpy()
    fp_arr = fp_per_label.numpy()
    fn_arr = fn_per_label.numpy()
    tn_arr = tn_per_label.numpy()

    # Compute precision, recall, F1 per label at given threshold
    eps = 1e-8
    precision = tp_arr / (tp_arr + fp_arr + eps)
    recall = tp_arr / (tp_arr + fn_arr + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # Handle division by zero cases
    precision = np.where((tp_arr + fp_arr) > 0, precision, 0.0)
    recall = np.where((tp_arr + fn_arr) > 0, recall, 0.0)
    f1 = np.where((precision + recall) > 0, f1, 0.0)
    
    # Find best threshold per label
    thresholds = torch.arange(0.1, 0.95, 0.05)
    best_thresholds = np.zeros(num_labels)
    best_f1_scores = np.zeros(num_labels)
    best_precisions = np.zeros(num_labels)
    best_recalls = np.zeros(num_labels)
    best_accuracies = np.zeros(num_labels)
    average_precisions = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        label_probs = all_probs[:, label_idx]
        label_targets = all_targets[:, label_idx]
        
        # Find best threshold for this label
        best_f1 = 0.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        best_accuracy = 0.0
        
        for thresh in thresholds:
            preds_thresh = (label_probs >= thresh).float()
            
            tp = ((preds_thresh == 1) & (label_targets == 1)).sum().item()
            fp = ((preds_thresh == 1) & (label_targets == 0)).sum().item()
            fn = ((preds_thresh == 0) & (label_targets == 1)).sum().item()
            tn = ((preds_thresh == 0) & (label_targets == 0)).sum().item()
            
            prec = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
            f1_score = 2 * prec * rec / (prec + rec + eps) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = thresh.item()
                best_precision = prec
                best_recall = rec
                best_accuracy = acc
        
        best_thresholds[label_idx] = best_threshold
        best_f1_scores[label_idx] = best_f1
        best_precisions[label_idx] = best_precision
        best_recalls[label_idx] = best_recall
        best_accuracies[label_idx] = best_accuracy
        
        # Compute Average Precision (AP) for this label
        # Sort by probability descending
        sorted_indices = torch.argsort(label_probs, descending=True)
        sorted_targets = label_targets[sorted_indices].numpy()
        sorted_probs = label_probs[sorted_indices].numpy()
        
        # Compute precision at each recall level
        tp_cumsum = np.cumsum(sorted_targets)
        fp_cumsum = np.cumsum(1 - sorted_targets)
        
        precisions_ap = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        recalls_ap = tp_cumsum / (sorted_targets.sum() + eps)
        
        # Compute AP using trapezoidal rule
        if sorted_targets.sum() > 0:
            # Add sentinel values
            recalls_ap = np.concatenate([[0], recalls_ap, [1]])
            precisions_ap = np.concatenate([[0], precisions_ap, [0]])
            
            # Make precision monotonically decreasing
            for i in range(len(precisions_ap) - 2, -1, -1):
                precisions_ap[i] = max(precisions_ap[i], precisions_ap[i + 1])
            
            # Compute AP
            ap = np.sum((recalls_ap[1:] - recalls_ap[:-1]) * precisions_ap[1:])
            average_precisions[label_idx] = ap
        else:
            average_precisions[label_idx] = 0.0
    
    # Build DataFrame
    df = pd.DataFrame({
        'label': label_columns,
        'true_positives': tp_arr,
        'false_positives': fp_arr,
        'false_negatives': fn_arr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_threshold': best_thresholds,
        'best_f1': best_f1_scores,
        'best_precision': best_precisions,
        'best_recall': best_recalls,
        'best_accuracy': best_accuracies,
        'average_precision': average_precisions,
    })
    
    return df


