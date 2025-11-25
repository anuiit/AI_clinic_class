import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix


def plot_training_history(history: List[Dict], save_dir: Path):
    """Create subplots for training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = [h['epoch'] for h in history]
    
    # Loss
    axes[0, 0].plot(epochs, [h['train_loss'] for h in history], label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, [h['val_loss'] for h in history], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 scores
    axes[0, 1].plot(epochs, [h['val_f1_micro'] for h in history], label='F1 Micro', marker='o', color='blue')
    axes[0, 1].plot(epochs, [h['val_f1_macro'] for h in history], label='F1 Macro', marker='s', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best threshold
    axes[0, 2].plot(epochs, [h['best_threshold'] for h in history], marker='o', color='purple')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Threshold')
    axes[0, 2].set_title('Best Threshold per Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[0, 2].legend()
    
    # Precision & Recall
    axes[1, 0].plot(epochs, [h['val_precision_micro'] for h in history], label='Precision', marker='o', color='orange')
    axes[1, 0].plot(epochs, [h['val_recall_micro'] for h in history], label='Recall', marker='s', color='cyan')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_label_analysis(df_labels: pd.DataFrame, save_dir: Path):
    """Create visualizations for per-label metrics."""
    
    # Top 10 and worst 10 by F1
    df_sorted = df_labels.sort_values('f1', ascending=False)
    top_10 = df_sorted.head(10)
    worst_10 = df_sorted.tail(10)
    
    # Labels with F1 < 0.3
    low_f1 = df_labels[df_labels['f1'] < 0.3].sort_values('f1')
    
    # Plot 1: Top 10 labels
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_10))
    width = 0.25
    
    ax.bar(x - width, top_10['precision'].values, width, label='Precision', alpha=0.8)
    ax.bar(x, top_10['recall'].values, width, label='Recall', alpha=0.8)
    ax.bar(x + width, top_10['f1'].values, width, label='F1', alpha=0.8)
    
    ax.set_xlabel('Label')
    ax.set_ylabel('Score')
    ax.set_title('Top 10 Labels by F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10['label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'top_10_labels.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Worst 10 labels
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(worst_10))
    width = 0.25
    
    ax.bar(x - width, worst_10['precision'].values, width, label='Precision', alpha=0.8)
    ax.bar(x, worst_10['recall'].values, width, label='Recall', alpha=0.8)
    ax.bar(x + width, worst_10['f1'].values, width, label='F1', alpha=0.8)
    
    ax.set_xlabel('Label')
    ax.set_ylabel('Score')
    ax.set_title('Worst 10 Labels by F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(worst_10['label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'worst_10_labels.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Low F1 labels (if any)
    if len(low_f1) > 0:
        # Adjust figure size based on number of labels
        fig_width = max(14, len(low_f1) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        x = np.arange(len(low_f1))
        width = 0.25
        
        ax.bar(x - width, low_f1['precision'].values, width, label='Precision', alpha=0.8)
        ax.bar(x, low_f1['recall'].values, width, label='Recall', alpha=0.8)
        ax.bar(x + width, low_f1['f1'].values, width, label='F1', alpha=0.8)
        
        ax.set_xlabel('Label')
        ax.set_ylabel('Score')
        ax.set_title(f'Labels with F1 < 0.3 ({len(low_f1)} labels)')
        ax.set_xticks(x)
        ax.set_xticklabels(low_f1['label'], rotation=90, ha='center', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_dir / 'low_f1_labels.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Support (number of true positives + false negatives = total occurrences)
    df_with_support = df_labels.copy()
    df_with_support['support'] = df_with_support['true_positives'] + df_with_support['false_negatives']
    df_with_support = df_with_support.sort_values('support', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(df_with_support)), df_with_support['support'], alpha=0.7, color='steelblue')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Top 20 Labels by Support (Total Occurrences)')
    ax.set_xticks(range(len(df_with_support)))
    ax.set_xticklabels(df_with_support['label'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'label_support.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_summary(df_labels: pd.DataFrame, save_dir: Path):
    """Plot aggregate confusion matrix metrics as a bar chart and detailed breakdown."""
    total_tp = df_labels['true_positives'].sum()
    total_fp = df_labels['false_positives'].sum()
    total_fn = df_labels['false_negatives'].sum()
    
    # Create a simple confusion-style breakdown visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Bar chart
    categories = ['True Positives', 'False Positives', 'False Negatives']
    values = [total_tp, total_fp, total_fn]
    colors = ['green', 'orange', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Aggregate Confusion Metrics')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Right plot: Confusion matrix-style heatmap (2x2 for binary per-label)
    # For multilabel, we show aggregated metrics in matrix format
    cm_data = np.array([[total_tp, total_fn],
                        [total_fp, 0]])  # TN not easily computable for multilabel
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actually Positive', 'Actually Negative'],
                cbar_kws={'label': 'Count'}, ax=ax2)
    ax2.set_title('Confusion Matrix (Aggregated)\nNote: TN not shown for multilabel')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tla_comparison(history: List[Dict], save_dir: Path, use_adrw_tla: bool, adrw_start_epoch: Optional[int] = None):
    """Compare raw model performance vs TLA-adjusted performance."""
    if not use_adrw_tla:
        print("TLA not enabled, skipping TLA comparison plots")
        return
    
    # Filter history to only epochs with TLA data
    tla_history = [h for h in history if 'val_tla_f1_micro' in h]
    
    if not tla_history:
        print("No TLA metrics found in history, skipping TLA comparison")
        return
    
    epochs = [h['epoch'] for h in tla_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TLA Impact Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: F1 Micro comparison
    ax = axes[0, 0]
    ax.plot(epochs, [h['val_f1_micro'] for h in tla_history], 
            label='Raw Model', marker='o', linewidth=2, color='blue')
    ax.plot(epochs, [h['val_tla_f1_micro'] for h in tla_history], 
            label='With TLA', marker='s', linewidth=2, color='green')
    ax.axvline(x=adrw_start_epoch, color='red', linestyle='--', alpha=0.7, label=f'ADRW Start (Epoch {adrw_start_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Micro Score')
    ax.set_title('F1 Micro: Raw vs TLA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: F1 improvement delta
    ax = axes[0, 1]
    improvements = [h['val_tla_f1_micro'] - h['val_f1_micro'] for h in tla_history]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax.bar(epochs, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=adrw_start_epoch, color='red', linestyle='--', alpha=0.7, label=f'ADRW Start')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Improvement (TLA - Raw)')
    ax.set_title('TLA F1 Boost per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Recall comparison
    ax = axes[1, 0]
    ax.plot(epochs, [h['val_recall_micro'] for h in tla_history], 
            label='Raw Model', marker='o', linewidth=2, color='blue')
    ax.plot(epochs, [h['val_tla_recall_micro'] for h in tla_history], 
            label='With TLA', marker='s', linewidth=2, color='green')
    ax.axvline(x=adrw_start_epoch, color='red', linestyle='--', alpha=0.7, label=f'ADRW Start')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall Micro')
    ax.set_title('Recall: Raw vs TLA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative improvement statistics
    ax = axes[1, 1]
    avg_improvement = np.mean(improvements)
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)
    positive_epochs = sum(1 for x in improvements if x > 0)
    
    stats_text = f"""TLA Impact Summary:
    
Average F1 Improvement: {avg_improvement:+.4f}
Maximum Boost: {max_improvement:+.4f}
Minimum Change: {min_improvement:+.4f}
Epochs with Positive Impact: {positive_epochs}/{len(improvements)}
    
TLA appears to {"HELP" if avg_improvement > 0.01 else "have MINIMAL impact" if avg_improvement > 0 else "HURT"} performance
"""
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat' if avg_improvement > 0 else 'lightcoral', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tla_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ TLA comparison plot saved")


def plot_learning_rate_schedule(history: List[Dict], save_dir: Path, freeze_epochs: int = 5, adrw_start_epoch: Optional[int] = None):
    """Visualize learning rate evolution over training."""
    epochs = [h['epoch'] for h in history]
    lrs = [h['learning_rate'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Learning Rate Schedule', fontsize=16, fontweight='bold')
    
    # Plot 1: Linear scale
    ax1.plot(epochs, lrs, marker='o', linewidth=2, markersize=4, color='darkblue')
    ax1.axvline(x=freeze_epochs, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Unfreeze Backbone (Epoch {freeze_epochs})')
    if adrw_start_epoch:
        ax1.axvline(x=adrw_start_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'ADRW Start (Epoch {adrw_start_epoch})')
    
    # Add phase annotations
    ax1.axvspan(1, freeze_epochs, alpha=0.1, color='orange', label='Phase 1: Frozen Backbone')
    ax1.axvspan(freeze_epochs, adrw_start_epoch-1 if adrw_start_epoch else len(epochs), alpha=0.1, color='green', label='Phase 2: Full Training')
    if adrw_start_epoch:
        ax1.axvspan(adrw_start_epoch, len(epochs), alpha=0.1, color='blue', label='Phase 3: ADRW Fine-tuning')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate (Linear Scale)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for better visibility
    ax2.semilogy(epochs, lrs, marker='o', linewidth=2, markersize=4, color='darkblue')
    ax2.axvline(x=freeze_epochs, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Unfreeze Backbone')
    if adrw_start_epoch:
        ax2.axvline(x=adrw_start_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'ADRW Start')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate (log scale)')
    ax2.set_title('Learning Rate (Log Scale)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_rate_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Learning rate schedule plot saved")


def plot_per_label_thresholds(df_labels: pd.DataFrame, save_dir: Path):
    """Analyze and visualize per-label optimal thresholds."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Label Threshold Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Threshold distribution
    ax = axes[0, 0]
    ax.hist(df_labels['best_threshold'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default (0.5)')
    ax.axvline(x=df_labels['best_threshold'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean ({df_labels["best_threshold"].mean():.3f})')
    ax.set_xlabel('Best Threshold')
    ax.set_ylabel('Number of Labels')
    ax.set_title('Distribution of Optimal Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Threshold vs F1 score
    ax = axes[0, 1]
    scatter = ax.scatter(df_labels['best_threshold'], df_labels['best_f1'], 
                        c=df_labels['best_f1'], cmap='RdYlGn', alpha=0.6, s=50)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default Threshold')
    ax.set_xlabel('Best Threshold')
    ax.set_ylabel('Best F1 Score')
    ax.set_title('Threshold vs F1 Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Plot 3: Threshold vs label frequency (support)
    ax = axes[1, 0]
    df_with_support = df_labels.copy()
    df_with_support['support'] = df_with_support['true_positives'] + df_with_support['false_negatives']
    scatter = ax.scatter(df_with_support['support'], df_with_support['best_threshold'],
                        c=df_with_support['best_f1'], cmap='RdYlGn', alpha=0.6, s=50)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Default Threshold')
    ax.set_xlabel('Label Support (# Occurrences)')
    ax.set_ylabel('Best Threshold')
    ax.set_title('Label Frequency vs Optimal Threshold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    stats_text = f"""Threshold Statistics:
    
Mean Threshold: {df_labels['best_threshold'].mean():.3f}
Median Threshold: {df_labels['best_threshold'].median():.3f}
Std Dev: {df_labels['best_threshold'].std():.3f}
Min: {df_labels['best_threshold'].min():.3f}
Max: {df_labels['best_threshold'].max():.3f}

Labels needing higher threshold (>0.6): {(df_labels['best_threshold'] > 0.6).sum()}
Labels needing lower threshold (<0.4): {(df_labels['best_threshold'] < 0.4).sum()}
Labels near default (0.45-0.55): {((df_labels['best_threshold'] >= 0.45) & (df_labels['best_threshold'] <= 0.55)).sum()}

Insight: {"Significant variation in optimal thresholds suggests per-label thresholding could improve performance" if df_labels['best_threshold'].std() > 0.1 else "Thresholds are relatively consistent, global threshold is reasonable"}
"""
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_label_thresholds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-label threshold analysis plot saved")


def plot_label_difficulty_analysis(df_labels: pd.DataFrame, save_dir: Path):
    """Categorize and visualize labels by difficulty/performance."""
    
    # Categorize labels by F1 score
    df_labels['category'] = pd.cut(df_labels['f1'], 
                                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                                    labels=['Poor (<0.3)', 'Fair (0.3-0.6)', 'Good (0.6-0.8)', 'Excellent (0.8+)'])
    
    df_labels['support'] = df_labels['true_positives'] + df_labels['false_negatives']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Label Difficulty & Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Category distribution
    ax = axes[0, 0]
    category_counts = df_labels['category'].value_counts().sort_index()
    colors = ['red', 'orange', 'lightgreen', 'darkgreen']
    bars = ax.bar(range(len(category_counts)), category_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=15, ha='right')
    ax.set_ylabel('Number of Labels')
    ax.set_title('Label Performance Categories')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: F1 vs Support (colored by category)
    ax = axes[0, 1]
    for category, color in zip(['Poor (<0.3)', 'Fair (0.3-0.6)', 'Good (0.6-0.8)', 'Excellent (0.8+)'],
                                ['red', 'orange', 'lightgreen', 'darkgreen']):
        mask = df_labels['category'] == category
        if mask.any():
            ax.scatter(df_labels[mask]['support'], df_labels[mask]['f1'],
                      label=category, alpha=0.6, s=60, color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Label Support (# Occurrences)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance vs Frequency')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for category boundaries
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
    
    # Plot 3: Worst performers details
    ax = axes[1, 0]
    worst = df_labels.nsmallest(15, 'f1')
    y_pos = np.arange(len(worst))
    bars = ax.barh(y_pos, worst['f1'].values, color='red', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(worst['label'].values, fontsize=8)
    ax.set_xlabel('F1 Score')
    ax.set_title('15 Most Difficult Labels')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 4: Performance summary by support level
    ax = axes[1, 1]
    # Bin by support level
    df_labels['support_bin'] = pd.cut(df_labels['support'], 
                                      bins=[0, 10, 50, 100, 500, float('inf')],
                                      labels=['1-10', '11-50', '51-100', '101-500', '500+'])
    
    support_performance = df_labels.groupby('support_bin', observed=True).agg({
        'f1': ['mean', 'std', 'count']
    }).round(3)
    
    x_pos = np.arange(len(support_performance))
    means = support_performance['f1']['mean'].values
    stds = support_performance['f1']['std'].fillna(0).values
    
    bars = ax.bar(x_pos, means, yerr=stds, alpha=0.7, color='steelblue', 
                  capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(support_performance.index, rotation=15, ha='right')
    ax.set_xlabel('Support Range')
    ax.set_ylabel('Mean F1 Score')
    ax.set_title('Performance by Label Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, support_performance['f1']['count'].values)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + stds[i],
                f'n={int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'label_difficulty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Label difficulty analysis plot saved")


def plot_threshold_f1_improvement(df_labels: pd.DataFrame, save_dir: Path, df_labels_tla: Optional[pd.DataFrame] = None):
    """Show F1 improvement from using best threshold vs default 0.5, and TLA impact."""
    
    # Calculate F1 at default threshold (0.5) for comparison
    # We use the current f1 as the optimized one, and estimate what it would be at 0.5
    # This is approximate since we don't have the actual per-threshold data here
    
    df_labels['f1_improvement'] = df_labels['best_f1'] - df_labels['f1']
    df_sorted = df_labels.sort_values('f1_improvement', ascending=False)
    
    # Determine grid size based on whether TLA data is available
    if df_labels_tla is not None:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impact of Per-Label Threshold Optimization', fontsize=16, fontweight='bold')
    
    # Plot 1: Top improvers
    ax = axes[0, 0]
    top_20 = df_sorted.head(20)
    y_pos = np.arange(len(top_20))
    bars = ax.barh(y_pos, top_20['f1_improvement'].values, color='green', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20['label'].values, fontsize=8)
    ax.set_xlabel('F1 Improvement')
    ax.set_title('Top 20 Labels Benefiting from Threshold Tuning')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 2: Distribution of improvements
    ax = axes[0, 1]
    ax.hist(df_labels['f1_improvement'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax.axvline(x=df_labels['f1_improvement'].mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean ({df_labels["f1_improvement"].mean():.4f})')
    ax.set_xlabel('F1 Improvement')
    ax.set_ylabel('Number of Labels')
    ax.set_title('Distribution of F1 Improvements')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Improvement vs threshold deviation from 0.5
    ax = axes[1, 0]
    df_labels['threshold_deviation'] = (df_labels['best_threshold'] - 0.5).abs()
    scatter = ax.scatter(df_labels['threshold_deviation'], df_labels['f1_improvement'],
                        c=df_labels['f1'], cmap='RdYlGn', alpha=0.6, s=50)
    ax.set_xlabel('Threshold Deviation from 0.5 (absolute)')
    ax.set_ylabel('F1 Improvement')
    ax.set_title('Threshold Change vs Performance Gain')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Final F1 Score')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    total_improvement = df_labels['f1_improvement'].sum()
    mean_improvement = df_labels['f1_improvement'].mean()
    labels_improved = (df_labels['f1_improvement'] > 0.01).sum()
    labels_hurt = (df_labels['f1_improvement'] < -0.01).sum()
    
    stats_text = f"""Threshold Optimization Impact:
    
Total F1 Improvement: {total_improvement:+.4f}
Mean per Label: {mean_improvement:+.4f}
Median: {df_labels['f1_improvement'].median():+.4f}

Labels Significantly Improved (>0.01): {labels_improved}
Labels Slightly Hurt (<-0.01): {labels_hurt}
Labels Unchanged (±0.01): {len(df_labels) - labels_improved - labels_hurt}

Best Single Improvement: {df_labels['f1_improvement'].max():.4f}
  Label: {df_sorted.iloc[0]['label']}

Conclusion: {"Per-label thresholds provide substantial benefit" if mean_improvement > 0.02 else "Per-label thresholds provide moderate benefit" if mean_improvement > 0.005 else "Per-label thresholds provide minimal benefit"}
"""
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if mean_improvement > 0.01 else 'wheat', alpha=0.5))
    ax.axis('off')
    
    # Plot 5 & 6: Progressive improvement comparison (if TLA data available)
    if df_labels_tla is not None:
        # Ensure we're comparing the same labels
        common_labels = df_labels['label'].values
        
        # Get F1 scores for each scenario (convert to numpy arrays explicitly)
        f1_default_threshold = np.array(df_labels['f1'].values, dtype=float)  # Using default 0.5 threshold
        f1_per_label_threshold = np.array(df_labels['best_f1'].values, dtype=float)  # Optimized per-label threshold
        
        # Match TLA metrics to the same label order
        df_tla_sorted = df_labels_tla.set_index('label').reindex(common_labels).reset_index()
        f1_tla_per_label = np.array(df_tla_sorted['best_f1'].values, dtype=float)  # TLA + per-label threshold
        
        # Calculate improvements
        improvement_stage1 = f1_per_label_threshold - f1_default_threshold  # Default → Per-label
        improvement_stage2 = f1_tla_per_label - f1_per_label_threshold  # Per-label → TLA+Per-label
        total_improvement = f1_tla_per_label - f1_default_threshold  # Default → TLA+Per-label
        
        # Plot 5: Stacked bar chart showing progressive improvements
        ax = axes[2, 0]
        
        # Sort by total improvement for better visualization
        sort_idx = np.argsort(total_improvement)[::-1][:30]  # Top 30 labels
        
        x_pos = np.arange(len(sort_idx))
        baseline = f1_default_threshold[sort_idx]
        stage1_heights = improvement_stage1[sort_idx]
        stage2_heights = improvement_stage2[sort_idx]
        
        # Create stacked bars
        ax.bar(x_pos, baseline, label='Default Threshold (0.5)', alpha=0.7, color='lightcoral')
        ax.bar(x_pos, stage1_heights, bottom=baseline, label='+ Per-Label Threshold', alpha=0.8, color='gold')
        ax.bar(x_pos, stage2_heights, bottom=baseline + stage1_heights, 
               label='+ TLA Adjustment', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Top 30 Labels (by total improvement)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Progressive F1 Improvement: Default → Per-Label → TLA+Per-Label')
        ax.set_xticks([])  # Too many labels to show individually
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Plot 6: Summary comparison
        ax = axes[2, 1]
        
        # Calculate mean F1 scores across all labels for each scenario
        mean_f1_default = f1_default_threshold.mean()
        mean_f1_per_label = f1_per_label_threshold.mean()
        mean_f1_tla_per_label = f1_tla_per_label.mean()
        
        scenarios = ['Default\nThreshold\n(0.5)', 'Per-Label\nThreshold', 'TLA +\nPer-Label\nThreshold']
        means = [mean_f1_default, mean_f1_per_label, mean_f1_tla_per_label]
        colors_bar = ['lightcoral', 'gold', 'lightgreen']
        
        bars = ax.bar(scenarios, means, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add improvement arrows and percentages
        stage1_improvement = mean_f1_per_label - mean_f1_default
        stage2_improvement = mean_f1_tla_per_label - mean_f1_per_label
        total_improvement_mean = mean_f1_tla_per_label - mean_f1_default
        
        # Arrow 1: Default to Per-Label
        ax.annotate('', xy=(1, mean_f1_per_label), xytext=(0, mean_f1_default),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax.text(0.5, (mean_f1_default + mean_f1_per_label) / 2,
                f'+{stage1_improvement:.4f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Arrow 2: Per-Label to TLA+Per-Label
        ax.annotate('', xy=(2, mean_f1_tla_per_label), xytext=(1, mean_f1_per_label),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.text(1.5, (mean_f1_per_label + mean_f1_tla_per_label) / 2,
                f'{stage2_improvement:+.4f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('Mean F1 Score', fontweight='bold')
        ax.set_title('Progressive Optimization Impact (Mean Across All Labels)', fontweight='bold')
        ax.set_ylim(0, max(means) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add summary text
        summary = f'Total Improvement: {total_improvement_mean:+.4f} ({(total_improvement_mean/mean_f1_default)*100:+.1f}%)'
        ax.text(0.5, 0.95, summary, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_optimization_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Threshold optimization impact plot saved")


def plot_comprehensive_metrics_correlation(df_labels: pd.DataFrame, save_dir: Path):
    """Create correlation heatmap and scatter matrix for all metrics."""
    
    # Select numerical columns for correlation
    df_labels['support'] = df_labels['true_positives'] + df_labels['false_negatives']
    
    metrics_cols = ['f1', 'precision', 'recall', 'best_f1', 'best_precision', 
                   'best_recall', 'best_threshold', 'support', 'average_precision']
    df_metrics = df_labels[metrics_cols].copy()
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main correlation heatmap
    ax_heatmap = fig.add_subplot(gs[0:2, 0:2])
    corr_matrix = df_metrics.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax_heatmap)
    ax_heatmap.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Scatter plots for key relationships
    # Support vs F1
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.scatter(df_labels['support'], df_labels['f1'], alpha=0.5, s=30, color='steelblue')
    ax1.set_xlabel('Support (log scale)', fontsize=9)
    ax1.set_ylabel('F1 Score', fontsize=9)
    ax1.set_xscale('log')
    ax1.set_title('Support vs F1', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Precision vs Recall
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.scatter(df_labels['recall'], df_labels['precision'], alpha=0.5, s=30, color='green')
    ax2.set_xlabel('Recall', fontsize=9)
    ax2.set_ylabel('Precision', fontsize=9)
    ax2.set_title('Precision vs Recall', fontsize=10)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Perfect Balance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Best threshold vs F1
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(df_labels['best_threshold'], df_labels['f1'], alpha=0.5, s=30, color='purple')
    ax3.set_xlabel('Best Threshold', fontsize=9)
    ax3.set_ylabel('F1 Score', fontsize=9)
    ax3.set_title('Threshold vs F1', fontsize=10)
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # Average Precision distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(df_labels['average_precision'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Average Precision (AP)', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('AP Distribution', fontsize=10)
    ax4.axvline(x=df_labels['average_precision'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_labels["average_precision"].mean():.3f}')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # mAP summary
    ax5 = fig.add_subplot(gs[2, 2])
    mAP = df_labels['average_precision'].mean()
    
    stats_text = f"""Summary Statistics:
    
mAP (mean AP): {mAP:.4f}

Metrics Analyzed: {len(metrics_cols)}
Labels: {len(df_labels)}

Key Insights:
• Heatmap shows metric relationships
• Strong corr = redundant metrics
• Weak corr = complementary info
• Support affects performance
"""
    
    ax5.text(0.05, 0.5, stats_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax5.axis('off')
    
    fig.suptitle('Comprehensive Metrics Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_dir / 'metrics_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Metrics correlation analysis plot saved")


def visualize_predictions(
    model: torch.nn.Module,
    dataset,
    label_columns: List[str],
    device: torch.device,
    save_dir: Path,
    num_samples: int = 10,
    threshold: float = 0.5,
    prob_threshold: float = 0.1
):
    """Visualize predictions on sample images."""
    model.eval()
    
    # Randomly select samples with time-based seed to ensure different samples each run
    # while keeping the rest of training reproducible
    import time
    rng = np.random.default_rng(seed=int(time.time() * 1000) % (2**32))
    indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        image_tensor, true_labels = dataset[sample_idx]
        
        # Get prediction
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        
        # Get predicted labels
        pred_labels = (probs >= threshold).astype(int)
        
        # Get true label names
        true_label_names = [label_columns[i] for i, val in enumerate(true_labels) if val == 1]
        
        # Get predicted label names
        pred_label_names = [label_columns[i] for i in range(len(probs)) if pred_labels[i] == 1]
        
        # Get all labels with prob > prob_threshold
        high_prob_labels = [(label_columns[i], probs[i]) for i in range(len(probs)) if probs[i] > prob_threshold]
        high_prob_labels = sorted(high_prob_labels, key=lambda x: x[1], reverse=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display image
        # Denormalize image for display
        img_display = image_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        
        ax1.imshow(img_display)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Display labels info
        ax2.axis('off')
        
        info_text = "TRUE LABELS:\n"
        info_text += "\n".join([f"  • {name}" for name in true_label_names[:15]])
        if len(true_label_names) > 15:
            info_text += f"\n  ... and {len(true_label_names) - 15} more"
        
        info_text += "\n\nPREDICTED LABELS:\n"
        info_text += "\n".join([f"  • {name}" for name in pred_label_names[:15]])
        if len(pred_label_names) > 15:
            info_text += f"\n  ... and {len(pred_label_names) - 15} more"
        
        info_text += f"\n\nTOP PROBABILITIES (>{prob_threshold*100:.0f}%):\n"
        for name, prob in high_prob_labels[:15]:
            info_text += f"  • {name}: {prob*100:.1f}%\n"
        if len(high_prob_labels) > 15:
            info_text += f"  ... and {len(high_prob_labels) - 15} more"
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_dir / f'prediction_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
