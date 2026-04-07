# Author: Francesco Chiumento
# License: MIT

"""
Evaluation Metrics Visualization for Binary Classification

Generates confusion matrices and ROC curves with optimal thresholds.
Computes AUC, sensitivity, specificity, PPV, and NPV.

Output: High-resolution plots (300 DPI) organized by validation/test folders.
"""
#========================================
#========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import os

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix", threshold=0.5):
    """
    create and save a confusion matrix visualization
    
    Args:
        y_true: ground truth label
        y_pred: predicted probabilities
        save_path: Path to save the figure
        title: plot title
        threshold: threshold to convert probabilities to binary predictions

    Returns:
        cm: Confusion matrix array
    """
    # convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # create figure
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 16})
    
    plt.title(f'{title}\n(Threshold: {threshold:.3f})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    
    # Add metrics annotations
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}',
             ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def plot_roc_curve(y_true, y_scores, save_path, title="ROC Curve"):

    #roc calculation
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # find optimal point (Youden's J statistic)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    optimal_point = (fpr[youden_idx], tpr[youden_idx])
    
    # create figure
    plt.figure(figsize=(8, 8))
    
    # Plot ROC
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    
    # diagonal line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    # optimal point
    plt.scatter(*optimal_point, c='red', s=100, zorder=5, 
                label=f'Optimal Point (threshold={optimal_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'{title}\nAUC = {roc_auc:.3f}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # add annotation for optimal point
    plt.annotate(f'Sens: {tpr[youden_idx]:.3f}\nSpec: {1-fpr[youden_idx]:.3f}',
                xy=optimal_point,
                xytext=(optimal_point[0] + 0.1, optimal_point[1] - 0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc, optimal_threshold

def create_evaluation_plots(y_true, y_probs, save_dir, prefix="", threshold=0.5):

    # Determine subfolder based on prefix

    if prefix.startswith("val_"):
        plot_subdir = os.path.join(save_dir, "evaluation_plots", "validation")
    elif prefix.startswith("test_"):
        plot_subdir = os.path.join(save_dir, "evaluation_plots", "test")
    else:
        plot_subdir = os.path.join(save_dir, "evaluation_plots", "other")
    
    os.makedirs(plot_subdir, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(plot_subdir, f"{prefix}confusion_matrix.png")
    cm = plot_confusion_matrix(y_true, y_probs, cm_path, 
                              title=f"{prefix.replace('_', ' ').title()}Confusion Matrix",
                              threshold=threshold)
    
    # ROC curve
    roc_path = os.path.join(plot_subdir, f"{prefix}roc_curve.png")
    auc_score, optimal_thresh = plot_roc_curve(y_true, y_probs, roc_path,
                                              title=f"{prefix.replace('_', ' ').title()}ROC Curve")
    
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'auc': auc_score,
        'optimal_threshold': optimal_thresh,
        'confusion_matrix': cm,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
    }
    
    return metrics