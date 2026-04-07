# Author: Francesco Chiumento
# License: MIT
"""
ROC Curve Generator for Multi-Sequence MRI Classification

Generates ROC curves comparing classification performance across MRI sequences.
Computes AUC, optimal thresholds (Youden's J), sensitivity, and specificity.

Input: CSV files with predictions (patient_id, true_label, predicted_label, probability)
Output: Multi-sequence ROC plots (PDF/PNG/SVG) and LaTeX tables
"""
#========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path


CSV_PATHS = {
    'T1w': "/path/to/results/T1w/test_predictions_T1w.csv",
    'T2w': "/path/to/results/T2w/test_predictions_T2w.csv",
    'FLAIR': "/path/to/results/FLAIR/test_predictions_FLAIR.csv",
    'T2*': "/path/to/results/T2star/test_predictions_T2star.csv"
}

OUTPUT_DIR = "/path/to/output/ROC_graphs"

# colors and style

SEQUENCE_COLORS = {
    'T1w': '#2E86AB',    
    'T2w': '#A23B72',   
    'FLAIR': '#F18F01',  
    'T2*': '#06A77D' 
}


plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.linewidth': 1.3,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0
})


def load_predictions_from_csv(csv_path):

    try:
        df = pd.read_csv(csv_path)
        
        # verify essential columns
        required_cols = ['amyloid_true', 'prediction_probability']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}\n"
                           f"Found columns: {df.columns.tolist()}")
        
        # Extract data
        y_true = df['amyloid_true'].values
        y_scores = df['prediction_probability'].values
        
        # statistics
        n_samples = len(y_true)
        n_positive = y_true.sum()
        prevalence = n_positive / n_samples
        
        # validation
        assert len(y_true) == len(y_scores), "Mismatch in y_true/y_scores length"
        assert set(y_true).issubset({0, 1}), "y_true must contain only 0/1"
        assert y_scores.min() >= 0 and y_scores.max() <= 1, "y_scores out of range [0,1]"
        
        print(f"Loaded {csv_path.split('/')[-1]}")
        print(f" N={n_samples}, Pos={n_positive} ({prevalence:.1%})")
        
        return y_true, y_scores, n_samples, prevalence
        
    except FileNotFoundError:
        print(f" File not found: {csv_path}")
        return None, None, 0, 0.0
    except Exception as e:
        print(f" Error during loading: {e}")
        raise

def compute_roc_metrics(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sens = tpr[optimal_idx]
    optimal_spec = 1 - fpr[optimal_idx]
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': optimal_sens,
        'optimal_specificity': optimal_spec
    }

# PLOTTING

def plot_oasis_roc_curves(roc_data, output_path):

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Reference diagonal (Random Classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, 
                label='Random Classifier', alpha=0.5, zorder=1)
    
    # 2. Plot curve for each sequence
    for seq_name in ['T1w', 'T2w', 'FLAIR', 'T2*']:  # Specific order
        if seq_name not in roc_data or roc_data[seq_name] is None:
            print(f"Sequence {seq_name} not available, skip.")
            continue
        
        metrics = roc_data[seq_name]
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        roc_auc = metrics['auc']
        
        # Plot
        ax.plot(fpr, tpr, 
                color=SEQUENCE_COLORS[seq_name],
                linewidth=2.0,
                label=f'{seq_name} (AUC = {roc_auc:.2f})', 
                zorder=3)
    
    ax.set_xlabel('False Positive Rate (1 − Specificity)', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                  fontsize=11, fontweight='bold')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend = ax.legend(loc='lower right', fontsize=9.5, 
                      frameon=True, shadow=False, 
                      fancybox=False, framealpha=0.98) 
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black') 
    legend.get_frame().set_linewidth(0.8)
    
    for ext in ['pdf', 'png', 'svg']:
        output_file = f"{output_path}.{ext}"
        plt.savefig(output_file, dpi=300 if ext == 'png' else None, 
                   bbox_inches='tight', facecolor='white')
        print(f" saved: {output_file}")
    
    plt.close()

def main():
    
    print("=" * 70)
    print("OASIS-3 ROC CURVE GENERATOR")
    print("=" * 70)
    print()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # 1. load all data from CSV
    print("STEP 1: Load predictions from CSV")
    print("-" * 70)
    roc_data = {}
    
    for seq_name, csv_path in CSV_PATHS.items():
        print(f"\n[{seq_name}]")
        y_true, y_scores, n_samples, prevalence = load_predictions_from_csv(csv_path)
        
        if y_true is not None:
            metrics = compute_roc_metrics(y_true, y_scores)
            roc_data[seq_name] = metrics
            
            print(f"   AUC: {metrics['auc']:.2f}")
            print(f"   Optimal threshold: {metrics['optimal_threshold']:.4f}")
            print(f"   @ Optimal: Sens={metrics['optimal_sensitivity']:.2f}, "
                  f"Spec={metrics['optimal_specificity']:.2f}")
        else:
            roc_data[seq_name] = None
    
    # 2. plot graph
    print("\n" + "=" * 70)
    print("STEP 2: Generating ROC curve plot")
    print("-" * 70)
    
    output_base = Path(OUTPUT_DIR) / "oasis_roc_multisequence"
    plot_oasis_roc_curves(roc_data, str(output_base))
    
    # 3. Generate summary LaTeX table
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Test set ROC-AUC across MRI sequences. "
          "All models are MRI-only students trained via PET-guided knowledge distillation.}")
    print("\\label{tab:roc_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Metric & T1w & T2w & FLAIR & T2* \\\\")
    print("\\midrule")
    
# AUC row
    auc_row = ["AUC"]
    for seq in ['T1w', 'T2w', 'FLAIR', 'T2*']:
        if seq in roc_data and roc_data[seq]:
            auc_row.append(f"{roc_data[seq]['auc']:.2f}")
        else:
            auc_row.append("--")
    print(" & ".join(auc_row) + " \\\\")
    
# Optimal Sens row
    sens_row = ["Sens$_{opt}$"]
    for seq in ['T1w', 'T2w', 'FLAIR', 'T2*']:
        if seq in roc_data and roc_data[seq]:
            sens_row.append(f"{roc_data[seq]['optimal_sensitivity']:.2f}")
        else:
            sens_row.append("--")
    print(" & ".join(sens_row) + " \\\\")
    
# Optimal Spec row
    spec_row = ["Spec$_{opt}$"]
    for seq in ['T1w', 'T2w', 'FLAIR', 'T2*']:
        if seq in roc_data and roc_data[seq]:
            spec_row.append(f"{roc_data[seq]['optimal_specificity']:.2f}")
        else:
            spec_row.append("--")
    print(" & ".join(spec_row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    print("=" * 70)
    print("completed ! file saved in:")
    print(f"   {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()