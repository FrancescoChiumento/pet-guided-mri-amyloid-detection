# Author: Francesco Chiumento
# License: MIT
"""
Stratified Group K-Fold Cross-Validation Split Generator

Creates 5-fold CV splits for multi-session data with patient-level grouping
to prevent data leakage. Maintains class balance across folds.

Method: StratifiedGroupKFold (scikit-learn)
- Groups by subject_id (all sessions from same patient in same fold)
- Stratifies by amyloid_positive label (binary classification)

Split Convention:
- Fold 0: Validation | Fold 1: Test | Folds 2-4: Training

Input: CSV with subject_id, amyloid_positive, imaging metadata
Output: CSV with 'split' column + JSON files per fold (subject IDs)
"""
#========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os
import json
# paths

combined_path = "/path/to/dataset/selected_patients_combined_ALL_TIMEPOINTS.csv"
output_path = "/path/to/dataset/selected_patients_combined_ALL_TIMEPOINTS_with_split.csv"
split_dir = "/path/to/dataset/splits_ALL_TIMEPOINTS"

# load csv
df = pd.read_csv(combined_path, sep="\t")

# prepare input for StratifiedGroupKFold
subject_ids = df["subject_id"].values
labels = df["amyloid_positive"].values
X_dummy = np.zeros(len(df))  

# split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(X_dummy, labels, groups=subject_ids))

# Maps subject id to fold
subject_to_fold = {}
for fold_idx, (_, test_idx) in enumerate(splits):
    fold_subjects = df.iloc[test_idx]["subject_id"].unique()
    for subj in fold_subjects:
        subject_to_fold[subj] = fold_idx

# apply split to dataframe
df["split"] = df["subject_id"].map(subject_to_fold)

# save final CSV
df.to_csv(output_path, sep="\t", index=False)
print(f"CSV saved in: {output_path}")

# save subjects per fold
os.makedirs(split_dir, exist_ok=True)
for fold in range(5):
    ids = [subj for subj, f in subject_to_fold.items() if f == fold]
    with open(os.path.join(split_dir, f"split_{fold}_subjects.json"), "w") as f:
        json.dump(ids, f)
