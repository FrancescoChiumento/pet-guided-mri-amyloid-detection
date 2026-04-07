# Author: Francesco Chiumento
# License: MIT

"""
Dataset Folder Cleanup Utility

Removes imaging session folders not present in the filtered dataset CSV.
Ensures only valid PET sessions (within 365-day matching window) are retained.

Input: CSV with session_id_pet column
Output: Cleaned directory with only matching session folders
"""
#========================================

import os
import pandas as pd
import shutil

# paths
base_folder = "/path/to/data/PUP_msum_SUVR"
csv_path = "/path/to/data/selected_pup_patients_all_365_days_baseline.csv"

# load data
df = pd.read_csv(csv_path, sep='\t')

# Sessions to keep
valid_sessions = set(df["session_id_pet"].str.replace("_PUPTIMECOURSE", "", regex=False))

all_folders = sorted(os.listdir(base_folder))

# Separation
folders_to_keep = [f for f in all_folders if f in valid_sessions]
folders_to_remove = [f for f in all_folders if f not in valid_sessions]

# === Report ===
print(f"\n Total number of folders in directory: {len(all_folders)}")
print(f" Folders to keep: {len(folders_to_keep)}")
print(f" Folders to delete: {len(folders_to_remove)}\n")

print("Folders to delete:")
for folder in folders_to_remove:
    print(" -", folder)

confirmation = input("\nProceed with deletion? (y/n): ").strip().lower()

if confirmation == 'y':
    for folder in folders_to_remove:
        full_path = os.path.join(base_folder, folder)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
    print("\n folders successfully deleted.")
else:
    print("\nOperation cancelled. No folders were deleted")
