# Author: Francesco Chiumento
# License: MIT

"""
Multi-Tracer Dataset Merger

Combines PET datasets from different tracers (PIB and AV45) into a single
unified CSV file with tracer identification column.

Input: Separate CSV files for PIB and AV45 tracers
Output: Combined CSV with tracer column
"""
#========================================

import pandas as pd

# paths
pib_path = "/path/to/data/selected_pup_PIB_patients_ALL_TIMEPOINTS.csv"
av45_path = "/path/to/data/selected_pup_AV45_patients_ALL_TIMEPOINTS.csv"

# Output path
combined_path = "/path/to/output/selected_pup_patients_combined_ALL_TIMEPOINTS.csv"

# load both datasets
df_pib = pd.read_csv(pib_path, sep="\t")
df_av45 = pd.read_csv(av45_path, sep="\t")

# add tracer column
df_pib["tracer"] = "PIB"
df_av45["tracer"] = "AV45"

# combine datasets
df_combined = pd.concat([df_pib, df_av45], ignore_index=True)

# save combined CSV
df_combined.to_csv(combined_path, sep="\t", index=False)

print(f"Combined CSV saved to: {combined_path}")
print(f"Total rows: {len(df_combined)}")
print(f"Unique patients: {df_combined['subject_id'].nunique()}")