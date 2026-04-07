# Author: Francesco Chiumento
# License: MIT

"""
PET Baseline Clinical Data Matcher

Matches baseline PET scans with temporally proximal clinical assessments.
Selects clinical evaluations within ±365 days of baseline PET scan.

Input: PET session list (PUP format), clinical assessment data (UDSb4)
Output: CSV with matched PET-clinical pairs within temporal window
"""
#========================================

import pandas as pd
import re

# load PUP file
pup_file_path = "/path/to/data/OASIS3_PUP_AV45_filtered.csv"
pup_df = pd.read_csv(pup_file_path, header=None, names=["session_id"])

# Extract subject_id and pet_days from PUP format string
def extract_pup_info(session_str):
    m = re.search(r'^(OAS\d+)_AV45_PUPTIMECOURSE_d(\d+)$', session_str)
    if m:
        return pd.Series([m.group(1), int(m.group(2))])
    return pd.Series([None, None])

pup_df[["subject_id", "pet_days"]] = pup_df["session_id"].apply(extract_pup_info)

# select baseline PET (first recorded exam)
pup_baseline = pup_df.groupby("subject_id", as_index=False)["pet_days"].min()
pup_baseline.rename(columns={"pet_days": "baseline_pet_days"}, inplace=True)

# Load clinical file 
clinical_file_path = "/path/to/data/clinical_data/OASIS3_UDSb4_cdr.csv"
clinical_df = pd.read_csv(clinical_file_path)

# Extract subject_id and clinical_days from "OASIS_session_label" column
def extract_clinical_info(label):
    m = re.search(r'^(OAS\d+)_UDSb4_d(\d+)$', label)
    if m:
        return pd.Series([m.group(1), int(m.group(2))])
    return pd.Series([None, None])

clinical_df[["subject_id", "clinical_days"]] = clinical_df["OASIS_session_label"].apply(extract_clinical_info)

# merge baseline pet data with clinical data
merged_df = pd.merge(clinical_df, pup_baseline, on="subject_id", how="inner")

# calculate day difference between PET baseline and clinical assessment
merged_df["day_difference"] = merged_df["clinical_days"] - merged_df["baseline_pet_days"]

# filter clinical assessments within ±365 days
selected = merged_df[merged_df["day_difference"].abs() <= 365].copy()

# select only required columns and rename
final_df = selected[["subject_id", "OASIS_session_label", "day_difference"]].copy()
final_df.rename(columns={"OASIS_session_label": "session_id_clinical"}, inplace=True)

# Add PET session
final_df = pd.merge(final_df, pup_df[["subject_id", "session_id"]], on="subject_id", how="left")
final_df.rename(columns={"session_id": "session_id_pet"}, inplace=True)

# save file
output_path = "/path/to/output/selected_PUP_patients_all_365_days_baseline.csv"
final_df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")
