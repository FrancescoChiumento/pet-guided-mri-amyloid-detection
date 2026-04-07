# Author: Francesco Chiumento
# License: MIT
"""
Script for processing and matching PET, clinical, and MRI data from OASIS-3 dataset.
This code performs temporal alignment between PET scans and clinical assessments,
extracts Centiloid values, and generates labels for dementia and amyloid positivity.
"""

import pandas as pd
import os
import re

# define the tracer: "AV45" or "PiB"
TRACER = "PIB"  

# path organization
DATA_ROOT = "/path/to/data/OASIS_3"
OUTPUT_ROOT = "/path/to/output/OASIS_3"

file_pet = f"{DATA_ROOT}/OASIS3_PUP_PIB_filtered.csv"
file_clinical = f"{DATA_ROOT}/clinical_data/OASIS3_UDSb4_cdr.csv"
file_mri = f"{DATA_ROOT}/OASIS3_PUP.csv"
file_centiloid = f"{DATA_ROOT}/clinical_data/OASIS3_amyloid_centiloid.csv"
output_file = f"{OUTPUT_ROOT}/selected_pup_PIB_patients_ALL_TIMEPOINTS.csv"

# complete loading of clinical file
clinical_df_full = pd.read_csv(file_clinical)
clinical_df_full.columns = clinical_df_full.columns.str.strip()

centiloid_df = pd.read_csv(file_centiloid)
centiloid_df.columns = centiloid_df.columns.str.strip()

# check if files exist
for file in [file_pet, file_clinical, file_mri]:
    if not os.path.exists(file):
        print(f"Error: the file {file} doesn't exist.")

# loading file PET one column 
pet_df = pd.read_csv(file_pet, header=None, names=["session_id_pet"])

#function to extract subject_id and pet_days from the string
def extract_pet_info(session_str):
    pattern = rf'^(OAS\d+)_({TRACER})_PUPTIMECOURSE_d(\d+)$'
    m = re.search(pattern, str(session_str))
    if m:
        return pd.Series([m.group(1), int(m.group(3))])
    return pd.Series([None, None])

pet_df[["subject_id", "pet_days"]] = pet_df["session_id_pet"].apply(extract_pet_info)

# check results
print(pet_df.head())

# loading clinical files
clinical_df = pd.read_csv(file_clinical)

# function to extract subject_id and clinical_days
def extract_clinical_info(label):
    m = re.search(r'^(OAS\d+)_UDSb4_d(\d+)$', str(label))
    if m:
        return pd.Series([m.group(1), int(m.group(2))])
    return pd.Series([None, None])

clinical_df[["subject_id", "clinical_days"]] = clinical_df["OASIS_session_label"].apply(extract_clinical_info)

# merging all clinical data with PET (not only baseline)
merged_df = pd.merge(clinical_df, pet_df, on="subject_id", how="inner")

# Calculate difference between pet and clinical session
merged_df["day_difference"] = merged_df["clinical_days"] - merged_df["pet_days"]

# select row with a maximum difference of 365 days
selected = merged_df[merged_df["day_difference"].abs() <= 365].copy()

final_df = selected[["subject_id", "OASIS_session_label", "day_difference", "session_id_pet"]].rename(columns={"OASIS_session_label": "session_id_clinical"})

print(final_df.head())

# loading MRI files
mri_df = pd.read_csv(file_mri, delimiter=",", engine="python", on_bad_lines='skip')

# remove extra spaces in column names
mri_df.columns = mri_df.columns.str.strip()

print("column of mri_df:", mri_df.columns.tolist())

# rename key column in MRI file
if "PUP_PUPTIMECOURSEDATA ID" in mri_df.columns:
    mri_df.rename(columns={"PUP_PUPTIMECOURSEDATA ID": "session_id_pet"}, inplace=True)
else:
    print("Error: key column not found in mri_df.")
    exit(1)

if "MRId" not in mri_df.columns:
    print("Error: Key column not found in mri_df.")
    exit(1)

#  merging clinical and PET data with MRIs
merged_final_df = final_df.merge(mri_df[['session_id_pet', 'MRId']], on='session_id_pet', how='left')

print(merged_final_df.head())

cols_to_keep = ["OASIS_session_label", "CDRTOT", "CDRSUM", "MMSE", "dx1"]

merged_final_df = merged_final_df.merge(
    clinical_df_full[cols_to_keep],
    left_on="session_id_clinical",
    right_on="OASIS_session_label",
    how="left"
)

# label: cognitively impaired = 1 if global CDR >= 1
merged_final_df["dementia_label"] = (merged_final_df["CDRTOT"] >= 1).astype(int)

merged_final_df["MMSE"] = pd.to_numeric(merged_final_df["MMSE"], errors="coerce")

print(" Clinical data:")
print(merged_final_df[["subject_id", "session_id_clinical", "CDRTOT", "MMSE", "dx1"]].head())

print("CDR distribution:")
print(merged_final_df["CDRTOT"].value_counts())

print("\nMMSE Statistic:")
print(merged_final_df["MMSE"].describe())

print("\nDiagnosis distribution (dx1):")
print(merged_final_df["dx1"].value_counts())

merged_final_df["oasis_session_id"] = merged_final_df["session_id_pet"].str.replace(
    r"_PUPTIMECOURSE", "", regex=True
)

# merging with centiloid values
merged_final_df = merged_final_df.merge(
    centiloid_df[["oasis_session_id", "Centiloid_fSUVR_rsf_TOT_CORTMEAN"]],
    on="oasis_session_id",
    how="left"
)

# label: positive if Centiloid > 20.6
merged_final_df["amyloid_positive"] = (merged_final_df["Centiloid_fSUVR_rsf_TOT_CORTMEAN"] > 20.6).astype(int)

print(" Centiloid merged")
print(merged_final_df[[
    "subject_id", "session_id_pet", "Centiloid_fSUVR_rsf_TOT_CORTMEAN", "amyloid_positive"
]].head())

print(f"Total combination number: {len(merged_final_df)}")
print(f"Unique patients number: {merged_final_df['subject_id'].nunique()}")
print(f"average patients sessions: {len(merged_final_df) / merged_final_df['subject_id'].nunique():.1f}")

print("\n Multiple patients sessions example:")
patient_counts = merged_final_df['subject_id'].value_counts()
multi_session_patients = patient_counts[patient_counts > 1].head(5)
for patient, count in multi_session_patients.items():
    print(f"  - {patient}: {count} sessions")
    
merged_final_df.to_csv(output_file, index=False, sep='\t')

print(f"\n file updated {output_file}")