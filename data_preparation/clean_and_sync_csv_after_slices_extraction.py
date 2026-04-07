# Author: Francesco Chiumento
# License: MIT
"""
CSV Deduplication and Validation Tool

Cleans dataset CSV by removing duplicate imaging sessions and validating
folder existence. Keeps most temporally proximal clinical timepoint for
each unique MRI-PET combination.

Input: CSV with MRI-PET associations, paths to MRI and PET slice directories
Output: Cleaned CSV with deduplicated entries and verified folder existence
"""
#========================================

import pandas as pd
import os

def clean_and_sync_csv(original_csv_path, mri_dir, pet_dir, output_csv_path):
    
    # load csv
    print("Loading original CSV")
    df = pd.read_csv(original_csv_path, sep='\t')
    print(f" Initial Rows: {len(df)}")
    
    print("\nIdentifying unique imaging combinations...")
    
    # extract sessions from columns
    df['mri_session'] = df['MRId'].str.split('_').str[-1] 
    df['pet_session'] = df['session_id_pet'].str.split('_').str[-1]
    
    # create unique key for each imaging combination
    df['imaging_key'] = (df['subject_id'] + '_' + 
                        df['mri_session'] + '_' + 
                        df['pet_session'] + '_' + 
                        df['tracer'])
    
    # count duplicates
    duplicates = df[df.duplicated('imaging_key', keep=False)]
    print(f"   Found {len(duplicates)} rows with duplicate imaging")

    df['days_diff_abs'] = df['day_difference'].abs() 
    
    # Sort by imaging_key and days_diff_abs
    df_sorted = df.sort_values(['imaging_key', 'days_diff_abs'])
    
    # keep only first row for each imaging_key
    df_unique = df_sorted.drop_duplicates('imaging_key', keep='first')
    print(f"   rows after removing duplicate imaging: {len(df_unique)}")
    
    print("\nVerifying folder existence")
    rows_to_keep = []
    missing_mri = 0
    missing_pet = 0
    missing_both = 0
    
    for idx, row in df_unique.iterrows():
        # Build paths
        mri_path = os.path.join(mri_dir, 
                               f"{row['subject_id']}_ses-{row['mri_session']}", 
                               "FLAIR_slices")
        pet_path = os.path.join(pet_dir, 
                               f"{row['subject_id']}_ses-{row['pet_session']}_{row['tracer']}", 
                               "PET_slices")
        
        mri_exists = os.path.exists(mri_path)
        pet_exists = os.path.exists(pet_path)
        
        if mri_exists and pet_exists:
            rows_to_keep.append(idx)
        else:
            if not mri_exists and not pet_exists:
                missing_both += 1
            elif not mri_exists:
                missing_mri += 1
            else:
                missing_pet += 1
    
    print(f"   MRI only missing: {missing_mri}")
    print(f"   PET only missing: {missing_pet}")
    print(f"   Both missing: {missing_both}")
    
    # create final dataframe
    df_final = df_unique.loc[rows_to_keep].copy()
    
    # remove temporary columns
    df_final = df_final.drop(['mri_session', 'pet_session', 'imaging_key', 'days_diff_abs'], axis=1)
    
    # final statistics
    print(f"\nSummary")
    print(f"   Original Rows: {len(df)}")
    print(f"   Unique rows (no duplicate imaging): {len(df_unique)}")
    print(f"   Final rows (with existing folders): {len(df_final)}")
    print(f"   Total rows removed: {len(df) - len(df_final)}")
    
    # Class balance analysis
    if 'amyloid_positive' in df_final.columns:
        pos_count = (df_final['amyloid_positive'] == 1).sum()
        neg_count = (df_final['amyloid_positive'] == 0).sum()
        print(f"\nClass balance")
        print(f"   Positive: {pos_count} ({pos_count/len(df_final)*100:.1f}%)")
        print(f"   Negative: {neg_count} ({neg_count/len(df_final)*100:.1f}%)")
        if pos_count > 0:
            print(f"   Neg/Pos ratio {neg_count/pos_count:.2f}")
    
    # save cleaned CSV
    df_final.to_csv(output_csv_path, sep='\t', index=False)
    print(f"\nCleaned CSV saved in: {output_csv_path}")
    
    return df_final

# Usage
if __name__ == "__main__":
    df_clean = clean_and_sync_csv(
        original_csv_path="/path/to/data/selected_patients_original.csv",
        mri_dir="/path/to/data/FLAIR_slices_extracted",
        pet_dir="/path/to/data/PET_slices_extracted",
        output_csv_path="/path/to/output/selected_patients_cleaned.csv"
    )