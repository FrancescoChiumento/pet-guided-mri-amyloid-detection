# Author: Francesco Chiumento
# License: MIT
"""
Script for T1w MRI Quality Assessment and Selection

This script processes a dataset to identify and select high-quality T1-weighted 
MRI scans based on orientation and voxel spacing criteria. 

Main functionality:
- Reads patient data from CSV with existing T2w information
- Scans T1w MRI files in BIDS format
- Evaluates image quality based on:
  * Orientation (RAS, LPS, RPS, LPI, RPI)
  * Maximum voxel spacing (≤1.5mm)
- Selects best available scan (standard or multi-run)
- Outputs enhanced CSV with T1w quality metadata

Input:  CSV file with patient IDs and T2w data
Output: CSV file with added T1w availability and quality columns
"""
#========================================

import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np

print(" Step 1: loading csv file with T2w already preprocessed")
csv_path = '/path/to/dataset/selected_patients_combined_ALL_TIMEPOINTS_with_split.csv'
df = pd.read_csv(csv_path, sep='\t')
print(f" Loaded {len(df)} patients")

print("\n Step 2: adding columns for T1w")

df['has_T1w'] = False
df['T1w_type'] = ''
df['T1w_files'] = ''
df['T1w_best_file'] = ''
df['T1w_orientation'] = ''
df['T1w_spacing'] = ''
df['T1w_usable'] = False

def check_nifti_quality(nifti_path):
    """check orientation and spacing"""
    try:
        img = nib.load(nifti_path)
        orientation = ''.join(nib.aff2axcodes(img.affine))
        voxel_sizes = img.header.get_zooms()[:3]
        max_spacing = max(voxel_sizes)
        
        # including only specific orientations
        good_orientations = ['RAS', 'LPS', 'RPS', 'LPI', 'RPI']  
        is_good = (orientation in good_orientations) and (max_spacing <= 1.5) 
        
        return {
            'orientation': orientation,
            'max_spacing': round(max_spacing, 2),
            'is_good': is_good
        }
    except Exception as e:
        print(f"   Error in reading {nifti_path}: {e}")
        return None

print("\n Step 3: check T1w and choose the best")
t1w_root = '/path/to/dataset/T1w_MRIs'

for idx, row in df.iterrows():
    if idx % 50 == 0:
        print(f"   Processing patient {idx}/{len(df)}")
    
    mr_id = row['MRId']
    parts = mr_id.split('_MR_d')
    if len(parts) != 2:
        continue
    
    subject_id = parts[0]
    session_day = parts[1]
    
    # Check T1w
    t1w_path = os.path.join(t1w_root, f'sub-{subject_id}', f'ses-d{session_day}', 'anat')
    if os.path.exists(t1w_path):
        t1w_files = glob.glob(os.path.join(t1w_path, '*T1w.nii.gz'))
        if t1w_files:
            df.at[idx, 'has_T1w'] = True
            file_names = [os.path.basename(f) for f in t1w_files]
            df.at[idx, 'T1w_files'] = ', '.join(file_names)
            
            best_file = None
            best_quality = None
            
            # 1. Check with first standard test
            for f in t1w_files:
                if 'run-' not in f:  # Standard T1w
                    full_path = os.path.join(t1w_path, f)
                    quality = check_nifti_quality(full_path)
                    if quality and quality['is_good']:
                        best_file = os.path.basename(f)
                        best_quality = quality
                        df.at[idx, 'T1w_type'] = 'standard'
                        break
            
            # 2. If there is no good standard T1w, compare different run
            if not best_file:
                run_files = [f for f in t1w_files if 'run-' in f]
                if run_files:
                    run_candidates = []
                    for f in run_files:
                        full_path = os.path.join(t1w_path, f)
                        quality = check_nifti_quality(full_path)
                        if quality and quality['is_good']:
                            run_candidates.append({
                                'file': os.path.basename(f),
                                'quality': quality,
                                'spacing': quality['max_spacing']
                            })
                    
                    if run_candidates:
                        # order for spacing
                        run_candidates.sort(key=lambda x: x['spacing'])
                        
                        # if there are multiple run with same spacing prefer run-01
                        best_spacing = run_candidates[0]['spacing']
                        best_candidates = [r for r in run_candidates if r['spacing'] == best_spacing]
                        
                        if len(best_candidates) > 1:
                            # search run-01
                            best_run = None
                            for r in best_candidates:
                                if 'run-01' in r['file']:
                                    best_run = r
                                    break
                            if not best_run:
                                best_run = best_candidates[0]
                        else:
                            best_run = run_candidates[0]
                        
                        best_file = best_run['file']
                        best_quality = best_run['quality']
                        df.at[idx, 'T1w_type'] = 'run'
                        
                        if len(run_candidates) > 1:
                            all_runs = ', '.join([f"{r['file']} (orient: {r['quality']['orientation']}, spacing: {r['spacing']}mm)" for r in run_candidates])
                            print(f"   {subject_id}: Compare run T1w: {all_runs}")
                            print(f"              selected T1w: {best_file}")
            
            # Save info
            if best_file and best_quality:
                df.at[idx, 'T1w_best_file'] = best_file
                df.at[idx, 'T1w_orientation'] = best_quality['orientation']
                df.at[idx, 'T1w_spacing'] = str(best_quality['max_spacing'])
                df.at[idx, 'T1w_usable'] = True
            else:
                df.at[idx, 'T1w_usable'] = False
                if t1w_files:
                    # Save quality info for excluded scans
                    first_file = t1w_files[0]
                    quality = check_nifti_quality(os.path.join(t1w_path, first_file))
                    if quality:
                        df.at[idx, 'T1w_orientation'] = quality['orientation']
                        df.at[idx, 'T1w_spacing'] = str(quality['max_spacing'])
                        print(f"    {subject_id}: Not possible to use T1w, orientation {quality['orientation']} - spacing {quality['max_spacing']}mm")
    else:
        df.at[idx, 'has_T1w'] = False

print("\n Final statistics:")
print(f"- Total patients: {len(df)}")
print(f"- With usable T1w: {df['T1w_usable'].sum()}")

print(f"\nT1w usable Types:")
usable_t1w = df[df['T1w_usable'] == True]
print(usable_t1w['T1w_type'].value_counts())

print(f"\nT1w orientations found:")
print(df[df['has_T1w']]['T1w_orientation'].value_counts())

print(f"\n Patients with usable T1w: {df['T1w_usable'].sum()}")  

output_path = '/path/to/dataset/selected_patients_ALL_TIMEPOINTS_with_T1w_info.csv'
df.to_csv(output_path, sep='\t', index=False)
print(f" Final CSV saved in: {output_path}")

# Excluded T1w examples
print("\n Example of T1w excluded for orientation/spacing:")
bad_t1w = df[(df['has_T1w'] == True) & (df['T1w_usable'] == False)].head(10)
for _, row in bad_t1w.iterrows():
    print(f"  {row['MRId']}: {row['T1w_files']} - Orient: {row['T1w_orientation']} - Spacing: {row['T1w_spacing']}mm")

# Show which orientations have been excluded
print("\n Orientations in T1w not usable:")
bad_orientations = df[(df['has_T1w'] == True) & (df['T1w_usable'] == False)]['T1w_orientation'].value_counts()
print(bad_orientations)