# Author: Francesco Chiumento
# License: MIT
"""
PET Slice Extractor with MRI Index Matching

Extracts 2D slices from 3D PET scans in MNI space.
Uses corresponding MRI slice indices for spatial alignment when available.

Features:
- Multi-tracer support (AV45, PIB)
- MRI-PET spatial correspondence via saved indices
- Parallel processing with multiprocessing
- Skip already processed patients

Input: NIfTI files (pet_in_mni.nii.gz), CSV with MRI-PET associations
Output: PNG slices organized by patient/session/tracer
"""
#========================================

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd

# Saving slices with required format
def save_slice(image_slice, patient_id, orientation, slice_num, output_dir):
    # remove 'sub-' prefix from patient name if present
    if patient_id.startswith("sub-"):
        patient_id = patient_id.replace("sub-", "")

    # create output directory if it doesn't exist
    patient_slice_dir = os.path.join(output_dir, patient_id, 'PET_slices')
    os.makedirs(patient_slice_dir, exist_ok=True)
    
    # save image with format: {orientation}_slice_{slice_num}.png
    file_name = f"{orientation}_slice_{slice_num}.png"
    save_path = os.path.join(patient_slice_dir, file_name)
    
    # saving image
    plt.imshow(image_slice.T, cmap="gray", origin="lower")
    plt.axis('off') 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def is_mostly_black(slice_img, threshold=0.01):
    mean_intensity = np.mean(slice_img)
    return mean_intensity < threshold

# selection of slices of interest
def select_slice_indices(total_slices, num_slices=10, start_fraction=0.4, end_fraction=0.6):
    start_slice = int(total_slices * start_fraction)
    end_slice = int(total_slices * end_fraction)
    
    if end_slice - start_slice < num_slices:
        num_slices = end_slice - start_slice

    return np.linspace(start_slice, end_slice, num=num_slices, dtype=int)

# extraction and saving of slices
def extract_and_save_slices(image_path, output_dir, num_slices=10, associations_df=None):
    # extract patient ID from file path
    sub = image_path.split("sub-")[1].split("/")[0]
    ses_part = image_path.split("ses-")[1].split("/")[0]
    
    # extract tracer if included in session name
    if "_AV45" in ses_part:
        ses = ses_part.split("_AV45")[0]
        tracer = "AV45"
    elif "_PIB" in ses_part:
        ses = ses_part.split("_PIB")[0]
        tracer = "PIB"
    else:
        ses = ses_part
        if "AV45" in image_path.upper():
            tracer = "AV45"
        elif "PIB" in image_path.upper():
            tracer = "PIB"
        else:
            tracer = "UNKNOWN"

    patient_id = f"sub-{sub}_ses-{ses}_{tracer}"

    print(f"Processing {image_path} for patient {patient_id}")
    if patient_id.startswith("sub-"):
        patient_id = patient_id.replace("sub-", "")

    # load image 
    img = nib.load(image_path)
    img_data = img.get_fdata()

    def save_if_not_black(slice_img, patient_id, i, orientation):
        if not is_mostly_black(slice_img):
            save_slice(slice_img, patient_id, orientation, i + 1, output_dir)
        else:
            print(f"Skipped black slice in {orientation} orientation, slice {i + 1}")

# search for saved MRI indices
    mri_indices = None
    if associations_df is not None:
        subject_id = sub
        pet_session = f"d{ses.split('_')[0].replace('d', '')}" 

        matching_rows = associations_df[
            (associations_df['subject_id'] == subject_id) & 
            (associations_df['session_id_pet'].str.endswith(pet_session)) &
            (associations_df['tracer'] == tracer)
        ]
        
        if not matching_rows.empty:
            mri_session = matching_rows.iloc[0]['MRId'].split('_')[-1] 
            
            mri_output_dir = '/path/to/data/T1w_slices_extracted'

            indices_file_path = os.path.join(mri_output_dir, f"{subject_id}_ses-{mri_session}", 'saved_slice_indices.txt')
            
            # read indices if file exists
            if os.path.exists(indices_file_path):
                print(f"Found MRI indices file: {indices_file_path}")
                with open(indices_file_path, 'r') as f:
                    mri_indices = [int(line.strip()) for line in f]
            else:
                print(f"Indices file not found: {indices_file_path}")
    
    # use MRI indices if available
    if mri_indices:
        print(f"Using {len(mri_indices)} indices from MRI")
        for i in mri_indices:
            slice_img = img_data[i, :, :]
            save_slice(slice_img, patient_id, 'sagittal', i + 1, output_dir)
    else:
        print(f"No MRI indices found, using standard selection")
        sagittal_indices = select_slice_indices(img_data.shape[0], num_slices, start_fraction=0.4, end_fraction=0.6)
        for i in sagittal_indices:
            slice_img = img_data[i, :, :]
            save_if_not_black(slice_img, patient_id, i, 'sagittal')

def process_patient(image_path, output_dir, num_slices, associations_df):
    try:
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            return
            
        if os.path.getsize(image_path) == 0:
            print(f"Warning: Empty file skipped: {image_path}")
            return
        
        sub = image_path.split("sub-")[1].split("/")[0]
        ses_part = image_path.split("ses-")[1].split("/")[0]
        
        if "_AV45" in ses_part:
            ses = ses_part.split("_AV45")[0]
            tracer = "AV45"
        elif "_PIB" in ses_part:
            ses = ses_part.split("_PIB")[0]
            tracer = "PIB"
        else:
            ses = ses_part
            if "AV45" in image_path.upper():
                tracer = "AV45"
            elif "PIB" in image_path.upper():
                tracer = "PIB"
            else:
                tracer = "UNKNOWN"
        
        patient_id = f"{sub}_ses-{ses}_{tracer}" 
        
        # Check if slices already exist for this patient
        patient_slice_dir = os.path.join(output_dir, patient_id, 'PET_slices')
        if os.path.exists(patient_slice_dir):
            existing_slices = [f for f in os.listdir(patient_slice_dir) if f.endswith('.png')]
            if len(existing_slices) >= 30:
                print(f"Slices already extracted for {patient_id} ({len(existing_slices)} slices found), skipping.")
                return
        # if no slices exist, proceed
        extract_and_save_slices(image_path, output_dir, num_slices, associations_df)


    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


def process_all_patients(preprocessed_dir, output_dir, num_slices=10, num_processes=None):
    # load csv with associations
    csv_path = "/path/to/data/selected_patients_associations.csv"
    associations_df = pd.read_csv(csv_path, sep='\t')
    
    # List of NIfTI files in preprocessed folder
    nifti_files = []
    for root, dirs, files in os.walk(preprocessed_dir):
        for file in files:
            if file == 'pet_in_mni.nii.gz':
                nifti_files.append(os.path.join(root, file))

    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count() 

    # Use multiprocessing to process patients
    with mp.Pool(processes=num_processes) as pool:
            pool.starmap(process_patient, [(file, output_dir, num_slices, associations_df) for file in nifti_files])

if __name__ == '__main__': 
    preprocessed_dir = '/path/to/data/PET_registered_MNI'
    output_dir = '/path/to/output/PET_slices_extracted'
    num_slices = 40
    process_all_patients(preprocessed_dir, output_dir, num_slices)