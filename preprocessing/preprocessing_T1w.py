# Author: Francesco Chiumento
# License: MIT

"""
MRI Preprocessing Pipeline with Skull Stripping

Performs N4 bias field correction and HD-BET skull stripping on T1-weighted
MRI scans. Includes intensity normalization and orientation standardization.

Features:
- N4 bias field correction (SimpleITK)
- HD-BET deep learning skull stripping
- Intensity normalization and clipping
- Multi-process parallel execution
- Automatic skip of already processed scans

Input: Raw T1w NIfTI files from BIDS structure, CSV with session info
Output: Preprocessed skull-stripped NIfTI files ready for registration
"""
#========================================

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import subprocess
import nibabel as nib
import numpy as np
import gc
import pandas as pd
import SimpleITK as sitk
import time
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor  
from hd_bet.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
import shutil

def standardize_orientation(img):
    return nib.as_closest_canonical(img)

def delayed_process(args):
    time.sleep(2)
    return process_patient(args)

def preprocess_image(image):
    p_low, p_high = np.percentile(image, (0.5, 99.5))  
    image = np.clip(image, p_low, p_high)                  
    image = (image - p_low) / (p_high - p_low)             
    image = np.power(image, 0.9)                           
    return image.astype(np.float32)                       


def process_patient(args):
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    
    import shutil
    import traceback

    # Avoid memory conflicts with torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.set_per_process_memory_fraction(0.4, 0)

    if len(args) == 6:
        subject_id, session_day, best_file, input_root, output_dir, predictor = args
    else:
        subject_id, session_day, best_file, input_root, output_dir = args
        predictor = get_hdbet_predictor(
            device=torch.device("cuda"),
            use_tta=True,
            verbose=True
        )

    sub_dir = f"sub-{subject_id}"
    ses_dir = f"ses-d{session_day}"
    anat_dir = os.path.join(input_root, sub_dir, ses_dir, "anat")
    file_name = best_file  
    image_path = os.path.join(anat_dir, file_name)

    # Check if file already preprocessed
    output_filename = "preprocessed_T1w.nii.gz"
    patient_output_dir = os.path.join(output_dir, sub_dir, ses_dir)
    output_path  = os.path.join(patient_output_dir,output_filename)
    if os.path.exists(output_path):
        print(f"MRI already preprocessed for {subject_id} - {session_day}, skipping")
        return

    if not os.path.exists(image_path):
        print(f"Error: File {file_name} not found for {subject_id} - {session_day}")
        return

    print(f"\nProcessing {subject_id} - day {session_day} - file: {best_file}...")
    # final directory for saving results
    patient_output_dir = os.path.join(output_dir, sub_dir, ses_dir)
    os.makedirs(patient_output_dir, exist_ok=True)

    try:
        # N4 bias field correction
        print("N4 bias field correction...")
        img = nib.load(image_path)
        img = nib.as_closest_canonical(img)
        img_affine = img.affine

        sitk_img = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
        mask_img = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_img = corrector.Execute(sitk_img, mask_img)
        corrected_array = sitk.GetArrayFromImage(corrected_img).transpose(2, 1, 0)

        # 2) Saving correct version to temporary file
        base_name = f"{subject_id}_{session_day}"
        temp_input = os.path.join(output_dir, f"{base_name}_tempinput.nii.gz")
        nib.save(nib.Nifti1Image(corrected_array, img_affine), temp_input)

        # 3) Run HD-BET (SKULL-STRIPPING)
        hd_bet_tmp_dir = os.path.join(output_dir, f"{base_name}_hd_bet_tmp")
        os.makedirs(hd_bet_tmp_dir, exist_ok=True)

        # Preparing output file for HD-BET
        out_bet = os.path.join(hd_bet_tmp_dir, f"{base_name}_BET.nii.gz")
        print(f"HD-BET on {temp_input} --> {out_bet}")

        # Fallback value if HD-BET fails
        brain_only_data = corrected_array

        try:
            hdbet_predict(
                input_file_or_folder=temp_input,
                output_file_or_folder=out_bet,  # single file
                predictor=predictor,
                keep_brain_mask=True,
                compute_brain_extracted_image=True
            )

            # Search for mask vs brain file by checking for 0/1 voxels
            bet_path = None
            mask_path = None
            for fname in os.listdir(hd_bet_tmp_dir):
                full_path = os.path.join(hd_bet_tmp_dir, fname)
                data = nib.load(full_path).get_fdata()
                uniques = np.unique(data)

                # if all voxels are 0 or q --> mask
                if np.all((uniques == 0) | (uniques == 1)):
                    mask_path = full_path
                    print(f"  File {fname} identified as binary mask")
                else:
                    bet_path = full_path
                    brain_only_data = data
                    print(f"  File {fname} identified as skull-stripped brain")

            # if no mask found --> fallback
            if mask_path is None:
                print(" Warning no mask found! Fallback: using corrected image not fully stripped")
                brain_only_data = corrected_array
            else:
                # moving mask in the final folder
                mask_final_path = os.path.join(patient_output_dir, "brain_mask.nii.gz")
                shutil.move(mask_path, mask_final_path)
                print(f"Mask saved in : {mask_final_path}")

        except Exception as e_hdbet:
            print(f"HD-BET failed for {subject_id} - {session_day}: {e_hdbet}")
            print("HD-BET traceback:\n", traceback.format_exc())
            # fallback
            brain_only_data = corrected_array

        to_remove = [temp_input]
        if os.path.exists(hd_bet_tmp_dir):
            for f in os.listdir(hd_bet_tmp_dir):
                fullf = os.path.join(hd_bet_tmp_dir, f)
                to_remove.append(fullf)
        for rf in to_remove:
            if os.path.exists(rf):
                os.remove(rf)
        if os.path.exists(hd_bet_tmp_dir):
            shutil.rmtree(hd_bet_tmp_dir, ignore_errors=True)

    except Exception as e_main:
        print(f"Warning: General error on {subject_id} - {session_day}: {e_main}")
        print("General error traceback:\n", traceback.format_exc())
        brain_only_data = corrected_array

    # 5) final preprocessing and saving
    try:
        processed_img = preprocess_image(brain_only_data)

        output_filename = "preprocessed_T1w.nii.gz"
        output_path = os.path.join(patient_output_dir, output_filename)

        nib.save(nib.Nifti1Image(processed_img, img_affine), output_path)
        print(f"Saved: {output_path}")

    except Exception as e_save:
        print(f"Blocking error on {subject_id} - {session_day}: {str(e_save)}")
        print("Save error traceback:\n", traceback.format_exc())
        raise e_save


def load_patient_list(csv_path):
    df = pd.read_csv(csv_path, sep='\t', on_bad_lines='warn')
    
    # filtering only patients with usable T1w
    df_usable = df[df['T1w_usable'] == True]
    df_usable = df_usable.drop_duplicates(subset=['MRId'])
    print(f"After removing MRI duplicates: {len(df_usable)} unique sessions")
    subjects_sessions = []
    for _, row in df_usable.iterrows():
        mr_id = row['MRId']
        parts = mr_id.split("_MR_d")
        if len(parts) == 2:
            subject_id, session_day = parts
            best_file = row['T1w_best_file']
            subjects_sessions.append((subject_id, session_day, best_file))
    
    return subjects_sessions

def process_all(csv_path, input_root, output_dir, num_processes=1):
    patient_list = load_patient_list(csv_path)
    args_list = [(subj, day, best_file, input_root, output_dir) 
                 for subj, day, best_file in patient_list]

    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            executor.map(delayed_process, args_list)
    except KeyboardInterrupt:
        print("Interrupted Manually. Closing Processes")
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == '__main__':
    csv_path = '/path/to/data/selected_patients_with_T1w_info.csv'

    # Verify dataset statistics
    df_check = pd.read_csv(csv_path, sep='\t')
    print(f"Total patients in CSV: {len(df_check)}")
    print(f"Patients with usable T1w: {df_check['T1w_usable'].sum()}")
    
    input_root = '/path/to/data/T1w_MRIs'
    output_dir = '/path/to/output/T1w_preprocessed_before_registration'
    
    print("Starting preprocessing of selected patients...")
    process_all(csv_path, input_root, output_dir, num_processes=4)