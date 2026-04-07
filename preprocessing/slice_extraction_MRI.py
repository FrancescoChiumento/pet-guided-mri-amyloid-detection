# Author: Francesco Chiumento
# License: MIT

"""
NIfTI Slice Extractor for 3D Medical Imaging

Extracts 2D slices from 3D T1-weighted MRI volumes in MNI space.
Supports parallel processing with automatic black slice detection.

Features:
- Multi-orientation extraction (axial, coronal, sagittal)
- Black slice filtering (mean intensity threshold)
- Parallel processing with multiprocessing
- Skip already processed patients

Input: NIfTI files (T1_in_mni.nii.gz)
Output: PNG slices organized by patient/session
"""
#========================================
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# save slices with correct format
def save_slice(image_slice, patient_id, orientation, slice_num, output_dir):
    """Save slices as PNG image with correct format"""
    # remove 'sub-' prefix from patient name if present
    if patient_id.startswith("sub-"):
        patient_id = patient_id.replace("sub-", "")

    # create output directory if it doesn't exist 
    patient_slice_dir = os.path.join(output_dir, patient_id, 'T1w_slices')
    os.makedirs(patient_slice_dir, exist_ok=True)
    
    #  Save image with format: {orientation}_slice_{slice_num}.png
    file_name = f"{orientation}_slice_{slice_num}.png"
    save_path = os.path.join(patient_slice_dir, file_name)
    
    plt.imshow(image_slice.T, cmap="gray", origin="lower")
    plt.axis('off') 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# function to see if the images are black
def is_mostly_black(slice_img, threshold=0.01): 
    mean_intensity = np.mean(slice_img)
    return mean_intensity < threshold

#selecting slices of interest
def select_slice_indices(total_slices, num_slices=10, start_fraction=0.4, end_fraction=0.6):
    start_slice = int(total_slices * start_fraction)
    end_slice = int(total_slices * end_fraction)
    
    if end_slice - start_slice < num_slices:
        num_slices = end_slice - start_slice
    return np.linspace(start_slice, end_slice, num=num_slices, dtype=int)

def extract_and_save_slices(image_path, output_dir, num_slices=10):
    sub = image_path.split("sub-")[1].split("/")[0]
    ses = image_path.split("ses-")[1].split("/")[0]
    patient_id = f"sub-{sub}_ses-{ses}"
    print(f"Processing {image_path} for patient {patient_id}")
    
    if patient_id.startswith("sub-"):
        patient_id = patient_id.replace("sub-", "")

    # load image using nibabel
    img = nib.load(image_path)
    img_data = img.get_fdata()

    indices_file = os.path.join(output_dir, patient_id, 'saved_slice_indices.txt')
    os.makedirs(os.path.dirname(indices_file), exist_ok=True)

    # saving image if not "black"
    def save_if_not_black(slice_img, patient_id, i, orientation):
        if not is_mostly_black(slice_img):
            save_slice(slice_img, patient_id, orientation, i + 1, output_dir)
            return True 
        else:
            print(f"Skipped black slice in {orientation} orientation, slice {i + 1}")
            return False 

    # selection of fixed number of slices for each orientation
    axial_indices = select_slice_indices(img_data.shape[2], num_slices, start_fraction=0.3, end_fraction=0.7)  # central axial slices
    coronal_indices = select_slice_indices(img_data.shape[1], num_slices, start_fraction=0.4, end_fraction=0.7)  # central coronal slices
    sagittal_indices = select_slice_indices(img_data.shape[0], num_slices, start_fraction=0.4, end_fraction=0.6)  # central sagittal slices

    # Extract and save selected slices (commented out axial/coronal, keeping sagittal)
#    for i in axial_indices:
#        slice_img = img_data[:, :, i]
#        save_if_not_black(slice_img, patient_id, i, 'axial')

#    for i in coronal_indices:
#        slice_img = img_data[:, i, :]
#        save_if_not_black(slice_img, patient_id, i, 'coronal')

    with open(indices_file, 'w') as f:
        for i in sagittal_indices:
            slice_img = img_data[i, :, :]
            if save_if_not_black(slice_img, patient_id, i, 'sagittal'):
                f.write(f"{i}\n")

def process_patient(image_path, output_dir, num_slices):
    try:
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            return
            
        if os.path.getsize(image_path) == 0:
            print(f"Warning: Empty file: {image_path}")
            return
        
        #skip if slices are already extracted
        sub = image_path.split("sub-")[1].split("/")[0]
        ses = image_path.split("ses-")[1].split("/")[0]
        patient_id = f"{sub}_ses-{ses}"
        
        patient_slice_dir = os.path.join(output_dir, patient_id, 'T1w_slices')
        if os.path.exists(patient_slice_dir):
            existing_slices = [f for f in os.listdir(patient_slice_dir) if f.endswith('.png')]
            if len(existing_slices) >= 30: 
                print(f"slices already extracted for {patient_id} ({len(existing_slices)} slices found), skip.")
                return
        
        extract_and_save_slices(image_path, output_dir, num_slices)
    except Exception as e:

        print(f"ERROR processing {image_path}: {str(e)}")

def process_all_patients(preprocessed_dir, output_dir, num_slices=10, num_processes=None):
    nifti_files = []
    for root, dirs, files in os.walk(preprocessed_dir):
        for file in files:
            if file == 'T1_in_mni.nii.gz':
                nifti_files.append(os.path.join(root, file))

    if num_processes is None:
        num_processes = mp.cpu_count()

    # multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(process_patient, [(file, output_dir, num_slices) for file in nifti_files])

if __name__ == '__main__': 

    preprocessed_dir = '/path/to/data/T1w_registered_MNI'
    output_dir = '/path/to/output/T1w_slices_extracted'
    
    num_slices = 40
    process_all_patients(preprocessed_dir, output_dir, num_slices)