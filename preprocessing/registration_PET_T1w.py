# Author: Francesco Chiumento
# License: MIT
"""
PET-MRI Registration Pipeline with MNI Normalization

Registers PET scans to skull-stripped MRI images and normalizes both modalities to MNI space.
Handles multiple PET tracers (AV45, PIB) with automated session matching.

Features:
- Multi-tracer PET registration (AV45, PIB)
- Rigid registration: PET --> skull-stripped MRI --> MNI template
- Automated MRI-PET session matching via CSV associations
- Skip already registered images to resume interrupted processing

Input: Preprocessed T1w MRI, PET SUVR maps, MNI template, CSV with MRI-PET associations
Output: Registered PET and MRI in native space and MNI space (NIfTI format)
"""
#========================================

import os
import ants
import pandas as pd

# Main paths
mri_base_dir = "/path/to/data/T1w_preprocessed_before_registration_ALL_TIMEPOINTS"
tracer_to_pet_base_dir = {
    "AV45": "/path/to/data/PUP_msum_SUVR",
    "PIB": "/path/to/data/PUP_PIB_msum_SUVR"
}
template_path = "/path/to/data/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"

# Output directories
output_mri_dir = "/path/to/data/T1w_registered_ALL_TIMEPOINTS"
output_pet_dir = "/path/to/data/PET_registered_ALL_TIMEPOINTS"

# Build subject --> correct PET session mapping from CSV
csv_path = "/path/to/data/selected_patients_ALL_TIMEPOINTS_with_T1w_info.csv"
df = pd.read_csv(csv_path, sep="\t")

# clean session names
df["session_id_pet_clean"] = df["session_id_pet"].str.replace("_PUPTIMECOURSE", "", regex=False)
df["subject_tracer"] = df["subject_id"] + "_" + df["tracer"]

# extract MRI day
df["mri_day"] = df["MRId"].str.extract(r"d(\d+)", expand=False)

# associate specific MRI session to specific tracer
mri_to_tracer = {}
for _, row in df.iterrows():
    mri_session = row["MRId"].split("_")[-1] 
    key = f"{row['subject_id']}_{mri_session}_{row['tracer']}"
    mri_to_tracer[key] = row["session_id_pet_clean"]

# Print for verification
print(f"Created map with {len(mri_to_tracer)} specific MRI-tracer associations")

# registration pipeline: PET --> skull-stripped MRI --> MNI

for subject_folder in sorted(os.listdir(mri_base_dir)):
    subject_path = os.path.join(mri_base_dir, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    subject_id = subject_folder.replace("sub-", "")

    for session_folder in sorted(os.listdir(subject_path)):
        session_path = os.path.join(subject_path, session_folder)
        if not os.path.isdir(session_path):
            continue

        session_id = session_folder.replace("ses-", "")
        print(f"\nProcessing {subject_id} - {session_id}")

        t1_path = os.path.join(session_path, "preprocessed_T1w.nii.gz")
        for tracer in ["AV45", "PIB"]:
            # Build key that uniquely identifies this MRI-tracer combination
            mri_tracer_key = f"{subject_id}_{session_id}_{tracer}"
            
            pet_folder = mri_to_tracer.get(mri_tracer_key)
            
            if not pet_folder:
                # Check if this specific combination exists in the original CSV
                continue
            
            pet_base_dir = tracer_to_pet_base_dir[tracer]
            selected_tracer = tracer
            
            pet_file = f"{pet_folder}n_msum_SUVR.nii"

            if "AV45" in pet_folder:
                pet_subfolder = pet_folder.replace("AV45", "AV45_PUPTIMECOURSE")
            elif "PIB" in pet_folder:
                pet_subfolder = pet_folder.replace("PIB", "PIB_PUPTIMECOURSE")
            else:
                print(f" Unrecognized tracer in {pet_folder}")
                continue

            pet_path = os.path.join(
                pet_base_dir,
                pet_folder,
                pet_subfolder,
                "pet_proc",
                pet_file
            )

            print(f" Looking for PET at path: {pet_path}")

            if not os.path.exists(t1_path):
                print(" T1 not found.")
                continue
            if not os.path.exists(pet_path):
                print(" PET not found.")
                continue
            
            subject_mri_out_dir = os.path.join(output_mri_dir, f"sub-{subject_id}", f"ses-{session_id}")
            pet_day = pet_folder.split("_")[-1] 
            subject_pet_out_dir = os.path.join(output_pet_dir, f"sub-{subject_id}", f"ses-{pet_day}_{selected_tracer}")
            out_pet_path = os.path.join(subject_pet_out_dir, "pet_registered_to_skull_stripped_t1.nii.gz")
            if os.path.exists(out_pet_path):
                print(" PET already registered on T1, skip.")
                continue

            # Load images
            t1_skull_stripped = ants.image_read(t1_path)
            pet = ants.image_read(pet_path)

            # Rigid registration
            rigid_reg = ants.registration(
                fixed=t1_skull_stripped,
                moving=pet,
                type_of_transform='Rigid',
                verbose=False
            )
            registered_pet = rigid_reg['warpedmovout']

            # Output folders
            os.makedirs(subject_mri_out_dir, exist_ok=True)
            os.makedirs(subject_pet_out_dir, exist_ok=True)

            ants.image_write(registered_pet, out_pet_path)
            print(f" PET registered on MRI skull-stripped and saved in: {out_pet_path}")

            out_t1_mni = os.path.join(subject_mri_out_dir, "T1_in_mni.nii.gz")
            if os.path.exists(out_t1_mni):
                print(" T1 already registered in MNI, skip.")
            else:
                mni_reg = ants.registration(
                    fixed=ants.image_read(template_path),
                    moving=t1_skull_stripped,
                    type_of_transform='Rigid',
                    verbose=False
                )
                t1_in_mni = mni_reg['warpedmovout']
                ants.image_write(t1_in_mni, out_t1_mni)
                print(f" T1 skull-stripped registered in MNI and saved in: {out_t1_mni}")

                # Transformation applied to PET

                out_pet_mni = os.path.join(subject_pet_out_dir, "pet_in_mni.nii.gz")
                if os.path.exists(out_pet_mni):
                    print(" PET already registered in MNI, skip.")
                else:
                    pet_in_mni = ants.apply_transforms(
                        fixed=ants.image_read(template_path),
                        moving=registered_pet,
                        transformlist=mni_reg['fwdtransforms'],
                        interpolator='linear'
                    )
                    ants.image_write(pet_in_mni, out_pet_mni)
                    print(f" PET registered in MNI and saved in: {out_pet_mni}")

