# PET-Guided MRI Amyloid Detection

**Cross-Modal Knowledge Distillation for PET-Free Amyloid-Beta Detection from MRI**

CVPR 2026 Workshop (PHAROS-AIF-MIH)

Author (code): Francesco Chiumento

Paper authors: Francesco Chiumento, Julia Dietlmeier, Ronan P. Killeen, Kathleen M. Curran, Noel E. O'Connor, Mingming Liu

## OASIS-3 PET/MRI Pipeline (T1w/T2w/FLAIR/T2*)

This document explains how to use the code to reproduce the experiments described in the paper. All script names and paths below match the files in this repository.

The pipeline is modular:

1. PET / clinical / MRI table construction (CSV-level).
2. Dataset splitting (patient-level).
3. MRI modality columns and quality check.
4. MRI & PET preprocessing and registration.
5. Slice extraction and CSV synchronization.
6. Training and evaluation.

Where possible the same scripts are reused across MRI modalities (T1w, T2w, FLAIR, T2*); only the input/output directories change.

---

The repository is organized into four folders: `data_preparation/`, `preprocessing/`, `training/`, and `utils/`.

Below we list all scripts and their role in the pipeline.

## 1. PET-clinical filtering and longitudinal tables

### 1.1 Baseline AV45 filtering
- **Script**: `data_preparation/filtering_av45_PUP.py` 
  - Matches AV45 PET sessions with clinical assessments within +/-365 days.  
  - **Input**: AV45 PUP table and clinical table (paths set in `pup_file_path` and `clinical_file_path`).  
  - **Output**: filtered baseline CSV of PET-clinical pairs (path set in `output_path` inside the script).

### 1.2 Multi-timepoint PIB table
- **Script**: `data_preparation/organize_all_data_365_all_timepoints.py`  
  - Builds a longitudinal PIB PET table aligned with clinical data and MRI sessions; adds amyloid and dementia labels.  
  - **Input**: paths configured in `DATA_ROOT`  
    (PUP, clinical, Centiloid).  
  - **Output**: multi-timepoint PIB CSV  
    (path set in `output_file`).

### 1.3 Multi-tracer merge
- **Script**: `data_preparation/combine_pib_av45.py`  
  - Merges the per-tracer CSVs (PIB + AV45) into one table with a `tracer` column.  
  - **Input**: `pib_path`, `av45_path`.  
  - **Output**: combined CSV with all PET sessions across tracers (`combined_path`).

### 1.4 Optional PET folder cleanup
- **Script**: `data_preparation/prune_unlisted_pup_sessions.py`  
  - Removes PET session folders on disk that are not present in the filtered CSV (keeps only valid PUP sessions).  
  - **Input**: filtered CSV and base PET directory.  
  - **Output**: cleaned PET directory tree and a report of deleted folders.

---

## 2. Dataset split (patient-level)

- **Script**: `data_preparation/create_subject_split.py`  
  - Creates 5-fold StratifiedGroupKFold splits, grouping all sessions of the same subject into the same fold and stratifying by the amyloid-positive label.  
  - **Input**: combined CSV without T1w columns (`combined_path`).  
  - **Output**:
    - CSV with an added `split` column (`output_path`).
    - JSON files with subject IDs per fold (`split_dir`).

The split column produced here is then used by `add_modality_columns.py` and by the training scripts.

---

## 3. MRI modality columns and quality check

### 3.1 Add T1w quality information
- **Script**: `data_preparation/add_modality_columns.py`  
  - Scans T1w MRI files in BIDS format and adds T1w-related columns to the patient CSV, including:
    - `T1w_files`, `T1w_best_file`
    - `T1w_orientation`, `T1w_spacing`
    - `T1w_usable` (boolean quality flag)  
  - **Input**: combined CSV (the version with a `split` column produced by `create_subject_split.py`)  
    (path set in `csv_path` inside the script).  
  - **Output**: updated CSV with T1w information  
    (`output_path` in the script).

> For T2w / FLAIR / T2* we reuse the same logic: we change the input MRI root and the column names, but the preprocessing and slice-extraction steps (below) are identical. To avoid code duplication, we provide one explicit implementation for T1w and apply the same sequence of operations to the other modalities.

---

## 4. MRI and PET preprocessing

### 4.1 PET format conversion
- **Script**: `preprocessing/convert_4dfp_to_nifti.sh`  
  - Converts PET volumes from 4dfp to NIfTI format.  
  - **Input**: raw 4dfp PET volumes.  
  - **Output**: PET NIfTI files in the same folder or in a specified
    NIfTI directory (paths configured at the top of the script).

### 4.2 T1w MRI preprocessing
- **Script**: `preprocessing/preprocessing_T1w.py`  
  - Applies N4 bias-field correction, HD-BET skull stripping and intensity normalization to T1w scans.  
  - **Input**:
    - CSV with T1w quality info (see `csv_path` at the bottom).
    - Input T1w root directory (`input_root`).  
  - **Output**: preprocessed T1w NIfTI files in
    `output_dir` (e.g. `T1w_preprocessed_before_registration/`).

> For T2w / FLAIR / T2* the same preprocessing steps are used in practice by pointing `input_root` and `output_dir` to the modality-specific folders. We do not duplicate the script for each modality, since the operations are identical.

### 4.3 T1w + PET registration to MNI
- **Script**: `preprocessing/registration_PET_T1w.py`  
  - Registers preprocessed T1w MRI and corresponding PET scans to an MNI template, producing MNI-space volumes for both modalities.  
  - **Input**: preprocessed T1w and PET folders, plus the patient CSV (paths configured in the script).  
  - **Output**: T1w and PET NIfTI files in MNI space for all selected sessions.

---

## 5. Slice extraction

### 5.1 MRI slice extraction (T1w and other MRI sequences)
- **Script**: `preprocessing/slice_extraction_MRI.py`  
  - Extracts sagittal 2D slices from preprocessed 3D MRI volumes in MNI space.  
  - Saves per-subject folders like `sub-XXX_ses-YYY/T1w_slices`.  
  - **Input**: directory with registered MRI NIfTI volumes (`preprocessed_dir`).  
  - **Output**: PNG slices and a text file with slice indices in `output_dir`.

> The same processing logic is reused for T2w, FLAIR and T2*. The scripts contain T1w-specific filenames (e.g. `T1w_slices`, `preprocessed_T1w.nii.gz`) that must be adapted for each modality.

### 5.2 PET slice extraction
- **Script**: `preprocessing/slice_extraction_PET.py`  
  - Extracts 2D PET slices in MNI space and, when available, reuses saved MRI slice indices to ensure spatial correspondence.  
  - **Input**:
    - PET NIfTI directory (`preprocessed_dir`).
    - MRI slices directory (for index matching, `mri_output_dir`).
    - Association CSV (`csv_path`).  
  - **Output**: per-subject PET slice folders `PET_slices`.

---

## 6. CSV synchronization after slice extraction

- **Script**: `data_preparation/clean_and_sync_csv_after_slices_extraction.py`  
  - Removes duplicate MRI-PET sessions, checks that MRI and PET slice folders exist on disk, and keeps the most temporally proximal clinical timepoint per MRI-PET pair.  
  - **Input**:
    - Original associations CSV (`original_csv_path`).
    - MRI slices directory (`mri_dir`).
    - PET slices directory (`pet_dir`).  
  - **Output**: cleaned and synchronized CSV (`output_csv_path`).

This script is used both for the T1w/T2w experiments and for the FLAIR/T2* setting by changing `mri_dir` and the MRI subdirectory name inside the script to match the target modality.

---

## 7. Training and evaluation

### 7.1 Single-sequence baseline (T1w + PET)
- **Script**: `training/train_main.py`  
  - Trains the main PET+T1w classifier and runs evaluation on the test set. Hyper-parameters, phase scheduling and paths are configured at the bottom of the file.

### 7.2 Multi-sequence model (T1w/T2w, optionally FLAIR/T2*)
- **Script**: `training/train_multi_sequence.py`  
  - Trains a model that can use multiple MRI sequences jointly (e.g. T1w and T2w), with PET registered on different MRI spaces as needed. Reuses the same architecture and training logic with modality-specific inputs. Directory paths for each modality are defined near the top of the script.

### 7.3 Ablation, metrics, and plots
- **Scripts**:
  - `training/train_ablation.py` -- runs ablation experiments by selectively disabling or modifying components of the model (e.g., loss terms, modalities, or attention).  
  - `utils/evaluation_utils.py` -- contains shared evaluation utilities (e.g., metrics computation, threshold selection, confusion matrices) used by the training scripts and by the supplementary analysis.  
  - `utils/generate_roc_multisequence.py` -- reads per-sequence prediction CSVs (T1w, T2w, FLAIR, T2*) and produces ROC curves and summary tables.  
  - `utils/visualization_utils.py` -- functions for visualizing model predictions, attention maps, and other interpretability outputs.

---

## 8. Notes on FLAIR and T2*

To avoid redundant code, we **do not** duplicate the entire pipeline for FLAIR and T2* in separate scripts. Instead:

- The same preprocessing (`preprocessing/preprocessing_T1w.py`), registration (`preprocessing/registration_PET_T1w.py`), slice extraction (`preprocessing/slice_extraction_MRI.py`, `preprocessing/slice_extraction_PET.py`) and CSV cleaning (`data_preparation/clean_and_sync_csv_after_slices_extraction.py`) steps are reused.
- The input/output folders and modality-specific names (e.g. `T1w_slices`, `preprocessed_T1w.nii.gz`) must be adapted for each modality.
- Evaluation across T1w, T2w, FLAIR and T2* uses the consolidated prediction CSVs in `utils/generate_roc_multisequence.py`.

Data-preparation and preprocessing scripts use placeholder paths (`/path/to/...`) that must be updated for the local environment. Training scripts read paths from environment variables (`MRI_ROOT`, `PET_ROOT`, `CSV_PATH`) with relative defaults under `data/OASIS_3/`. No real usernames, hostnames or absolute paths are required to run the code.

## Data Access

This repository does not distribute ADNI or OASIS-3 data. Users must obtain access directly from the official data providers:
- OASIS-3: https://sites.wustl.edu/oasisbrains/
- ADNI: https://adni.loni.usc.edu/

## Citation

If you use this code, please cite:
```bibtex
@InProceedings{Chiumento_2026_CVPR,
  author    = {Chiumento, Francesco and Dietlmeier, Julia and Killeen, Ronan P. and Curran, Kathleen M. and O'Connor, Noel E. and Liu, Mingming},
  title     = {Cross-Modal Knowledge Distillation for PET-Free Amyloid-Beta Detection from MRI},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) file for details.
