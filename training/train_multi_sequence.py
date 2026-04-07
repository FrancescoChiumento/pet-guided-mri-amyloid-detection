# Author: Francesco Chiumento
# License: MIT

"""
NOTE: Phase numbering correspondence between paper and code
------------------------------------------------------------
Code:  Phase 0 (pre-train) | Phase 1 (triplet) | Phase 2 (distillation)
Paper: Phase 1             | Phase 2           | Phase 3
------------------------------------------------------------
Code uses 0-based indexing. Methodology is identical.
"""

#========================================
#========================================

"""
Multi-Phase Training Pipeline for Cross-Modal Medical Image Classification

This script implements a three-phase training strategy for amyloid positivity 
classification using multi-modal neuroimaging data (T1w/T2w MRI and PET scans).

Architecture Overview:
- Base model: BiomedCLIP with LoRA fine-tuning
- Cross-modal attention mechanism for PET-MRI fusion
- Self-attention module for single-modality processing
- Binary classification head with focal loss

Training Phases:
Phase 0 (Pre-training):
  - Teacher model training with MRI+PET using classification loss
  - Establishes baseline feature representations
  - Duration: 30 epochs

Phase 1 (Contrastive Learning):
  - Teacher refinement using triplet loss with hard negatives
  - Learns discriminative embeddings guided by PET data
  - Combines triplet loss + classification loss
  - Duration: 15 epochs (configurable)

Phase 2 (Knowledge Distillation):
  - Student model training (MRI-only) via distillation from teacher (MRI+PET)
  - Multi-component loss: feature distillation + logit distillation + classification
  - Enables PET-free inference while retaining teacher knowledge
  - Duration: 100 epochs (configurable)

Key Features:
- Dual-modality support (T1w and T2w MRI with corresponding PET)
- Balanced sampling for class imbalance
- Mixed precision training (AMP)
- Early stopping with patience
- Comprehensive evaluation metrics (F1, precision, recall, AUC)
- Interpretability analysis with saliency maps
- Anti-data-leakage validation

Input:  
- T1w/T2w MRI slices (224×224, 25 slices per patient)
- PET slices (same dimensions, used only in teacher training)
- CSV metadata with patient labels and split assignments

Output: 
- Best teacher model (Phase 1)
- Best student models (Phase 2): optimized for F1 and similarity
- Training history plots and metrics
- Test set predictions and evaluation plots
- Saliency maps for model interpretation

Environment Variables:
- MRI_ROOT: path to MRI slice directory
- PET_ROOT: path to PET slice directory  
- CSV_PATH: path to metadata file
- OUTPUT_DIR: experiment output directory
- TEACHER_DIR: (optional) pre-trained teacher directory
"""
#========================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.5" #limit memory blocks to max 64 MB, garbage collector --> when 50% of memory is fragmented
import pandas as pd
import torch
import gc
import random
import hashlib
import json
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import copy
import logging
import pickle
import re

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim import AdamW
from datetime import datetime
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.visualization_utils import visualize_student_reasoning
from utils.evaluation_utils import create_evaluation_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 4
SKIP_PHASE1 = False

EXISTING_TEACHER_DIR = os.environ.get("TEACHER_DIR", "experiments/teacher")
EXISTING_TEACHER_PATH = os.path.join(EXISTING_TEACHER_DIR, "best_contrastive.pt")

BASE_SAVE_DIR = os.environ.get("OUTPUT_DIR", "experiments")

if SKIP_PHASE1 and os.path.exists(EXISTING_TEACHER_PATH):
    EXPERIMENT_DIR = EXISTING_TEACHER_DIR
    print(f"directory current experiment: {EXPERIMENT_DIR}")
    
    os.makedirs(os.path.join(EXPERIMENT_DIR, "train_analysis"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIR, "test_analysis"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIR, "saliency"), exist_ok=True)
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join(BASE_SAVE_DIR, f"exp_{timestamp}")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIR, "train_analysis"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIR, "test_analysis"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIR, "saliency"), exist_ok=True)
    print(f"new directory current experiment: {EXPERIMENT_DIR}")

logger.info(f"Saving results in: {EXPERIMENT_DIR}")

PHASE1_EPOCHS = 15  
PHASE2_EPOCHS = 100 

TEACHER_PATH = os.path.join(EXPERIMENT_DIR, "best_contrastive.pt")
BEST_STUDENT_PATH = os.path.join(EXPERIMENT_DIR, "best_student_model.pt")
FINAL_STUDENT_PATH = os.path.join(EXPERIMENT_DIR, "final_student_model.pt")

print(f"Training plan:")
print(f"   - Phase 1: {PHASE1_EPOCHS} epochs (triplet loss + classification)")
print(f"   - Phase 2: {PHASE2_EPOCHS} epochs (distillation + classification)")
print(f"   - Total: {PHASE1_EPOCHS + PHASE2_EPOCHS} epochs")

class PairedTransform: #apply same transformations to MRI and PET of the same patient
    def __init__(self, augment=True):
        self.augment = augment
        self.resize = transforms.Resize((224, 224))
        
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        
        if augment:
            self.spatial_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=12, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            ])
            self.color_transform = transforms.ColorJitter(brightness=0.35, contrast=0.3)
            self.blur_transform = transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.4)

            self.tensor_erase_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])         
            # Separate normalization
            self.normalize = transforms.Normalize(mean=_mean, std=_std)
        else:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=_mean, std=_std)
            ])
    
    def __call__(self, mri_img, pet_img):
        
        mri_resized = self.resize(mri_img) #224x224 for MRI
        pet_resized = self.resize(pet_img) # 224x224 for PET
        
        if self.augment:
            
            seed = torch.randint(0, 2**32, (1,)).item()
            
            
            torch_state = torch.get_rng_state()
            random_state = random.getstate()
            
            #apply MRI transformations with fixed seed
            torch.manual_seed(seed)
            random.seed(seed)
            mri_aug = self.spatial_transforms(mri_resized)
            mri_color = self.color_transform(mri_aug)
            mri_blur = self.blur_transform(mri_color)
            
            #Same seed for PET
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            torch.manual_seed(seed)
            random.seed(seed)
            pet_aug = self.spatial_transforms(pet_resized)
            pet_color = self.color_transform(pet_aug)
            pet_blur = self.blur_transform(pet_color)

            erase_seed = torch.randint(0, 2**32, (1,)).item()

            # MRI
            torch.manual_seed(erase_seed)
            mri_tensor_erased = self.tensor_erase_transform(mri_blur)
            
            # PET - same seed
            torch.manual_seed(erase_seed)
            pet_tensor_erased = self.tensor_erase_transform(pet_blur)
            
            # Normalize
            mri_tensor = self.normalize(mri_tensor_erased)
            pet_tensor = self.normalize(pet_tensor_erased)
        else:
            mri_tensor = self.normalize(mri_resized)
            pet_tensor = self.normalize(pet_resized)

        return mri_tensor, pet_tensor
    
def subj_from_pid(pid: str) -> str:
    m = re.search(r'OAS\d+', pid)
    return m.group(0) if m else pid.split("_")[0]

def verify_paired_transform():
    """Verify if PET and MRI receive same transformations"""
    print("\n Verify paired transforms")
    transform = PairedTransform(augment=True)
    
    test_img = Image.new('RGB', (224, 224), color='red') 
    
    max_diff = 0
    all_diffs = []
    
    for i in range(10):
        mri_t, pet_t = transform(test_img.copy(), test_img.copy()) # apply transformation to two identical copies
        
        
        diff = (mri_t - pet_t).abs().mean().item() 
        all_diffs.append(diff) 
        max_diff = max(max_diff, diff) 
        
        if i < 3:  
            print(f"  Test {i+1}: Average difference = {diff:.6f}")
    
    print(f"  ...")
    print(f"  Average difference on 10 tests: {sum(all_diffs)/len(all_diffs):.6f}")
    print(f"  Max difference on 10 tests: {max_diff:.6f}")
    
    
    if max_diff > 0.15:
        print("Error: Transformations not synchronized")
        print("   MRI and PET are receiving different transformations")
        return False
    else:
        print("Ok: Transformations correctly synchronized")
        return True   
    
class SoftTripletLoss(nn.Module): 
    def __init__(self, margin=2.0):  
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_pos = F.pairwise_distance(anchor, positive, p=2) # Euclidean distances
        distance_neg = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(distance_pos - distance_neg + self.margin)
        return loss.mean()
    
class MarginFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, margin=1.0, pos_weight=1.0,
                 reduction='mean', contrastive_weight=0.0, smoothing=0.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = float(margin)
        self.reduction = reduction
        self.contrastive_weight = float(contrastive_weight)
        self.smoothing = float(smoothing)
        self.eps = eps
        self.register_buffer("pos_weight_t", torch.tensor([float(pos_weight)]))
        
    def forward(self, logits, targets):
        targets = targets.to(logits.dtype)
        
        # Label smoothing toward 0.5
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
            
        if self.margin > 0:
            margin_vec = torch.where(targets > 0.5, -self.margin, self.margin)
            margin_logits = logits + margin_vec
        else:
            margin_logits = logits
            
        # BCE with focal weight
        bce = F.binary_cross_entropy_with_logits(
            margin_logits, targets,
            pos_weight=self.pos_weight_t.to(margin_logits.device),
            reduction='none'
        )
        
        prob = torch.sigmoid(margin_logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - p_t).clamp_(0, 1) ** self.gamma
        loss = focal_weight * bce
        
        if self.contrastive_weight > 0:
            with torch.no_grad():
                pos_mask = targets > 0.5
                neg_mask = ~pos_mask
                
            if pos_mask.any() and neg_mask.any():
                pos_mean = logits[pos_mask].mean()
                neg_mean = logits[neg_mask].mean()
                # We want that (pos_mean - neg_mean) >= margin
                gap_deficit = F.relu(self.margin - (pos_mean - neg_mean))
                
                # Scale the deficit to prevent it from dominating the loss
                gap_deficit = gap_deficit * 0.1  
                if self.reduction == 'mean':
                    base_loss = loss.mean()
                elif self.reduction == 'sum':
                    base_loss = loss.sum()
                else:
                    base_loss = loss
                    
                total_loss = base_loss + self.contrastive_weight * gap_deficit
                return torch.clamp(total_loss, min=self.eps)  # loss >= eps
                                
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class CrossModalAttention(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True) 
        
        self.pool = nn.Linear(dim, 1) # 128--> 1 
        nn.init.xavier_uniform_(self.pool.weight, gain=1.0) # weights initialization
        nn.init.constant_(self.pool.bias, 0.0) 

        self.norm = nn.LayerNorm(dim) 
        self.dropout = nn.Dropout(0.1)  
        
    def forward(self, query, key):
        assert query.shape[0] == key.shape[0], "Batch size mismatch"
        assert query.shape[2] == key.shape[2], "Feature dimension mismatch"
        attn_out, _ = self.attn(query, key, key)
        attn_out = self.norm(attn_out + query)
        attn_out = self.dropout(attn_out)  
        
        weights = torch.softmax(self.pool(attn_out) / 2.0, dim=1)  
        pooled = (attn_out * weights).sum(dim=1)
        
        return pooled

def monitor_embedding_stats(embeddings, name=""):
    with torch.no_grad():
        norms = torch.norm(embeddings.view(-1, embeddings.size(-1)), dim=1)
        print(f"{name} Embedding Stats:")
        print(f"  - Average Norm: {norms.mean().item():.4f}")
        print(f"  - Std Norm: {norms.std().item():.4f}")
        print(f"  - Range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")
        
        flat_emb = embeddings.view(-1, embeddings.size(-1))
        if flat_emb.size(0) > 20:
            sample_idx = torch.randperm(flat_emb.size(0))[:20]
            sample = flat_emb[sample_idx] # select 20 random embeddings using indices
            try:
                corr_matrix = torch.corrcoef(sample.T)
                print(f"  - Average correlation between dimensions: {corr_matrix.abs().mean().item():.4f}")
            except RuntimeError:
                print("  - Undefined correlation (variance ~0 in some features).")


def evaluate_model_with_diagnostics(model, data_loader, dataset_metadata, threshold=0.5, 
                                    use_this_threshold_for_opt=False, 
                                    value_for_opt_threshold=None, 
                                    prefix=""):
    model.eval()  
    all_logits = []
    all_probs = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()): # mixed precision
            for batch_idx, batch in enumerate(data_loader):
                mri = batch["mri_slices"].to(device)
                patient_ids = batch["patient_id"] # extracts the list of patient IDs in the current batch
                all_patient_ids.extend(patient_ids) #Adds all items in patient_ids to the global all_patient_ids list
                
                labels = get_labels_from_patient_ids(patient_ids, dataset_metadata) 
                all_labels.extend(labels)
                                
                mri_emb = model.encode_batch(mri) #convert mri in embeddings
                student_emb = model.self_attn(query=mri_emb, key=mri_emb) # self attention 
                logits = model.classifier_head(student_emb).squeeze(1) 

                probs = torch.sigmoid(logits).cpu().numpy() #converts logits in probability [0,1]
                            
                all_logits.append(logits.detach().cpu())
                all_probs.extend(probs)
                
                if batch_idx % 100 == 0:
                    print(f"\n Diagnostic {prefix} (batch {batch_idx}) ===")
                    print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                    print(f"Probability: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    
    all_logits_tensor = torch.cat(all_logits) 
    all_labels_array = np.array(all_labels)
    all_probs_array = np.array(all_probs)
    
    preds_fixed = (all_probs_array > threshold).astype(int) # binary predictions
    metrics_fixed = {
        'accuracy': accuracy_score(all_labels_array, preds_fixed),
        'precision': precision_score(all_labels_array, preds_fixed, zero_division=0),
        'recall': recall_score(all_labels_array, preds_fixed, zero_division=0),
        'f1': f1_score(all_labels_array, preds_fixed, zero_division=0)
    }
    if metrics_fixed['precision'] + metrics_fixed['recall'] > 0:
        calculated_f1 = 2 * (metrics_fixed['precision'] * metrics_fixed['recall']) / (metrics_fixed['precision'] + metrics_fixed['recall'])
        print(f"F1 (threshold 0.5): calculated={calculated_f1:.4f}, reported={metrics_fixed['f1']:.4f}")
    
    if use_this_threshold_for_opt and value_for_opt_threshold is not None:
        best_thresh = value_for_opt_threshold
        preds_opt = (all_probs_array > best_thresh).astype(int)
        best_f1 = f1_score(all_labels_array, preds_opt, zero_division=0)
        print(f" Using provided threshold for optimal metrics: {best_thresh:.3f}")
    else:
        print(f" Calculating the optimal threshold on the data '{prefix}'...")
        best_thresh, best_f1 = find_best_threshold(all_probs_array, all_labels_array)
        preds_opt = (all_probs_array > best_thresh).astype(int)
        
    metrics_opt = {
        'threshold': best_thresh,
        'accuracy': accuracy_score(all_labels_array, preds_opt),
        'precision': precision_score(all_labels_array, preds_opt, zero_division=0),
        'recall': recall_score(all_labels_array, preds_opt, zero_division=0),
        'f1': best_f1
    }
    if metrics_opt['precision'] + metrics_opt['recall'] > 0:
        calculated_f1 = 2 * (metrics_opt['precision'] * metrics_opt['recall']) / (metrics_opt['precision'] + metrics_opt['recall'])
        print(f"F1 (threshold {best_thresh:.3f}): calculated={calculated_f1:.4f}, reported={metrics_opt['f1']:.4f}")
    
    class_stats = analyze_class_separation(
        all_logits_tensor, 
        torch.tensor(all_labels_array, device=device), 
        -1,  
        prefix=prefix
    )
    
    return {
        'metrics_fixed': metrics_fixed,
        'metrics_opt': metrics_opt,
        'class_stats': class_stats,
        'logits': all_logits_tensor,
        'probs': all_probs_array,
        'labels': all_labels_array,
        'patient_ids': all_patient_ids
    }

def save_predictions_to_csv(probs, labels, patient_ids, threshold, filepath):
    """Save amyloid predictions for each patient in a CSV file."""    
    predictions = (np.array(probs) > threshold).astype(int) 
    
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'amyloid_true': labels,
        'amyloid_predicted': predictions,
        'prediction_probability': probs,
        'correct': (predictions == labels).astype(int)
    })
    
    results_df.to_csv(filepath, index=False)
    
    correct_count = sum(predictions == labels)
    accuracy = correct_count / len(labels) * 100
    
    print(f"Predictions saved in: {filepath}")
    print(f"   - Total patients: {len(patient_ids)}")
    print(f"   - Correct predictions: {correct_count} ({accuracy:.1f}%)")
    
    return results_df

class NeuroMultimodalDataset(Dataset):
    def __init__(self, mri_root, pet_root_t1, pet_root_t2, csv_path, transform=None, max_slices=40,
                 num_slices_to_use=None, verbose=True,
                 mri_t2_root=None, modality_policy="random"):
        self.mri_root = mri_root                  # T1 root
        self.mri_t2_root = mri_t2_root            # T2 root 
        self.modality_policy = modality_policy    # random | force_t1 | force_t2
        self.pet_root_t1 = pet_root_t1           
        self.pet_root_t2 = pet_root_t2          
        self.transform = transform
        self.max_slices = max_slices
        self.num_slices_to_use = num_slices_to_use
                
        self.metadata = pd.read_csv(csv_path, sep="\t")

        # normalization
        self.metadata["tracer"] = self.metadata["tracer"].astype(str).str.upper().str.strip()

        # remove incomplete rows
        self.metadata = self.metadata.dropna(
            subset=["Centiloid_fSUVR_rsf_TOT_CORTMEAN", "MRId", "session_id_pet", "tracer"]
        )

        original_len = len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(
            subset=['subject_id', 'MRId', 'session_id_pet', 'tracer']
        )
        if verbose:
            print(f" Removed {original_len - len(self.metadata)} duplicates from CSV")

        self.patient_ids = self._get_valid_patient_ids()
        
        valid_subject_ids = [pid.split('_')[0] for pid in self.patient_ids]
        self.metadata = self.metadata[self.metadata['subject_id'].isin(valid_subject_ids)]
        
        self.healthy_patients = self.metadata[self.metadata['amyloid_positive'] == 0]
        self.demented_patients = self.metadata[self.metadata['amyloid_positive'] == 1]
        
        if verbose:
            print(f" Total valid patients: {len(self.patient_ids)}")
            print(f" Dataframe filtered: {len(self.metadata)} rows")
            print(f" Healthy patients: {len(self.healthy_patients)}, unhealthy patients: {len(self.demented_patients)}")  
            
    def get_healthy_patients(self):
        return self.healthy_patients

    def _pet_dir_for(self, patient_id, modality):
        base = self.pet_root_t1 if modality == "T1" else self.pet_root_t2
        return os.path.join(base, patient_id, "PET_slices")

    
    def get_demented_patients(self):
        return self.demented_patients
        
    def _get_valid_patient_ids(self):
        ids = []
        for _, row in self.metadata.iterrows():
            sub = row['subject_id']
            mri_day = row['MRId'].split("_")[-1]
            pet_day = row['session_id_pet'].split("_")[-1]

            if "AV45" in row["tracer"]:
                tracer = "AV45"
            elif "PIB" in row["tracer"]:
                tracer = "PIB"
            else:
                continue

            pid = f"{sub}_ses-{pet_day}_{tracer}"

            # MRI folders
            t1_dir = os.path.join(self.mri_root, f"{sub}_ses-{mri_day}", "T1w_slices")
            t2_dir = os.path.join(self.mri_t2_root, f"{sub}_ses-{mri_day}", "T2w_slices") if self.mri_t2_root else None
            has_t1 = os.path.isdir(t1_dir)
            has_t2 = (t2_dir is not None) and os.path.isdir(t2_dir)

            # PET folders
            pet_dir_t1 = os.path.join(self.pet_root_t1, pid, "PET_slices")
            pet_dir_t2 = os.path.join(self.pet_root_t2, pid, "PET_slices")
            has_pet_t1 = os.path.isdir(pet_dir_t1)
            has_pet_t2 = os.path.isdir(pet_dir_t2)

            has_pair_t1 = has_t1 and has_pet_t1
            has_pair_t2 = has_t2 and has_pet_t2

            if self.modality_policy == "force_t1":
                if has_pair_t1:
                    ids.append(pid)
                continue
            if self.modality_policy == "force_t2":
                if has_pair_t2:
                    ids.append(pid)
                continue
            if has_pair_t1 or has_pair_t2:
                ids.append(pid)
        return ids

    
    def _load_slices(self, folder_path): 
        if not os.path.isdir(folder_path):
            logger.warning(f"Directory not found: {folder_path}")
            return []  
        
        files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")])
        if len(files) != self.max_slices:
            print(f" Slice count mismatch: {len(files)} instead of {self.max_slices} in {folder_path}")
        
        imgs = []
        for f in files:
            try:
                im = Image.open(f)
                img_array = np.array(im.convert("RGB"))
                img_new = Image.fromarray(img_array)
                im.close()
                imgs.append(img_new)
            except Exception as e:
                logger.warning(f"Error opening {f}: {e}")
        return imgs
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        parts = patient_id.split("_")
        sub = parts[0]
        pet_ses = parts[1].replace("ses-", "")  # Data PET
        tracer = parts[2].strip().upper()
        
        rows = self.metadata[
            (self.metadata["subject_id"] == sub) &
            (self.metadata["session_id_pet"].str.endswith(pet_ses)) &
            (self.metadata["tracer"] == tracer)
        ]
        if rows.empty:
            raise ValueError(f"No metadata found for patient_id={patient_id}")
        row = rows.iloc[0]
        
        # Get the correct MRI date from the metadata
        mri_ses = row['MRId'].split("_")[-1]

        # Robust T1/T2 choice versus disk availability
        t1_dir = os.path.join(self.mri_root, f"{sub}_ses-{mri_ses}", "T1w_slices")
        t2_dir = os.path.join(self.mri_t2_root, f"{sub}_ses-{mri_ses}", "T2w_slices") if self.mri_t2_root else None

        has_t1 = os.path.isdir(t1_dir)
        has_t2 = (t2_dir is not None) and os.path.isdir(t2_dir)

        
        if self.modality_policy == "force_t1":
            if not has_t1:
                raise FileNotFoundError(f"T1 missing for {patient_id}: {t1_dir}")
            mri_dir, chosen_modality = t1_dir, "T1"

        elif self.modality_policy == "force_t2":
            if not has_t2:
                raise FileNotFoundError(f"T2 missing for {patient_id}: {t2_dir}")
            mri_dir, chosen_modality = t2_dir, "T2"

        else:  # "random"
            base = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)
            cur_epoch = getattr(self, "epoch", 0)
            rng = random.Random(base + cur_epoch)
            want_t2 = (rng.random() < 0.5)
            if want_t2 and has_t2:
                mri_dir, chosen_modality = t2_dir, "T2"
            elif (not want_t2) and has_t1:
                mri_dir, chosen_modality = t1_dir, "T1"
            elif has_t1:
                mri_dir, chosen_modality = t1_dir, "T1"
            elif has_t2:
                mri_dir, chosen_modality = t2_dir, "T2"
            else:
                raise FileNotFoundError(f"Neither T1 nor T2 available for {patient_id}")

        mri_imgs = self._load_slices(mri_dir)

        # PET in space consistent with the chosen modality
        pet_dir = self._pet_dir_for(patient_id, chosen_modality)
        pet_imgs = self._load_slices(pet_dir)

        if len(mri_imgs) == 0 or len(pet_imgs) == 0:
            raise ValueError(f" No slice found for patient {patient_id}")
        assert len(mri_imgs) == len(pet_imgs), f"Mismatch in slices: {len(mri_imgs)} MRI vs {len(pet_imgs)} PET"
        
        if self.num_slices_to_use is not None and self.num_slices_to_use < len(mri_imgs):
            total_slices = len(mri_imgs)
            indices = []
            if self.num_slices_to_use > 1:
                indices = np.round(np.linspace(0, total_slices - 1, num=self.num_slices_to_use)).astype(int)
                indices = np.unique(indices)  # Remove duplicates
            else:
                indices = [total_slices // 2]
            
            mri_imgs = [mri_imgs[i] for i in indices] 
            pet_imgs = [pet_imgs[i] for i in indices]
        
        if self.transform:
            mri_tensor, pet_tensor = zip(*[self.transform(mri, pet) for mri, pet in zip(mri_imgs, pet_imgs)])
        else:
            to_tensor = transforms.ToTensor()
            mri_tensor = [to_tensor(img) for img in mri_imgs]
            pet_tensor = [to_tensor(img) for img in pet_imgs]

        return {
            "mri_slices": torch.stack(mri_tensor),
            "pet_slices": torch.stack(pet_tensor),
            "patient_id": patient_id,
            "modality": chosen_modality 
        }
            
    def __len__(self):
        return len(self.patient_ids)
    
    def set_epoch(self, epoch: int):
        """Set the current epoch to change the T1/T2 choice"""
        self.epoch = int(epoch)

def get_labels_from_patient_ids(patient_ids, metadata_df):
    labels = []
    for pid in patient_ids:
        parts = pid.split("_")
        subject_id = parts[0]
        session_day = parts[1].replace("ses-d", "")  
        tracer = parts[2].strip().upper()
        
        row = metadata_df[
            (metadata_df["subject_id"] == subject_id) &
            (metadata_df["session_id_pet"].str.endswith(f"d{session_day}")) &
            (metadata_df["tracer"] == tracer)
        ]

        
        if row.empty:
            raise ValueError(f"label not found for {pid}")
        labels.append(int(row["amyloid_positive"].values[0]))
    return labels

def find_best_threshold(probs, targets):
    best_f1 = 0
    best_thresh = 0.5
    
    if len(np.unique(targets)) < 2:  
        return 0.5, 0.0
    
    thresholds = np.linspace(0.01, 0.99, 200)
    
    for thresh in thresholds:
        preds = (np.array(probs) > thresh).astype(int)
        
        # Skip if all 0 or 1
        if np.sum(preds) == 0 or np.sum(preds) == len(preds):
            continue
            
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1

def analyze_classifier_outputs(logits, labels, epoch, output_dir):
    with torch.no_grad():
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        pos_logits = logits_np[labels_np == 1]
        neg_logits = logits_np[labels_np == 0]
        
        pos_mean = np.mean(pos_logits) if len(pos_logits) > 0 else 0
        neg_mean = np.mean(neg_logits) if len(neg_logits) > 0 else 0
        separation = pos_mean - neg_mean
        
        print(f"  Average logits: Positives={pos_mean:.4f}, Negatives={neg_mean:.4f}")
        print(f"  Separation={separation:.4f}")
        
        plt.figure(figsize=(10, 6))
        if len(neg_logits) > 0:
            plt.hist(neg_logits, bins=20, alpha=0.5, label='Negatives', color='green')
        if len(pos_logits) > 0:
            plt.hist(pos_logits, bins=20, alpha=0.5, label='Positives', color='red')
        
        plt.axvline(x=0, color='black', linestyle='--')
        plt.title(f'Logits distribution (Epoch {epoch+1})')
        plt.xlabel('Logits')
        plt.ylabel('Counts')
        plt.legend()
        plt.close()
                
        probs_pos = 1 / (1 + np.exp(-pos_logits)) if len(pos_logits) > 0 else np.array([])
        probs_neg = 1 / (1 + np.exp(-neg_logits)) if len(neg_logits) > 0 else np.array([])
        
        plt.figure(figsize=(10, 6))
        if len(probs_neg) > 0:
            plt.hist(probs_neg, bins=20, alpha=0.5, label='Negatives', color='green')
        if len(probs_pos) > 0:
            plt.hist(probs_pos, bins=20, alpha=0.5, label='Positives', color='red')
        
        plt.axvline(x=0.5, color='black', linestyle='--')
        plt.title(f'Probability distribution (Epoch {epoch+1})')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        
        plt.close()

class MRITextPETContrastive(nn.Module):
    def __init__(self, model_name="chuhac/BiomedCLIP-vit-bert-hf", freeze_projection=False):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        for name, param in base_model.vision_model.named_parameters():
            if any(f"encoder.layers.{i}." in name for i in range(6)):
                param.requires_grad = False
                
        lora_config = LoraConfig(
            r=32,  
            lora_alpha=32,  
            lora_dropout=0.1,  
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=[
                *[f"vision_model.encoder.layers.{i}.self_attn.q_proj" for i in range(6, 12)],
                *[f"vision_model.encoder.layers.{i}.self_attn.k_proj" for i in range(6, 12)],
                *[f"vision_model.encoder.layers.{i}.self_attn.v_proj" for i in range(6, 12)],
                *[f"vision_model.encoder.layers.{i}.self_attn.out_proj" for i in range(6, 12)],  
            ]
        )
        self.model = get_peft_model(base_model, lora_config).to(device)
        print(f" Vision model with LoRA: {type(self.model.vision_model)}")
        
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.4)
        )
        
        self.cross_attn = CrossModalAttention(dim=128, heads=4)
        self.self_attn = CrossModalAttention(dim=128, heads=4)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(64, 1)
        )
        
        if freeze_projection:
            print(" Projection freezing active!")
            for param in self.projection.parameters():
                param.requires_grad = False
                
    # inside MRITextPETContrastive
    def encode_batch(self, slices, chunk_size=None, require_grad=False):
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        B, S, C, H, W = slices.shape
        slices = slices.view(B * S, C, H, W)

        dev = next(self.parameters()).device
        outs = []
        for i in range(0, slices.size(0), chunk_size):
            chunk = slices[i:i+chunk_size].to(dev)

            if self.training or require_grad:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = self.model.vision_model(pixel_values=chunk)
                    emb = out.last_hidden_state[:, 0, :]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        out = self.model.vision_model(pixel_values=chunk)
                        emb = out.last_hidden_state[:, 0, :]

            outs.append(emb)
            del out, chunk

        all_emb = torch.cat(outs, dim=0).to(dev)
        all_emb = self.projection(all_emb)
        all_emb = all_emb.view(B, S, -1)
        return all_emb

def analyze_class_separation(logits, labels, epoch, prefix=""):
    """Analyze class separation in logits"""
    with torch.no_grad():
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        pos_mask = labels_np == 1
        neg_mask = labels_np == 0
        
        pos_logits = logits_np[pos_mask]
        neg_logits = logits_np[neg_mask]
        
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            print(f"{prefix} Warning: one of the classes has no examples!")
            return {'overlap_pct': 0}
        
        pos_mean, pos_std = np.mean(pos_logits), np.std(pos_logits)
        neg_mean, neg_std = np.mean(neg_logits), np.std(neg_logits)
        separation = pos_mean - neg_mean
        
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        if pooled_std > 0:
            separation_significance = separation / pooled_std
        else:
            separation_significance = float('inf') if separation != 0 else 0
        
        epoch_label = f"Epoch {epoch+1}" if epoch >= 0 else "Analysis"
        print(f"\n{prefix} Classes separation - {epoch_label}")
        print(f"  Positives: average={pos_mean:.4f}, std={pos_std:.4f}, n={len(pos_logits)}")
        print(f"  Negatives: average={neg_mean:.4f}, std={neg_std:.4f}, n={len(neg_logits)}")
        print(f"  Absolute separation: {separation:.4f}")
        print(f"  Significance (d'): {separation_significance:.4f}")
        
        pos_min = np.min(pos_logits)
        neg_max = np.max(neg_logits)
        
        overlap_percentage = 0  
        if pos_min < neg_max:
            overlap_percentage = sum(1 for p in pos_logits if p < neg_max) / len(pos_logits) * 100
            print(f"   Overlap: {overlap_percentage:.1f}% positives under negative max")
        else:
            print(f"   NO OVERLAP: positive min ({pos_min:.4f}) > negative max ({neg_max:.4f})")
                
        optimal_threshold = (pos_mean + neg_mean) / 2
        prob_threshold = 1 / (1 + np.exp(-optimal_threshold))
        print(f"  Theoretical optimal threshold(logit): {optimal_threshold:.4f} (prob: {prob_threshold:.4f})")
        
        return {
            'pos_mean': pos_mean, 
            'neg_mean': neg_mean,
            'separation': separation,
            'significance': separation_significance,
            'overlap_pct': overlap_percentage,
            'optimal_threshold': optimal_threshold
        }

def select_negative_for_patient(anchor_patient, all_patients, suvr_threshold=5.0):
    anchor_id = anchor_patient["subject_id"]
    
    if pd.isna(anchor_patient["Centiloid_fSUVR_rsf_TOT_CORTMEAN"]):
        print(f" Anchor patient {anchor_id} has a missing SUVR value")
        all_other_patients = all_patients[all_patients["subject_id"] != anchor_id]
        if all_other_patients.empty:
            return None
        return all_other_patients.sample(n=1).iloc[0]
    
    anchor_suvr = anchor_patient["Centiloid_fSUVR_rsf_TOT_CORTMEAN"]
    
    possible_negatives = all_patients[
        (all_patients["subject_id"] != anchor_id) &
        (abs(all_patients["Centiloid_fSUVR_rsf_TOT_CORTMEAN"] - anchor_suvr) >= suvr_threshold)
    ]
    
    if possible_negatives.empty:
        print(f" No negative found for anchor {anchor_id} with threshold {suvr_threshold}")
        
        all_other_patients = all_patients[all_patients["subject_id"] != anchor_id].copy()
        
        if all_other_patients.empty:
            print(f" No other available patients for {anchor_id}")
            return None
        
        valid_patients = all_other_patients.dropna(subset=["Centiloid_fSUVR_rsf_TOT_CORTMEAN"])
        
        if valid_patients.empty:
            print(f" No patients with valid SUVR found for {anchor_id}")
            return all_other_patients.sample(n=1).iloc[0]
        
        valid_patients = valid_patients.copy()  
        valid_patients["suvr_diff"] = abs(valid_patients["Centiloid_fSUVR_rsf_TOT_CORTMEAN"] - anchor_suvr)
        found = False
        for threshold in [15, 10, 5, 2.5, 1.0]:
            valid_patients = all_other_patients.copy()
            valid_patients["suvr_diff"] = abs(valid_patients["Centiloid_fSUVR_rsf_TOT_CORTMEAN"] - anchor_suvr)
            valid_patients = valid_patients.dropna(subset=["suvr_diff"])
            valid_patients = valid_patients[valid_patients["suvr_diff"] > threshold]
            if not valid_patients.empty:
                most_different = valid_patients.sort_values("suvr_diff", ascending=False).iloc[0]
                found = True
                break
        if not found:
            most_different = all_other_patients.sample(n=1).iloc[0]
        print(f" Using fallback: patient {most_different['subject_id']} with diff SUVR {most_different.get('suvr_diff', -1):.2f}")
        return most_different
    
    return possible_negatives.sample(n=1).iloc[0]

def set_seed(seed=42):
    """Set all seeds to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_negative_pet_embeddings(batch, dataset, healthy_df, demented_df, model, device, chunk_size=2):
    """
    Helper function to get negative PET embeddings for a batch.
    
    Args:
        batch: The current batch with patient_ids
        dataset: The dataset (train/val) to access paths
        healthy_df: DataFrame of healthy patients
        demented_df: DataFrame of patients with dementia
        model: The model for encode_batch
        device: The device (cuda/cpu)
        chunk_size: Chunk size for encode_batch
        
    Returns:
        pet_neg_embeddings_list: List of negative embeddings
        valid_indices: Valid indices in the original batch
    """
    batch_size_actual = batch["mri_slices"].size(0)
    pet_neg_embeddings_list = []
    valid_indices = []
    
    for i in range(batch_size_actual):
        anchor_full_id = batch["patient_id"][i]
        
        # Parse patient ID
        anchor_parts = anchor_full_id.split("_")
        anchor_subject_id = anchor_parts[0]
        anchor_session_day = anchor_parts[1].replace("ses-d", "")
        anchor_tracer = anchor_parts[2]
        
        # Find anchor in metadata
        anchor_df = dataset.metadata[
            (dataset.metadata["subject_id"] == anchor_subject_id) &
            (dataset.metadata["session_id_pet"].str.endswith(f"d{anchor_session_day}")) &
            (dataset.metadata["tracer"] == anchor_tracer)
        ]
        
        if anchor_df.empty:
            print(f" Anchor patient {anchor_full_id} not found in metadata")
            continue
            
        anchor_patient = anchor_df.iloc[0]
        
        # Get all patients for negative selection
        all_patients_df = pd.concat([healthy_df, demented_df])
        negative_patient = select_negative_for_patient(anchor_patient, all_patients_df, suvr_threshold=5.0)
        
        if negative_patient is None:
            continue
        
        # Build PET directory path
        session_str = negative_patient["session_id_pet"]
        tracer = str(negative_patient["tracer"]).strip().upper()
        
        try:
            day_part = session_str.split("_")[-1]
            if not day_part.startswith("d"):
                print(f" session_id_pet malformed: {session_str}")
                continue
            day = day_part[1:]
            
            # Determine the mode of the i-th anchor sample
            current_modality = batch["modality"][i]  # "T1" or "T2"

            pet_base = dataset.pet_root_t1 if current_modality == "T1" else dataset.pet_root_t2
            pet_dir = os.path.join(
                pet_base,
                f"{negative_patient['subject_id']}_ses-d{day}_{tracer}",
                "PET_slices"
            )

        except Exception as e:
            logger.error(f"Error parsing session_id_pet '{session_str}': {e}")
            continue
        
        if not os.path.isdir(pet_dir):
            print(f" Directory PET not found: {pet_dir}")
            continue
            
        # Load PET slices
        neg_pet_slices = dataset._load_slices(pet_dir)
        if len(neg_pet_slices) == 0:
            print(f" No PET slice found for {negative_patient['subject_id']}")
            continue
            
        # Apply slice selection if needed
        if dataset.num_slices_to_use is not None and dataset.num_slices_to_use < len(neg_pet_slices):
            total_slices = len(neg_pet_slices)
            indices = []
            if dataset.num_slices_to_use > 1:
                step = total_slices / dataset.num_slices_to_use
                indices = np.round(np.linspace(0, total_slices - 1, num=dataset.num_slices_to_use)).astype(int)
                indices = np.unique(indices)
            else:
                indices = [total_slices // 2]
            
            neg_pet_slices = [neg_pet_slices[j] for j in indices]
        
        # Transform slices
        resize = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        
        neg_pet_tensor = torch.stack([norm(to_tensor(resize(img))) for img in neg_pet_slices])
        single_neg = neg_pet_tensor.unsqueeze(0).to(device)
                
        # Encode with model
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                single_emb = model.encode_batch(single_neg, chunk_size=chunk_size)
                
            result_emb = single_emb.squeeze(0).detach().cpu().clone()

        pet_neg_embeddings_list.append(result_emb)
        valid_indices.append(i)

        # Clean up memory
        del neg_pet_tensor, single_neg, single_emb, neg_pet_slices

        if i % 5 == 0:
            torch.cuda.empty_cache()
        
    return pet_neg_embeddings_list, valid_indices

def main():
    global batch_size, num_slices_to_use
    batch_size = 6
    num_slices_to_use = 25
    
    print("\nData preparation")
    
    # Metrics for graphs
    pretrain_history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_f1': []}
    phase1_history = {'epochs': [], 'train_loss': [], 'val_margin': [], 'val_f1': []}
    phase2_history = {
        'epochs': [], 
        'train_loss': [], 
        'val_sim': [], 
        'val_f1': [],
        'val_auc_t1': [],  
        'val_auc_t2': []  
    }

    print("\n" + "="*60)
    print(" Training configuration")
    print("="*60)
    print(f"- Batch size: {batch_size}")
    print(f"- LoRA rank: {32}")
    print(f"- Slices for patient: {num_slices_to_use}")
    print(f"- SKIP_PHASE1: {SKIP_PHASE1}")
    print(f"- Phase 1 epochs: {PHASE1_EPOCHS}")
    print(f"- Phase 2 epochs: {PHASE2_EPOCHS}")
    print(f"- Device: {device}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name()}")
        print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print("="*60 + "\n")
    
    train_transform = PairedTransform(augment=True)
    val_transform = PairedTransform(augment=False)
    
    print("\n" + "="*60)
    print(" Check integrity of transformations")
    print("="*60)
    is_transform_ok = verify_paired_transform()
    if not is_transform_ok:
        print("\nWarning: PairedTransform detected, applying fallback")
    else:
        print("   Starting training")
    print("="*60 + "\n")
    
    mri_root_t1 = os.environ.get("MRI_ROOT_T1", "data/OASIS_3/T1w_slices")
    mri_root_t2 = os.environ.get("MRI_ROOT_T2", "data/OASIS_3/T2w_slices")
    pet_root_t1 = os.environ.get("PET_ROOT_T1", "data/OASIS_3/PET_slices_T1w")
    pet_root_t2 = os.environ.get("PET_ROOT_T2", "data/OASIS_3/PET_slices_T2w")
    csv_path = os.environ.get("CSV_PATH", "data/OASIS_3/selected_patients.tsv")
    
    train_dataset = NeuroMultimodalDataset(
        mri_root=mri_root_t1, pet_root_t1=pet_root_t1, pet_root_t2=pet_root_t2, csv_path=csv_path,
        transform=train_transform, num_slices_to_use=num_slices_to_use, verbose=True,
        mri_t2_root=mri_root_t2, modality_policy="random"
    )

    val_dataset_t1 = NeuroMultimodalDataset(
        mri_root=mri_root_t1, pet_root_t1=pet_root_t1, pet_root_t2=pet_root_t2, csv_path=csv_path,
        transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False,
        mri_t2_root=mri_root_t2, modality_policy="force_t1"
    )
    val_dataset_t2 = NeuroMultimodalDataset(
        mri_root=mri_root_t1, pet_root_t1=pet_root_t1, pet_root_t2=pet_root_t2, csv_path=csv_path,
        transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False,
        mri_t2_root=mri_root_t2, modality_policy="force_t2"
    )

    test_dataset_t1 = NeuroMultimodalDataset(
        mri_root=mri_root_t1, pet_root_t1=pet_root_t1, pet_root_t2=pet_root_t2, csv_path=csv_path,
        transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False,
        mri_t2_root=mri_root_t2, modality_policy="force_t1"
    )
    test_dataset_t2 = NeuroMultimodalDataset(
        mri_root=mri_root_t1, pet_root_t1=pet_root_t1, pet_root_t2=pet_root_t2, csv_path=csv_path,
        transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False,
        mri_t2_root=mri_root_t2, modality_policy="force_t2"
    )

    # Create split map FROM FULL CSV (not from filtered dataset)
    print("\nLoading split map from full CSV")
    df_splits = pd.read_csv(csv_path, sep="\t", usecols=["subject_id", "split"]).drop_duplicates()
    split_map = {sid: int(s) for sid, s in zip(df_splits["subject_id"], df_splits["split"])}
    print(f"Split map created with {len(split_map)} unique subjects")
    print("\nCalculating the correct indices for each dataset")

    train_idx = []
    for i, pid in enumerate(train_dataset.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value in [2, 3, 4]:  # Train splits
            train_idx.append(i)

    # T1 validation indices
    val_idx_t1 = []
    for i, pid in enumerate(val_dataset_t1.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 0:
            val_idx_t1.append(i)

    # T2 validation indices
    val_idx_t2 = []
    for i, pid in enumerate(val_dataset_t2.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 0:
            val_idx_t2.append(i)

    # T1 test indices
    test_idx_t1 = []
    for i, pid in enumerate(test_dataset_t1.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 1:
            test_idx_t1.append(i)

    # T2 test indices
    test_idx_t2 = []
    for i, pid in enumerate(test_dataset_t2.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 1:
            test_idx_t2.append(i)

    # Verify that the indexes are correct
    print(f" Indices correctly calculated:")
    print(f"  Train: {len(train_idx)} indices out {len(train_dataset.patient_ids)} total")
    print(f"  Val T1:   {len(val_idx_t1)} indices out {len(val_dataset_t1.patient_ids)} total")
    print(f"  Val T2:   {len(val_idx_t2)} indices out {len(val_dataset_t2.patient_ids)} total")
    print(f"  Test T1:  {len(test_idx_t1)} indices out {len(test_dataset_t1.patient_ids)} total")
    print(f"  Test T2:  {len(test_idx_t2)} indices out {len(test_dataset_t2.patient_ids)} total")

    from torch.utils.data import Subset
    train_set = Subset(train_dataset, train_idx)
    val_set_t1 = Subset(val_dataset_t1, val_idx_t1)
    val_set_t2 = Subset(val_dataset_t2, val_idx_t2)
    test_set_t1 = Subset(test_dataset_t1, test_idx_t1)
    test_set_t2 = Subset(test_dataset_t2, test_idx_t2)

    # Aliases for compatibility with existing code
    val_set = val_set_t1
    test_set = test_set_t1

    val_dataset = val_dataset_t1 
    test_dataset = test_dataset_t1  

    print(f"\n Final set dimensions:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

    # Check that there are no overlaps
    print("\nCheck the integrity of the splits")

    train_subjects = set()
    for idx in train_idx:
        pid = train_dataset.patient_ids[idx]
        train_subjects.add(subj_from_pid(pid))

    val_subjects = set()
    for idx in val_idx_t1:
        pid = val_dataset_t1.patient_ids[idx]  # use val_dataset_t1
        val_subjects.add(subj_from_pid(pid))
        
    test_subjects = set()
    for idx in test_idx_t1:
        pid = test_dataset_t1.patient_ids[idx]  # use test_dataset_t1
        test_subjects.add(subj_from_pid(pid))

    # Check overlaps
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(" Critical error: Find overlaps between sets!")
        print(f"  Train-Val: {train_val_overlap}")
        print(f"  Train-Test: {train_test_overlap}")
        print(f"  Val-Test: {val_test_overlap}")
        raise ValueError("Split not valid! Duplicate patients between different datasets!")
    else:
        print(" No overlap between sets - Correct splits")

    print("\nCheck anti-leakage")

    train_pids_subset = [train_dataset.patient_ids[i] for i in train_idx]
    val_pids_subset = [val_dataset_t1.patient_ids[i] for i in val_idx_t1]  
    test_pids_subset = [test_dataset_t1.patient_ids[i] for i in test_idx_t1]  

    # Extract subject_ids from real subsets
    train_subjects_real = {subj_from_pid(pid) for pid in train_pids_subset}
    val_subjects_real = {subj_from_pid(pid) for pid in val_pids_subset}
    test_subjects_real = {subj_from_pid(pid) for pid in test_pids_subset}

    # Verify that there are no subjects shared between splits
    assert len(train_subjects_real & val_subjects_real) == 0, " LEAK: subjects shared between train-val"
    assert len(train_subjects_real & test_subjects_real) == 0, " LEAK: subjects shared between train-test"
    assert len(val_subjects_real & test_subjects_real) == 0, " LEAK: subjects shared between val-test"
    print(" Assert 1: No shared subject between splits")

    # Verify that ALL records for a subject are in the same split
    print(" Multi-session/multi-tracer consistency check")
    leak_found = False

    #Helper function to count records of a subject in a list of pids
    def count_subject_records(pid_list, subject):
        return sum(1 for pid in pid_list if subj_from_pid(pid) == subject)

    # Check for each subject in the train
    for subj in train_subjects_real:
        val_count = count_subject_records(val_pids_subset, subj)
        test_count = count_subject_records(test_pids_subset, subj)
        
        if val_count > 0 or test_count > 0:
            print(f" LEAK: Subject {subj} (assigned to TRAIN) has record also in other splits!")
            print(f"   - Record in validation: {val_count}")
            print(f"   - Record in test: {test_count}")
            leak_found = True

    for subj in val_subjects_real:
        train_count = count_subject_records(train_pids_subset, subj)
        test_count = count_subject_records(test_pids_subset, subj)
        
        if train_count > 0 or test_count > 0:
            print(f" LEAK: Subject {subj} (assigned to VAL) has record also in other splits!")
            print(f"   - Record in train: {train_count}")
            print(f"   - Record in test: {test_count}")
            leak_found = True

    for subj in test_subjects_real:
        train_count = count_subject_records(train_pids_subset, subj)
        val_count = count_subject_records(val_pids_subset, subj)
        
        if train_count > 0 or val_count > 0:
            print(f" LEAK: Subject {subj} (assigned to TEST) has record also in other!")
            print(f"   - Record in train: {train_count}")
            print(f"   - Record in validation: {val_count}")
            leak_found = True

    if leak_found:
        raise ValueError("DATA LEAKAGE: Some subjects have records in multiple splits!")
    else:
        print(" All records for each subject are consistently in the same split")
    total_unique_subjects = len(train_subjects_real | val_subjects_real | test_subjects_real)
    print(f"\nFinal summary:")
    print(f"   - Unique total subjects: {total_unique_subjects}")
    print(f"   - Subjects in train: {len(train_subjects_real)}")
    print(f"   - Subjects in val: {len(val_subjects_real)}")
    print(f"   - Subjects in test: {len(test_subjects_real)}")
    print(f"   - Sum: {len(train_subjects_real) + len(val_subjects_real) + len(test_subjects_real)}")

    assert total_unique_subjects == len(train_subjects_real) + len(val_subjects_real) + len(test_subjects_real), \
        "The sum of the subjects does not correspond to the single total!"

    print("\nNo possible data leakage!")
    print("="*60)
        
    print(f" Split dimension:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Worker seed (reproducibility)
    def seed_worker(worker_id):
        import random  
        import numpy as np 
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Generator with fixed seed
    g = torch.Generator()
    g.manual_seed(42)
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker, generator=g)
    val_loader_t1 = DataLoader(val_set_t1, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker)
    val_loader_t2 = DataLoader(val_set_t2, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker)
    test_loader_t1 = DataLoader(test_set_t1, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker)
    test_loader_t2 = DataLoader(test_set_t2, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker)

    val_loader = val_loader_t1
    test_loader = test_loader_t1

    print(f"\n Final set dimensions:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val T1: {len(val_set_t1)} | Val T2: {len(val_set_t2)}")
    print(f"  Test T1: {len(test_set_t1)} | Test T2: {len(test_set_t2)}")
    
    print("\n Pre-training check:")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"- Free GPU memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f} GB")
        print(f"- Free GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("\n Weight calculation for classes")
    all_train_labels = []
    for batch in tqdm(train_loader, desc="Scanning training set for class weights"):
        patient_ids = batch["patient_id"]
        batch_labels = get_labels_from_patient_ids(patient_ids, train_dataset.metadata)
        all_train_labels.extend(batch_labels)
        
    print(f"\nClass distribution in the training set:")
    unique, counts = np.unique(all_train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  - Class {label}: {count} samples ({count/len(all_train_labels)*100:.1f}%)")
        
    neg_count = sum(1 for l in all_train_labels if l == 0)
    pos_count = sum(1 for l in all_train_labels if l == 1)
    global_pos_weight = torch.tensor([neg_count / max(1, pos_count)], device=device)
    global_pos_weight = torch.clamp(global_pos_weight, 1.0, 3.0)
    print(f" Class ratio (neg:pos): {neg_count}:{pos_count}, pos_weight: {global_pos_weight.item():.2f}")
        
    def filter_metadata_by_indices(dataset, indices):
        selected_patient_ids = [dataset.patient_ids[i] for i in indices]
        selected_subject_ids = [subj_from_pid(pid) for pid in selected_patient_ids]
        
        filtered_metadata = dataset.metadata[dataset.metadata['subject_id'].isin(selected_subject_ids)]
        
        healthy = filtered_metadata[filtered_metadata["amyloid_positive"] == 0]
        demented = filtered_metadata[filtered_metadata["amyloid_positive"] == 1]
        
        return healthy, demented
    
    train_healthy, train_demented = filter_metadata_by_indices(train_dataset, train_idx)
    val_healthy, val_demented = filter_metadata_by_indices(val_dataset_t1, val_idx_t1)
    
    triplet_loss_fn = SoftTripletLoss(margin=0.5)
    
    set_seed(42)
    
    model = MRITextPETContrastive()
    model = model.to(device)
    
    print(f"\n Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            print(f" {name} is not on {device}!")

    PRETRAIN_EPOCHS = 30

    if not SKIP_PHASE1:
        print(f"\n phase 0: Teacher pre-training with only Classification ({PRETRAIN_EPOCHS} epochs)")
        
        print("\n Detailed model check:")
        with torch.no_grad():
            for name, param in model.classifier_head.named_parameters():
                print(f"Classifier {name}: mean={param.mean():.4f}, std={param.std():.4f}, "
                    f"min={param.min():.4f}, max={param.max():.4f}")
            
            test_batch = next(iter(train_loader))
            test_mri = test_batch["mri_slices"].to(device)
            test_pet = test_batch["pet_slices"].to(device)
            
            print(f"\nInput shapes: MRI={test_mri.shape}, PET={test_pet.shape}")
            
            test_mri_emb = model.encode_batch(test_mri[:1])
            print(f"After encode MRI: shape={test_mri_emb.shape}, has_nan={torch.isnan(test_mri_emb).any()}")
            
            test_pet_emb = model.encode_batch(test_pet[:1])
            print(f"After encode PET: shape={test_pet_emb.shape}, has_nan={torch.isnan(test_pet_emb).any()}")
            
            teacher_emb = model.cross_attn(query=test_pet_emb, key=test_mri_emb)
            print(f"After cross_attn: shape={teacher_emb.shape}, has_nan={torch.isnan(teacher_emb).any()}")
            print(f"Teacher emb stats: mean={teacher_emb.mean():.4f}, std={teacher_emb.std():.4f}")
            
            x = teacher_emb
            for i, layer in enumerate(model.classifier_head):
                x = layer(x)
                print(f"After classifier layer {i} ({type(layer).__name__}): "
                    f"shape={x.shape}, has_nan={torch.isnan(x).any()}, "
                    f"mean={x.mean():.4f}, std={x.std():.4f}")
        
        pretrain_optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3) 
        print(f" Optimizer configured with lr=5e-5")  
        best_pretrain_f1 = 0.0

        PRETRAIN_BEST_PATH = os.path.join(EXPERIMENT_DIR, "best_pretrain_teacher.pt")
        patience_pretrain = 5 
        no_improve_pretrain = 0
        print(f" Best pre-training model will be saved in: {PRETRAIN_BEST_PATH}")
        
        for epoch in range(PRETRAIN_EPOCHS):
            train_dataset.set_epoch(epoch) 
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS}")):
                mri = batch["mri_slices"].to(device)
                pet = batch["pet_slices"].to(device)
                patient_ids = batch["patient_id"]
                
                labels = get_labels_from_patient_ids(patient_ids, train_dataset.metadata)
                labels = torch.tensor(labels, device=device).float()
                
                pretrain_optimizer.zero_grad()
                
                with autocast(enabled=torch.cuda.is_available()):
                    mri_emb = model.encode_batch(mri)
                    pet_emb = model.encode_batch(pet)
                    
                    teacher_emb = model.cross_attn(query=pet_emb, key=mri_emb)
                    
                    logits = model.classifier_head(teacher_emb).squeeze(1)

                    loss = F.binary_cross_entropy_with_logits(
                        logits, labels, 
                        pos_weight=global_pos_weight
                    )
                    
                scaler.scale(loss).backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f" NaN/Inf in gradient of {name}! Max: {param.grad.max().item():.2e}")
                            param.grad.zero_()
                            
                scaler.unscale_(pretrain_optimizer)
                
                if batch_idx % 100 == 0:  
                    grad_stats = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_stats[name] = grad_norm
                    
                    top_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("\n Top 5 higher gradients (Pre-training):")
                    for name, value in top_grads:
                        print(f"  {name}: {value:.2f}")

                if epoch < 2:
                    clip_value = 1.0  
                elif epoch < 5:
                    clip_value = 2.0  
                else:
                    clip_value = 5.0  
                    
                max_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                    
                scaler.step(pretrain_optimizer)
                scaler.update()
                
                running_loss += loss.item()
                
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    if batch_idx % 100 == 0:
                        print(f"   Batch {batch_idx}: loss={loss.item():.4f}")
            
            epoch_acc = accuracy_score(all_labels, all_preds)  
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)  
            
            print(f"\n Pre-train Epoch {epoch+1} completed:")
            print(f"   Average loss: {running_loss/len(train_loader):.4f}")
            print(f"   Accuracy: {epoch_acc:.3f}")
            print(f"   F1 Score: {epoch_f1:.3f}")

            pretrain_history['epochs'].append(epoch + 1)
            pretrain_history['train_loss'].append(running_loss/len(train_loader))
            pretrain_history['train_acc'].append(epoch_acc)
            
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Pre-train validation", leave=False):
                    mri = batch["mri_slices"].to(device)
                    pet = batch["pet_slices"].to(device)
                    
                    patient_ids = batch["patient_id"] 
                    labels = get_labels_from_patient_ids(patient_ids, val_dataset_t1.metadata) 
                    labels = torch.tensor(labels, device=device).float()
                    
                    mri_emb = model.encode_batch(mri)
                    pet_emb = model.encode_batch(pet)
                    teacher_emb = model.cross_attn(query=pet_emb, key=mri_emb)
                    logits = model.classifier_head(teacher_emb).squeeze(1)
                    
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)  
            print(f"   Validation F1: {val_f1:.3f}")

            pretrain_history['val_f1'].append(val_f1)
            
            if val_f1 > best_pretrain_f1:
                best_pretrain_f1 = val_f1
                torch.save(model.state_dict(), PRETRAIN_BEST_PATH) 
                print(f"    New best F1 pre-training! Checkpoint saved.")
                no_improve_pretrain = 0  # Reset counter
            else:
                no_improve_pretrain += 1
                print(f"    No improvements. Epochs without improvements: {no_improve_pretrain}/{patience_pretrain}")
                
                # Early stopping (optional)
                if no_improve_pretrain >= patience_pretrain:
                    print(f"    Early stopping pre-training after {epoch+1} epochs")
                    break
                    
        print(f"\n Pre-training completed! Best Validation F1: {best_pretrain_f1:.3f}")

        if os.path.exists(PRETRAIN_BEST_PATH):
            print(f"\n Using best model from pre-training...")
            model.load_state_dict(torch.load(PRETRAIN_BEST_PATH, map_location=device))
            print(f" Model from checkpoint with F1={best_pretrain_f1:.3f}")
            
            model.eval()
            quick_check_f1 = 0
            model.train()
        else:
            print(f" Pre-training checkpoint not found, using last state")

        print("   Proceeding with Phase 1 (Triplet Loss + Classification)\n")
    
    if not SKIP_PHASE1:
        print("\n Phase 1: Teacher training (PET-guided) with Triplet Loss")
        optimizer_phase1 = AdamW([
            {'params': model.model.parameters(), 'lr': 5e-6, 'weight_decay': 1e-2}, 
            {'params': model.projection.parameters(), 'lr': 5e-6, 'weight_decay': 1e-2}, 
            {'params': model.cross_attn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-2}, 
            {'params': model.self_attn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-2},  
            {'params': model.classifier_head.parameters(), 'lr': 2e-5, 'weight_decay': 5e-3} 
        ], eps=1e-8)

        scheduler_phase1 = CosineAnnealingWarmRestarts(
            optimizer_phase1,
            T_0=5,      
            T_mult=2,   
            eta_min=1e-7
        )
        best_val_score = -float('inf')
        phase1_train_losses = []
        phase1_val_margins = []
        early_stopping_patience = 3
        no_improve_epochs = 0
        
        for epoch in range(PHASE1_EPOCHS):
            train_dataset.set_epoch(epoch)
            print(f"\n Phase 1 - Epoch {epoch+1}/{PHASE1_EPOCHS} -------------------------")
            model.train()
            total_loss = 0
            batch_count = 0
            epoch_margins = []
            epoch_dist_pos = []
            epoch_dist_neg = []
            epoch_sim_pos = []
            epoch_sim_neg = []
            gc.collect()
            
            accumulation_steps = 4  
 
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}", position=0, leave=True)):
                mri = batch["mri_slices"].to(device, non_blocking=True)
                pet_pos = batch["pet_slices"].to(device, non_blocking=True)
                batch_size_actual = mri.size(0)
                
                
                filtered_patient_ids = batch["patient_id"]
                
                if batch_idx % accumulation_steps == 0:
                    optimizer_phase1.zero_grad()
                
                with autocast(enabled=torch.cuda.is_available()):
                    mri_embeddings = model.encode_batch(mri)
                    if batch_idx % 100 == 0:
                        mean_val = mri_embeddings.mean().item()
                        std_val = mri_embeddings.std().item()
                        print(f" [Batch {batch_idx}] MRI Embedding --> Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                        if std_val < 0.01:
                            print(f" MRI embedding almost constant!")
                                        
                    pet_pos_embeddings = model.encode_batch(pet_pos)

                    if batch_size_actual < 2:
                        print(" Batch size too small, skipping this batch")
                        continue

                    pet_neg_embeddings_list, valid_indices = get_negative_pet_embeddings(
                        batch, train_dataset, train_healthy, train_demented, model, device  
                    )
                    if len(pet_neg_embeddings_list) == 0:
                        print(" No negative sample for this batch. Skipping")
                        continue

                    if len(valid_indices) < batch_size_actual:
                        print(f" Insufficient negative samples. Found {len(valid_indices)}, using only these indices.")
                        mri = mri[valid_indices]
                        pet_pos = pet_pos[valid_indices]
                        mri_embeddings = mri_embeddings[valid_indices]
                        pet_pos_embeddings = pet_pos_embeddings[valid_indices]
                        batch_size_actual = len(valid_indices)
                        filtered_patient_ids = [batch["patient_id"][i] for i in valid_indices]

                    pet_neg_embeddings = torch.stack([emb.to(device) for emb in pet_neg_embeddings_list])
                    del pet_neg_embeddings_list
                    
                    anchor_emb = model.cross_attn(query=pet_pos_embeddings, key=mri_embeddings)
                    positive_emb = model.self_attn(query=pet_pos_embeddings, key=pet_pos_embeddings)
                    negative_emb = model.self_attn(query=pet_neg_embeddings, key=pet_neg_embeddings)
                    
                    if batch_idx % 100 == 0:
                        monitor_embedding_stats(anchor_emb, "Anchor")
                        monitor_embedding_stats(positive_emb, "Positive")
                        monitor_embedding_stats(negative_emb, "Negative")
                    
                    triplet_loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                    
                    teacher_logits = model.classifier_head(anchor_emb).squeeze(1)
                    
                    labels = get_labels_from_patient_ids(filtered_patient_ids, train_dataset.metadata)

                    labels = torch.tensor(labels, device=device).float()
                    
                    class_loss = F.binary_cross_entropy_with_logits(
                        teacher_logits, 
                        labels,
                        pos_weight=global_pos_weight  
                    )
                    
                    classification_weight = 1.0  
                    triplet_weight = 1.0
                    loss = triplet_weight * triplet_loss + classification_weight * class_loss
                    
                    if batch_idx % 100 == 0:
                        with torch.no_grad():
                            teacher_probs = torch.sigmoid(teacher_logits)
                            print(f"   Teacher classification: loss={class_loss.item():.4f}, "
                                f"mean_prob={teacher_probs.mean().item():.4f}")
                            if labels.sum() > 0:
                                pos_mask = labels == 1
                                neg_mask = labels == 0
                                if pos_mask.sum() > 0:
                                    print(f"     Pos logits mean: {teacher_logits[pos_mask].mean().item():.4f}")
                                if neg_mask.sum() > 0:
                                    print(f"     Neg logits mean: {teacher_logits[neg_mask].mean().item():.4f}")
                    
                    cos_sim_pos = F.cosine_similarity(anchor_emb, positive_emb)
                    cos_sim_neg = F.cosine_similarity(anchor_emb, negative_emb)
                    l2_anchor = torch.norm(anchor_emb, p=2, dim=1).mean()
                    l2_positive = torch.norm(positive_emb, p=2, dim=1).mean()
                    l2_negative = torch.norm(negative_emb, p=2, dim=1).mean()
                    l2_weight = 0.01
                    l2_reg = l2_weight * (l2_anchor + l2_positive + l2_negative)
                    
                    with torch.no_grad():
                        anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
                        sim_matrix = torch.mm(anchor_norm, anchor_norm.t())
                        mask = torch.eye(sim_matrix.size(0), device=device).bool()
                        sim_matrix.masked_fill_(mask, 0)
                        avg_similarity = sim_matrix.sum() / (sim_matrix.size(0) * (sim_matrix.size(0) - 1))
                        
                    if epoch < 3:
                        penalty_scale = 0.1  
                    else:
                        penalty_scale = min(1.0, 0.1 + (epoch - 3) * 0.1)  
                        
                    collapse_penalty = torch.relu(avg_similarity - 0.5) * 1.0 * penalty_scale
                    neg_similarity_penalty = torch.relu(cos_sim_neg.mean() + 0.1) * 0.5 * penalty_scale
                    
                    if batch_idx % 100 == 0:
                        print(f"   Penalty scale: {penalty_scale:.2f}")
                    
                    if batch_idx == 0:  
                        print(f"\n Loss components:")
                        print(f"  Base loss (triplet + class): {loss.item():.4f}")
                        print(f"  L2 regularization: {l2_reg.item():.6f}")
                        print(f"  Collapse penalty: {collapse_penalty.item():.4f}")
                        print(f"  Negative similarity penalty: {neg_similarity_penalty.item():.4f}")
                        print(f"  Average similarity value: {avg_similarity.item():.4f}")
                        print(f"  Cosine sim negative mean: {cos_sim_neg.mean().item():.4f}")
                        
                    
                    loss_unscaled = loss + l2_reg + collapse_penalty + neg_similarity_penalty
                    loss_scaled = loss_unscaled / accumulation_steps
  
                    loss_for_tracking = loss_scaled.item()
                         
                scaler.scale(loss_scaled).backward()
                
                total_loss += loss_unscaled.item()  

                batch_count += 1
                
                if batch_idx % 100 == 0:
                    
                    print(f" Batch {batch_idx}: Total Loss = {loss_unscaled.item():.4f}")
                    
                if torch.isnan(loss_scaled) or torch.isinf(loss_scaled) or loss_scaled.item() > 100:
                    print(f" Invalid loss detected: {loss_scaled.item()}")
                    optimizer_phase1.zero_grad()
                    continue
                    
                if batch_idx % 100 == 0:  
                    grad_stats = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_stats[name] = grad_norm
                    
                    top_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("\n Top 5 Highest Gradients (Phase 1):")
                    for name, value in top_grads:
                        print(f"  {name}: {value:.2f}")
                        
                if batch_idx % 100 == 0:
                    grad_norms = {
                        'projection': [],
                        'cross_attn': [],
                        'self_attn': [],
                        'classifier': [],
                        'lora': []
                    }
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            
                            if 'projection' in name:
                                grad_norms['projection'].append(grad_norm)
                            elif 'cross_attn' in name:
                                grad_norms['cross_attn'].append(grad_norm)
                            elif 'self_attn' in name:
                                grad_norms['self_attn'].append(grad_norm)
                            elif 'classifier' in name:
                                grad_norms['classifier'].append(grad_norm)
                            elif 'lora' in name.lower():
                                grad_norms['lora'].append(grad_norm)
                    
                    print(f"\n Gradient stats for components:")
                    for component, norms in grad_norms.items():
                        if norms:
                            print(f"  {component}: max={max(norms):.2f}, mean={np.mean(norms):.2f}")
                    
                    if grad_norms:
                        all_grad_values = []
                        for component_norms in grad_norms.values():
                            all_grad_values.extend(component_norms)
                        
                        if all_grad_values:  
                            max_grad = max(all_grad_values)
                            min_grad = min(all_grad_values)
                            print(f" Global gradient stats: max={max_grad:.6f}, min={min_grad:.6f}")
                            if max_grad > 10:
                                print(" Warning: very high gradients!")
                            if min_grad < 1e-6:
                                print(" Warning: gradients almost zero!")
                        else:
                            print(" No gradient detected")
                            
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    scaler.unscale_(optimizer_phase1)
                    max_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if max_grad_norm > 1.0:
                        print(f" Gradient norm before clipping: {max_grad_norm:.2f}")
                    scaler.step(optimizer_phase1)
                    scaler.update()
                    optimizer_phase1.zero_grad()
                    scheduler_phase1.step(epoch + (batch_idx + 1) / len(train_loader))
                
                with torch.no_grad():
                    dist_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                    dist_neg = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                    margin_achieved = dist_neg - dist_pos
                    
                    epoch_dist_pos.append(dist_pos.mean().item())
                    epoch_dist_neg.append(dist_neg.mean().item())
                    epoch_margins.extend(margin_achieved.cpu().tolist())
                    epoch_sim_pos.append(cos_sim_pos.mean().item())
                    epoch_sim_neg.append(cos_sim_neg.mean().item())
                    
                    if batch_idx % 100 == 0:
                        
                        print(f" Batch {batch_idx}: Triplet Loss = {triplet_loss.item():.4f}")
                        print(f" Distance Anchor-Pos: {dist_pos.mean().item():.4f}, Anchor-Neg: {dist_neg.mean().item():.4f}")
                        print(f" Average margin: {margin_achieved.mean().item():.4f}")
                        
            avg_loss = total_loss / max(1, batch_count)
            phase1_train_losses.append(avg_loss)

            current_epoch_data = {'epoch': epoch + 1, 'train_loss': avg_loss}
            
            valid_triplets_pct = 100 * sum(1 for m in epoch_margins if m > 0) / max(1, len(epoch_margins))
            avg_margin = sum(epoch_margins) / max(1, len(epoch_margins))
            
            print(f"\nEpoch Statistics {epoch+1} ===")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Average Margin: {avg_margin:.4f}")
            print(f"Valid triplets: {valid_triplets_pct:.1f}%")
            avg_dist_pos = sum(epoch_dist_pos) / max(1, len(epoch_dist_pos))
            avg_dist_neg = sum(epoch_dist_neg) / max(1, len(epoch_dist_neg))
            avg_sim_pos = sum(epoch_sim_pos) / max(1, len(epoch_sim_pos))
            avg_sim_neg = sum(epoch_sim_neg) / max(1, len(epoch_sim_neg))
            print(f"Average distance Anchor-Positive: {avg_dist_pos:.4f}")
            print(f"Average distance Anchor-Negative: {avg_dist_neg:.4f}")
            print(f"Average cosine similarity Anchor-Positive: {avg_sim_pos:.4f}")
            print(f"Average cosine similarity Anchor-Negative: {avg_sim_neg:.4f}")
            
            model.eval()
            model.projection.eval()
            val_margins = []
            val_dist_pos = []
            val_dist_neg = []
            val_sim_pos = []
            val_sim_neg = []
            val_loss = 0.0  
            val_batch_count = 0  
            
            with torch.inference_mode():
                with autocast(enabled=torch.cuda.is_available()):
                    for batch in val_loader:
                        mri_val = batch["mri_slices"].to(device, non_blocking=True)
                        pet_val = batch["pet_slices"].to(device, non_blocking=True)
                        batch_size_actual = mri_val.size(0)
                        
                        pet_neg_embeddings_list, valid_indices = get_negative_pet_embeddings(
                            batch, val_dataset_t1, val_healthy, val_demented, model, device
                        )

                        if len(pet_neg_embeddings_list) == 0:
                            continue
                            
                        if len(valid_indices) < batch_size_actual:
                            mri_val = mri_val[valid_indices]
                            pet_val = pet_val[valid_indices]
                            
                        if len(valid_indices) == 0:
                            continue  

                        pet_neg_embeddings = torch.stack([emb.to(device) for emb in pet_neg_embeddings_list])

                        with autocast(enabled=torch.cuda.is_available()):
                            mri_embeddings = model.encode_batch(mri_val)
                            pet_pos_embeddings = model.encode_batch(pet_val)
                            
                            anchor_emb = model.cross_attn(query=pet_pos_embeddings, key=mri_embeddings)
                            positive_emb = model.self_attn(query=pet_pos_embeddings, key=pet_pos_embeddings)
                            negative_emb = model.self_attn(query=pet_neg_embeddings, key=pet_neg_embeddings)
                            loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                            
                        val_loss += loss.item()
                        val_batch_count += 1
                        
                        dist_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                        dist_neg = F.pairwise_distance(anchor_emb, negative_emb, p=2)
                        margin = dist_neg - dist_pos
                        cos_pos = F.cosine_similarity(anchor_emb, positive_emb)
                        cos_neg = F.cosine_similarity(anchor_emb, negative_emb)
                                
                        val_dist_pos.extend(dist_pos.cpu().tolist())
                        val_dist_neg.extend(dist_neg.cpu().tolist())
                        val_margins.extend(margin.cpu().tolist())
                        val_sim_pos.extend(cos_pos.cpu().tolist())
                        val_sim_neg.extend(cos_neg.cpu().tolist())
        
            avg_val_loss = val_loss / max(1, val_batch_count) if val_batch_count > 0 else 0.0
            if val_margins:
                avg_val_margin = sum(val_margins) / len(val_margins)
                avg_val_dist_pos = sum(val_dist_pos) / len(val_dist_pos)
                avg_val_dist_neg = sum(val_dist_neg) / len(val_dist_neg)
                avg_val_sim_pos = sum(val_sim_pos) / len(val_sim_pos)
                avg_val_sim_neg = sum(val_sim_neg) / len(val_sim_neg)
                val_valid_triplets_pct = 100 * sum(1 for m in val_margins if m > 0) / len(val_margins)
                phase1_val_margins.append(avg_val_margin)
                
                print(f"\nValidation epochs statistics {epoch+1} ===")
                print(f"Average validation margin: {avg_val_margin:.4f}")
                print(f"Average distance Anchor-Positive: {avg_val_dist_pos:.4f}")
                print(f"Average distance Anchor-Negative: {avg_val_dist_neg:.4f}")
                print(f"Cosine similarity Anchor-Positive: {avg_val_sim_pos:.4f}")
                print(f"Cosine similarity Anchor-Negative: {avg_val_sim_neg:.4f}")
                print(f"Valid triplets validation: {val_valid_triplets_pct:.1f}%")
                validation_score = val_valid_triplets_pct / 100.0 - 0.1 * avg_val_loss
                
                print("\nTeacher Classification Evaluation")
                teacher_correct = 0
                teacher_total = 0
                teacher_probs_all = []
                teacher_labels_all = []
                
                
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        mri_val = batch["mri_slices"].to(device)
                        pet_val = batch["pet_slices"].to(device)
                        
                        mri_emb = model.encode_batch(mri_val)
                        pet_emb = model.encode_batch(pet_val)
                        teacher_emb = model.cross_attn(query=pet_emb, key=mri_emb)
                        
                        teacher_logits = model.classifier_head(teacher_emb).squeeze(1)
                        teacher_probs = torch.sigmoid(teacher_logits)
                        
                        labels = get_labels_from_patient_ids(batch["patient_id"], val_dataset_t1.metadata)
                        labels = torch.tensor(labels, device=device).float()
                        
                        preds = (teacher_probs > 0.5).float()
                        teacher_correct += (preds == labels).sum().item()
                        teacher_total += len(labels)
                        
                        teacher_probs_all.extend(teacher_probs.cpu().numpy())
                        teacher_labels_all.extend(labels.cpu().numpy())
                        
                teacher_acc = teacher_correct / max(1, teacher_total)
                
                teacher_f1 = f1_score(teacher_labels_all, 
                                            (np.array(teacher_probs_all) > 0.5).astype(int), 
                                            zero_division=0)
                print(f" Teacher accuracy (PET+MRI): {teacher_acc:.3f}")
                print(f" Teacher F1 score: {teacher_f1:.3f}")
                
                phase1_history['epochs'].append(epoch + 1)
                phase1_history['train_loss'].append(avg_loss)
                phase1_history['val_margin'].append(avg_val_margin if 'avg_val_margin' in locals() else 0)
                phase1_history['val_f1'].append(teacher_f1)

                classification_score = teacher_f1
                combined_score = validation_score + 0.5 * classification_score
                
                if combined_score > best_val_score + 0.001:
                    best_val_score = combined_score
                    torch.save(model.state_dict(), TEACHER_PATH)
                    print(f" New best teacher saved! Score: {combined_score:.4f} "
                        f"(Triplet: {validation_score:.4f}, Class: {classification_score:.4f})")
                    no_improve_epochs = 0  
                else:
                    no_improve_epochs += 1 
                
                if epoch % 2 == 0:
                    print(f"\n Analysis embedding collapse:")
                    with torch.no_grad():
                        sample_anchors = []
                        sample_positives = []
                        sample_negatives = []
                        
                        for batch_idx, batch in enumerate(val_loader):
                            if batch_idx >= 5: break  
                            
                            mri_val = batch["mri_slices"].to(device)
                            pet_val = batch["pet_slices"].to(device)
                            batch_size_actual = mri_val.size(0)
                            
                            pet_neg_tensors = []
                            valid_indices = []
                            
                            for i in range(batch_size_actual):
                                anchor_full_id = batch["patient_id"][i]
                                anchor_parts = anchor_full_id.split("_")
                                anchor_subject_id = anchor_parts[0]
                                anchor_session_day = anchor_parts[1].replace("ses-d", "")
                                anchor_tracer = anchor_parts[2]
                                anchor_df = val_dataset_t1.metadata[
                                    (val_dataset_t1.metadata["subject_id"] == anchor_subject_id) &  
                                    (val_dataset_t1.metadata["session_id_pet"].str.endswith(f"d{anchor_session_day}")) &
                                    (val_dataset_t1.metadata["tracer"] == anchor_tracer)
                                ]
                                if anchor_df.empty:
                                    continue
                                    
                                anchor_patient = anchor_df.iloc[0]
                                all_patients_df = pd.concat([val_healthy, val_demented])
                                negative_patient = select_negative_for_patient(anchor_patient, all_patients_df, suvr_threshold=5.0)
                                
                                if negative_patient is None:
                                    continue
                                session_str = negative_patient["session_id_pet"]
                                tracer = str(negative_patient["tracer"]).strip().upper()
                                try:
                                    day_part = session_str.split("_")[-1]
                                    if not day_part.startswith("d"):
                                        continue
                                    day = day_part[1:]
                                    negative_patient_id = f"{negative_patient['subject_id']}_ses-d{day}_{tracer}"
                                except Exception as e:
                                    continue
                                pet_dir = os.path.join(val_dataset_t1.pet_root_t1, negative_patient_id, "PET_slices")

                                if not os.path.isdir(pet_dir):
                                    continue
                                neg_pet_slices = val_dataset_t1._load_slices(pet_dir)  # uses val_dataset_t1
                                if len(neg_pet_slices) == 0:
                                    continue
                                    
                                if val_dataset_t1.num_slices_to_use is not None and val_dataset_t1.num_slices_to_use < len(neg_pet_slices):  # uses val_dataset_t1
                                    total_slices = len(neg_pet_slices)
                                    indices = []
                                    if val_dataset_t1.num_slices_to_use > 1:
                                        step = total_slices / val_dataset_t1.num_slices_to_use
                                        indices = [min(total_slices-1, int(j * step)) for j in range(val_dataset_t1.num_slices_to_use)]
                                    else:
                                        indices = [total_slices // 2]
                                    
                                    neg_pet_slices = [neg_pet_slices[j] for j in indices]
                                resize = transforms.Resize((224, 224))
                                to_tensor = transforms.ToTensor()
                                norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                        std=[0.26862954, 0.26130258, 0.27577711])
                                neg_pet_tensor = torch.stack([norm(to_tensor(resize(img))) for img in neg_pet_slices])
                                pet_neg_tensors.append(neg_pet_tensor)
                                valid_indices.append(i)
                            
                            if len(pet_neg_tensors) == 0:
                                continue
                                
                            if len(valid_indices) < batch_size_actual:
                                mri_val = mri_val[valid_indices]
                                pet_val = pet_val[valid_indices]
                                
                            pet_neg = torch.stack(pet_neg_tensors).to(device)
                            
                            mri_embeddings = model.encode_batch(mri_val)
                            pet_pos_embeddings = model.encode_batch(pet_val)
                            pet_neg_embeddings = model.encode_batch(pet_neg)
                            
                            anchor_emb = model.cross_attn(query=pet_pos_embeddings, key=mri_embeddings)
                            positive_emb = model.self_attn(query=pet_pos_embeddings, key=pet_pos_embeddings)
                            negative_emb = model.self_attn(query=pet_neg_embeddings, key=pet_neg_embeddings)
                            
                            sample_anchors.append(anchor_emb)
                            sample_positives.append(positive_emb)
                            sample_negatives.append(negative_emb)
                        
                        if len(sample_anchors) > 0:
                            all_anchors = torch.cat(sample_anchors)
                            all_positives = torch.cat(sample_positives)
                            all_negatives = torch.cat(sample_negatives)
                            
                            anchor_std = all_anchors.std(dim=0).mean().item()
                            positive_std = all_positives.std(dim=0).mean().item()
                            negative_std = all_negatives.std(dim=0).mean().item()
                            
                            print(f"  Average std anchor embeddings: {anchor_std:.4f}")
                            print(f"  Average std positive embeddings: {positive_std:.4f}")
                            print(f"  Average std negative embeddings: {negative_std:.4f}")
                            
                            if anchor_std < 0.1 or positive_std < 0.1 or negative_std < 0.1:
                                print("   Warning: Possible embedding collapse!")
                            
                            anchor_norm = F.normalize(all_anchors, p=2, dim=1)
                            similarity_matrix = torch.mm(anchor_norm, anchor_norm.t())
                            
                            mask = torch.eye(similarity_matrix.size(0), device=device).bool()
                            similarity_matrix.masked_fill_(mask, 0)
                            avg_similarity = similarity_matrix.sum() / (similarity_matrix.size(0) * (similarity_matrix.size(0) - 1))
                            
                            print(f"  Average similarity between anchors: {avg_similarity:.4f}")
                            if avg_similarity > 0.9:
                                print("   Warning: Anchor too similar between them!")
                                
                            avg_pos_dist = F.pairwise_distance(all_anchors, all_positives, p=2).mean().item()
                            avg_neg_dist = F.pairwise_distance(all_anchors, all_negatives, p=2).mean().item()
                            print(f" Average distance A-P: {avg_pos_dist:.4f}, A-N: {avg_neg_dist:.4f}")
                            
                            if avg_neg_dist < avg_pos_dist * 1.5:
                                print("   Warning: negatives not enough separated!")
            else: 
                print(f" No validation margin calculated this epoch, considered as no improvement")
                no_improve_epochs += 1 
                
            print(f" Status end epoch {epoch+1}: Epochs without improvements: {no_improve_epochs}/{early_stopping_patience}")
            
            if no_improve_epochs >= early_stopping_patience:
                print(f" Early stopping activated! Phase 1 interrupted after {epoch+1} epochs.")
                if best_val_score > 0 : 
                    print(f" Best teacher model saved in: {TEACHER_PATH} with score: {best_val_score:.4f}")
                else:
                    print(f" No model teacher has been saved as 'best'.")
                break  
            
            current_lr_phase1 = optimizer_phase1.param_groups[0]['lr']
            
            log_msg_score_details = f"LR: {current_lr_phase1:.2e}."
            if val_margins and 'combined_score' in locals() and 'validation_score' in locals():
                log_msg_score_details += f" Last Val Score (triplet): {validation_score:.4f}, Combined: {combined_score:.4f}"
            
            logger.info(f"Scheduler phase 1 updated. {log_msg_score_details}")
            
            train_metrics = {'loss': avg_loss, 'margin': avg_margin} 
            if val_margins and 'avg_val_margin' in locals(): 
                val_metrics = {'loss': avg_val_loss, 'margin': avg_val_margin}
                overfitting_score = (train_metrics['margin'] - val_metrics['margin']) / max(train_metrics['margin'], 1e-6)
                print(f" Overfitting index: {overfitting_score:.2%}")
            else:
                print(" Overfitting index not calculable (no validation margin/metric).")
                    
            if epoch == PHASE1_EPOCHS - 1:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epoch + 2), phase1_train_losses, label='Train Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Phase 1: Loss teacher')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.close()
                
                if phase1_val_margins:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(phase1_val_margins) + 1), phase1_val_margins, label='Validation Margin')
                    plt.xlabel('Epoch')
                    plt.ylabel('Average margin')
                    plt.title('Phase 1: Validation margin')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.close()
                torch.cuda.empty_cache()  
                gc.collect()  
        
    else:
        print(f"\n Phase 1 skipped! Checking if there is a pre-trained teacher in:: {TEACHER_PATH}")
        if not os.path.exists(TEACHER_PATH):
            print(f" Error: teacher model not found in {TEACHER_PATH}. Set SKIP_PHASE1 = False to train the teacher.")
            return 
            
        print(f" Loading pre-trained teacher")
        
        state_dict_old_model = torch.load(TEACHER_PATH, map_location=device)
        result_load = model.load_state_dict(state_dict_old_model, strict=False)
        print(f"Teacher weight loaded from: {TEACHER_PATH}")
        print(f"  Missing keys in checkpoint (OK if only classifier_head): {result_load.missing_keys}")
        print(f"  Unexpected keys in checkpoint (will be ignored): {result_load.unexpected_keys}")
        if not all(k.startswith("classifier_head") for k in result_load.missing_keys):
            print(f"Warning: there are missing keys other than the classifier_head: {result_load.missing_keys}")
            
    print("\n Phase 2: Distillation from Teacher (PET-guided) to Student (MRI-only)")
    
    print("\n Teacher loading for Phase 2")
    teacher_temp = MRITextPETContrastive().to(device)
    if os.path.exists(TEACHER_PATH):
        print(f" Loading teacher from: {TEACHER_PATH}")
        teacher_state = torch.load(TEACHER_PATH, map_location=device)

        result_load_teacher = teacher_temp.load_state_dict(teacher_state, strict=False)

        print(" Teacher weights loaded.")
        print(f" Missing keys in teacher: {result_load_teacher.missing_keys}")

    else:
        print(" Error: teacher checkpoint not found!")
        raise FileNotFoundError(f"Teacher model not found in {TEACHER_PATH}")
    
    teacher = copy.deepcopy(teacher_temp)

    print("\n Check teacher-student architectural compatibility")
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_mri = test_batch["mri_slices"][:2].to(device)
        test_pet = test_batch["pet_slices"][:2].to(device)
        
        # Teacher output
        mri_enc_t = teacher.encode_batch(test_mri)
        pet_enc_t = teacher.encode_batch(test_pet)
        teacher_out = teacher.cross_attn(query=pet_enc_t, key=mri_enc_t)
        
        # Student output 
        mri_enc_s = model.encode_batch(test_mri)
        student_out_correct = model.self_attn(query=mri_enc_s, key=mri_enc_s)
        
        # Student output
        weights_wrong = F.softmax(model.self_attn.pool(mri_enc_s), dim=1)
        student_out_wrong = (mri_enc_s * weights_wrong).sum(dim=1)
        
        print(f"Teacher output shape: {teacher_out.shape}")
        print(f"Student output shape: {student_out_correct.shape}")
        print(f"Student output shape: {student_out_wrong.shape}")
        print(f"\nDifference between correct and wrong method: {(student_out_correct - student_out_wrong).abs().mean():.4f}")
        print(f"If this difference is > 0.1, confirm that you are correcting a significant error.")
        
        if teacher_out.shape != student_out_correct.shape:
            raise ValueError("Error: Teacher and Student produce output of different shapes!")
        else:
            print(" Dimensional compatibility verified")
        del teacher_temp  
    
    print("\n Check architecture teacher:")
    for i, module in enumerate(teacher.projection):
        print(f"  Layer {i}: {module}")
    
    print("\n FULL TEACHER CHECK BEFORE DISTILLATION")
    teacher.eval()
    
    print("\n Teacher projection architecture:")
    for i, module in enumerate(teacher.projection):
        print(f"  Layer {i}: {module}")
        
    print("\n Teacher's complete evaluation (MRI+PET) on the validation set")
    teacher.eval() 
    teacher_all_logits = []
    teacher_all_probs = []
    teacher_all_labels_list = []
    teacher_all_patient_ids = []
    
    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()):
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Correct teacher evaluation")):
                mri_val = batch["mri_slices"].to(device)
                pet_val = batch["pet_slices"].to(device) 
                patient_ids = batch["patient_id"]
                teacher_all_patient_ids.extend(patient_ids)
                labels = get_labels_from_patient_ids(patient_ids, val_dataset_t1.metadata)  # uses val_dataset_t1

                teacher_all_labels_list.extend(labels)
                
                mri_emb_teacher = teacher.encode_batch(mri_val)
                pet_emb_teacher = teacher.encode_batch(pet_val)
                teacher_emb_for_classification = teacher.cross_attn(query=pet_emb_teacher, key=mri_emb_teacher)
                logits = teacher.classifier_head(teacher_emb_for_classification).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                teacher_all_logits.append(logits.detach().cpu())
                teacher_all_probs.extend(probs)
                                
    teacher_all_logits_tensor = torch.cat(teacher_all_logits)
    teacher_all_labels_array = np.array(teacher_all_labels_list)
    teacher_all_probs_array = np.array(teacher_all_probs)
    
    teacher_preds_fixed = (teacher_all_probs_array > 0.5).astype(int)
    teacher_metrics_fixed = {
        'accuracy': accuracy_score(teacher_all_labels_array, teacher_preds_fixed),
        'precision': precision_score(teacher_all_labels_array, teacher_preds_fixed, zero_division=0),
        'recall': recall_score(teacher_all_labels_array, teacher_preds_fixed, zero_division=0),
        'f1': f1_score(teacher_all_labels_array, teacher_preds_fixed, zero_division=0)
    }
    
    print(f" Calculating the optimal threshold for the Teacher on the validation set")
    teacher_best_thresh, _ = find_best_threshold(teacher_all_probs_array, teacher_all_labels_array)
    teacher_preds_opt = (teacher_all_probs_array > teacher_best_thresh).astype(int)
    teacher_metrics_opt = {
        'threshold': teacher_best_thresh,
        'accuracy': accuracy_score(teacher_all_labels_array, teacher_preds_opt),
        'precision': precision_score(teacher_all_labels_array, teacher_preds_opt, zero_division=0),
        'recall': recall_score(teacher_all_labels_array, teacher_preds_opt, zero_division=0),
        'f1': f1_score(teacher_all_labels_array, teacher_preds_opt, zero_division=0)
    }
    
    teacher_results = {'metrics_opt': teacher_metrics_opt, 'metrics_fixed': teacher_metrics_fixed} 
    print(f"\nCorrect teacher evaluation results (Threshold {teacher_metrics_opt['threshold']:.3f}):")
    print(f"  F1 Score: {teacher_metrics_opt['f1']:.3f}")
    print(f"  Precision: {teacher_metrics_opt['precision']:.3f}")
    print(f"  Recall: {teacher_metrics_opt['recall']:.3f}")
    
    print(f"\n Teacher F1 Score: {teacher_results['metrics_opt']['f1']:.3f}")
    print(f" Teacher Precision: {teacher_results['metrics_opt']['precision']:.3f}")
    print(f" Teacher Recall: {teacher_results['metrics_opt']['recall']:.3f}")
    
    if teacher_results['metrics_opt']['f1'] < 0.5:
        print("\n Critical error: Teacher has F1 < 0.5!")
        return  
        
    teacher.eval() 
    
    for param in teacher.parameters():
        param.requires_grad = False
        
    gc.collect()
    print(" Student maintains the SAME projection layer as the teacher (full architecture)")
    
    print(" Projection fully unfreezed for training")

    print("\n Reducing student dropout")
    model.projection[3] = nn.Dropout(0.2)  # was 0.6
    model.projection[6] = nn.Dropout(0.1)  # was 0.5
    # Reducing also classifier head dropout
    model.classifier_head[2] = nn.Dropout(0.2)  # was 0.5
    print(" Dropout classifier_head reduced: 0.5-->0.2")
        
    print("\n Critical verification of architectures:")
    print("="*60)
    
    for component_name in ['projection', 'classifier_head', 'self_attn', 'cross_attn']:
        if hasattr(teacher, component_name) and hasattr(model, component_name):
            teacher_comp = getattr(teacher, component_name)
            student_comp = getattr(model, component_name)
            
            teacher_params = sum(p.numel() for p in teacher_comp.parameters())
            student_params = sum(p.numel() for p in student_comp.parameters())
            
            if teacher_params != student_params:
                print(f" Error: {component_name} - Teacher: {teacher_params} params, Student: {student_params} params")
                raise ValueError(f"Incompatible architectures in {component_name}!")
            else:
                print(f" {component_name}: {teacher_params} parameters")
    print("="*60)
    
    print("\n Test forward pass")
    with torch.no_grad():
        test_input = torch.randn(2, num_slices_to_use, 3, 224, 224).to(device)   
        try:
            teacher_out = teacher.encode_batch(test_input)
            student_out = model.encode_batch(test_input)
            print(f" Forward pass OK - Teacher output: {teacher_out.shape}, Student output: {student_out.shape}")
            if teacher_out.shape != student_out.shape:
                raise ValueError("Output shapes not corresponding!")
        except Exception as e:
            print(f" Error in forward pass: {e}")
            raise
    print("\n Architecture verification completed successfully!")
    print("="*60 + "\n")
        
    for p in model.projection.parameters():
        p.requires_grad = True
            
    print(" Optimizer configuration for student")

    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'lora' in n.lower()], 
        'lr': 2e-4, 'weight_decay': 0.0},  
        
        {'params': [p for n, p in model.projection.named_parameters() if 'lora' not in n.lower()], 
        'lr': 1e-4, 'weight_decay': 1e-4},  
        
        {'params': [p for n, p in model.self_attn.named_parameters() if 'lora' not in n.lower()], 
        'lr': 1e-4, 'weight_decay': 1e-3},  
        
        {'params': [p for n, p in model.cross_attn.named_parameters() if 'lora' not in n.lower()], 
        'lr': 1e-4, 'weight_decay': 1e-3},  
        
        {'params': model.classifier_head.parameters(),
        'lr': 1e-4, 'weight_decay': 1e-3}  
    ], eps=1e-8)

    clf_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.7,     
        patience=15,    
        threshold=0.005,
        min_lr=1e-5,    
        cooldown=2,     
        verbose=True
    )
        
    best_f1_score = 0.0 
    phase2_no_improve = 0 
    best_student_similarity = -float('inf')
    best_f1_path = os.path.join(EXPERIMENT_DIR, "best_student_f1_model.pt")
    phase2_losses = []
    phase2_similarities = []
    phase2_patience = 25
    phase2_no_improve = 0
    convergence_monitor = {
        'epoch_f2': [],  
        'separation': [],
        'overlap_pct': [],
        'f1_score': []
    }

    import random
    random.seed(42)  # For reproducibility 
    fixed_val_indices = random.sample(range(len(val_set)), min(6, len(val_set)))
    print(f" Fixed indices selected for visualization: {fixed_val_indices}")
    
    all_train_labels_f2 = [] 
    for batch in tqdm(train_loader, desc="Scanning training set for pos_weight (Phase 2)"):
        patient_ids = batch["patient_id"]
        batch_labels = get_labels_from_patient_ids(patient_ids, train_dataset.metadata)
        all_train_labels_f2.extend(batch_labels)
        
    neg_count_f2 = sum(1 for l in all_train_labels_f2 if l == 0)
    pos_count_f2 = sum(1 for l in all_train_labels_f2 if l == 1)
    
    if pos_count_f2 == 0:
        print(" Warning: no positive samples in the training set!")
        raw_pos_weight_f2 = 1.0
    else:
        raw_pos_weight_f2 = neg_count_f2 / pos_count_f2
        
    print(f" Class ratio (neg:pos) for phase 2: {neg_count_f2}:{pos_count_f2}, raw_pos_weight: {raw_pos_weight_f2:.2f}")
    effective_pos_weight_f2 = torch.tensor([min(raw_pos_weight_f2, 2.5)], device=device)
    print(f" Using balanced pos_weight: {effective_pos_weight_f2.item():.2f} (real: {raw_pos_weight_f2:.2f})")
    print(f"Info Phase 2: using MarginFocalLoss (gamma=2.0, margin=1.5, pos_weight={effective_pos_weight_f2.item():.2f}) for all epochs.")

    print("\n Bias initialization for unbalanced dataset")
    with torch.no_grad():
        neg_ratio = neg_count_f2 / (neg_count_f2 + pos_count_f2)
        pos_ratio = pos_count_f2 / (neg_count_f2 + pos_count_f2)
        
        initial_bias = 0.0
        
        # Initializing weights with moderate gain
        nn.init.xavier_uniform_(model.classifier_head[-1].weight, gain=0.5)
        model.classifier_head[-1].bias.data.fill_(initial_bias)
        
        print(f" Classifier initialized: bias={initial_bias:.3f}, weight gain=0.5")
        print(f"   Dataset: {neg_ratio:.0%} negatives, {pos_ratio:.0%} positives")
        print(f"   pos_weight={effective_pos_weight_f2.item():.2f} manages the imbalance")

        model.classifier_head[-1].bias.data.fill_(initial_bias)
        print(f" Bias classifier initialized to {initial_bias:.3f} (dataset: {neg_ratio:.0%} negatives, {pos_ratio:.0%} positives)")

    for epoch_f2 in range(PHASE2_EPOCHS): 
        train_dataset.set_epoch(epoch_f2)
        print(f"\nPhase 2 - Epoch {epoch_f2+1}/{PHASE2_EPOCHS} -------------------------")
        model.train()
        teacher.eval()
        
        if epoch_f2 < 3:
            temperature = 4.0  
        elif epoch_f2 < 15:
            temperature = 4.0 - (epoch_f2 - 3) * 0.2  
        else:
            temperature = 1.5  
        
        if epoch_f2 % 10 == 0:
            print(f" Distillation temperature: {temperature}")
        
        if epoch_f2 < 5:
            margin_value = 0.3
        elif epoch_f2 < 20:
            margin_value = 0.3 + (epoch_f2 - 5) * 0.06  # Max 1.2
        else:
            margin_value = 1.2 
        
        current_epoch_classification_loss_fn = MarginFocalLoss(
            alpha=1.0,
            gamma=2.0,
            margin=margin_value, 
            pos_weight=effective_pos_weight_f2.item(),
            contrastive_weight=0.1 if epoch_f2 < 10 else 0.3,  
            smoothing=0.0
        )

        if epoch_f2 == 0 or epoch_f2 % 10 == 0:
            print(f" MarginFocalLoss params: margin={margin_value}, contrastive_weight={0.1 if epoch_f2 < 10 else 0.3}")

        if hasattr(current_epoch_classification_loss_fn, 'margin'):
            print(f"    Loss used for epoch {epoch_f2+1}: {type(current_epoch_classification_loss_fn).__name__} with margin={current_epoch_classification_loss_fn.margin}")
        else:
            print(f"    Loss used for epoch {epoch_f2+1}: {type(current_epoch_classification_loss_fn).__name__}")
            
        total_combined_loss = 0
        total_distill_loss  = 0
        total_class_loss    = 0
        batch_count         = 0

        accumulation_steps = 4 

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Distillation epoch {epoch_f2+1}")):
            mri = batch["mri_slices"].to(device, non_blocking=True)
            pet = batch["pet_slices"].to(device, non_blocking=True)
            B = mri.size(0)
            if B < 1:
                continue
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
          
            loss_unscaled_tensor = None 
            loss_for_backward = None 

            with autocast(enabled=torch.cuda.is_available()):
                mri_embeddings_student = model.encode_batch(mri)
                student_emb = model.self_attn(query=mri_embeddings_student, key=mri_embeddings_student)
                student_emb_for_distillation = student_emb 

                student_logits = model.classifier_head(student_emb).squeeze(1)

                with torch.no_grad(): 
                    mri_emb_for_teacher = teacher.encode_batch(mri) 
                    pet_emb_for_teacher = teacher.encode_batch(pet) 
                    teacher_emb_target_for_distillation = teacher.cross_attn(query=pet_emb_for_teacher, key=mri_emb_for_teacher)
                    teacher_logits_raw = teacher.classifier_head(teacher_emb_target_for_distillation).squeeze(1)
                
                patient_ids = batch["patient_id"]
                labels = get_labels_from_patient_ids(patient_ids, train_dataset.metadata)
                labels = torch.tensor(labels, device=device).float()

                student_emb_norm = F.normalize(student_emb, p=2, dim=1) 
                teacher_emb_target_norm = F.normalize(teacher_emb_target_for_distillation.detach(), p=2, dim=1) 
                feature_distill_loss = F.mse_loss(student_emb_norm, teacher_emb_target_norm)

                class_loss = current_epoch_classification_loss_fn(student_logits, labels)

                # Loss breakdown
                if batch_idx % 50 == 0 and epoch_f2 > 0:
                    print(f" Loss (Batch {batch_idx}):")
                    print(f"   class_loss: {class_loss.item():.4f}")
                    if class_loss.item() < 0:
                        print(f"    Negative loss detected!")

                w_clf = 0.5 
                w_feature_distill = 0.3
                w_logit_distill = 0.2 

                if epoch_f2 < 5:
                    warmup_progress = epoch_f2 / 5.0
                    w_clf = 0.8 - 0.3 * warmup_progress  # 0.8 → 0.5
                    w_feature_distill = 0.1 + 0.2 * warmup_progress  # 0.1 → 0.3
                    w_logit_distill = 0.1 + 0.1 * warmup_progress  # 0.1 → 0.2
                    if batch_idx == 0:
                        print(f" Warmup (epoch {epoch_f2+1}/5): "
                            f"clf={w_clf:.2f}, feat={w_feature_distill:.2f}, "
                            f"logit={w_logit_distill:.2f}")
                else:
                    w_clf = 0.5
                    w_feature_distill = 0.3
                    w_logit_distill = 0.2
                    if batch_idx == 0: 
                        print(f" Warmup (epoch {epoch_f2+1}/3): clf={w_clf}, feat={w_feature_distill}, logit={w_logit_distill}")

                loss_unscaled_tensor = (w_clf * class_loss) + (w_feature_distill * feature_distill_loss)

                if w_logit_distill > 0:
                    # BCE with correct temperature for binary tasks
                    logit_distill_loss = F.binary_cross_entropy_with_logits(
                        student_logits / temperature, 
                        torch.sigmoid(teacher_logits_raw.detach() / temperature), 
                        reduction='mean'
                    ) * (temperature * temperature) 
                    
                    loss_unscaled_tensor += w_logit_distill * logit_distill_loss
                else:
                    logit_distill_loss = torch.tensor(0.0)
                loss_for_backward = loss_unscaled_tensor / accumulation_steps

                if batch_idx % 10 == 0 and epoch_f2 >= 100: 
                    print(f"\n Loss components (Batch {batch_idx}):")
                    print(f"  - class_loss raw: {class_loss.item():.4f}")
                    print(f"  - w_clf * class_loss: {(w_clf * class_loss).item():.4f}")
                    print(f"  - feature_distill: {(w_feature_distill * feature_distill_loss).item():.4f}")
                    print(f"  - loss_unscaled: {loss_unscaled_tensor.item():.4f}")
                    
                    if hasattr(current_epoch_classification_loss_fn, 'contrastive_weight'):
                        print(f"  - Contrastive weight in loss: {current_epoch_classification_loss_fn.contrastive_weight}")
                if torch.isnan(loss_for_backward) or torch.isinf(loss_for_backward):
                    print(f" Loss NaN/Inf: skip batch")
                    optimizer.zero_grad()
                    continue

                if loss_for_backward.item() < -10:
                    print(f" Loss still negative: {loss_for_backward.item():.4f} - Unexpected error!")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss_for_backward).backward()

                if batch_idx % 100 == 0:
                    print(f"\n Loss Components (Epoch {epoch_f2+1}, Batch {batch_idx}):")
                    print(f"   Loss divided for {accumulation_steps} (accumulation steps)")
                    print(f"  Feature distill: {feature_distill_loss.item():.4f} (weight: {w_feature_distill:.2f})")
                    print(f"  Logit distill: {logit_distill_loss.item():.4f} (weight: {w_logit_distill:.2f})")
                    print(f"  Classification: {class_loss.item():.4f} (weight: {w_clf:.2f})")

                with torch.no_grad():
                    pos_mask = labels > 0.5
                    neg_mask = labels <= 0.5
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_mean = student_logits[pos_mask].mean().item()
                        neg_mean = student_logits[neg_mask].mean().item()
                        separation = pos_mean - neg_mean

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g['params']], 
                    max_norm=5.0  
                )
                if grad_norm > 5.0:
                    print(f" Grad norm {grad_norm:.1f} clipped to 5.0")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if loss_for_backward is not None: 
                total_combined_loss += loss_unscaled_tensor.item() 
            total_distill_loss += (w_logit_distill * logit_distill_loss.item() + 
                                w_feature_distill * feature_distill_loss.item())
            total_class_loss += (w_clf * class_loss.item())            
            batch_count += 1            
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pos_mask = labels > 0.5
                    neg_mask = labels <= 0.5
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        neg_median = student_logits[neg_mask].median().item()
                        pos_above_median = (student_logits[pos_mask] > neg_median).sum().item()
                        total_pos = pos_mask.sum().item()
                        print(f" Positives above median negative: {pos_above_median}/{total_pos} ({pos_above_median/total_pos*100:.1f}%)")
        gc.collect()
    
        avg_loss = total_combined_loss / max(1, batch_count)
        avg_distill = total_distill_loss / max(1, batch_count)
        avg_class = total_class_loss / max(1, batch_count)

        model.eval()
        train_logits = []
        train_labels_tensor = []

        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                for i, batch in enumerate(train_loader):
                    if i >= 10: 
                        break
                        
                    mri = batch["mri_slices"].to(device)
                    patient_ids = batch["patient_id"]
                    
                    labels = get_labels_from_patient_ids(patient_ids, train_dataset.metadata)
                    labels = torch.tensor(labels, device=device).float()
                                                            
                    mri_emb = model.encode_batch(mri)
                    student_emb = model.self_attn(query=mri_emb, key=mri_emb)

                    logits = model.classifier_head(student_emb).squeeze(1)

                    if i % 10 == 0:  
                        print(f"\nEmbedding training diagnostic (batch {i}) ===")
                        print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        
                        probs = torch.sigmoid(logits)
                        print(f"Probability: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
                        
                        norms = torch.norm(student_emb, dim=1)
                        print(f"Embedding norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
                                        
                    train_logits.append(logits)
                    train_labels_tensor.append(labels.cpu()) 

        if len(train_logits) > 0:
            try:
                all_train_logits = torch.cat(train_logits)
                all_train_labels = torch.cat(train_labels_tensor)

                print("\nDistribution logits analysis on training set")
                analyze_classifier_outputs(
                    all_train_logits, all_train_labels, epoch_f2,
                    os.path.join(EXPERIMENT_DIR, "train_analysis")
                )
            except Exception as e:
                print(f"Error analysing training logits: {e}")

        print(f"\nDistillation statistics epoch {epoch_f2+1} ===")
        print(f"Total average loss: {avg_loss:.4f}")
        print(f" - Distillation: {avg_distill:.4f}")
        print(f" - Classification: {avg_class:.4f}")

        phase2_losses.append(avg_loss)
        
        print(f"\nDistillation statistics epoch {epoch_f2+1} ===")
        print(f"Average loss: {avg_loss:.4f}")

        current_phase2_data = {'epoch': epoch_f2 + 1, 'train_loss': avg_loss}

        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                
        model.eval()
        val_similarities = []
        val_probs = []
        val_targets = []
        all_val_logits = [] 
        all_val_labels = [] 
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                for batch_idx, batch in enumerate(val_loader): 
                    mri_val = batch["mri_slices"].to(device, non_blocking=True)
                    pet_val = batch["pet_slices"].to(device, non_blocking=True)
                    patient_ids = batch["patient_id"]

                    
                    mri_emb_student_val = model.encode_batch(mri_val)
                    student_emb_val = model.self_attn(query=mri_emb_student_val, key=mri_emb_student_val)
                    student_emb_norm_val = normalize(student_emb_val, p=2, dim=1)
                  
                    mri_emb_for_teacher_val = teacher.encode_batch(mri_val)
                    pet_emb_for_teacher_val = teacher.encode_batch(pet_val)
                    teacher_emb_for_sim = teacher.cross_attn(query=pet_emb_for_teacher_val, key=mri_emb_for_teacher_val)
                    teacher_emb_for_sim_norm = normalize(teacher_emb_for_sim.detach(), p=2, dim=1)
                   
                    batch_sim = F.cosine_similarity(student_emb_norm_val, teacher_emb_for_sim_norm).cpu().numpy()
                    val_similarities.extend(batch_sim)

                    logits = model.classifier_head(student_emb_val).squeeze(1)

                    if batch_idx % 100 == 0:  
                        print(f"\nEmbedding diagnostic (batch {batch_idx}) ===")
                        print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        
                        probs = torch.sigmoid(logits)
                        print(f"Probability: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
                        
                        norms = torch.norm(student_emb_val, dim=1)
                        print(f"Embedding norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")

                        labels_tensor = torch.tensor(get_labels_from_patient_ids(patient_ids, val_dataset_t1.metadata), device=device).float()

                        pos_mask = labels_tensor == 1
                        neg_mask = labels_tensor == 0
                        
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_mean = logits[pos_mask].mean().item()
                            neg_mean = logits[neg_mask].mean().item()
                            separation = pos_mean - neg_mean
                            print(f" Separation diagnostic: pos={pos_mean:.4f}, neg={neg_mean:.4f}, sep={separation:.4f}")
                            
                            pos_min = logits[pos_mask].min().item() if pos_mask.sum() > 0 else float('inf')
                            neg_max = logits[neg_mask].max().item() if neg_mask.sum() > 0 else float('-inf')
                            if pos_min < neg_max:
                                print(f" Overlap: pos_min={pos_min:.4f} < neg_max={neg_max:.4f}")
           
                    all_val_logits.append(logits.detach().cpu())                                       
                    probs = torch.sigmoid(logits).cpu().numpy()
                    labels = get_labels_from_patient_ids(patient_ids, val_dataset_t1.metadata)
                               
                    val_probs.extend(probs)
                    val_targets.extend(labels)                    
                    labels_tensor = torch.tensor(labels, device=device).float() 
                    all_val_labels.append(labels_tensor.cpu()) 

        val_stats = None  
        if len(all_val_logits) > 0 and len(all_val_labels) > 0:
            try:
                all_val_logits_tensor = torch.cat(all_val_logits)
                all_val_labels_tensor = torch.cat(all_val_labels)

                val_stats = analyze_class_separation(
                    all_val_logits_tensor, 
                    all_val_labels_tensor, 
                    epoch_f2, 
                    prefix="[VAL]"
                )

                print("\n Class separation analysis:")
                print(f"   - Epoch: {epoch_f2 + 1}")
                print(f"   - val_stats type: {type(val_stats)}")
                print(f"   - val_stats keys: {val_stats.keys() if val_stats else 'None'}")

                if torch.cuda.is_available():
                    print(f"   - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                    print(f"   - GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                                
                if val_stats:
                    optimal_logit_thresh = val_stats['optimal_threshold']
                    optimal_prob_thresh = 1 / (1 + np.exp(-optimal_logit_thresh))
                    print(f"\n Estimated optimal threshold: {optimal_prob_thresh:.4f}")
                    
                    if optimal_prob_thresh < 0.05:
                        print(f" Very low threshold, limited to 0.05")
                        optimal_prob_thresh = 0.05
                    
                    preds_opt = (np.array(val_probs) > optimal_prob_thresh).astype(int)
                    acc_opt_new = accuracy_score(val_targets, preds_opt)
                    prec_opt_new = precision_score(val_targets, preds_opt, zero_division=0)
                    rec_opt_new = recall_score(val_targets, preds_opt, zero_division=0)
                    f1_opt_new = f1_score(val_targets, preds_opt, zero_division=0)
                    
                    print(f"\nValiation with analitical threshold = {optimal_prob_thresh:.3f} ===")
                    print(f"  Accuracy:  {acc_opt_new:.3f}")
                    print(f"  Precision: {prec_opt_new:.3f}")
                    print(f"  Recall:    {rec_opt_new:.3f}")
                    print(f"  F1 score:  {f1_opt_new:.3f}")
                
                print("\nDistribution logits analysis")
                analyze_classifier_outputs(all_val_logits_tensor, all_val_labels_tensor, epoch_f2, EXPERIMENT_DIR)
            except Exception as e:
                print(f"Error in logit analysis: {e}")

        print(f"\n Embedding similarity check:")
        print(f"   - len(val_similarities): {len(val_similarities)}")
        print(f"   - type(val_similarities): {type(val_similarities)}")
        if len(val_similarities) > 0:
            print(f"   - First element: {val_similarities[0]}")
            print(f"   - First element type: {type(val_similarities[0])}")
        print(f"   - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        if val_similarities:
            avg_val_similarity = sum(val_similarities) / len(val_similarities)
            print(f"\nStudent Validation Statistics Epoch {epoch_f2+1} ===")
            print(f"Average validation similarity: {avg_val_similarity:.4f}")

            current_phase2_data['val_sim'] = avg_val_similarity
                    
            if avg_val_similarity > best_student_similarity:
                    best_student_similarity = avg_val_similarity
                    best_similarity_state = copy.deepcopy(model.state_dict())
                    torch.save(best_similarity_state, BEST_STUDENT_PATH)
                    print(f" Saved new best student similarity model! Sim: {best_student_similarity:.4f}")
                    
                    eval_results_sim = evaluate_model_with_diagnostics(
                        model, val_loader, val_dataset_t1.metadata,  # use val_dataset_t1
                        threshold=0.5, 
                        use_this_threshold_for_opt=False, 
                        prefix="VAL (Similarity)"
                    )
                                        
                    threshold_data_for_similarity_model = {
                        "threshold": eval_results_sim['metrics_opt']['threshold'],
                        "precision": eval_results_sim['metrics_opt']['precision'],
                        "recall":    eval_results_sim['metrics_opt']['recall'],
                        "f1":        eval_results_sim['metrics_opt']['f1'],
                        "accuracy":  eval_results_sim['metrics_opt']['accuracy']
                    }
                    
                    with open(os.path.join(EXPERIMENT_DIR, "best_threshold_similarity.pkl"), "wb") as f:
                        pickle.dump(threshold_data_for_similarity_model, f)
                    with open(os.path.join(EXPERIMENT_DIR, "best_threshold_similarity.txt"), "w") as f:
                        f.write(f"{eval_results_sim['metrics_opt']['threshold']}\t"
                                f"{eval_results_sim['metrics_opt']['precision']:.4f}\t"
                                f"{eval_results_sim['metrics_opt']['recall']:.4f}\t"
                                f"{eval_results_sim['metrics_opt']['f1']:.4f}")

                    with open(os.path.join(EXPERIMENT_DIR, "best_threshold_similarity_t1.txt"), "w") as f:
                        f.write(f"{eval_results_sim['metrics_opt']['threshold']}\t"
                                f"{eval_results_sim['metrics_opt']['precision']:.4f}\t"
                                f"{eval_results_sim['metrics_opt']['recall']:.4f}\t"
                                f"{eval_results_sim['metrics_opt']['f1']:.4f}")

                    if (val_loader_t2 is not None) and hasattr(val_loader_t2, "dataset") and (len(val_loader_t2.dataset) > 0):
                        eval_results_sim_t2 = evaluate_model_with_diagnostics(
                            model, val_loader_t2, val_dataset_t2.metadata,
                            threshold=0.5, use_this_threshold_for_opt=False, prefix="VAL (Similarity T2)"
                        )
                        
                        with open(os.path.join(EXPERIMENT_DIR, "best_threshold_similarity_t2.txt"), "w") as f:
                            f.write(f"{eval_results_sim_t2['metrics_opt']['threshold']}\t"
                                    f"{eval_results_sim_t2['metrics_opt']['precision']:.4f}\t"
                                    f"{eval_results_sim_t2['metrics_opt']['recall']:.4f}\t"
                                    f"{eval_results_sim_t2['metrics_opt']['f1']:.4f}")
                        
                        print(f" Threshold (Similarity) T2 saved: {eval_results_sim_t2['metrics_opt']['threshold']:.3f}")

                    print(f" Threshold (Similarity) successfully saved: {eval_results_sim['metrics_opt']['threshold']:.3f}")

        preds = (np.array(val_probs) > 0.5).astype(int)

        acc = accuracy_score(val_targets, preds)
        prec = precision_score(val_targets, preds, zero_division=0)
        rec = recall_score(val_targets, preds, zero_division=0)
        f1 = f1_score(val_targets, preds, zero_division=0)

        print(f"\n Phase 2 Validation - Classification")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 score:  {f1:.3f}")

        if prec > 0 and rec > 0:
            prec_recall_ratio = prec / rec
            balanced_score = f1 * (1 - abs(0.5 - prec_recall_ratio) * 0.5)
            print(f"  Precision/recall ratio: {prec_recall_ratio:.3f}")
            print(f"  Balanced score: {balanced_score:.3f}")

        best_thresh, best_f1 = find_best_threshold(val_probs, val_targets)
        preds_opt = (np.array(val_probs) > best_thresh).astype(int)

        acc_opt = accuracy_score(val_targets, preds_opt)
        prec_opt = precision_score(val_targets, preds_opt, zero_division=0)
        rec_opt = recall_score(val_targets, preds_opt, zero_division=0)

        print("\n Model validation")
        eval_results = evaluate_model_with_diagnostics(
            model, val_loader, val_dataset_t1.metadata, 
            threshold=0.5, 
            use_this_threshold_for_opt=False,  
            prefix="VAL"
        )

        metrics_fixed = eval_results['metrics_fixed']
        metrics_opt = eval_results['metrics_opt']
        class_stats = eval_results['class_stats']

        if val_stats:  
            convergence_monitor['epoch_f2'].append(epoch_f2+ 1)
            convergence_monitor['separation'].append(val_stats['separation'])
            convergence_monitor['overlap_pct'].append(val_stats['overlap_pct'])
            convergence_monitor['f1_score'].append(metrics_opt['f1'])  
            
            
            if (epoch_f2+ 1) % 10 == 0:
                print("\n Convergence summary:")
                print(f"  Separation trend: {convergence_monitor['separation'][-10:]}")
                print(f"  Overlap trend: {convergence_monitor['overlap_pct'][-10:]}")
                print(f"  F1 trend: {convergence_monitor['f1_score'][-10:]}")

        print(f"\nPhase 2 Validation - Classification (FIXED THRESHOLD 0.5)")
        print(f"Accuracy:  {metrics_fixed['accuracy']:.3f}")
        print(f"Precision: {metrics_fixed['precision']:.3f}")
        print(f"Recall:    {metrics_fixed['recall']:.3f}")
        print(f"F1 score:  {metrics_fixed['f1']:.3f}")

        if metrics_fixed['precision'] > 0 and metrics_fixed['recall'] > 0:
            prec_recall_ratio = metrics_fixed['precision'] / metrics_fixed['recall']
            balanced_score = metrics_fixed['f1'] * (1 - abs(0.5 - prec_recall_ratio) * 0.5)
            print(f"  Precision/recall ratio: {prec_recall_ratio:.3f}")
            print(f"  Balanced score: {balanced_score:.3f}")

        print(f"\nValidation (Optimal threshold {metrics_opt['threshold']:.3f}) ===")
        print(f"Accuracy:  {metrics_opt['accuracy']:.3f}")
        print(f"Precision: {metrics_opt['precision']:.3f}")
        print(f"Recall:    {metrics_opt['recall']:.3f}")
        print(f"F1 score:  {metrics_opt['f1']:.3f}")

        phase2_history['epochs'].append(epoch_f2 + 1)
        phase2_history['train_loss'].append(avg_loss)
        phase2_history['val_sim'].append(avg_val_similarity if val_similarities else 0)
        phase2_history['val_f1'].append(metrics_opt['f1'])

        if metrics_opt['precision'] > 0 and metrics_opt['recall'] > 0:
            prec_recall_ratio = metrics_opt['precision'] / metrics_opt['recall']
            balanced_score = metrics_opt['f1'] * (1 - abs(0.5 - prec_recall_ratio) * 0.5)
            print(f"  Precision/recall ratio: {prec_recall_ratio:.3f}")
            print(f"  Balanced score: {balanced_score:.3f}")

        clf_scheduler.step(metrics_opt['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate updated: {current_lr:.2e}")
                    
        eval_results_t2_val = None
        if (val_loader_t2 is not None) and hasattr(val_loader_t2, "dataset") and (len(val_loader_t2.dataset) > 0):
            eval_results_t2_val = evaluate_model_with_diagnostics(
                model, val_loader_t2, val_dataset_t2.metadata,
                threshold=0.5, use_this_threshold_for_opt=False, prefix="VAL[T2]"
            )
            f1_t2 = eval_results_t2_val['metrics_opt']['f1']
        else:
            f1_t2 = None

        from sklearn.metrics import roc_auc_score

        # AUC T1
        current_auc_t1 = 0.0
        if len(val_probs) > 0 and len(np.unique(val_targets)) > 1:
            try:
                current_auc_t1 = roc_auc_score(val_targets, val_probs)
                print(f" [T1w] Validation AUC: {current_auc_t1:.3f}") 
            except Exception as e:
                print(f" Error AUC T1: {e}")

        # AUC T2
        current_auc_t2 = 0.0
        if eval_results_t2_val is not None:
            val_probs_t2 = []
            val_targets_t2 = []
            
            model.eval()
            with torch.no_grad():
                for batch in val_loader_t2:
                    mri = batch["mri_slices"].to(device)
                    pids = batch["patient_id"]
                    
                    mri_emb = model.encode_batch(mri)
                    student_emb = model.self_attn(query=mri_emb, key=mri_emb)
                    logits = model.classifier_head(student_emb).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                    labels = get_labels_from_patient_ids(pids, val_dataset_t2.metadata)
                    val_probs_t2.extend(probs)
                    val_targets_t2.extend(labels)
            
            if len(val_probs_t2) > 0 and len(np.unique(val_targets_t2)) > 1:
                try:
                    current_auc_t2 = roc_auc_score(val_targets_t2, val_probs_t2)
                    print(f" [T2w] Validation AUC: {current_auc_t2:.3f}") 
                except Exception as e:
                    print(f" Error AUC T2: {e}")

        phase2_history['val_auc_t1'].append(current_auc_t1)
        phase2_history['val_auc_t2'].append(current_auc_t2)

        # Calculate combined F1
        f1_t1 = metrics_opt['f1']
        f1_combined = (f1_t1 + f1_t2)/2 if f1_t2 is not None else f1_t1

        if f1_combined > best_f1_score:
            best_f1_score = f1_combined
            best_f1_state = copy.deepcopy(model.state_dict())
            torch.save(best_f1_state, best_f1_path)
            
            with open(os.path.join(EXPERIMENT_DIR, "best_threshold_f1_t1.txt"), "w") as f:
                f.write(f"{metrics_opt['threshold']}\t"
                        f"{metrics_opt['precision']:.4f}\t"
                        f"{metrics_opt['recall']:.4f}\t"
                        f"{metrics_opt['f1']:.4f}")
            
            if f1_t2 is not None:
                with open(os.path.join(EXPERIMENT_DIR, "best_threshold_f1_t2.txt"), "w") as f:
                    f.write(f"{eval_results_t2_val['metrics_opt']['threshold']}\t"
                            f"{eval_results_t2_val['metrics_opt']['precision']:.4f}\t"
                            f"{eval_results_t2_val['metrics_opt']['recall']:.4f}\t"
                            f"{eval_results_t2_val['metrics_opt']['f1']:.4f}")
            
            t2_str = f"{f1_t2:.3f}" if f1_t2 is not None else "N/A"
            print(f" New best saved! F1_combined={f1_combined:.4f} (T1={f1_t1:.3f}, T2={t2_str})")

            phase2_no_improve = 0
        else:
            phase2_no_improve += 1
            print(f" No improvement in F1. Epochs without improvement.: {phase2_no_improve}/{phase2_patience}")

            if phase2_no_improve >= phase2_patience:
                print(f" Early stopping activated!")
                break
        should_visualize = False

        if f1_combined > best_f1_score - 0.001: 
            should_visualize = True

        if epoch_f2 < 10 and (epoch_f2 + 1) % 2 == 0: 
            should_visualize = True
        elif epoch_f2 < 40 and (epoch_f2 + 1) % 5 == 0: 
            should_visualize = True
        elif (epoch_f2 + 1) % 10 == 0:
            should_visualize = True

        if epoch_f2 == PHASE2_EPOCHS - 1:
            should_visualize = True

        if should_visualize:
            # T1
            logger.info(f" Generation of T1 interpretability maps for the epoch {epoch_f2+1}")
            visualize_student_reasoning(
                student_model=model,
                data_loader=val_loader_t1,
                epoch=epoch_f2,
                EXPERIMENT_DIR=EXPERIMENT_DIR,
                prefix="val_t1_",
                fixed_indices=fixed_val_indices
            )
            logger.info(f" T1 maps saved in: {os.path.join(EXPERIMENT_DIR, 'saliency', f'val_t1_epoch_{epoch_f2+1}')}")
            
            # T2
            if len(val_set_t2) > 0:
                logger.info(f" Generation of T2 interpretability maps for the epoch {epoch_f2+1}...")
                visualize_student_reasoning(
                    student_model=model,
                    data_loader=val_loader_t2,
                    epoch=epoch_f2,
                    EXPERIMENT_DIR=EXPERIMENT_DIR,
                    prefix="val_t2_",
                    fixed_indices=fixed_val_indices
                )
                logger.info(f" T2 maps saved in: {os.path.join(EXPERIMENT_DIR, 'saliency', f'val_t2_epoch_{epoch_f2+1}')}")

    torch.save(model.state_dict(), FINAL_STUDENT_PATH)
    print(f" Phase 2 completed! Final model saved to: {FINAL_STUDENT_PATH}")
    
    print("\n Final analysis of the distillation efficiency")
    
    print("\n Final evaluation of the models saved during training")

    if os.path.isfile(BEST_STUDENT_PATH):
        model_sim = MRITextPETContrastive().to(device) 
        
        print(f"     Loading weights from {BEST_STUDENT_PATH}...")
        model_sim.load_state_dict(torch.load(BEST_STUDENT_PATH)) 
        print("     Weights loaded for model_sim.")

        print("\n Creating a model instance for 'F1 Optimized Model'")
        model_f1 = MRITextPETContrastive().to(device)
        
        print(f"     Loading weights from {best_f1_path}...")
        try:
            model_f1.load_state_dict(torch.load(best_f1_path), strict=True)
            print("     Weights loaded correctly for model_f1.")
        except RuntimeError as e:
            print(f"     Error loading: {e}")
            
            result = model_f1.load_state_dict(torch.load(best_f1_path), strict=False)
            print(f"    Missing keys: {result.missing_keys}")
            print(f"    Unexpected keys: {result.unexpected_keys}")
        
        diff_count = 0
        total_params = 0
        max_diff = 0
        for (name_sim, param_sim), (name_f1, param_f1) in zip(
            model_sim.named_parameters(), model_f1.named_parameters()
        ):
            total_params += param_sim.numel()
            param_diff = (param_sim - param_f1).abs().sum().item()
            diff_count += param_diff
            max_diff = max(max_diff, param_diff)
        
        print(f"Difference between models: {diff_count:.1f} out {total_params} parameters")
        print(f"Average difference per parameter: {diff_count/max(1,total_params):.6f}")
        print(f"Max difference: {max_diff:.6f}")
        
        if diff_count < 0.1:
            print(" Attention: models seem to be identical!")
    else:
        print(f" File {BEST_STUDENT_PATH} not found, comparison between models skipped.")

    print(f"\n Comparison between models on test set:")
    for model_path, model_name in [(BEST_STUDENT_PATH, "Optimized similarity model"), 
                                (best_f1_path, "F1 optimized model")]:
        
        print(f"\nEvaluation of: {model_name} ===")
        
        if not os.path.isfile(model_path):
            print(f" Model file not found: {model_path}")
            print(f"Skipping this model evaluation.")
            continue
        
        test_model = MRITextPETContrastive().to(device)
        test_model.load_state_dict(torch.load(model_path))
        test_model.eval() 
        
        print(f"Model eval mode: training={test_model.training}") 
        print(f"Dropout active: {any(m.training for m in test_model.modules() if isinstance(m, nn.Dropout))}")  
        
        if "similarity" in model_name:
            base_name = "best_threshold_similarity"
        else:
            base_name = "best_threshold_f1"
                
        best_thresh_from_validation = None
        t1_path = os.path.join(EXPERIMENT_DIR, f"{base_name}_t1.txt")
        if os.path.exists(t1_path):
            with open(t1_path, "r") as f:
                content = f.read().strip()
                best_thresh_from_validation = float(content.split('\t')[0]) if '\t' in content else float(content)
            print(f" Threshold [T1] loaded by: {t1_path} → {best_thresh_from_validation:.3f}")
        else:
            print(" Threshold [T1] not found: recalculation on validation T1")
            # recalculation on val_loader_t1 / val_dataset_t1
            test_model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    mri = batch["mri_slices"].to(device)
                    pids = batch["patient_id"]
                    mri_enc = test_model.encode_batch(mri)
                    stud_emb = test_model.self_attn(query=mri_enc, key=mri_enc)
                    logits = test_model.classifier_head(stud_emb).squeeze(1)
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(get_labels_from_patient_ids(pids, val_dataset_t1.metadata))
            best_thresh_from_validation, _ = find_best_threshold(val_probs, val_labels)
            print(f" Threshold [T1] recalculated: {best_thresh_from_validation:.3f}")            
                    
        eval_results = evaluate_model_with_diagnostics(
            test_model, test_loader, test_dataset_t1.metadata,
            threshold=0.5,
            use_this_threshold_for_opt=True,
            value_for_opt_threshold=best_thresh_from_validation,
            prefix=f"TEST ({model_name})"
        )
                       
        metrics_fixed = eval_results['metrics_fixed']
        metrics_opt = eval_results['metrics_opt']
        
        print(f"\n Results with fixed threshold = 0.5")
        print(f"  Accuracy:  {metrics_fixed['accuracy']:.3f}")
        print(f"  Precision: {metrics_fixed['precision']:.3f}")
        print(f"  Recall:    {metrics_fixed['recall']:.3f}")
        print(f"  F1 Score:  {metrics_fixed['f1']:.3f}")
        
        print(f"\n  With optimal threshold ({metrics_opt['threshold']:.3f}):")
        print(f"  Accuracy:  {metrics_opt['accuracy']:.3f}")
        print(f"  Precision: {metrics_opt['precision']:.3f}")
        print(f"  Recall:    {metrics_opt['recall']:.3f}")
        print(f"  F1 Score:  {metrics_opt['f1']:.3f}")
        
        csv_path = os.path.join(EXPERIMENT_DIR, f"test_predictions_{model_name.replace(' ', '_')}.csv")
        save_predictions_to_csv(eval_results['probs'], eval_results['labels'], 
                            eval_results['patient_ids'], metrics_opt['threshold'], csv_path)

# Create evaluation plots for the test set
        test_eval_metrics = create_evaluation_plots(
            y_true=eval_results['labels'],
            y_probs=eval_results['probs'],
            save_dir=EXPERIMENT_DIR,
            prefix=f"test_{model_name.replace(' ', '_').lower()}_",
            threshold=metrics_opt['threshold']
        )
        print(f" [T1w] Test AUC: {test_eval_metrics['auc']:.3f}") 
        print(f" Test PPV: {test_eval_metrics['ppv']:.3f}, NPV: {test_eval_metrics['npv']:.3f}")
        
        print(f"\nTEST T2 evaluation for {model_name} ===")
        if len(test_set_t2) == 0:
            print(" No T2 data in test, skip.")
        else:
            if "Similarity" in model_name:
                base_name = "best_threshold_similarity"
            else:
                base_name = "best_threshold_f1"
            
            best_thresh_from_validation_t2 = None
            t2_path = os.path.join(EXPERIMENT_DIR, f"{base_name}_t2.txt")
            if os.path.exists(t2_path):
                with open(t2_path, "r") as f:
                    content = f.read().strip()
                    best_thresh_from_validation_t2 = float(content.split('\t')[0]) if '\t' in content else float(content)
                print(f" Threshold [T2] loaded by: {t2_path} → {best_thresh_from_validation_t2:.3f}")
            else:
                print(" Threshold [T2] not found: recalculation on T2 validation")
                val_probs_t2, val_labels_t2 = [], []
                with torch.no_grad():
                    for batch in val_loader_t2:
                        mri = batch["mri_slices"].to(device)
                        pids = batch["patient_id"]
                        mri_enc = test_model.encode_batch(mri)
                        stud_emb = test_model.self_attn(query=mri_enc, key=mri_enc)
                        logits = test_model.classifier_head(stud_emb).squeeze(1)
                        val_probs_t2.extend(torch.sigmoid(logits).cpu().numpy())
                        val_labels_t2.extend(get_labels_from_patient_ids(pids, val_dataset_t2.metadata))
                best_thresh_from_validation_t2, _ = find_best_threshold(val_probs_t2, val_labels_t2)
                print(f" Threshold [T2] recalculated: {best_thresh_from_validation_t2:.3f}")

            eval_results_t2 = evaluate_model_with_diagnostics(
                test_model, test_loader_t2, test_dataset_t2.metadata,
                threshold=0.5, use_this_threshold_for_opt=True,
                value_for_opt_threshold=best_thresh_from_validation_t2, 
                prefix=f"[T2] TEST ({model_name})"
            )

            test_eval_metrics_t2 = create_evaluation_plots(
                y_true=eval_results_t2['labels'],
                y_probs=eval_results_t2['probs'],
                save_dir=EXPERIMENT_DIR,
                prefix=f"test_t2_{model_name.replace(' ', '_').lower()}_",
                threshold=eval_results_t2['metrics_opt']['threshold']
            )
            print(f" [T2w] Test AUC: {test_eval_metrics_t2['auc']:.3f}") 
            print(f" [T2w] Test PPV: {test_eval_metrics_t2['ppv']:.3f}, NPV: {test_eval_metrics_t2['npv']:.3f}")
        
            print(f"\n [T2] Results with fixed threshold = 0.5")
            print(f"  Accuracy:  {eval_results_t2['metrics_fixed']['accuracy']:.3f}")
            print(f"  Precision: {eval_results_t2['metrics_fixed']['precision']:.3f}")
            print(f"  Recall:    {eval_results_t2['metrics_fixed']['recall']:.3f}")
            print(f"  F1 Score:  {eval_results_t2['metrics_fixed']['f1']:.3f}")
            
            print(f"\n  [T2] With optimal threshold ({eval_results_t2['metrics_opt']['threshold']:.3f}):")
            print(f"  Accuracy:  {eval_results_t2['metrics_opt']['accuracy']:.3f}")
            print(f"  Precision: {eval_results_t2['metrics_opt']['precision']:.3f}")
            print(f"  Recall:    {eval_results_t2['metrics_opt']['recall']:.3f}")
            print(f"  F1 Score:  {eval_results_t2['metrics_opt']['f1']:.3f}")
            
            csv_path_t2 = os.path.join(EXPERIMENT_DIR, f"test_predictions_T2_{model_name.replace(' ', '_')}.csv")
            save_predictions_to_csv(eval_results_t2['probs'], eval_results_t2['labels'],
                                eval_results_t2['patient_ids'], eval_results_t2['metrics_opt']['threshold'], csv_path_t2)
        
        test_model.eval() 
        teacher.eval()

        
        test_model.eval()
        teacher.eval()
        sim_for_this_model = []

        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                for batch in tqdm(test_loader, desc=f"Similarity for {model_name}"):
                    mri_test = batch["mri_slices"].to(device)
                    pet_test = batch["pet_slices"].to(device)
                    
                    mri_emb_student = test_model.encode_batch(mri_test)
                    student_emb = test_model.self_attn(query=mri_emb_student, key=mri_emb_student) 
                    student_emb_norm = normalize(student_emb, p=2, dim=1)

                    mri_emb_teacher = teacher.encode_batch(mri_test)  # Encoder TEACHER
                    pet_emb_teacher = teacher.encode_batch(pet_test)  # Encoder TEACHER
                    teacher_emb = teacher.cross_attn(query=pet_emb_teacher, key=mri_emb_teacher)
                    teacher_emb_norm = normalize(teacher_emb, p=2, dim=1)
                    batch_sim = F.cosine_similarity(student_emb_norm, teacher_emb_norm).cpu().numpy() 
                    sim_for_this_model.extend(batch_sim)
        
        avg_sim = sum(sim_for_this_model) / max(1, len(sim_for_this_model))
        print(f"\n Average similarity on test set for {model_name}: {avg_sim:.4f}")
        
        
        plt.figure(figsize=(10, 6))
        plt.hist(sim_for_this_model, bins=20, alpha=0.7)
        plt.axvline(x=avg_sim, color='r', linestyle='--', 
                    label=f'Average: {avg_sim:.4f}')
        plt.xlabel('Cosine similarity')
        plt.ylabel('Count')
        plt.title(f'Similarity Student-Teacher ({model_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
      
        plt.close()

    print(f"\n Final analysis completed! All results are saved in: {EXPERIMENT_DIR}")

    if os.path.isfile(best_f1_path):
        print(f" Loading the model with best F1 score from: {best_f1_path} for embedding extraction.")
        model.load_state_dict(torch.load(best_f1_path, map_location=device))
    else:
        print(f" Best F1 Model Files ({best_f1_path}) not found. Using the latest model state for extraction.")
   
    print(f"The results of the entire experiment are available in: {EXPERIMENT_DIR}")

    create_training_plots(pretrain_history, phase1_history, phase2_history, EXPERIMENT_DIR)


def create_training_plots(pretrain_hist, phase1_hist, phase2_hist, save_dir):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    if pretrain_hist['epochs']:
        ax1_twin = ax1.twinx()

        line1 = ax1.plot(pretrain_hist['epochs'], pretrain_hist['train_loss'], 
                        'b-', label='Train Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Accuracy and F1 on right axis
        line2 = ax1_twin.plot(pretrain_hist['epochs'], pretrain_hist['train_acc'], 
                             'g--', label='Train Acc', linewidth=2)
        line3 = ax1_twin.plot(pretrain_hist['epochs'], pretrain_hist['val_f1'], 
                             'r-', label='Val F1', linewidth=2)
        ax1_twin.set_ylabel('Accuracy/F1', color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        ax1_twin.set_ylim(0, 1)
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
    ax1.set_title('Pre-training (Only classification)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Phase 1 (Triplet + Classification)
    ax2 = axes[1]
    if phase1_hist['epochs']:
        ax2_twin = ax2.twinx()
        
        # Loss on left axis
        line1 = ax2.plot(phase1_hist['epochs'], phase1_hist['train_loss'], 
                        'b-', label='Train Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Margin and F1 on right axis
        line2 = ax2_twin.plot(phase1_hist['epochs'], phase1_hist['val_margin'], 
                             'g--', label='Val Margin', linewidth=2)
        line3 = ax2_twin.plot(phase1_hist['epochs'], phase1_hist['val_f1'], 
                             'r-', label='Val F1', linewidth=2)
        ax2_twin.set_ylabel('Margin/F1', color='g')
        ax2_twin.tick_params(axis='y', labelcolor='g')
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')
        
    ax2.set_title('Phase 1: Teacher Training (Triplet)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Phase 2 (Distillation)
    ax3 = axes[2]
    if phase2_hist['epochs']:
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(phase2_hist['epochs'], phase2_hist['train_loss'], 
                        'b-', label='Train Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        
        line2 = ax3_twin.plot(phase2_hist['epochs'], phase2_hist['val_sim'], 
                             'g--', label='Val Similarity', linewidth=2)
        line3 = ax3_twin.plot(phase2_hist['epochs'], phase2_hist['val_f1'], 
                             'r-', label='Val F1', linewidth=2)
        ax3_twin.set_ylabel('Similarity/F1', color='g')
        ax3_twin.tick_params(axis='y', labelcolor='g')
        ax3_twin.set_ylim(0, 1)
        
        # Legend combined
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')
        
    ax3.set_title('Phase 2: Student Distillation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_history_3phases.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n Training graphs saved in: {plot_path}")
    
    # Also save numeric data
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'pretrain': pretrain_hist,
            'phase1': phase1_hist,
            'phase2': phase2_hist
        }, f, indent=2)
    print(f" Numerical data saved in: {history_path}")

if __name__ == "__main__":
    main()