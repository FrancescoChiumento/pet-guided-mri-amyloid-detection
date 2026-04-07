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
classification using multi-modal neuroimaging data (T1w MRI and PET scans).

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
- Balanced sampling for class imbalance
- Mixed precision training (AMP)
- Early stopping with patience
- Comprehensive evaluation metrics (F1, precision, recall, AUC)
- Interpretability analysis with saliency maps
- Anti-data-leakage validation

Input:  
- T1w MRI slices (224×224, 25 slices per patient)
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.5" # max memory block to 64 MB
import pandas as pd
import torch
import gc
import random
import json
import pickle
import re
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import logging
import numpy as np
import torch.nn as nn

from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
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

scaler = GradScaler(enabled=torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_EBS = 32  # Effective Batch Size desired

CHUNK_SIZE = 4
SKIP_PHASE1 = False

EXISTING_TEACHER_DIR = os.environ.get("TEACHER_DIR", "experiments/teacher")
EXISTING_TEACHER_PATH = os.path.join(EXISTING_TEACHER_DIR, "best_contrastive.pt")

BASE_SAVE_DIR = os.environ.get("OUTPUT_DIR", "experiments")

if SKIP_PHASE1 and os.path.exists(EXISTING_TEACHER_PATH):
    EXPERIMENT_DIR = EXISTING_TEACHER_DIR
    print(f"directory current experiment: {EXPERIMENT_DIR}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join(BASE_SAVE_DIR, f"exp_{timestamp}")
    print(f"new directory current experiment: {EXPERIMENT_DIR}")

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_DIR, "train_analysis"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_DIR, "test_analysis"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_DIR, "saliency"), exist_ok=True)

PHASE1_EPOCHS = 15  
PHASE2_EPOCHS = 100 
TEACHER_PATH = os.path.join(EXPERIMENT_DIR, "best_contrastive.pt")
BEST_STUDENT_PATH = os.path.join(EXPERIMENT_DIR, "best_student_model.pt")
FINAL_STUDENT_PATH = os.path.join(EXPERIMENT_DIR, "final_student_model.pt")

print(f" training plan")
print(f"   - Phase 1: {PHASE1_EPOCHS} epochs (triplet loss + classification)")
print(f"   - Phase 2: {PHASE2_EPOCHS} epochs (distillation + classification)")
print(f"   - Total: {PHASE1_EPOCHS + PHASE2_EPOCHS} epochs")

# --- helpers top-level, picklable with num_workers>0 ---
def random_gamma_tensor(t, p=0.5, min_gamma=0.9, max_gamma=1.1):
    if torch.rand(()) < p:
        g = torch.empty(()).uniform_(min_gamma, max_gamma).item()
        t = t.clamp(min=1e-6).pow(g)
    return t

def add_gaussian_noise_tensor(t, p=0.5, sigma_min=0.01, sigma_max=0.03):
    if torch.rand(()) < p:
        sigma = torch.empty(()).uniform_(sigma_min, sigma_max).item()
        t = (t + torch.randn_like(t) * sigma).clamp(0, 1)
    return t

class PairedTransform: # same transformations to MRI and PET of same patient
    def __init__(self, augment=True):
        self.augment = augment
        self.resize = transforms.Resize((224, 224))
        
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        
        
        if augment:
            # no flip 
            self.spatial_transforms = transforms.Compose([
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
            # slight variation in brightness and contrast
            self.color_transform = transforms.ColorJitter(brightness=0.1, contrast=0.1)
            # slight blur
            self.blur_transform = transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3)

            self.tensor_erase_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: random_gamma_tensor(t, p=0.5, min_gamma=0.9, max_gamma=1.1)),
                transforms.Lambda(lambda t: add_gaussian_noise_tensor(t, p=0.5, sigma_min=0.01, sigma_max=0.03)),
                transforms.RandomErasing(p=0.25, scale=(0.05, 0.12), value=0)
            ])
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
            
            #same transformation to MRI with fixed seed
            torch.manual_seed(seed)
            random.seed(seed)
            mri_aug = self.spatial_transforms(mri_resized)
            mri_color = self.color_transform(mri_aug)
            mri_blur = self.blur_transform(mri_color)
            
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
            
            # PET, same seed
            torch.manual_seed(erase_seed)
            pet_tensor_erased = self.tensor_erase_transform(pet_blur)
            
            # normalization 
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
    print("\nverify paired transform")
    transform = PairedTransform(augment=True)
    
    
    test_img = Image.new('RGB', (224, 224), color='red')
    
    
    max_diff = 0
    all_diffs = []
    
    for i in range(10):
        mri_t, pet_t = transform(test_img.copy(), test_img.copy())
        
        diff = (mri_t - pet_t).abs().mean().item() 
        all_diffs.append(diff) #
        max_diff = max(max_diff, diff)
        
        if i < 3:  
            print(f"  Test {i+1}: average difference = {diff:.6f}")
    
    print(f"  ...")
    print(f"  Average difference on 10 tests: {sum(all_diffs)/len(all_diffs):.6f}")
    print(f"  Max difference on 10 tests: {max_diff:.6f}")
    
    
    if max_diff > 0.15:
        print(" Not synchronized transformation")
        return False
    else:
        print(" OK: transformation correctly synchronized")
        return True   
    
class SoftTripletLoss(nn.Module):
    def __init__(self, margin=1.0):  
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
        # Creates buffer for pos_weight 
        self.register_buffer("pos_weight_t", torch.tensor([float(pos_weight)]))
        
    def forward(self, logits, targets):
        targets = targets.to(logits.dtype)
        
        # # Label smoothing towards 0.5
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
                # (pos_mean - neg_mean) >= margin
                gap_deficit = F.relu(self.margin - (pos_mean - neg_mean))
                
                gap_deficit = gap_deficit * 0.1  # reduce impact
                if self.reduction == 'mean':
                    base_loss = loss.mean()
                elif self.reduction == 'sum':
                    base_loss = loss.sum()
                else:
                    base_loss = loss
                    
                #no negative loss
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
        nn.init.constant_(self.pool.bias, 0.0) #bias 0

        self.norm = nn.LayerNorm(dim) # training stabilization
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
    with torch.no_grad(): # no gradients computing
        norms = torch.norm(embeddings.view(-1, embeddings.size(-1)), dim=1)
        print(f"{name} Embedding Stats:")
        print(f"  - Average norm: {norms.mean().item():.4f}")
        print(f"  - std norm: {norms.std().item():.4f}")
        print(f"  - Range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")
        
        flat_emb = embeddings.view(-1, embeddings.size(-1))
        if flat_emb.size(0) > 20:
            sample_idx = torch.randperm(flat_emb.size(0))[:20]
            sample = flat_emb[sample_idx] 
            corr_matrix = torch.corrcoef(sample.T)
            print(f"  average correlation between dimensions: {corr_matrix.abs().mean().item():.4f}")

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
                patient_ids = batch["patient_id"] # patients list
                all_patient_ids.extend(patient_ids) 
                
                labels = get_labels_from_patient_ids(patient_ids, dataset_metadata)
                all_labels.extend(labels)
                                
                mri_emb = model.encode_batch(mri) # mri conversion in embeddings
                student_emb = model.self_attn(query=mri_emb, key=mri_emb) # self attention 
                logits = model.classifier_head(student_emb).squeeze(1) # final layer for prediction
                probs = torch.sigmoid(logits).cpu().numpy() 
                            
                all_logits.append(logits.detach().cpu())
                all_probs.extend(probs)
                
                if batch_idx % 100 == 0:
                    print(f"\n diagnostic {prefix} (batch {batch_idx}) ===")
                    print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                    print(f"Probability: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    
    all_logits_tensor = torch.cat(all_logits)
    all_labels_array = np.array(all_labels)
    all_probs_array = np.array(all_probs)
    
    preds_fixed = (all_probs_array > threshold).astype(int)
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
        print(f"threshold from optimal metrics: {best_thresh:.3f}")
    else:
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
    
    print(f"Predictions saved in {filepath}")
    print(f"   total patients: {len(patient_ids)}")
    print(f"   correct predictions: {correct_count} ({accuracy:.1f}%)")
    
    return results_df

class NeuroMultimodalDataset(Dataset):
    def __init__(self, mri_root, pet_root, csv_path, transform=None, max_slices=40, num_slices_to_use=None, verbose=True):
        self.mri_root = mri_root
        self.pet_root = pet_root
        self.transform = transform
        self.max_slices = max_slices
        self.num_slices_to_use = num_slices_to_use
        
        self.metadata = pd.read_csv(csv_path, sep="\t")
        # remove duplicates
        original_len = len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(subset=['MRId', 'session_id_pet'])
        if verbose:
            print(f"Removed {original_len - len(self.metadata)} duplicates from CSV")

        self.metadata["tracer"] = self.metadata["tracer"].astype(str).str.upper().str.strip()
        self.metadata = self.metadata.dropna(
    subset=["Centiloid_fSUVR_rsf_TOT_CORTMEAN", "MRId", "session_id_pet", "tracer"]
)
        self.patient_ids = self._get_valid_patient_ids()
        
        valid_subject_ids = [pid.split('_')[0] for pid in self.patient_ids]
        self.metadata = self.metadata[self.metadata['subject_id'].isin(valid_subject_ids)]
        
        self.healthy_patients = self.metadata[self.metadata['amyloid_positive'] == 0]
        self.demented_patients = self.metadata[self.metadata['amyloid_positive'] == 1]
        
        if verbose:
            print(f"Total valid patients: {len(self.patient_ids)}")
            print(f"Filtered Dataframe: {len(self.metadata)} rows")
            print(f"Healthy patients: {len(self.healthy_patients)}, unhealthy patients: {len(self.demented_patients)}")  
            
    def get_healthy_patients(self):
        return self.healthy_patients
    
    def get_demented_patients(self):
        return self.demented_patients
        
    def _get_valid_patient_ids(self):
        ids = []
        for _, row in self.metadata.iterrows():
            sub = row['subject_id']
            
            # MRI data 
            mri_day = row['MRId'].split("_")[-1] 
            
            # PET data
            pet_day = row['session_id_pet'].split("_")[-1]
            
            if "AV45" in row["tracer"]:
                tracer = "AV45"
            elif "PIB" in row["tracer"]:
                tracer = "PIB"
            else:
                continue
            pid = f"{sub}_ses-{pet_day}_{tracer}"
            mri_dir = os.path.join(self.mri_root, f"{sub}_ses-{mri_day}", "T1w_slices")
            pet_dir = os.path.join(self.pet_root, pid, "PET_slices")
            
            if os.path.isdir(mri_dir) and os.path.isdir(pet_dir):
                ids.append(pid)
            else:
                continue
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
                logger.warning(f"Error {f}: {e}")
        return imgs
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        parts = patient_id.split("_")
        sub = parts[0]
        pet_ses = parts[1].replace("ses-", "")  # PET data
        tracer = parts[2].strip().upper()
        
        rows = self.metadata[
            (self.metadata["subject_id"] == sub) &
            (self.metadata["session_id_pet"].str.endswith(pet_ses)) &
            (self.metadata["tracer"] == tracer)
        ]
        if rows.empty:
            raise ValueError(f"No metadata for patient_id={patient_id}")
        row = rows.iloc[0]
        
        # get correct MRI data from file
        mri_ses = row['MRId'].split("_")[-1]
        
        mri_dir = os.path.join(self.mri_root, f"{sub}_ses-{mri_ses}", "T1w_slices")
        mri_imgs = self._load_slices(mri_dir)
        
        pet_imgs = self._load_slices(os.path.join(self.pet_root, patient_id, "PET_slices"))
        if len(mri_imgs) == 0 or len(pet_imgs) == 0:
            raise ValueError(f"No slices for {patient_id}")
        assert len(mri_imgs) == len(pet_imgs), f"Mismatch in slice: {len(mri_imgs)} MRI vs {len(pet_imgs)} PET"
        
        if self.num_slices_to_use is not None and self.num_slices_to_use < len(mri_imgs):
            total_slices = len(mri_imgs)
            indices = []
            if self.num_slices_to_use > 1:
                indices = np.round(np.linspace(0, total_slices - 1, num=self.num_slices_to_use)).astype(int)
                indices = np.unique(indices)  # remove duplicates
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
            "patient_id": patient_id
        }
    
    def __len__(self):
        return len(self.patient_ids)

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
            (metadata_df["tracer"].str.upper().str.strip() == tracer)
        ]
        
        if row.empty:
            raise ValueError(f"label not found in {pid}")
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
        
        # skip if 0 or 1
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
        
        print(f"  average logits: Positives={pos_mean:.4f}, Negatives={neg_mean:.4f}")
        print(f"  separation={separation:.4f}")
        
        plt.figure(figsize=(10, 6))
        if len(neg_logits) > 0:
            plt.hist(neg_logits, bins=20, alpha=0.5, label='Negatives', color='green')
        if len(pos_logits) > 0:
            plt.hist(pos_logits, bins=20, alpha=0.5, label='Positives', color='red')
        
        plt.axvline(x=0, color='black', linestyle='--')
        plt.title(f'Logits distribution (Epochs {epoch+1})')
        plt.xlabel('Logits')
        plt.ylabel('counts')
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
        plt.title(f'Probability Distribution (Epoch {epoch+1})')
        plt.xlabel('Probability')
        plt.ylabel('Counts')
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
        print(f"Vision model with LoRA: {type(self.model.vision_model)}")
        
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
            nn.Dropout(0.6),  
            nn.Linear(64, 1)
        )
        
        if freeze_projection:
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
    with torch.no_grad():
        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        pos_mask = labels_np == 1
        neg_mask = labels_np == 0
        
        pos_logits = logits_np[pos_mask]
        neg_logits = logits_np[neg_mask]
        
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            print(f"{prefix} Attention: some class doesn't have examples")
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
        print(f"\n{prefix} separation classes - {epoch_label}")
        print(f"  Positives: average={pos_mean:.4f}, std={pos_std:.4f}, n={len(pos_logits)}")
        print(f"  Negatives: average={neg_mean:.4f}, std={neg_std:.4f}, n={len(neg_logits)}")
        print(f"  Absolute separation: {separation:.4f}")
        print(f"  Significance (d'): {separation_significance:.4f}")
        
        pos_min = np.min(pos_logits)
        neg_max = np.max(neg_logits)
        
        overlap_percentage = 0  
        if pos_min < neg_max:
            overlap_percentage = sum(1 for p in pos_logits if p < neg_max) / len(pos_logits) * 100
            print(f"   OVERLAP: {overlap_percentage:.1f}% of positives under negative max")
        else:
            print(f" NO OVERLAP: min positive ({pos_min:.4f}) > max negative ({neg_max:.4f})")
                
        optimal_threshold = (pos_mean + neg_mean) / 2
        prob_threshold = 1 / (1 + np.exp(-optimal_threshold))
        print(f"  Optimal threshold (logit): {optimal_threshold:.4f} (prob: {prob_threshold:.4f})")
        
        return {
            'pos_mean': pos_mean, 
            'neg_mean': neg_mean,
            'separation': separation,
            'significance': separation_significance,
            'overlap_pct': overlap_percentage,
            'optimal_threshold': optimal_threshold
        }

def test_cerebellum_dependency(model, val_loader, num_samples=20):
    model.eval()
    deltas = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            mri = batch["mri_slices"].to(device)
            
            # normal prediction
            mri_emb = model.encode_batch(mri)
            emb = model.self_attn(query=mri_emb, key=mri_emb)
            logits_normal = model.classifier_head(emb).squeeze(1)
            
            # with cerebellum masked
            mri_masked = mri.clone()
            h_cutoff = int(mri.shape[3] * 0.72)  # masked 28% inferior
            mri_masked[:, :, :, h_cutoff:, :] = 0
            
            mri_emb_masked = model.encode_batch(mri_masked)
            emb_masked = model.self_attn(query=mri_emb_masked, key=mri_emb_masked)
            logits_masked = model.classifier_head(emb_masked).squeeze(1)
            
            # differences in prediction calculation
            prob_change = torch.abs(torch.sigmoid(logits_normal) - torch.sigmoid(logits_masked))
            deltas.extend(prob_change.cpu().numpy())
    
    mean_delta = np.mean(deltas)
    print(f"\n Cerebellum prediction influence:")
    print(f"Changing in probabilities when masked: {mean_delta:.3f}")
    print(f"Max change: {np.max(deltas):.3f}")
    
    if mean_delta > 0.15:
        print("High dependence from cerebellum!")
    elif mean_delta > 0.05:
        print("Moderate dependence")
    else:
        print("Low dependence")
        
    return mean_delta

def select_negative_for_patient(anchor_patient, all_patients, suvr_threshold=5.0):
    anchor_id = anchor_patient["subject_id"]
    
    if pd.isna(anchor_patient["Centiloid_fSUVR_rsf_TOT_CORTMEAN"]):
        print(f"anchor patient {anchor_id} with SUVR value missing")
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
        print(f" no negative for anchor {anchor_id} with threshold {suvr_threshold}")
        
        all_other_patients = all_patients[all_patients["subject_id"] != anchor_id].copy()
        
        if all_other_patients.empty:
            print(f"no other available patients for {anchor_id}")
            return None
        
        valid_patients = all_other_patients.dropna(subset=["Centiloid_fSUVR_rsf_TOT_CORTMEAN"])
        
        if valid_patients.empty:
            print(f"no available SUVR patient for {anchor_id}")
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
        print(f" fallback: patient {most_different['subject_id']} with diff SUVR {most_different.get('suvr_diff', -1):.2f}")
        return most_different
    
    return possible_negatives.sample(n=1).iloc[0]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_negative_pet_embeddings(batch, dataset, healthy_df, demented_df, model, device, chunk_size=2):
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
            print(f"Anchor patients {anchor_full_id} not found in metadata")
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
                continue
            day = day_part[1:]
            
            pet_dir = os.path.join(dataset.pet_root, 
                                f"{negative_patient['subject_id']}_ses-d{day}_{tracer}", 
                                "PET_slices")
        except Exception as e:
            logger.error(f"Error parsing session_id_pet '{session_str}': {e}")
            continue
        
        if not os.path.isdir(pet_dir):
            print(f" PET directory not found for: {pet_dir}")
            continue
            
        # Load PET slices
        neg_pet_slices = dataset._load_slices(pet_dir)
        if len(neg_pet_slices) == 0:
            print(f" no slice for patients {negative_patient['subject_id']}")
            continue
            
        #Apply slice selection if needed
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

        del neg_pet_tensor, single_neg, single_emb, neg_pet_slices

        if i % 5 == 0:
            torch.cuda.empty_cache()
        
    return pet_neg_embeddings_list, valid_indices

def main():
    global batch_size, num_slices_to_use
    global batch_size_phase0, batch_size_phase1, batch_size_phase2
    batch_size_phase0 = 6  # Pre-training (MRI+PET)
    batch_size_phase1 = 6  # Phase 1 (MRI+PET+Triplet)
    batch_size_phase2 = 10  # Phase 2 (only MRI student)
    
    batch_size = batch_size_phase0
    num_slices_to_use = 25
        
    # metrics
    pretrain_history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_f1': []}
    phase1_history = {'epochs': [], 'train_loss': [], 'val_margin': [], 'val_f1': []}
    phase2_history = {'epochs': [], 'train_loss': [], 'val_sim': [], 'val_f1': []}

    print("\n" + "="*60)
    print(" TRAINING CONFIGURATION")
    print("="*60)
    print(f"- Batch size Pre-training: {batch_size_phase0}")
    print(f"- Batch size Phase 1: {batch_size_phase1}")
    print(f"- Batch size Phase 2: {batch_size_phase2}")
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
    print("="*60)
    is_transform_ok = verify_paired_transform()
    if not is_transform_ok:
        print("\n Warning: PairedTransform detected, applying fallback")
    else:
        print("   training")
    print("="*60 + "\n")
        
    mri_root = os.environ.get("MRI_ROOT", "data/OASIS_3/T1w_slices")
    pet_root = os.environ.get("PET_ROOT", "data/OASIS_3/PET_slices")
    csv_path = os.environ.get("CSV_PATH", "data/OASIS_3/selected_patients.tsv")  

    
    # 3 dataset
    train_dataset = NeuroMultimodalDataset(mri_root, pet_root, csv_path, transform=train_transform, num_slices_to_use=num_slices_to_use, verbose=True)
    val_dataset = NeuroMultimodalDataset(mri_root, pet_root, csv_path, transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False)
    test_dataset = NeuroMultimodalDataset(mri_root, pet_root, csv_path, transform=val_transform, num_slices_to_use=num_slices_to_use, verbose=False)

    print("\nloading split maps from csv")
    df_splits = pd.read_csv(csv_path, sep="\t", usecols=["subject_id", "split"]).drop_duplicates()
    split_map = {sid: int(s) for sid, s in zip(df_splits["subject_id"], df_splits["split"])}

    print("\ncalculate correct indices from dataset")

    # train dataset indices
    train_idx = []
    for i, pid in enumerate(train_dataset.patient_ids): 
        subject_id = subj_from_pid(pid) 
        split_value = split_map.get(subject_id, -1)
        if split_value in [2, 3, 4]:  # Train splits
            train_idx.append(i)

    # VAL dataset indices
    val_idx = []
    for i, pid in enumerate(val_dataset.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 0:  # Validation split
            val_idx.append(i)

    # TEST dataset indices
    test_idx = []
    for i, pid in enumerate(test_dataset.patient_ids):
        subject_id = subj_from_pid(pid)
        split_value = split_map.get(subject_id, -1)
        if split_value == 1:  # Test split
            test_idx.append(i)

    print(f"  Train: {len(train_idx)} indices out of {len(train_dataset.patient_ids)} total")
    print(f"  Val:   {len(val_idx)}  indices out of {len(val_dataset.patient_ids)} total")
    print(f"  Test:  {len(test_idx)}  indices out of {len(test_dataset.patient_ids)} total")

    # subset with correct indices
    from torch.utils.data import Subset

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)      
    test_set = Subset(test_dataset, test_idx)

    print(f"\nfinal set dimensions")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

    # test no overlaps
    print("\n Split verification")

    train_subjects = set()
    for idx in train_idx:
        pid = train_dataset.patient_ids[idx]
        train_subjects.add(subj_from_pid(pid))

    val_subjects = set()
    for idx in val_idx:
        pid = val_dataset.patient_ids[idx]
        val_subjects.add(subj_from_pid(pid))
        
    test_subjects = set()
    for idx in test_idx:
        pid = test_dataset.patient_ids[idx]
        test_subjects.add(subj_from_pid(pid))

    # check overlaps
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("Error: overlaps found")
        print(f"  Train-Val: {train_val_overlap}")
        print(f"  Train-Test: {train_test_overlap}")
        print(f"  Val-Test: {val_test_overlap}")
        raise ValueError(" No valid splits, duplicates between different splits")
    else:
        print(" No overlaps, correct splits")

    print("\n Anti-leakage check")

    train_pids_subset = [train_dataset.patient_ids[i] for i in train_idx]
    val_pids_subset = [val_dataset.patient_ids[i] for i in val_idx]
    test_pids_subset = [test_dataset.patient_ids[i] for i in test_idx]

    train_subjects_real = {subj_from_pid(pid) for pid in train_pids_subset}
    val_subjects_real = {subj_from_pid(pid) for pid in val_pids_subset}
    test_subjects_real = {subj_from_pid(pid) for pid in test_pids_subset}

    # check patients between different splits
    assert len(train_subjects_real & val_subjects_real) == 0, "LEAK: train val"
    assert len(train_subjects_real & test_subjects_real) == 0, "LEAK: train test"
    assert len(val_subjects_real & test_subjects_real) == 0, "LEAK: val-test"
    print("Assert 1: no subject shared between splits")

    print("multi-session, multi-sequence coherence")
    leak_found = False

    def count_subject_records(pid_list, subject):
        return sum(1 for pid in pid_list if subj_from_pid(pid) == subject)

    # verify for each subject
    for subj in train_subjects_real:
        val_count = count_subject_records(val_pids_subset, subj)
        test_count = count_subject_records(test_pids_subset, subj)
        
        if val_count > 0 or test_count > 0:
            print(f"LEAK: subject {subj} (assigned to train) has records in other sets")
            print(f"   - Record in validation: {val_count}")
            print(f"   - Record in test: {test_count}")
            leak_found = True

    for subj in val_subjects_real:
        train_count = count_subject_records(train_pids_subset, subj)
        test_count = count_subject_records(test_pids_subset, subj)
        
        if train_count > 0 or test_count > 0:
            print(f" LEAK subject {subj} (assigned in val) has records in other sets")
            print(f"   - Record in train: {train_count}")
            print(f"   - Record in test: {test_count}")
            leak_found = True

    # verify for each subjects in test
    for subj in test_subjects_real:
        train_count = count_subject_records(train_pids_subset, subj)
        val_count = count_subject_records(val_pids_subset, subj)
        
        if train_count > 0 or val_count > 0:
            print(f" LEAK subject {subj} (assigned in TEST) has records in other sets")
            print(f"   - Record in train: {train_count}")
            print(f"   - Record in validation: {val_count}")
            leak_found = True

    if leak_found:
        raise ValueError("DATA LEAKAGE")
    else:
        print(" NO LEAKAGE, ALL PATIENTS IN CORRECT SPLITS")

    # number of unique subjects
    total_unique_subjects = len(train_subjects_real | val_subjects_real | test_subjects_real)
    print(f"   - total unique subjects: {total_unique_subjects}")
    print(f"   - subjects in train: {len(train_subjects_real)}")
    print(f"   - subjects in val: {len(val_subjects_real)}")
    print(f"   - subjects in test: {len(test_subjects_real)}")
    print(f"   - sum: {len(train_subjects_real) + len(val_subjects_real) + len(test_subjects_real)}")

    assert total_unique_subjects == len(train_subjects_real) + len(val_subjects_real) + len(test_subjects_real), \
        "sum does not equal the total unique count"

    print("\n No leakage")
    print("="*60)
        
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # seed for reproducibility
    def seed_worker(worker_id):
        import random  
        import numpy as np  
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # generator with fixed seed
    g = torch.Generator()
    g.manual_seed(42)
        
    from torch.utils.data import WeightedRandomSampler

    print("\n creation balanced sampler")

    #label for each index in train-set
    labels = []
    for i in range(len(train_set)):
        real_idx = train_set.indices[i]
        pid = train_dataset.patient_ids[real_idx]
        parts = pid.split("_")
        subject_id = parts[0]
        session_day = parts[1].replace("ses-d", "")
        tracer = parts[2].strip().upper()
        
        #find in metadata
        row = train_dataset.metadata[
            (train_dataset.metadata["subject_id"] == subject_id) &
            (train_dataset.metadata["session_id_pet"].str.endswith(f"d{session_day}")) &
            (train_dataset.metadata["tracer"] == tracer)
        ]
        
        label = int(row["amyloid_positive"].values[0])
        labels.append(label)

    labels = np.array(labels, dtype=np.int64)

    class_sample_count = np.bincount(labels, minlength=2)
    neg_count = int(class_sample_count[0])
    pos_count = int(class_sample_count[1])
    print(f"Train set (real, pre-sampler): {neg_count} negatives, {pos_count} positives")

    weights_per_class = 1.0 / np.maximum(class_sample_count, 1)
    weights = torch.as_tensor(weights_per_class[labels], dtype=torch.double)

    # sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )

    print(" Pre-computing label mapping for efficiency...")
    pid_to_label = {}
    for i in range(len(train_set)):
        real_idx = train_set.indices[i]
        pid = train_dataset.patient_ids[real_idx]
        pid_to_label[pid] = int(labels[i]) 
    print(f" Mapping created for {len(pid_to_label)} patient IDs")

    print(" Pre-computing label mapping for validation...")
    val_pid_to_label = {}

    for idx in val_idx:
        pid = val_dataset.patient_ids[idx]

        parts = pid.split("_")
        subject_id = parts[0]
        session_day = parts[1].replace("ses-d", "")
        tracer = parts[2].strip().upper()
        
        row = val_dataset.metadata[
            (val_dataset.metadata["subject_id"] == subject_id) &
            (val_dataset.metadata["session_id_pet"].str.endswith(f"d{session_day}")) &
            (val_dataset.metadata["tracer"] == tracer)
        ]
        
        label = int(row["amyloid_positive"].values[0])
        val_pid_to_label[pid] = label
    print(f" Mapping validation created for {len(val_pid_to_label)} patient IDs")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True,
                        worker_init_fn=seed_worker,generator=g)  # No generator for shuffle=False
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker,generator=g)

    print(" Pre-computing label mapping for test...")
    test_pid_to_label = {}
    for idx in test_idx:
        pid = test_dataset.patient_ids[idx]
        parts = pid.split("_")
        subject_id = parts[0]
        session_day = parts[1].replace("ses-d", "")
        tracer = parts[2].strip().upper()
        
        row = test_dataset.metadata[
            (test_dataset.metadata["subject_id"] == subject_id) &
            (test_dataset.metadata["session_id_pet"].str.endswith(f"d{session_day}")) &
            (test_dataset.metadata["tracer"] == tracer)
        ]
        
        label = int(row["amyloid_positive"].values[0])
        test_pid_to_label[pid] = label
    print(f" Mapping test created for {len(test_pid_to_label)} patient IDs")

    print("\n CHECK PRE-TRAINING:")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"- GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3:.2f} GB")
        print(f"- GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    neg_count = class_sample_count[0]  
    pos_count = class_sample_count[1]
    print(f"  - Negatives: {neg_count} ({neg_count/(neg_count+pos_count)*100:.1f}%)")
    print(f"  - Positives: {pos_count} ({pos_count/(neg_count+pos_count)*100:.1f}%)")

    global_pos_weight = torch.tensor([1.0], device=device)
        
    def filter_metadata_by_indices(dataset, indices):
        selected_patient_ids = [dataset.patient_ids[i] for i in indices]
        selected_subject_ids = [subj_from_pid(pid) for pid in selected_patient_ids]
        
        filtered_metadata = dataset.metadata[dataset.metadata['subject_id'].isin(selected_subject_ids)]
        
        healthy = filtered_metadata[filtered_metadata["amyloid_positive"] == 0]
        demented = filtered_metadata[filtered_metadata["amyloid_positive"] == 1]
        
        return healthy, demented
    
    train_healthy, train_demented = filter_metadata_by_indices(train_dataset, train_idx)
    val_healthy, val_demented = filter_metadata_by_indices(val_dataset, val_idx) 
    
    triplet_loss_fn = SoftTripletLoss(margin=1.0)
    
    set_seed(42)
    
    model = MRITextPETContrastive()
    model = model.to(device)
    
    print(f"\n Total parameters in the model: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            print(f" {name} is not on {device}!")

    PRETRAIN_EPOCHS = 30

    if not SKIP_PHASE1:
        print(f"\n Phase 0: Pre-training teacher with only classification ({PRETRAIN_EPOCHS} epochs)")
        
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
        
        pretrain_optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3) 
        print(f" Optimizer configured with lr=2e-5")  
        best_pretrain_f1 = 0.0


        PRETRAIN_BEST_PATH = os.path.join(EXPERIMENT_DIR, "best_pretrain_teacher.pt")
        patience_pretrain = 5 
        no_improve_pretrain = 0
        print(f" Best pre-training model will be saved to: {PRETRAIN_BEST_PATH}")
        
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS}")):
                mri = batch["mri_slices"].to(device)
                pet = batch["pet_slices"].to(device)
                patient_ids = batch["patient_id"]
                
                labels = torch.tensor(
                    [pid_to_label[pid] for pid in batch["patient_id"]], 
                    device=device, dtype=torch.float32
                )
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
                        bad = (not torch.isfinite(param.grad).all().item())
                        if bad:
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
            print(f"   average loss: {running_loss/len(train_loader):.4f}")
            print(f"   Accuracy: {epoch_acc:.3f}")
            print(f"   F1 Score: {epoch_f1:.3f}")

            pretrain_history['epochs'].append(epoch + 1)
            pretrain_history['train_loss'].append(running_loss/len(train_loader))
            pretrain_history['train_acc'].append(epoch_acc)
            
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation pre-train", leave=False):
                    mri = batch["mri_slices"].to(device)
                    pet = batch["pet_slices"].to(device)
                    
                    labels = torch.tensor(
                        [val_pid_to_label[pid] for pid in batch["patient_id"]], 
                        device=device, dtype=torch.float32
                    )
                    
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
                print(f"    New best F1 in pre-training!")
                no_improve_pretrain = 0 
            else:
                no_improve_pretrain += 1
                print(f"    No improvements epochs without improvement: {no_improve_pretrain}/{patience_pretrain}")
                
                if no_improve_pretrain >= patience_pretrain:
                    print(f"    Early stopping pre-training after {epoch+1} epochs")
                    break
                    
        print(f"\n Pre-training completed! Best Validation F1: {best_pretrain_f1:.3f}")

        if os.path.exists(PRETRAIN_BEST_PATH):
            print(f"\n using best model from pre-training")
            model.load_state_dict(torch.load(PRETRAIN_BEST_PATH, map_location=device))
            print(f" model from checkpoint with F1={best_pretrain_f1:.3f}")
            
            model.eval()
            quick_check_f1 = 0  
            model.train()
        else:
            print(f" Checkpoint pre-training not found, last checkpoint")

        if batch_size != batch_size_phase1:
            print(f"\n batch size update for phase 1: {batch_size} → {batch_size_phase1}")
            batch_size = batch_size_phase1

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
                generator=g
            )
            val_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g
            )
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f" Loader (train/val/test) created with batch size {batch_size}")
    
    if not SKIP_PHASE1:
        print("\n phase 1 teacher training (pet guided) with triplet loss")
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
            print(f"\n phase 1 epoch: {epoch+1}/{PHASE1_EPOCHS} -------------------------")
            model.train()
            total_loss = 0
            batch_count = 0
            epoch_margins = []
            epoch_dist_pos = []
            epoch_dist_neg = []
            epoch_sim_pos = []
            epoch_sim_neg = []
            gc.collect()
            
            accumulation_steps = max(1, TARGET_EBS // max(1, batch_size))
 
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
                        print(f" [Batch {batch_idx}] MRI Embedding Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                        if std_val < 0.01:
                            print(f" MRI embedding almost constant!")
                                        
                    pet_pos_embeddings = model.encode_batch(pet_pos)
                    if pet_pos.shape[3] != 224 or pet_pos.shape[4] != 224:
                        pet_pos = F.interpolate(
                            pet_pos.view(-1, pet_pos.size(2), pet_pos.size(3), pet_pos.size(4)),
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        ).view(pet_pos.size(0), pet_pos.size(1), pet_pos.size(2), 224, 224)
                    
                    if batch_size_actual < 2:
                        print(" Batch size too small, skipping this batch")
                        continue

                    pet_neg_embeddings_list, valid_indices = get_negative_pet_embeddings(
                        batch, train_dataset, train_healthy, train_demented, model, device
                    )

                    if len(pet_neg_embeddings_list) == 0:
                        print(" no valid patients, skipping this batch")
                        continue

                    if len(valid_indices) < batch_size_actual:
                        print(f" not enough negative examples found {len(valid_indices)}, using only these indices.")
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
                    
                    labels = torch.tensor(
                        [pid_to_label[pid] for pid in filtered_patient_ids], 
                        device=device, dtype=torch.float32
                    )
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

                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all().item():
                            print(f" NaN/Inf in gratient of {name}! Max abs: {p.grad.detach().abs().max().item():.2e}")
                            p.grad.zero_()
         
                total_loss += loss_unscaled.item() 

                batch_count += 1
                
                if batch_idx % 100 == 0:
                    
                    print(f" Batch {batch_idx}: Total Loss = {loss_unscaled.item():.4f}")
                    
                if (not torch.isfinite(loss_scaled).item()) or loss_scaled.item() > 100:
                    print(f" Loss not valid detected: {loss_scaled.item()}")
                    optimizer_phase1.zero_grad()
                    continue
                    
                if batch_idx % 100 == 0:  
                    grad_stats = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_stats[name] = grad_norm
                    
                    top_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("\n Top 5 higher gradients")
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
                    
                    print(f"\n Gradient stats for component:")
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
                            print(f" Gradient stats global: max={max_grad:.6f}, min={min_grad:.6f}")
                            if max_grad > 10:
                                print(" WARNING:very high gradients")
                            if min_grad < 1e-6:
                                print(" WARNING:gradients almost zero")
                        else:
                            print(" No gradient found")
                            
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
                        print(f" Average Margin: {margin_achieved.mean().item():.4f}")
                        
            avg_loss = total_loss / max(1, batch_count)
            phase1_train_losses.append(avg_loss)

            current_epoch_data = {'epoch': epoch + 1, 'train_loss': avg_loss}
            
            valid_triplets_pct = 100 * sum(1 for m in epoch_margins if m > 0) / max(1, len(epoch_margins))
            avg_margin = sum(epoch_margins) / max(1, len(epoch_margins))
            
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Average Margin: {avg_margin:.4f}")
            print(f"Valid Triplets: {valid_triplets_pct:.1f}%")
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
                            batch, val_dataset, val_healthy, val_demented, model, device
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
                
                print(f"Average margin Validation: {avg_val_margin:.4f}")
                print(f"Average Distance Anchor-Positive: {avg_val_dist_pos:.4f}")
                print(f"Average Distance Anchor-Negative: {avg_val_dist_neg:.4f}")
                print(f"Cosine similarity Anchor-Positive: {avg_val_sim_pos:.4f}")
                print(f"Cosine similarity Anchor-Negative: {avg_val_sim_neg:.4f}")
                print(f"Valid triplets validation: {val_valid_triplets_pct:.1f}%")

                validation_score = val_valid_triplets_pct / 100.0 - 0.1 * avg_val_loss   

                print("\n Classification teacher validation")
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
                        
                        labels = torch.tensor(
                            [val_pid_to_label[pid] for pid in batch["patient_id"]], 
                            device=device, dtype=torch.float32
                        )
                        
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
                

                #save metrics for the graph
                phase1_history['epochs'].append(epoch + 1)
                phase1_history['train_loss'].append(avg_loss)
                phase1_history['val_margin'].append(avg_val_margin if 'avg_val_margin' in locals() else 0)
                phase1_history['val_f1'].append(teacher_f1)

                classification_score = teacher_f1
                combined_score = validation_score + 0.5 * classification_score
                
                if combined_score > best_val_score + 0.001:
                    best_val_score = combined_score
                    torch.save(model.state_dict(), TEACHER_PATH)
                    print(f" Saved new best teacher. score: {combined_score:.4f} "
                        f"(Triplet: {validation_score:.4f}, Class: {classification_score:.4f})")
                    no_improve_epochs = 0  
                else:
                    no_improve_epochs += 1 
                
                if epoch % 2 == 0:
                    print(f"\n Test embedding collapse:")
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
                                anchor_df = val_dataset.metadata[
                                    (val_dataset.metadata["subject_id"] == anchor_subject_id) &
                                    (val_dataset.metadata["session_id_pet"].str.endswith(f"d{anchor_session_day}")) &
                                    (val_dataset.metadata["tracer"] == anchor_tracer)
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
                                pet_dir = os.path.join(val_dataset.pet_root, negative_patient_id, "PET_slices")
                                if not os.path.isdir(pet_dir):
                                    continue
                                neg_pet_slices = val_dataset._load_slices(pet_dir)
                                if len(neg_pet_slices) == 0:
                                    continue
                                    
                                if val_dataset.num_slices_to_use is not None and val_dataset.num_slices_to_use < len(neg_pet_slices):
                                    total_slices = len(neg_pet_slices)
                                    indices = []
                                    if val_dataset.num_slices_to_use > 1:
                                        step = total_slices / val_dataset.num_slices_to_use
                                        indices = [min(total_slices-1, int(j * step)) for j in range(val_dataset.num_slices_to_use)]
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
                                print("   WARNING: Possible embedding collapse!")
                            
                            anchor_norm = F.normalize(all_anchors, p=2, dim=1)
                            similarity_matrix = torch.mm(anchor_norm, anchor_norm.t())
                            
                            mask = torch.eye(similarity_matrix.size(0), device=device).bool()
                            similarity_matrix.masked_fill_(mask, 0)
                            avg_similarity = similarity_matrix.sum() / (similarity_matrix.size(0) * (similarity_matrix.size(0) - 1))
                            
                            print(f"  average similarity between anchor: {avg_similarity:.4f}")
                            if avg_similarity > 0.9:
                                print("   warning: anchor too similar between each others")
                                
                            avg_pos_dist = F.pairwise_distance(all_anchors, all_positives, p=2).mean().item()
                            avg_neg_dist = F.pairwise_distance(all_anchors, all_negatives, p=2).mean().item()
                            print(f"  Average distance A-P: {avg_pos_dist:.4f}, A-N: {avg_neg_dist:.4f}")
                            
                            if avg_neg_dist < avg_pos_dist * 1.5:
                                print("  warning! negatives not enough separated")
            else: 
                print(f" No improvement in these epochs ")
                no_improve_epochs += 1 
                
            print(f" End epoch {epoch+1}: Epochs without improvement: {no_improve_epochs}/{early_stopping_patience}")
            
            if no_improve_epochs >= early_stopping_patience:
                print(f" Early stopping! Phase 1 stopped after {epoch+1} epochs.")
                if best_val_score > 0 : 
                    print(f" Best teacher model saved in: {TEACHER_PATH} with score: {best_val_score:.4f}")
                else:
                    print(f" No model teacher saved as best.")
                break  
            
            current_lr_phase1 = optimizer_phase1.param_groups[0]['lr']
            
            log_msg_score_details = f"LR: {current_lr_phase1:.2e}."
            if val_margins and 'combined_score' in locals() and 'validation_score' in locals():
                log_msg_score_details += f" Last Val Score (triplet): {validation_score:.4f}, Combined: {combined_score:.4f}"
            
            logger.info(f"Scheduler Phase 1 updated. {log_msg_score_details}")
            
            train_metrics = {'loss': avg_loss, 'margin': avg_margin} 
            if val_margins and 'avg_val_margin' in locals(): 
                val_metrics = {'loss': avg_val_loss, 'margin': avg_val_margin}
                overfitting_score = (train_metrics['margin'] - val_metrics['margin']) / max(train_metrics['margin'], 1e-6)
                print(f" overfitting index: {overfitting_score:.2%}")
            else:
                print(" not possible to calculate overfitting index")
                    
            if epoch == PHASE1_EPOCHS - 1:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epoch + 2), phase1_train_losses, label='Train Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('phase1 loss teacher')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.close()
                
                if phase1_val_margins:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(phase1_val_margins) + 1), phase1_val_margins, label='Validation Margin')
                    plt.xlabel('Epoch')
                    plt.ylabel('average margin')
                    plt.title('phase 1 average margin validation')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.close()
                torch.cuda.empty_cache()  
                gc.collect()  
        
    else:
        print(f"\n phase1 skipped, verifying if teacher exists in: {TEACHER_PATH}")
        if not os.path.exists(TEACHER_PATH):
            print(f" error: teacher model not found in {TEACHER_PATH}.")
            return 
            
        print(f" Loading the pre-trained teacher.")
        
        old_model_state_dict = torch.load(TEACHER_PATH, map_location=device)
        load_result = model.load_state_dict(old_model_state_dict, strict=False)
        print(f"Teacher weights loaded from: {TEACHER_PATH}")
        print(f"  teacher keys missing in checkpoint: {load_result.missing_keys}")
        print(f" unexpected keys in checkpoint: {load_result.unexpected_keys}")
        if not all(k.startswith("classifier_head") for k in load_result.missing_keys):
            print(f"attention: other missing keys {load_result.missing_keys}")
    

    #batch size update for phase2
    if batch_size != batch_size_phase2:
        print(f"\n updating batch size: {batch_size} → {batch_size_phase2}")
        batch_size = batch_size_phase2
        
        # new dataloader with new batch 
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
                    
    print("\n phase 2: distillation from teacher (PET-guided) to Student (MRI-only)")
    
    print("\n loading teacher for phase 2")
    teacher_temp = MRITextPETContrastive().to(device)
    if os.path.exists(TEACHER_PATH):
        print(f" loading teacher from: {TEACHER_PATH}")
        teacher_state = torch.load(TEACHER_PATH, map_location=device)

        teacher_load_result = teacher_temp.load_state_dict(teacher_state, strict=False)

        print(f" missing keys in teacher: {teacher_load_result.missing_keys}")

    else:
        print("error teacher not found!")
        raise FileNotFoundError(f"teacher model not found {TEACHER_PATH}")
    
    teacher = copy.deepcopy(teacher_temp)

    print("\n test compatibility architecture teacher student")
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
        
        # Student output (for comparison)
        weights_wrong = F.softmax(model.self_attn.pool(mri_enc_s), dim=1)
        student_out_wrong = (mri_enc_s * weights_wrong).sum(dim=1)
        
        print(f"Teacher output shape: {teacher_out.shape}")
        print(f"Student output shape: {student_out_correct.shape}")
        print(f"Student output shape: {student_out_wrong.shape}")
        print(f"\nDifference between correct and wrong method: {(student_out_correct - student_out_wrong).abs().mean():.4f}")
        
        if teacher_out.shape != student_out_correct.shape:
            raise ValueError("ERROR: teacher's and student's output shapes are different")
        else:
            print(" dimensional compatibility verified")
        del teacher_temp  
    
    print("\n test teacher's architecture")
    for i, module in enumerate(teacher.projection):
        print(f"  Layer {i}: {module}")

    teacher.eval()
    
    print("\n projection architecture teacher:")
    for i, module in enumerate(teacher.projection):
        print(f"  Layer {i}: {module}")
        
    teacher.eval() 
    teacher_all_logits = []
    teacher_all_probs = []
    teacher_all_labels_list = []
    teacher_all_patient_ids = []
    
    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()):
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Teacher validation correct")):
                mri_val = batch["mri_slices"].to(device)
                pet_val = batch["pet_slices"].to(device) 
                patient_ids = batch["patient_id"]
                teacher_all_patient_ids.extend(patient_ids)
                labels = get_labels_from_patient_ids(patient_ids, val_dataset.metadata)
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
    
    print(f" Optimal threshold calculation on the validation set")
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
    print(f"\nResults validation teacher (threshold {teacher_metrics_opt['threshold']:.3f}):")
    print(f"  F1 Score: {teacher_metrics_opt['f1']:.3f}")
    print(f"  Precision: {teacher_metrics_opt['precision']:.3f}")
    print(f"  Recall: {teacher_metrics_opt['recall']:.3f}")
    
    print(f"\n Teacher F1 Score: {teacher_results['metrics_opt']['f1']:.3f}")
    print(f" Teacher Precision: {teacher_results['metrics_opt']['precision']:.3f}")
    print(f" Teacher Recall: {teacher_results['metrics_opt']['recall']:.3f}")
    
    if teacher_results['metrics_opt']['f1'] < 0.5:
        print("\n Error teacher has F1 < 0.5!")
        return  
          
    teacher.eval() 
    
    for param in teacher.parameters():
        param.requires_grad = False
        
    gc.collect()
    print(" Student with the same projection layer of teacher")
    
    print("\n dropout reduction to improve distillation")
    # dropout modification in projection
    model.projection[3] = nn.Dropout(0.3) 
    model.projection[6] = nn.Dropout(0.2) 

    # dropout modification in classifier
    model.classifier_head[2] = nn.Dropout(0.4) 

    print(" Dropout reduced for phase2:")
    print("   - Projection layer 3: 0.5 → 0.3")
    print("   - Projection layer 6: 0.4 → 0.2")
    print("   - Classifier head: 0.6 → 0.4")    
        
    print("\n verification of architectures")
    print("="*60)
    
    for component_name in ['projection', 'classifier_head', 'self_attn', 'cross_attn']:
        if hasattr(teacher, component_name) and hasattr(model, component_name):
            teacher_comp = getattr(teacher, component_name)
            student_comp = getattr(model, component_name)
            
            teacher_params = sum(p.numel() for p in teacher_comp.parameters())
            student_params = sum(p.numel() for p in student_comp.parameters())
            
            if teacher_params != student_params:
                print(f" Error: {component_name} - Teacher: {teacher_params} params, Student: {student_params} params")
                raise ValueError(f"architectures not compatible in {component_name}!")
            else:
                print(f" {component_name}: {teacher_params} parameters (OK)")
    print("="*60)
    
    print("\n Test forward pass...")
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
    print("\n Architecture verification completed successfully")
    print("="*60 + "\n")
        
    for p in model.projection.parameters():
        p.requires_grad = True
            
    print(" Configuration optimizer for student")

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
    convergence_monitor = {
        'epoch_f2': [],  
        'separation': [],
        'overlap_pct': [],
        'f1_score': []
    }

    # Phase 2 - Initial configuration
    import random
    random.seed(42)
    fixed_val_indices = random.sample(range(len(val_set)), min(6, len(val_set)))
    print(f" Fixed indices selected for visualization: {fixed_val_indices}")

    # pos_weight=1.0 because sampler already balanced
    effective_pos_weight_f2 = torch.tensor([1.0], device=device)
    print(f" phase2 using pos_weight=1.0")

    print("\n Initialization bias for unbalanced dataset")
    with torch.no_grad():
        neg_ratio = neg_count / (neg_count + pos_count)  # neg_count
        pos_ratio = pos_count / (neg_count + pos_count)  # pos_count
        
        initial_bias = 0.0 
        nn.init.xavier_uniform_(model.classifier_head[-1].weight, gain=0.5)
        model.classifier_head[-1].bias.data.fill_(initial_bias)
        
        print(f" Initialization classifier: bias={initial_bias:.3f}, weight gain=0.5")
        print(f"   Real dataset: {neg_ratio:.0%} negatives, {pos_ratio:.0%} positives")
        print(f"   Sampler produce batch ~50/50, pos_weight=1.0")

        model.classifier_head[-1].bias.data.fill_(initial_bias)
        print(f" Bias classifier initialized at {initial_bias:.3f} (dataset: {neg_ratio:.0%} negatives, {pos_ratio:.0%} positives)")

    for epoch_f2 in range(PHASE2_EPOCHS): 
        print(f"\n phase 2 - Epoch {epoch_f2+1}/{PHASE2_EPOCHS} -------------------------")
        model.train()
        teacher.eval()
        
        if epoch_f2 < 5:
            temperature = 2.5  
        elif epoch_f2 < 20:
            temperature = 2.5 - (epoch_f2 - 5) * 0.1  # reduces gradually to 1.0
        else:
            temperature = 1.0  # final target 1.0

        if epoch_f2 % 5 == 0: 
            print(f" Distillation temperature: {temperature:.1f}")
        
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

        accumulation_steps = max(1, TARGET_EBS // max(1, batch_size))

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Distillation Epoch {epoch_f2+1}")):
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
                labels = torch.tensor(
                    [pid_to_label[pid] for pid in batch["patient_id"]], 
                    device=device, dtype=torch.float32
                )
                student_emb_norm = F.normalize(student_emb, p=2, dim=1) 
                teacher_emb_target_norm = F.normalize(teacher_emb_target_for_distillation.detach(), p=2, dim=1) 
                feature_distill_loss = F.mse_loss(student_emb_norm, teacher_emb_target_norm)

                # class_loss calculation
                class_loss = current_epoch_classification_loss_fn(student_logits, labels)

                # Loss breakdown
                if batch_idx % 50 == 0 and epoch_f2 > 0:
                    print(f" Loss (Batch {batch_idx}):")
                    print(f"   class_loss: {class_loss.item():.4f}")
                    if class_loss.item() < 0:
                        print(f"    negative loss detected!")

                w_clf = 0.5  # more weight on classification
                w_feature_distill = 0.3
                w_logit_distill = 0.2  # less weight on distillation

                # more weight on distillation
                if epoch_f2 < 10:  # Warmup
                    warmup_progress = epoch_f2 / 10.0
                    w_clf = 0.3 + 0.1 * warmup_progress  
                    w_feature_distill = 0.5 - 0.1 * warmup_progress  # more feature matching
                    w_logit_distill = 0.2 
                    if batch_idx == 0:
                        print(f" warmup distillation (epoch {epoch_f2+1}/10): "
                            f"clf={w_clf:.2f}, feat={w_feature_distill:.2f}, "
                            f"logit={w_logit_distill:.2f}")
                else:
                    # fixed weight after warmup
                    w_clf = 0.4 
                    w_feature_distill = 0.4 
                    w_logit_distill = 0.2
                    if batch_idx == 0:
                        print(f" stable weights: clf={w_clf}, feat={w_feature_distill}, logit={w_logit_distill}")
                loss_unscaled_tensor = (w_clf * class_loss) + (w_feature_distill * feature_distill_loss)

                if w_logit_distill > 0:
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

                if (not torch.isfinite(loss_for_backward).item()) or (loss_for_backward.item() < 0):
                     print(f" Loss NaN/Inf/negative detected: skip batch")
                     optimizer.zero_grad()
                     continue
                                    
                # in case of too high losses
                if loss_for_backward.item() < -10:
                    print(f" Loss still negative: {loss_for_backward.item():.4f} - unexpected error!")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss_for_backward).backward()

                # check Nan/inf
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all().item():
                            print(f" NaN/Inf in the gradient of {name}! Max abs: {p.grad.detach().abs().max().item():.2e}")
                            p.grad.zero_()

                if batch_idx % 100 == 0:
                    print(f"\n Loss Components (Epoch {epoch_f2+1}, Batch {batch_idx}):")
                    print(f"   loss divided by {accumulation_steps} (accumulation steps)")
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
                        print(f" positives over neg median: {pos_above_median}/{total_pos} ({pos_above_median/total_pos*100:.1f}%)")
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
                    
                    labels = torch.tensor(
                        [pid_to_label[pid] for pid in patient_ids], 
                        device=device, dtype=torch.float32
                    )
                                                            
                    mri_emb = model.encode_batch(mri)
                    student_emb = model.self_attn(query=mri_emb, key=mri_emb)

                    # new layers as in the principal training
                    logits = model.classifier_head(student_emb).squeeze(1)

                    if i % 10 == 0:  
                        print(f"\n=== Diagnostic training embeddings (batch {i}) ===")
                        print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        
                        probs = torch.sigmoid(logits)
                        print(f"probability: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
                        
                        norms = torch.norm(student_emb, dim=1)
                        print(f"embeddings norm: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
                                        
                    train_logits.append(logits)
                    train_labels_tensor.append(labels.cpu()) 

        if len(train_logits) > 0:
            try:
                all_train_logits = torch.cat(train_logits)
                all_train_labels = torch.cat(train_labels_tensor)

                print("\n=== analysis logits distribution on training set ===")
                analyze_classifier_outputs(
                    all_train_logits, all_train_labels, epoch_f2,
                    os.path.join(EXPERIMENT_DIR, "train_analysis")
                )
            except Exception as e:
                print(f"Error in training logits analysis: {e}")

        print(f"\n=== distillation statistics epoch{epoch_f2+1} ===")
        print(f"total average loss: {avg_loss:.4f}")
        print(f" - distillation: {avg_distill:.4f}")
        print(f" - classification: {avg_class:.4f}")

        phase2_losses.append(avg_loss)
        
        print(f"\n=== Distillation statistics epoch {epoch_f2+1} ===")
        print(f"average loss: {avg_loss:.4f}")

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
                        print(f"\n=== embeddings diagnostic (batch {batch_idx}) ===")
                        print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        
                        probs = torch.sigmoid(logits)
                        print(f"probability: min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
                        
                        norms = torch.norm(student_emb_val, dim=1)
                        print(f"embeddings norm: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
                            
                        labels_tensor = torch.tensor(get_labels_from_patient_ids(patient_ids, val_dataset.metadata), device=device).float()
                        pos_mask = labels_tensor == 1
                        neg_mask = labels_tensor == 0
                        
                        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                            pos_mean = logits[pos_mask].mean().item()
                            neg_mean = logits[neg_mask].mean().item()
                            separation = pos_mean - neg_mean
                            print(f" diagnostic separation: pos={pos_mean:.4f}, neg={neg_mean:.4f}, sep={separation:.4f}")
                            
                            pos_min = logits[pos_mask].min().item() if pos_mask.sum() > 0 else float('inf')
                            neg_max = logits[neg_mask].max().item() if neg_mask.sum() > 0 else float('-inf')
                            if pos_min < neg_max:
                                print(f" OVERLAP: pos_min={pos_min:.4f} < neg_max={neg_max:.4f}")
           
                    all_val_logits.append(logits.detach().cpu())
                                        
                    probs = torch.sigmoid(logits).cpu().numpy()

                    labels = [val_pid_to_label[pid] for pid in patient_ids]
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
                print(f"   - Epochs: {epoch_f2 + 1}")
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
                        print(f" very low threshold limited to 0.05")
                        optimal_prob_thresh = 0.05
                    
                    preds_opt = (np.array(val_probs) > optimal_prob_thresh).astype(int)
                    acc_opt_new = accuracy_score(val_targets, preds_opt)
                    prec_opt_new = precision_score(val_targets, preds_opt, zero_division=0)
                    rec_opt_new = recall_score(val_targets, preds_opt, zero_division=0)
                    f1_opt_new = f1_score(val_targets, preds_opt, zero_division=0)
                    
                    print(f"\n=== validation with analytical threshold = {optimal_prob_thresh:.3f} ===")
                    print(f"  Accuracy:  {acc_opt_new:.3f}")
                    print(f"  Precision: {prec_opt_new:.3f}")
                    print(f"  Recall:    {rec_opt_new:.3f}")
                    print(f"  F1 score:  {f1_opt_new:.3f}")
                
                print("\n=== distribution logits analysis ===")
                analyze_classifier_outputs(all_val_logits_tensor, all_val_labels_tensor, epoch_f2, EXPERIMENT_DIR)
            except Exception as e:
                print(f"error in logits analysis: {e}")

        print(f"\n Embedding similarity check:")
        print(f"   - len(val_similarities): {len(val_similarities)}")
        print(f"   - type(val_similarities): {type(val_similarities)}")
        if len(val_similarities) > 0:
            print(f"   - first element: {val_similarities[0]}")
            print(f"   - first type element: {type(val_similarities[0])}")
        print(f"   - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        if val_similarities:
            avg_val_similarity = sum(val_similarities) / len(val_similarities)
            print(f"\n=== Student statistics validation epoch {epoch_f2+1} ===")
            print(f"average similarity validation: {avg_val_similarity:.4f}")

            current_phase2_data['val_sim'] = avg_val_similarity
                          
            if avg_val_similarity > best_student_similarity:
                    best_student_similarity = avg_val_similarity
                    best_similarity_state = copy.deepcopy(model.state_dict())
                    torch.save(best_similarity_state, BEST_STUDENT_PATH)
                    print(f" new similarity student saved! Sim: {best_student_similarity:.4f}")
                    
                    eval_results_sim = evaluate_model_with_diagnostics(
                        model, val_loader, val_dataset.metadata, 
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
                    print(f" Similarity threshold saved successfully: {eval_results_sim['metrics_opt']['threshold']:.3f}")

        preds = (np.array(val_probs) > 0.5).astype(int)

        acc = accuracy_score(val_targets, preds)
        prec = precision_score(val_targets, preds, zero_division=0)
        rec = recall_score(val_targets, preds, zero_division=0)
        f1 = f1_score(val_targets, preds, zero_division=0)

        print(f"\n=== validation phase 2 classification ===")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 score:  {f1:.3f}")

        if prec > 0 and rec > 0:
            prec_recall_ratio = prec / rec
            balanced_score = f1 * (1 - abs(0.5 - prec_recall_ratio) * 0.5)
            print(f"  precision/recall: {prec_recall_ratio:.3f}")
            print(f"  Score balanced: {balanced_score:.3f}")

        best_thresh, best_f1 = find_best_threshold(val_probs, val_targets)
        preds_opt = (np.array(val_probs) > best_thresh).astype(int)

        acc_opt = accuracy_score(val_targets, preds_opt)
        prec_opt = precision_score(val_targets, preds_opt, zero_division=0)
        rec_opt = recall_score(val_targets, preds_opt, zero_division=0)

        print("\n model validation")
        eval_results = evaluate_model_with_diagnostics(
            model, val_loader, val_dataset.metadata, 
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
                print("\n CONVERGENCE SUMMARY:")
                print(f"  Separation trend: {convergence_monitor['separation'][-10:]}")
                print(f"  Overlap trend: {convergence_monitor['overlap_pct'][-10:]}")
                print(f"  F1 trend: {convergence_monitor['f1_score'][-10:]}")

        print(f"\n=== Validation phase 2- classification (fixed threshold 0.5) ===")
        print(f"Accuracy:  {metrics_fixed['accuracy']:.3f}")
        print(f"Precision: {metrics_fixed['precision']:.3f}")
        print(f"Recall:    {metrics_fixed['recall']:.3f}")
        print(f"F1 score:  {metrics_fixed['f1']:.3f}")

        if metrics_fixed['precision'] > 0 and metrics_fixed['recall'] > 0:
            prec_recall_ratio = metrics_fixed['precision'] / metrics_fixed['recall']
            balanced_score = metrics_fixed['f1'] * (1 - abs(0.5 - prec_recall_ratio) * 0.5)
            print(f"   precision/recall: {prec_recall_ratio:.3f}")
            print(f"  Score balanced: {balanced_score:.3f}")

        print(f"\n=== Validation (OPTIMAL THRESHOLD {metrics_opt['threshold']:.3f}) ===")
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
            print(f"  precision/recall: {prec_recall_ratio:.3f}")
            print(f"  balanced score: {balanced_score:.3f}")

        clf_scheduler.step(metrics_opt['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate updated: {current_lr:.2e}")
                        
        if metrics_opt['f1'] > best_f1_score:
            best_f1_score = metrics_opt['f1']
            
            best_f1_state = copy.deepcopy(model.state_dict())
            torch.save(best_f1_state, best_f1_path)

            eval_metrics = create_evaluation_plots(
                y_true=np.array(val_targets),
                y_probs=np.array(val_probs),
                save_dir=EXPERIMENT_DIR,
                prefix=f"val_epoch_{epoch_f2+1}_",
                threshold=metrics_opt['threshold']
            )
            print(f" AUC Score: {eval_metrics['auc']:.3f}")
            print(f" Sensitivity: {eval_metrics['sensitivity']:.3f}, Specificity: {eval_metrics['specificity']:.3f}")
            
            current_metrics_for_f1_model = {
                "threshold": metrics_opt['threshold'],
                "precision": metrics_opt['precision'],
                "recall":    metrics_opt['recall'],
                "f1":        metrics_opt['f1'],
                "accuracy":  metrics_opt['accuracy'] 
            }
            
            try:
                with open(os.path.join(EXPERIMENT_DIR, "best_threshold_f1.pkl"), "wb") as f:
                    pickle.dump(current_metrics_for_f1_model, f)

                with open(os.path.join(EXPERIMENT_DIR, "best_threshold_f1.txt"), "w") as f:
                    f.write(f"{metrics_opt['threshold']}\t"
                            f"{metrics_opt['precision']:.4f}\t"
                            f"{metrics_opt['recall']:.4f}\t"
                            f"{metrics_opt['f1']:.4f}")
                print(f" threshold (F1) saved successfully: {metrics_opt['threshold']:.3f}")
            except Exception as e:
                print(f" error in saving the threshold (F1): {e}")
                print(f"  F1 threshold not saved: {metrics_opt['threshold']:.3f}")
            
            print(f" saved new best student! F1: {metrics_opt['f1']:.4f}, P: {metrics_opt['precision']:.3f}, R: {metrics_opt['recall']:.3f}")
            phase2_no_improve = 0
        else:
            phase2_no_improve += 1
            print(f" no F1 improvement epochs without improvements: {phase2_no_improve}/{phase2_patience}")

            if phase2_no_improve >= phase2_patience:
                print(f" Early stopping activated!")
                break
        should_visualize = False

        if metrics_opt['f1'] > best_f1_score - 0.001: 
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
            logger.info(f" interpretability maps generated for epoch {epoch_f2+1}...")
            visualize_student_reasoning(
                student_model=model,
                data_loader=val_loader,
                epoch=epoch_f2,
                EXPERIMENT_DIR=EXPERIMENT_DIR, 
                prefix="val_",
                fixed_indices=fixed_val_indices 
            )
            logger.info(f" Maps saved in: {os.path.join(EXPERIMENT_DIR, 'saliency', f'val_epoch_{epoch_f2+1}')}")

    torch.save(model.state_dict(), FINAL_STUDENT_PATH)
    print(f" phase 2 completed, final model saved to: {FINAL_STUDENT_PATH}")
        

    print("\n final models evaluation saved during training")


    if os.path.isfile(BEST_STUDENT_PATH):
        model_sim = MRITextPETContrastive().to(device) 
        
        print(f"     loading weights from {BEST_STUDENT_PATH}")
        model_sim.load_state_dict(torch.load(BEST_STUDENT_PATH)) 

        model_f1 = MRITextPETContrastive().to(device)
        
        print(f"     loading weights from {best_f1_path}...")
        try:
            model_f1.load_state_dict(torch.load(best_f1_path), strict=True)
            print("     weights loaded correctly for model_f1.")
        except RuntimeError as e:
            print(f"     error in loading: {e}")
            
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
        
        print(f"Difference between models: {diff_count:.1f} over {total_params} parameters")
        print(f"Average difference per parameter: {diff_count/max(1,total_params):.6f}")
        print(f"Max difference: {max_diff:.6f}")
        
        if diff_count < 0.1:
            print(" attention: models seem to be identical")
    else:
        print(f" File {BEST_STUDENT_PATH} not found, comparison between model skipped.")

    print(f"\n comparison between models in the test set:")
    for model_path, model_name in [(BEST_STUDENT_PATH, "Model optimized for similarity"), 
                                (best_f1_path, "Model optimized for F1 score")]:
        
        print(f"\n=== evaluation of {model_name} ===")
        
        if not os.path.isfile(model_path):
            print(f" model file not found: {model_path}")
            print(f"Skipping validation of this model.")
            continue
        
        # new instance for model testing
        test_model = MRITextPETContrastive().to(device)
        test_model.load_state_dict(torch.load(model_path))
        test_model.eval()  
        
        print(f"Model eval mode: training={test_model.training}") 
        print(f"Dropout active: {any(m.training for m in test_model.modules() if isinstance(m, nn.Dropout))}")   

        if "similarity" in model_name:
            threshold_filename = "best_threshold_similarity.txt"
        else:
            threshold_filename = "best_threshold_f1.txt"
                
        threshold_pkl_filename = threshold_filename.replace('.txt', '.pkl')
        threshold_pkl_path = os.path.join(EXPERIMENT_DIR, threshold_pkl_filename)
        threshold_txt_path = os.path.join(EXPERIMENT_DIR, threshold_filename)
        threshold_loaded = False
        
        if os.path.exists(threshold_pkl_path):
            with open(threshold_pkl_path, "rb") as f:
                threshold_data = pickle.load(f)
                best_thresh_from_validation = threshold_data['threshold']
                print(f" loaded threshold for {model_name}: {best_thresh_from_validation:.3f}")
                print(f"   (F1 validation: {threshold_data.get('f1', 'N/A')})")
                threshold_loaded = True
                
        elif os.path.exists(threshold_txt_path):
            with open(threshold_txt_path, "r") as f:
                content = f.read().strip()
                if '\t' in content:
                    best_thresh_from_validation = float(content.split('\t')[0])
                else:
                    best_thresh_from_validation = float(content)
            print(f" loaded threshold from txt for {model_name}: {best_thresh_from_validation:.3f}")
            threshold_loaded = True
            
        if not threshold_loaded:
            print(f"  threshold not found, recalculate from validation set")
            print(f" optimal threshold calculation for {model_name}")
            test_model.eval()  #  use test_model, not model!
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    mri = batch["mri_slices"].to(device)
                    patient_ids = batch["patient_id"]
                    
                    mri_emb = test_model.encode_batch(mri)
                    student_emb = test_model.self_attn(query=mri_emb, key=mri_emb)
                    logits = test_model.classifier_head(student_emb).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    labels = get_labels_from_patient_ids(patient_ids, val_dataset.metadata)
                    val_probs.extend(probs)
                    val_labels.extend(labels)
            
            best_thresh_from_validation, _ = find_best_threshold(val_probs, val_labels) 
            print(f" threshold recalculated: {best_thresh_from_validation:.3f}")              
        
        eval_results = evaluate_model_with_diagnostics(
            test_model, test_loader, test_dataset.metadata, 
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
        
        print(f"\n  with optimal threshold ({metrics_opt['threshold']:.3f}):")
        print(f"  Accuracy:  {metrics_opt['accuracy']:.3f}")
        print(f"  Precision: {metrics_opt['precision']:.3f}")
        print(f"  Recall:    {metrics_opt['recall']:.3f}")
        print(f"  F1 Score:  {metrics_opt['f1']:.3f}")
        
        csv_path = os.path.join(EXPERIMENT_DIR, f"test_predictions_{model_name.replace(' ', '_')}.csv")
        save_predictions_to_csv(eval_results['probs'], eval_results['labels'], 
                            eval_results['patient_ids'], metrics_opt['threshold'], csv_path)

        test_eval_metrics = create_evaluation_plots(
            y_true=eval_results['labels'],
            y_probs=eval_results['probs'],
            save_dir=EXPERIMENT_DIR,
            prefix=f"test_{model_name.replace(' ', '_').lower()}_",
            threshold=metrics_opt['threshold']
        )
        print(f" Test AUC: {test_eval_metrics['auc']:.3f}")
        print(f" Test PPV: {test_eval_metrics['ppv']:.3f}, NPV: {test_eval_metrics['npv']:.3f}")
        
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

                    mri_emb_teacher = teacher.encode_batch(mri_test) 
                    pet_emb_teacher = teacher.encode_batch(pet_test)  
                    teacher_emb = teacher.cross_attn(query=pet_emb_teacher, key=mri_emb_teacher)
                    teacher_emb_norm = normalize(teacher_emb, p=2, dim=1)
                    batch_sim = F.cosine_similarity(student_emb_norm, teacher_emb_norm).cpu().numpy() 
                    sim_for_this_model.extend(batch_sim)
        
        avg_sim = sum(sim_for_this_model) / max(1, len(sim_for_this_model))
        print(f"\n average similarity on test set for {model_name}: {avg_sim:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(sim_for_this_model, bins=20, alpha=0.7)
        plt.axvline(x=avg_sim, color='r', linestyle='--', 
                    label=f'Average: {avg_sim:.4f}')
        plt.xlabel('Cosine similarity')
        plt.ylabel('Counts')
        plt.title(f'Similarity Student-Teacher ({model_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
      
        plt.close()

    # test cerebellum 
    print("\n" + "="*60)
    print(" Test dependence on Cerebellum")
    print("="*60)

    # Test on best F1 model
    if os.path.exists(best_f1_path):
        test_model_cerv = MRITextPETContrastive().to(device)
        test_model_cerv.load_state_dict(torch.load(best_f1_path))
        test_model_cerv.eval()
        
        # Test on validations set
        print("\nTest on VALIDATION set:")
        val_dependency = test_cerebellum_dependency(test_model_cerv, val_loader, num_samples=20)
        
        # test on test set
        print("\nTest on TEST set:")
        test_dependency = test_cerebellum_dependency(test_model_cerv, test_loader, num_samples=20)
        
        # save results
        with open(os.path.join(EXPERIMENT_DIR, "cerebellum_dependency_analysis.txt"), "w") as f:
            f.write(f"Model: best_f1_model\n")
            f.write(f"Validation dependency: {val_dependency:.4f}\n")
            f.write(f"Test dependency: {test_dependency:.4f}\n")
            f.write(f"Threshold for concern: > 0.15\n")
    else:
        print(" f1 best model not found for test")

    print("="*60)
        
    print(f"\n final analysis completed! All results are saved in: {EXPERIMENT_DIR}")

    if os.path.isfile(best_f1_path):
        print(f" loading model with best score from: {best_f1_path} for embeddings extraction.")
        model.load_state_dict(torch.load(best_f1_path, map_location=device))
    else:
        print(f" file best f1 model not found ({best_f1_path}) using last model for extraction.")
        
    
    print(f"the results of the experiment are available in: {EXPERIMENT_DIR}")

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
        
        line2 = ax1_twin.plot(pretrain_hist['epochs'], pretrain_hist['train_acc'], 
                             'g--', label='Train Acc', linewidth=2)
        line3 = ax1_twin.plot(pretrain_hist['epochs'], pretrain_hist['val_f1'], 
                             'r-', label='Val F1', linewidth=2)
        ax1_twin.set_ylabel('Accuracy/F1', color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        ax1_twin.set_ylim(0, 1)
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
    ax1.set_title('Pre-training (only classification)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    

    ax2 = axes[1]
    if phase1_hist['epochs']:
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(phase1_hist['epochs'], phase1_hist['train_loss'], 
                        'b-', label='Train Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        line2 = ax2_twin.plot(phase1_hist['epochs'], phase1_hist['val_margin'], 
                             'g--', label='Val Margin', linewidth=2)
        line3 = ax2_twin.plot(phase1_hist['epochs'], phase1_hist['val_f1'], 
                             'r-', label='Val F1', linewidth=2)
        ax2_twin.set_ylabel('Margin/F1', color='g')
        ax2_twin.tick_params(axis='y', labelcolor='g')
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')
        
    ax2.set_title('Phase 1: Teacher Training (Triplet)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
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
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')
        
    ax3.set_title('Phase 2: Student Distillation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save graph
    plot_path = os.path.join(save_dir, 'training_history_3phases.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n Training graphs saved in: {plot_path}")
    
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'pretrain': pretrain_hist,
            'phase1': phase1_hist,
            'phase2': phase2_hist
        }, f, indent=2)
    print(f" numerical data saved in: {history_path}")

if __name__ == "__main__":
    main()