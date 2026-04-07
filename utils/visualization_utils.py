# Author: Francesco Chiumento
# License: MIT

"""
Interpretability Visualization Utilities

Generates saliency maps and Grad-CAM/HiResCAM visualizations for model
predictions. Supports side-by-side comparison of MRI input, PET target,
saliency, and attention overlays.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import logging

# Import PYTORCH-GRAD-CAM library
from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

logger = logging.getLogger(__name__)

# compute_saliency_map as original
def compute_saliency_map(model, input_image):
    was_training = model.training
    model.eval()             
    torch.set_grad_enabled(True)

    device = next(model.parameters()).device
    x = input_image.unsqueeze(0).unsqueeze(1).to(device).float()
    x = x.detach().requires_grad_(True)

    # freeze weights: grad pass only through input
    req = []
    for p in model.parameters():
        req.append(p.requires_grad)
        p.requires_grad_(False)

    mri_emb = model.encode_batch(x, require_grad=True) 
    student_emb = model.self_attn(query=mri_emb, key=mri_emb)
    score = model.classifier_head(student_emb)
    score.mean().backward()

    g = x.grad.detach().abs().max(dim=2).values.squeeze(0).squeeze(0)
    saliency_np = g.cpu().numpy()

    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    model.train(was_training)
    return saliency_np

def compute_gradcam(model, input_image):
    """Grad-CAM/HiResCAM for ViT (with fallback)"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    input_4d = input_image.unsqueeze(0).to(device).float()

    class ModelWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
        def forward(self, x):
            x_5d = x.unsqueeze(1)  # [B,1,3,H,W]
            B, S, C, H, W = x_5d.shape
            x_flat = x_5d.view(B*S, C, H, W)
            with torch.cuda.amp.autocast(enabled=False):
                out = self.model.model.vision_model(pixel_values=x_flat)
                tokens = out.last_hidden_state          # [B*S, 1+N, D]
                emb = self.model.projection(tokens[:, 0, :])
                emb = emb.view(B, S, -1)
                student_emb = self.model.self_attn(query=emb, key=emb)
                score = self.model.classifier_head(student_emb)   # [B,1]
            return score

    wrapped = ModelWrapper(model).to(device).eval()

    # most informative layer of ViT
    try_layers = [
        "self_attn",      
        "mlp.fc2",     
        "layer_norm1",    
        "layer_norm2",
    ]

    def get_target_layer(layer_name):
        enc_last = wrapped.model.model.vision_model.encoder.layers[-1]
        if layer_name == "self_attn":
            return enc_last.self_attn
        if layer_name == "mlp.fc2":
            return enc_last.mlp.fc2
        if layer_name == "layer_norm1":
            return enc_last.layer_norm1
        if layer_name == "layer_norm2":
            return enc_last.layer_norm2
        return None

    cam_np = None
    last_err = None

    for lname in try_layers:
        target_layer = get_target_layer(lname)
        if target_layer is None:
            continue
        try:
            with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                # 2) explicit target for binary
                targets = [ClassifierOutputTarget(0)]

                # 3) no smoothing
                cam = HiResCAM(
                    model=wrapped,
                    target_layers=[target_layer],
                    reshape_transform=vit_reshape_transform
                )
                grayscale_cam = cam(input_tensor=input_4d, targets=targets)
                cam_np = grayscale_cam[0]
            # if all zero try next layer
            if np.max(cam_np) <= 1e-12:
                last_err = f"Layer {lname} has returned CAM ~0"
                cam_np = None
                continue
            break
        except Exception as e:
            last_err = f"{lname} -> {e}"
            cam_np = None
            continue

    # Fallback: try classic GradCAM
    if cam_np is None:
        try:
            with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                target_layer = get_target_layer("self_attn") or get_target_layer("mlp.fc2")
                if target_layer is None:
                    raise RuntimeError("No valid target layer found for GradCAM fallback")
                cam = GradCAM(
                    model=wrapped,
                    target_layers=[target_layer],
                    reshape_transform=vit_reshape_transform
                )
                grayscale_cam = cam(input_tensor=input_4d, targets=[ClassifierOutputTarget(0)])
                cam_np = grayscale_cam[0]
        except Exception as e:
            logger.error(f"GradCAM fallback failed: {e}; last error HiResCAM: {last_err}")
            cam_np = np.zeros((input_image.shape[1], input_image.shape[2]))

    model.train(was_training)
    return cam_np

# overlay_saliency_on_image as original
def overlay_saliency_on_image(image_tensor, saliency_map, output_path=None, title="Saliency", gamma=0.5):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    if "Saliency" in title and saliency_map.max() > 0:
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        saliency_map = np.power(saliency_map, gamma)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def compare_visualization_methods(image_tensor, saliency_map, cam_map, 
                                  output_path=None, cam_label="HiResCAM",
                                  pet_tensor=None):
    
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
    
    if pet_tensor is not None:
        pet_np = pet_tensor.permute(1, 2, 0).cpu().numpy()
        pet_np = (pet_np - pet_np.min()) / (pet_np.max() - pet_np.min() + 1e-8)
    else:
        pet_np = image_np  # Fallback to MRI if PET not available

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # changed 4 subplot instead of 3
    
    # Subplot 1: original MRI
    axes[0].imshow(image_np)
    axes[0].set_title("MRI (Input)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Subplot 2: corresponding PET
    axes[1].imshow(pet_np)
    axes[1].set_title("PET (Target)", fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Subplot 3: Saliency on MRI
    axes[2].imshow(image_np)
    axes[2].imshow(saliency_map, cmap='jet', alpha=0.5)
    axes[2].set_title("Saliency Map", fontsize=14)
    axes[2].axis('off')
    
    # Subplot 4: GradCAM on MRI
    axes[3].imshow(image_np)
    axes[3].imshow(cam_map, cmap='jet', alpha=0.5)
    axes[3].set_title(cam_label, fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_student_reasoning(student_model, data_loader, epoch, EXPERIMENT_DIR, prefix="", fixed_indices=None):
    was_training_outer = student_model.training

    student_model.eval()
    
    saliency_dir = os.path.join(EXPERIMENT_DIR, "saliency", f"{prefix}epoch_{epoch+1}")
    os.makedirs(saliency_dir, exist_ok=True)
    
    dataset = data_loader.dataset

    if fixed_indices is None:
        print("No fixed index for visualization, choosing random patient.")
        indices = np.random.choice(len(dataset), size=1, replace=False)
    else:
        indices = fixed_indices

    for idx in indices:
        sample = dataset[idx]
        patient_id = sample["patient_id"]
        
        mri_slices = sample["mri_slices"]
        pet_slices = sample["pet_slices"]
        
        num_mri = mri_slices.shape[0]  
        num_pet = pet_slices.shape[0]

        # calculate central index using pet as reference
        slice_idx = num_pet // 2            # 25 // 2 = 12 (central slice)

        mri_slice = mri_slices[slice_idx] 
        pet_slice = pet_slices[slice_idx] 

        # Log for debug
        logger.info(f"Slices - MRI: {num_mri}, PET: {num_pet}, used index: {slice_idx}")

        saliency_mri = compute_saliency_map(student_model, mri_slice)
        gradcam_mri = compute_gradcam(student_model, mri_slice)
        
        logger.info(f"Stats per {patient_id}, Epoch {epoch+1} ({prefix.upper()})")
        logger.info(f"Saliency (raw) - Min: {saliency_mri.min():.6f}, Max: {saliency_mri.max():.6f}, "
                   f"Mean: {saliency_mri.mean():.6f}, Std: {saliency_mri.std():.6f}")
        logger.info(f"Grad-CAM (raw) - Min: {gradcam_mri.min():.6f}, Max: {gradcam_mri.max():.6f}, "
                   f"Mean: {gradcam_mri.mean():.6f}, Std: {gradcam_mri.std():.6f}")
        
        # compute focus area percentage
        saliency_focus = (saliency_mri > saliency_mri.mean() + saliency_mri.std()).sum() / saliency_mri.size
        gradcam_focus = (gradcam_mri > gradcam_mri.mean() + gradcam_mri.std()).sum() / gradcam_mri.size
        logger.info(f"Focus area - Saliency: {saliency_focus:.1%}, GradCAM: {gradcam_focus:.1%}")

        output_path = os.path.join(saliency_dir, f"{prefix}{patient_id}_slice{slice_idx}.png")
        
        compare_visualization_methods(
            mri_slice.cpu(),
            saliency_mri,
            gradcam_mri,
            output_path=output_path,
            cam_label="HiResCAM",
            pet_tensor=pet_slice.cpu() 
        )

        logger.info(f"Interpretability maps saved in: {output_path}")
        
        raw_maps_dir = os.path.join(saliency_dir, "raw_maps")
        os.makedirs(raw_maps_dir, exist_ok=True)
        np.save(os.path.join(raw_maps_dir, f"{patient_id}_slice{slice_idx}_saliency.npy"), saliency_mri)
        np.save(os.path.join(raw_maps_dir, f"{patient_id}_slice{slice_idx}_gradcam.npy"), gradcam_mri)

    student_model.train(was_training_outer)