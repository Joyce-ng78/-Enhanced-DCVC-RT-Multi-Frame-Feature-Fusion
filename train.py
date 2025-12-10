# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import time
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import deque

# Import your models
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, get_state_dict

# ============================================================
# 1. Configuration & Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCVC-RT with Multi-Frame Fusion")
    
    # Dataset & Paths
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to Vimeo90k or video dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')
    
    # Pretrained Weights (Start from standard DCVC-RT checkpoints)
    parser.add_argument('--checkpoint_i', type=str, required=True, help='Pretrained I-frame model')
    parser.add_argument('--checkpoint_p', type=str, required=True, help='Pretrained P-frame model')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_val', type=float, default=256, help='Lambda for Rate-Distortion trade-off')
    parser.add_argument('--gop_size', type=int, default=7, help='Training sequence length (I + 6 P)')
    parser.add_argument('--patch_size', type=int, default=256, help='Crop size for training')
    
    # Fusion Settings
    parser.add_argument('--train_fusion_only', type=str2bool, default=False, 
                        help='If True, freezes backbone and trains only Fusion Module')
    
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--worker', type=int, default=4)
    
    args = parser.parse_args()
    return args

# ============================================================
# 2. Dataset Loader (Sequence Based)
# ============================================================
class VideoSequenceDataset(Dataset):
    """
    Loads a sequence of frames for training.
    Expected structure: root/sequence_name/im1.png, im2.png...
    """
    def __init__(self, root, gop_size=7, patch_size=256):
        self.root = root
        self.gop_size = gop_size
        self.patch_size = patch_size
        self.sequences = self._scan_sequences(root)
        self.transform = transforms.ToTensor()

    def _scan_sequences(self, root):
        # Simplified scanner: assumes subfolders contain image sequences
        sequences = []
        for subdir in sorted(os.listdir(root)):
            path = os.path.join(root, subdir)
            if os.path.isdir(path):
                frames = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')])
                if len(frames) >= self.gop_size:
                    sequences.append(frames)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        # Random start index ensuring we have enough frames for GOP
        start_idx = random.randint(0, len(frame_paths) - self.gop_size)
        selected_paths = frame_paths[start_idx : start_idx + self.gop_size]
        
        # Load images
        frames = [Image.open(p).convert('RGB') for p in selected_paths]
        
        # Random Crop (Applied identically to all frames in sequence)
        w, h = frames[0].size
        th, tw = self.patch_size, self.patch_size
        if w < tw or h < th:
            # Resize if too small
            frames = [f.resize((max(w, tw), max(h, th))) for f in frames]
            w, h = frames[0].size
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        tensor_frames = []
        for img in frames:
            crop = img.crop((j, i, j + tw, i + th))
            tensor_frames.append(self.transform(crop))
            
        return torch.stack(tensor_frames) # Shape: [GOP, 3, H, W]

# ============================================================
# 3. Rate-Distortion Loss
# ============================================================
class RateDistortionLoss(torch.nn.Module):
    def __init__(self, lambda_val):
        super().__init__()
        self.lambda_val = lambda_val
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, output, target):
        # Distortion (MSE)
        mse = self.mse_loss(output['x_hat'], target)
        
        # Rate (Bits)
        # Assuming bit_stream length or estimated bits provided by model
        # If model returns estimated bits (likelihoods):
        bpp = output['bit_bpp']
        
        loss = self.lambda_val * mse + bpp
        return loss, mse, bpp

# ============================================================
# 4. Training Engine
# ============================================================
def train_one_epoch(model_i, model_p, dataloader, optimizer, criterion, epoch, args):
    model_i.train()
    model_p.train()
    
    avg_loss = 0
    avg_mse = 0
    avg_bpp = 0
    
    # We clear the buffer at start of every batch (new sequences)
    model_p.clear_dpb() 

    for batch_idx, frames in enumerate(dataloader):
        # frames shape: [Batch, GOP, 3, H, W]
        B, GOP, C, H, W = frames.shape
        frames = frames.to(args.device)
        
        optimizer.zero_grad()
        
        total_loss = 0
        
        # === 1. Encode I-Frame (First frame of GOP) ===
        x_i = frames[:, 0] # [B, 3, H, W]
        out_i = model_i(x_i) # Should return x_hat and likelihoods
        
        # Calculate I-frame loss (Simplified wrapper call)
        # Note: You need to implement a proper wrapper if model returns raw likelihoods
        # For this script, we assume model_i returns 'x_hat' and 'likelihoods'
        mse_i = torch.mean((out_i['x_hat'] - x_i) ** 2)
        bpp_i = torch.mean(out_i['likelihoods']) # Placeholder for actual entropy estimation
        loss_i = args.lambda_val * mse_i + bpp_i
        
        total_loss += loss_i
        
        # Prepare for P-frames: Add I-frame recon to DPB
        # The modified DMC class needs manual management if using external loop
        model_p.clear_dpb()
        model_p.add_ref_frame(None, out_i['x_hat'].detach()) # Detach to stop gradient flowing back to I-frame model indefinitely
        
        # === 2. Encode P-Frames (Rest of GOP) ===
        for t in range(1, GOP):
            x_curr = frames[:, t]
            
            # The model internally handles fusion via get_fused_context()