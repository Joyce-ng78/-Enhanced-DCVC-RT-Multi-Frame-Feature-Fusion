# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm

from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.layers.cuda_inference import replicate_pad
from src.utils.common import str2bool, create_folder, get_state_dict, set_torch_env
from src.utils.transforms import rgb2ycbcr
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.metrics import calc_psnr


class VideoDataset(Dataset):
    def __init__(self, config_path, gop_size=12, frame_limit=None):
        """
        Video dataset for training
        Args:
            config_path: Path to dataset config JSON
            gop_size: Group of pictures size
            frame_limit: Maximum frames per sequence (for debugging)
        """
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.gop_size = gop_size
        self.frame_limit = frame_limit
        self.sequences = []
        
        # Build sequence list
        root_path = self.config['root_path']
        for ds_name in self.config['train_classes']:
            ds_config = self.config['train_classes'][ds_name]
            if ds_config.get('train', 1) == 0:
                continue
                
            for seq_name in ds_config['sequences']:
                seq_info = ds_config['sequences'][seq_name]
                seq_path = os.path.join(root_path, ds_config['base_path'], seq_name)
                
                self.sequences.append({
                    'path': seq_path,
                    'type': ds_config['src_type'],
                    'width': seq_info['width'],
                    'height': seq_info['height'],
                    'frames': seq_info['frames'] if frame_limit is None else min(seq_info['frames'], frame_limit),
                    'name': seq_name
                })
        
        # Calculate total GOPs
        self.gops = []
        for seq in self.sequences:
            num_gops = seq['frames'] // gop_size
            for gop_idx in range(num_gops):
                self.gops.append({
                    'seq': seq,
                    'gop_idx': gop_idx
                })
        
        print(f"Loaded {len(self.sequences)} sequences, {len(self.gops)} GOPs")
    
    def __len__(self):
        return len(self.gops)
    
    def __getitem__(self, idx):
        gop_info = self.gops[idx]
        seq = gop_info['seq']
        gop_idx = gop_info['gop_idx']
        
        # Read frames
        frames = []
        start_frame = gop_idx * self.gop_size
        
        if seq['type'] == 'png':
            reader = PNGReader(seq['path'], seq['width'], seq['height'], start_num=start_frame + 1)
        else:  # yuv420
            reader = YUV420Reader(seq['path'], seq['width'], seq['height'], skip_frame=start_frame)
        
        for _ in range(self.gop_size):
            if seq['type'] == 'png':
                rgb = reader.read_one_frame()
                if rgb is None:
                    break
                # Convert to tensor and normalize
                frame = torch.from_numpy(rgb).float() / 255.0
            else:  # yuv420
                from src.utils.transforms import ycbcr420_to_444_np
                y, uv = reader.read_one_frame()
                if y is None:
                    break
                yuv = ycbcr420_to_444_np(y, uv)
                frame = torch.from_numpy(yuv).float() / 255.0
            
            frames.append(frame)
        
        reader.close()
        
        if len(frames) < self.gop_size:
            return None
        
        # Stack frames: [gop_size, 3, H, W]
        frames = torch.stack(frames, dim=0)
        
        return {
            'frames': frames,
            'height': seq['height'],
            'width': seq['width'],
            'seq_name': seq['name']
        }


def collate_fn(batch):
    """Custom collate function to handle None samples"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch[0]  # Return single GOP


class RateDistortionLoss(torch.nn.Module):
    """Rate-distortion loss for video compression"""
    def __init__(self, lmbda=0.01, metric='mse'):
        super().__init__()
        self.lmbda = lmbda
        self.metric = metric
    
    def forward(self, x, x_hat, bits):
        """
        Args:
            x: Original frames [B, C, H, W]
            x_hat: Reconstructed frames [B, C, H, W]
            bits: Total bits (scalar)
        """
        N, C, H, W = x.shape
        num_pixels = N * H * W
        
        if self.metric == 'mse':
            distortion = F.mse_loss(x, x_hat)
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented")
        
        # Normalize bits by number of pixels
        bpp = bits / num_pixels
        
        # Rate-distortion loss
        rd_loss = self.lmbda * distortion + bpp
        
        return {
            'loss': rd_loss,
            'distortion': distortion,
            'bpp': bpp,
            'bits': bits
        }


def estimate_bits_from_likelihoods(y, scales, means):
    """Estimate bits from Gaussian likelihood"""
    # Gaussian likelihood
    values = y - means
    scales = torch.clamp(scales, min=0.11, max=16.0)
    
    # -log2(likelihood)
    log_scales = torch.log(scales)
    log_likelihood = 0.5 * torch.log(2 * np.pi * torch.tensor(1.0)) + log_scales + \
                     0.5 * ((values / scales) ** 2)
    bits = log_likelihood / np.log(2.0)
    
    return bits.sum()


def train_one_epoch(p_frame_net, i_frame_net, dataloader, optimizer, criterion, 
                    device, epoch, args):
    """Train for one epoch"""
    p_frame_net.train()
    i_frame_net.eval()  # Keep I-frame model frozen
    
    metrics = defaultdict(list)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        frames = batch['frames'].to(device)  # [gop_size, 3, H, W]
        gop_size = frames.shape[0]
        
        # Get padding
        pic_height = batch['height']
        pic_width = batch['width']
        padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
        
        optimizer.zero_grad()
        
        total_bits = 0
        reconstructed = []
        
        # Process GOP
        p_frame_net.clear_dpb()
        
        for frame_idx in range(gop_size):
            x = frames[frame_idx:frame_idx+1]  # [1, 3, H, W]
            x = rgb2ycbcr(x)  # Convert to YCbCr
            x_padded = replicate_pad(x, padding_b, padding_r)
            
            if frame_idx == 0:
                # I-frame (frozen)
                with torch.no_grad():
                    qp_i = args.qp_i
                    encoded_i = i_frame_net.compress(x_padded, qp_i)
                    p_frame_net.add_ref_frame(None, encoded_i['x_hat'])
                    x_hat = encoded_i['x_hat'][:, :, :pic_height, :pic_width]
                    reconstructed.append(x_hat)
                continue
            
            # P-frame
            fa_idx = [0, 1, 0, 2, 0, 2, 0, 2][frame_idx % 8]
            qp_p = p_frame_net.shift_qp(args.qp_p, fa_idx)
            
            # Forward pass
            q_encoder = p_frame_net.q_encoder[qp_p:qp_p+1, :, :, :]
            q_decoder = p_frame_net.q_decoder[qp_p:qp_p+1, :, :, :]
            q_feature = p_frame_net.q_feature[qp_p:qp_p+1, :, :, :]
            
            feature = p_frame_net.apply_feature_adaptor()
            ctx, ctx_t = p_frame_net.feature_extractor(feature, q_feature)
            y = p_frame_net.encoder(x_padded, ctx, q_encoder)
            
            # Hyperprior
            hyper_inp = p_frame_net.pad_for_y(y)
            z = p_frame_net.hyper_encoder(hyper_inp)
            
            # Round z (straight-through estimator)
            z_hat = torch.round(z)
            z_bits = estimate_bits_from_likelihoods(z_hat, 
                                                    torch.ones_like(z_hat), 
                                                    torch.zeros_like(z_hat))
            
            # Get parameters
            params = p_frame_net.res_prior_param_decoder(z_hat, ctx_t)
            
            # Separate prior parameters
            y_padded, q_dec, scales, means = p_frame_net.separate_prior_for_video_encoding(params, y)
            
            # Round y (straight-through estimator)
            y_hat = torch.round(y_padded - means) + means
            y_bits = estimate_bits_from_likelihoods(y_hat, scales, means)
            
            total_bits += z_bits + y_bits
            
            # Decode
            feature = p_frame_net.decoder(y_hat, ctx, q_decoder)
            p_frame_net.add_ref_frame(feature, None)
            
            # Reconstruct
            q_recon = p_frame_net.q_recon[qp_p:qp_p+1, :, :, :]
            x_hat_full = p_frame_net.recon_generation_net(feature, q_recon)
            x_hat = x_hat_full[:, :, :pic_height, :pic_width]
            reconstructed.append(x_hat)
        
        # Calculate loss (only for P-frames)
        if len(reconstructed) > 1:
            x_orig = rgb2ycbcr(frames[1:])  # Skip I-frame
            x_recon = torch.cat(reconstructed[1:], dim=0)
            
            loss_dict = criterion(x_orig, x_recon, total_bits)
            loss = loss_dict['loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(p_frame_net.parameters(), args.clip_max_norm)
            
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                psnr = calc_psnr(x_orig[0].cpu().numpy() * 255, 
                                x_recon[0].detach().cpu().numpy() * 255)
            
            metrics['loss'].append(loss.item())
            metrics['distortion'].append(loss_dict['distortion'].item())
            metrics['bpp'].append(loss_dict['bpp'].item())
            metrics['psnr'].append(psnr)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bpp': f"{loss_dict['bpp'].item():.4f}",
                'psnr': f"{psnr:.2f}"
            })
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train video compression model")
    
    # Model paths
    parser.add_argument('--model_path_i', type=str, required=True,
                       help='Path to pretrained I-frame model')
    parser.add_argument('--model_path_p', type=str, default=None,
                       help='Path to pretrained P-frame model (optional)')
    
    # Dataset
    parser.add_argument('--train_config', type=str, required=True,
                       help='Path to training dataset config')
    parser.add_argument('--gop_size', type=int, default=12,
                       help='Group of pictures size')
    parser.add_argument('--frame_limit', type=int, default=None,
                       help='Max frames per sequence (for debugging)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (currently only 1 GOP supported)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr_steps', type=int, nargs='+', default=[60, 80],
                       help='Epochs to decay learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Learning rate decay factor')
    parser.add_argument('--lmbda', type=float, default=0.01,
                       help='Rate-distortion tradeoff parameter')
    parser.add_argument('--clip_max_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Quality parameters
    parser.add_argument('--qp_i', type=int, default=37,
                       help='QP for I-frames')
    parser.add_argument('--qp_p', type=int, default=37,
                       help='Base QP for P-frames')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_zero_thres', type=float, default=None)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set environment
    set_torch_env()
    
    # Device
    if args.cuda and torch.cuda.is_available():
        device = f"cuda:{args.cuda_idx}"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_idx)
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    create_folder(args.checkpoint_dir, True)
    
    # Load I-frame model (frozen)
    print("Loading I-frame model...")
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    i_frame_net.update(args.force_zero_thres)
    
    # Freeze I-frame model
    for param in i_frame_net.parameters():
        param.requires_grad = False
    
    # Load P-frame model
    print("Loading P-frame model...")
    p_frame_net = DMC()
    if args.model_path_p is not None:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net.load_state_dict(p_state_dict)
    p_frame_net = p_frame_net.to(device)
    p_frame_net.update(args.force_zero_thres)
    
    # Dataset
    print("Loading dataset...")
    train_dataset = VideoDataset(args.train_config, 
                                 gop_size=args.gop_size,
                                 frame_limit=args.frame_limit)
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)
    
    # Optimizer
    optimizer = Adam(p_frame_net.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    
    # Loss
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        p_frame_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        metrics = train_one_epoch(p_frame_net, i_frame_net, train_loader,
                                 optimizer, criterion, device, epoch, args)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} - Time: {epoch_time:.1f}s - "
              f"Loss: {metrics['loss']:.4f} - "
              f"BPP: {metrics['bpp']:.4f} - "
              f"PSNR: {metrics['psnr']:.2f} dB")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, 
                                          f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': p_frame_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save(p_frame_net.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
