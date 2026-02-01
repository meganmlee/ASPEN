"""
Fusion Ablation Study for EEG Classification using AdaptiveSCALENet Architecture

Tests 7 different fusion strategies:
1. static - Equal 0.5/0.5 weighting
2. global_attention - Trial-level dynamic weighting (same as AdaptiveSCALENet)
3. spatial_attention - Per-channel weighting before global pooling
4. glu - Gated Linear Unit (noise suppression)
5. multiplicative - Element-wise feature interaction
6. bilinear - Low-rank bilinear interaction
7. cross_attention - Multi-head cross-attention between streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import mne
# Import utilities
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale_net')
sys.path.insert(0, scale_net_path)

from seed_utils import seed_everything
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders

# Import shared components from scale_net_adaptive_v2
from scale_net_adaptive_v2 import (
    SqueezeExcitation,
    SpectralResidualBlock,
    setup_device,
    wrap_model_multi_gpu,
    unwrap_model,
)


# ==================== Fusion Strategies ====================

class StaticFusion(nn.Module):
    """Static equal weighting (0.5/0.5)"""
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x_s, x_t):
        fused = 0.5 * x_s + 0.5 * x_t
        fused = self.bn(fused)
        return fused, None


class GlobalAttentionFusion(nn.Module):
    """
    Trial-level global attention fusion (same as AdaptiveSCALENet).
    Dynamically weights spectral and temporal streams per trial.
    """
    def __init__(self, dim, temperature=2.0, dropout=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        
        self.bn = nn.BatchNorm1d(dim)
        
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_s, x_t):
        """
        Args:
            x_s: Spectral features (B, dim)
            x_t: Temporal features (B, dim)
        
        Returns:
            fused: Fused features (B, dim)
            weights: Attention weights (B, 2)
        """
        # Concatenate
        ctx = torch.cat([x_s, x_t], dim=-1)  # (B, dim*2)
        
        # Calculate attention weights
        attn_logits = self.attn(ctx) / self.temperature
        w = torch.softmax(attn_logits, dim=-1)  # (B, 2)
        
        # Weighted combination
        fused = w[:, 0:1] * x_s + w[:, 1:2] * x_t
        
        # Add residual for stability
        residual = (x_s + x_t) * 0.5
        fused = fused + torch.sigmoid(self.alpha) * residual
        
        # Batch normalize
        fused = self.bn(fused)
        
        return fused, w


class GLUFusion(nn.Module):
    """Gated Linear Unit fusion for noise suppression"""
    def __init__(self, dim, dropout=0.1, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_s, x_t):
        combined = x_s + x_t
        ctx = torch.cat([x_s, x_t], dim=-1)
        gate_val = self.gate(ctx)
        fused = combined * gate_val
        fused = self.dropout(fused)
        fused = self.bn(fused)
        return fused, gate_val


class MultiplicativeFusion(nn.Module):
    """Element-wise multiplicative feature interaction"""
    def __init__(self, dim, dropout=0.1, **kwargs):
        super().__init__()
        self.proj_s = nn.Linear(dim, dim)
        self.proj_t = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_s, x_t):
        fused = self.proj_s(x_s) * self.proj_t(x_t)
        fused = self.dropout(fused)
        fused = self.bn(fused)
        return fused, None


class BilinearFusion(nn.Module):
    """Low-rank bilinear fusion for full pairwise interaction"""
    def __init__(self, dim, rank=16, dropout=0.1, **kwargs):
        super().__init__()
        self.U_s = nn.Linear(dim, rank, bias=False)
        self.U_t = nn.Linear(dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_s, x_t):
        z_s = self.U_s(x_s)  # (B, rank)
        z_t = self.U_t(x_t)  # (B, rank)
        z = z_s * z_t       # Element-wise product
        fused = self.out_proj(z)
        fused = self.dropout(fused)
        fused = self.bn(fused)
        return fused, None


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention fusion between streams"""
    def __init__(self, dim, num_heads=4, dropout=0.1, temperature=2.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Cross-attention: spectral attends to temporal
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Global weighting after cross-attention
        self.global_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )
        
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_s, x_t):
        B = x_s.size(0)
        
        # Reshape for multi-head attention: (B, num_heads, head_dim)
        q = self.q_proj(x_s).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(x_t).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(x_t).view(B, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bnh,bnh->bn', q, k) * scale  # (B, num_heads)
        attn = F.softmax(attn / self.temperature, dim=-1)
        
        # Apply attention to values
        attended = attn.unsqueeze(-1) * v  # (B, num_heads, head_dim)
        attended = attended.view(B, -1)  # (B, dim)
        attended = self.out_proj(attended)
        
        # Global weighting between original spectral and cross-attended features
        ctx = torch.cat([x_s, attended], dim=-1)
        global_weights = F.softmax(self.global_attn(ctx) / self.temperature, dim=-1)
        
        fused = global_weights[:, 0:1] * x_s + global_weights[:, 1:2] * attended
        
        # Residual
        residual = (x_s + x_t) * 0.5
        fused = fused + torch.sigmoid(self.alpha) * residual
        
        fused = self.dropout(fused)
        fused = self.bn(fused)
        
        return fused, global_weights


class SpatialAttentionFusion(nn.Module):
    """
    Spatial (channel-level) attention fusion.
    NOTE: This requires modification to the model forward pass to work with
    per-channel features BEFORE the channel averaging step.
    """
    def __init__(self, dim, n_channels, temperature=2.0, dropout=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.n_channels = n_channels
        
        # Per-channel attention weights
        self.channel_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )
        
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x_s, x_t, x_s_per_channel=None):
        """
        Args:
            x_s: Spectral features after channel averaging (B, dim)
            x_t: Temporal features (B, dim)
            x_s_per_channel: Optional per-channel spectral features (B, C, dim)
        
        Returns:
            fused: Fused features (B, dim)
            weights: Attention weights (B, 2) or (B, C, 2) if per-channel
        """
        if x_s_per_channel is not None:
            # Per-channel attention
            B, C, D = x_s_per_channel.shape
            x_t_expanded = x_t.unsqueeze(1).expand(-1, C, -1)  # (B, C, dim)
            ctx = torch.cat([x_s_per_channel, x_t_expanded], dim=-1)  # (B, C, dim*2)
            
            attn_logits = self.channel_attn(ctx) / self.temperature  # (B, C, 2)
            w = F.softmax(attn_logits, dim=-1)  # (B, C, 2)
            
            # Weighted combination per channel, then average
            fused_per_ch = w[:, :, 0:1] * x_s_per_channel + w[:, :, 1:2] * x_t_expanded
            fused = fused_per_ch.mean(dim=1)  # (B, dim)
            fused = self.bn(fused)
            
            return fused, w
        else:
            # Fall back to global attention if no per-channel features
            ctx = torch.cat([x_s, x_t], dim=-1)
            attn_logits = self.channel_attn(ctx) / self.temperature
            w = F.softmax(attn_logits, dim=-1)
            fused = w[:, 0:1] * x_s + w[:, 1:2] * x_t
            fused = self.bn(fused)
            return fused, w


def get_fusion_layer(mode, dim, n_channels=None, temperature=2.0, rank=16, num_heads=4, dropout=0.1):
    """Factory function to create fusion layer based on mode"""
    if mode == 'static':
        return StaticFusion(dim=dim, dropout=dropout)
    elif mode == 'global_attention':
        return GlobalAttentionFusion(dim=dim, temperature=temperature, dropout=dropout)
    elif mode == 'spatial_attention':
        return SpatialAttentionFusion(dim=dim, n_channels=n_channels, temperature=temperature, dropout=dropout)
    elif mode == 'glu':
        return GLUFusion(dim=dim, dropout=dropout)
    elif mode == 'multiplicative':
        return MultiplicativeFusion(dim=dim, dropout=dropout)
    elif mode == 'bilinear':
        return BilinearFusion(dim=dim, rank=rank, dropout=dropout)
    elif mode == 'cross_attention':
        return CrossAttentionFusion(dim=dim, num_heads=num_heads, dropout=dropout, temperature=temperature)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


# ==================== AdaptiveSCALENet with Configurable Fusion ====================

class AdaptiveSCALENetAblation(nn.Module):
    """
    EXACT same architecture as AdaptiveSCALENet from scale_net_adaptive_v2.py
    with configurable fusion layer for ablation study.
    
    Architecture:
    - Spectral Stream: 2D CNN with ResNet blocks → avg over channels → (B, hidden_dim)
    - Temporal Stream: EEGNet-style CNN → (B, hidden_dim)
    - Fusion: Configurable (7 strategies)
    - Classifier: Optional hidden layer + final linear
    """
    def __init__(self,
                 freq_bins,
                 time_bins,
                 n_channels,
                 n_classes,
                 T_raw,
                 cnn_filters=16,
                 hidden_dim=128,
                 dropout=0.4,
                 cnn_dropout=0.25,
                 use_hidden_layer=True,
                 classifier_hidden_dim=64,
                 fusion_mode='global_attention',
                 fusion_temperature=2.0,
                 fusion_rank=16):
        super().__init__()
        
        self.n_channels = n_channels
        self.T_raw = T_raw
        self.fusion_mode = fusion_mode
        
        # ====== SPECTRAL STREAM (2D CNN with Residual Blocks) ======
        self.spec_cnn_filters = cnn_filters * 2
        
        # Stage 1: Initial Conv + Residual Blocks
        self.spec_conv1 = nn.Conv2d(1, cnn_filters, kernel_size=7, padding=3, bias=False)
        self.spec_bn1 = nn.BatchNorm2d(cnn_filters)
        self.spec_se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.spec_dropout_cnn1 = nn.Dropout2d(cnn_dropout)
        
        # Add residual blocks for Stage 1
        self.spec_res1 = SpectralResidualBlock(cnn_filters, dropout=cnn_dropout)
        self.spec_res2 = SpectralResidualBlock(cnn_filters, dropout=cnn_dropout)
        
        self.spec_pool1 = nn.MaxPool2d(2)  # (F, T) → (F/2, T/2)
        
        # Stage 2: Conv + Residual Blocks
        self.spec_conv2 = nn.Conv2d(cnn_filters, self.spec_cnn_filters, kernel_size=5, padding=2, bias=False)
        self.spec_bn2 = nn.BatchNorm2d(self.spec_cnn_filters)
        self.spec_se2 = SqueezeExcitation(self.spec_cnn_filters, reduction=4)
        self.spec_dropout_cnn2 = nn.Dropout2d(cnn_dropout)
        
        # Add residual blocks for Stage 2
        self.spec_res3 = SpectralResidualBlock(self.spec_cnn_filters, dropout=cnn_dropout)
        self.spec_res4 = SpectralResidualBlock(self.spec_cnn_filters, dropout=cnn_dropout)
        
        self.spec_pool2 = nn.MaxPool2d(2)  # (F/2, T/2) → (F/4, T/4)
        
        # Spectral CNN Output Dimension
        self.spec_out_dim = (freq_bins // 4) * (time_bins // 4) * self.spec_cnn_filters
        
        # ====== TEMPORAL STREAM (EEGNet Inspired) ======
        F1 = 16   # F1 filters per temporal kernel
        D = 2    # Depth multiplier (Spatial filters per temporal filter)
        F2 = F1 * D
        
        # Layer 1: Temporal Conv (Kernels along Time)
        self.temp_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn_temp = nn.BatchNorm2d(F1)
        
        # Layer 2: Spatial Conv (Kernels along Channels) - Depthwise
        self.spatial_conv = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F2)
        self.pool_spatial = nn.AvgPool2d((1, 4))
        
        # Layer 3: Separable Conv
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),  # Depthwise
            nn.Conv2d(F2, F2, (1, 1), bias=False),  # Pointwise
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(cnn_dropout),

            nn.Conv2d(F2, F2, (1, 8), padding=(0, 4), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate temporal stream output dimension
        self.time_out_dim = self._get_time_flattened_size(n_channels, T_raw)
        
        # ====== FEATURE PROJECTION (to common dimension) ======
        self.proj_spec = nn.Sequential(
            nn.Linear(self.spec_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.proj_time = nn.Sequential(
            nn.Linear(self.time_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # ====== FUSION LAYER (configurable) ======
        self.fusion_layer = get_fusion_layer(
            mode=fusion_mode,
            dim=hidden_dim,
            n_channels=n_channels,
            temperature=fusion_temperature,
            rank=fusion_rank,
            num_heads=4,
            dropout=dropout * 0.5
        )
        
        # ====== CLASSIFIER ======
        self.use_hidden_layer = use_hidden_layer
        if use_hidden_layer:
            self.hidden_layer = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, classifier_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            classifier_input = classifier_hidden_dim
        else:
            classifier_input = hidden_dim
        
        self.classifier = nn.Linear(classifier_input, 1 if n_classes == 2 else n_classes)
        
        self._init_weights()
    
    def _get_time_flattened_size(self, n_channels, T_raw):
        with torch.no_grad():
            # Create a dummy input
            dummy_x = torch.zeros(1, 1, n_channels, T_raw)
            # Pass through temporal layers
            x = self.temp_conv(dummy_x)
            x = self.bn_temp(x)
            x = self.spatial_conv(x)
            x = self.bn_spatial(x)
            x = self.pool_spatial(F.relu(x))
            x = self.separable_conv(x)
            return x.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_time, x_spec, chan_ids: Optional[torch.Tensor] = None):
        B, C, _, _ = x_spec.shape
        
        # 1. Spectral Stream (with Residual Blocks)
        x_s = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
        
        # Stage 1: Initial conv + SE + Dropout
        x_s = self.spec_conv1(x_s)
        x_s = F.relu(self.spec_bn1(x_s))
        x_s = self.spec_se1(x_s)
        x_s = self.spec_dropout_cnn1(x_s)
        
        # Stage 1: Residual blocks (improves gradient flow)
        x_s = self.spec_res1(x_s)
        x_s = self.spec_res2(x_s)
        
        # Stage 1: Pool
        x_s = self.spec_pool1(x_s)
        
        # Stage 2: Conv + SE + Dropout
        x_s = self.spec_conv2(x_s)
        x_s = F.relu(self.spec_bn2(x_s))
        x_s = self.spec_se2(x_s)
        x_s = self.spec_dropout_cnn2(x_s)
        
        # Stage 2: Residual blocks
        x_s = self.spec_res3(x_s)
        x_s = self.spec_res4(x_s)
        
        # Stage 2: Pool
        x_s = self.spec_pool2(x_s)
        
        # Project spectral features
        x_s = x_s.view(B, C, -1)  # (B, C, flattened_features)
        x_s_per_channel = self.proj_spec(x_s)  # (B, C, hidden_dim) - keep for spatial attention
        x_s = x_s_per_channel.mean(dim=1)  # Average over channels: (B, hidden_dim)
        
        # 2. Temporal Stream
        x_t = self.separable_conv(
            self.pool_spatial(
                F.relu(self.bn_spatial(
                    self.spatial_conv(
                        self.bn_temp(self.temp_conv(x_time.unsqueeze(1)))
                    )
                ))
            )
        )
        x_t = self.proj_time(x_t.view(B, -1))  # (B, hidden_dim)
        
        # 3. Fusion
        if self.fusion_mode == 'spatial_attention':
            features, weights = self.fusion_layer(x_s, x_t, x_s_per_channel)
        else:
            features, weights = self.fusion_layer(x_s, x_t)
        
        # 4. Classifier
        if self.use_hidden_layer:
            features = self.hidden_layer(features)
        
        return self.classifier(features), weights


# ==================== Training Functions ====================

def train_epoch(model, loader, criterion, optimizer, device, is_binary=False):
    """Train for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100, leave=False)
    for (x_time, x_spec), labels in pbar:
        x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
        
        if is_binary:
            labels_float = labels.float().unsqueeze(1)
        else:
            labels_float = labels
        
        optimizer.zero_grad()
        outputs, _ = model(x_time, x_spec)
        loss = criterion(outputs, labels_float)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if is_binary:
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
        else:
            preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, device, criterion, is_binary=False, return_metrics=False):
    """
    Evaluate model
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        criterion: Loss function
        is_binary: Whether this is binary classification
        return_metrics: If True, return additional metrics (f1, recall, auc, pr_auc)
    
    Returns:
        If return_metrics=False: (avg_loss, acc)
        If return_metrics=True: (avg_loss, acc, metrics_dict)
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in loader:
            x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
            
            if is_binary:
                labels_float = labels.float().unsqueeze(1)
            else:
                labels_float = labels
            
            outputs, _ = model(x_time, x_spec)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()
            
            if is_binary:
                probs = torch.sigmoid(outputs).squeeze(1)
                preds = (probs > 0.5).long()
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            else:
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            
            if return_metrics:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    
    if not return_metrics:
        return avg_loss, acc
    
    # Calculate additional metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    metrics = {}
    
    # F1 score (macro average for multi-class, binary for binary)
    if is_binary:
        metrics["f1"] = f1_score(all_labels, all_preds, average="binary", zero_division=0) * 100
        metrics["recall"] = recall_score(all_labels, all_preds, average="binary", zero_division=0) * 100
        try:
            metrics["auc"] = roc_auc_score(all_labels, all_probs) * 100
            metrics["pr_auc"] = average_precision_score(all_labels, all_probs) * 100
        except ValueError:
            metrics["auc"] = -1
            metrics["pr_auc"] = -1
    else:
        metrics["f1"] = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        metrics["recall"] = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        # Multi-class AUC calculation (one-vs-rest)
        try:
            if all_probs.ndim == 2 and all_probs.shape[1] > 2:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(all_labels, classes=np.arange(all_probs.shape[1]))
                if y_bin.shape[1] == 1:
                    metrics["auc"] = roc_auc_score(all_labels, all_probs[:, 1]) * 100
                else:
                    metrics["auc"] = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr') * 100
            else:
                metrics["auc"] = -1
        except:
            metrics["auc"] = -1
    
    return avg_loss, acc, metrics


def evaluate_comprehensive(model, loader, device, is_binary=False):
    """Evaluate with additional metrics (F1, Recall, AUC)"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in loader:
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            outputs, _ = model(x_time, x_spec)
            
            if is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds.flatten())
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    avg_type = 'binary' if is_binary else 'macro'
    metrics = {
        'acc': np.mean(np.array(all_preds) == np.array(all_labels)) * 100,
        'f1': f1_score(all_labels, all_preds, average=avg_type, zero_division=0) * 100,
        'recall': recall_score(all_labels, all_preds, average=avg_type, zero_division=0) * 100,
    }
    
    try:
        if is_binary:
            metrics['auc'] = roc_auc_score(all_labels, all_probs) * 100
            metrics['pr_auc'] = average_precision_score(all_labels, all_probs) * 100
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr') * 100
    except:
        metrics['auc'] = None
        metrics['pr_auc'] = None
    
    return metrics


# ==================== Feature Statistics Collection ====================

def collect_feature_statistics(model, loader, device, fusion_mode):
    """Collect fusion statistics and attention weights"""
    model.eval()
    
    # Unwrap DataParallel if needed
    actual_model = unwrap_model(model)
    
    stats = {
        'spec_contribution': [], 
        'time_contribution': [],
        'labels': [], 
        'predictions': [], 
        'confidences': []
    }
    
    if fusion_mode in ['global_attention', 'spatial_attention', 'cross_attention']:
        stats['attn_weights'] = []
    if fusion_mode == 'spatial_attention':
        stats['spatial_weights'] = []
    elif fusion_mode == 'glu':
        stats['gate_sparsity'] = []
        stats['gate_values'] = []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in tqdm(loader, desc='Collecting Stats', ncols=100, leave=False):
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            B, C = x_spec.shape[0], x_spec.shape[1]
            
            # Get projected features manually to calculate contributions
            # Spectral stream with ResNet blocks
            x_spec_f = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
            x_spec_f = actual_model.spec_conv1(x_spec_f)
            x_spec_f = F.relu(actual_model.spec_bn1(x_spec_f))
            x_spec_f = actual_model.spec_se1(x_spec_f)
            x_spec_f = actual_model.spec_dropout_cnn1(x_spec_f)
            # ResNet blocks Stage 1
            x_spec_f = actual_model.spec_res1(x_spec_f)
            x_spec_f = actual_model.spec_res2(x_spec_f)
            x_spec_f = actual_model.spec_pool1(x_spec_f)
            # Stage 2
            x_spec_f = actual_model.spec_conv2(x_spec_f)
            x_spec_f = F.relu(actual_model.spec_bn2(x_spec_f))
            x_spec_f = actual_model.spec_se2(x_spec_f)
            x_spec_f = actual_model.spec_dropout_cnn2(x_spec_f)
            # ResNet blocks Stage 2
            x_spec_f = actual_model.spec_res3(x_spec_f)
            x_spec_f = actual_model.spec_res4(x_spec_f)
            x_spec_f = actual_model.spec_pool2(x_spec_f)
            x_spec_proj = actual_model.proj_spec(x_spec_f.view(B, C, -1))  # (B, C, hidden_dim)
            x_spec_avg = x_spec_proj.mean(dim=1)  # (B, hidden_dim)
            
            # Temporal stream
            x_g = x_time.unsqueeze(1)
            x_g = actual_model.bn_temp(actual_model.temp_conv(x_g))
            x_g = actual_model.pool_spatial(F.relu(actual_model.bn_spatial(actual_model.spatial_conv(x_g))))
            x_g = actual_model.separable_conv(x_g)
            x_time_proj = actual_model.proj_time(x_g.view(B, -1))  # (B, hidden_dim)
            
            # Get fusion weights
            if fusion_mode == 'spatial_attention':
                _, weights = actual_model.fusion_layer(x_spec_avg, x_time_proj, x_spec_proj)
                if weights is not None:
                    # weights shape is (B, C, 2) - take spectral weights
                    stats['spatial_weights'].append(weights[:, :, 0].cpu().numpy())
            else:
                _, weights = actual_model.fusion_layer(x_spec_avg, x_time_proj)
            
            if fusion_mode in ['global_attention', 'cross_attention'] and weights is not None:
                # weights shape is (B, 2) - [Spectral, Temporal]
                stats['attn_weights'].append(weights.cpu().numpy())
            elif fusion_mode == 'glu' and weights is not None:
                stats['gate_values'].append(weights.cpu().numpy())
                stats['gate_sparsity'].append((weights < 0.1).float().mean().cpu().item())
            
            # Calculate contribution magnitudes
            spec_mag = torch.norm(x_spec_avg, dim=-1)
            time_mag = torch.norm(x_time_proj, dim=-1)
            total = spec_mag + time_mag + 1e-8
            
            stats['spec_contribution'].append((spec_mag / total).cpu().numpy())
            stats['time_contribution'].append((time_mag / total).cpu().numpy())
            
            # Get predictions
            outputs, _ = model(x_time, x_spec)
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)
            stats['confidences'].append(probs.max(dim=1)[0].cpu().numpy())
            stats['labels'].append(labels.numpy())

    # Concatenate all arrays
    return {
        k: np.concatenate(v, axis=0) if (len(v) > 0 and isinstance(v[0], np.ndarray))
        else np.array(v)
        for k, v in stats.items()
    }


def analyze_fusion_statistics(stats, fusion_mode, task):
    """Analyze and save fusion statistics"""
    save_dir = f'./ablation_{task}/analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    avg_spec = stats['spec_contribution'].mean()
    avg_time = stats['time_contribution'].mean()
    
    analysis = {
        'fusion_mode': fusion_mode,
        'spectral_dominance': avg_spec,
        'temporal_dominance': avg_time,
        'primary_modality': 'Spectral' if avg_spec > avg_time else 'Temporal'
    }
    
    if 'attn_weights' in stats and len(stats['attn_weights']) > 0:
        analysis['attn_variance'] = stats['attn_weights'].var()
    if 'gate_sparsity' in stats and len(stats['gate_sparsity']) > 0:
        analysis['noise_suppression'] = np.mean(stats['gate_sparsity'])
    
    pd.DataFrame([analysis]).to_csv(f'{save_dir}/{task}_{fusion_mode}_usage.csv', index=False)
    return analysis


# ==================== Visualization ====================

def plot_modality_contributions(task, results_df):
    """Plot modality contribution balance across strategies"""
    save_dir = f'./ablation_{task}/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    plot_data = []
    for _, row in results_df.iterrows():
        if 'spectral_dominance' in row and 'temporal_dominance' in row:
            plot_data.append({
                'Strategy': row['strategy'],
                'Contribution': row['spectral_dominance'],
                'Modality': 'Spectral'
            })
            plot_data.append({
                'Strategy': row['strategy'],
                'Contribution': row['temporal_dominance'],
                'Modality': 'Temporal'
            })
    
    if not plot_data:
        print(f"No modality contribution data to plot for {task}")
        return
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    df = pd.DataFrame(plot_data)
    sns.barplot(x='Strategy', y='Contribution', hue='Modality', data=df, palette='viridis')
    plt.title(f'Modality Usage Balance - {task}', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel('Contribution', fontsize=12)
    plt.xlabel('Fusion Strategy', fontsize=12)
    
    # Add accuracy annotations
    for i, row in results_df.iterrows():
        if 'val_acc' in row:
            plt.text(i, 0.95, f"{row['val_acc']:.1f}%", 
                    ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{task}_modality_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved modality balance plot to {save_dir}/{task}_modality_balance.png")


def plot_spatial_attention_heatmap(task, stats, n_channels):
    """
    Plots both a topographical brain map and a named grid map of attention weights.
    Categorizes channel types to prevent overlapping position errors in MNE.
    """
    if 'spatial_weights' not in stats or len(stats['spatial_weights']) == 0:
        print(f"No spatial weights found for {task}. Skipping heatmap.")
        return

    save_dir = f'./ablation_{task}/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Average weights across all trials in the test set
    mean_weights = stats['spatial_weights'].mean(axis=0)

    if n_channels == 64 or n_channels == 62: 
        # Base 60 EEG channels common to both
        eeg_names = [
            'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
        ]
        
        if n_channels == 64:  # Wang2016 specific additions
            ch_names = eeg_names + ['CB1', 'CB2', 'VEO', 'HEO']
        else:  # Lee2019 specific (62 channels total)
            ch_names = eeg_names + ['VEO', 'HEO']
            
        # Assign types: EEG for the first 60, EOG/Misc for the rest
        ch_types = ['eeg'] * 60 + ['eog'] * (n_channels - 60)
            
    elif n_channels == 22:  # MI: BNCI2014_001
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz'
        ]
        ch_types = ['eeg'] * 22
        
    elif n_channels == 32:  # P300: BI2014B
        ch_names = [
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 
            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 
            'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 
            'Oz', 'O2'
        ]
        ch_types = ['eeg'] * 32

    elif n_channels == 16:  # P300: BNCI2014_009
        ch_names = [
            'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz', 'F3', 'F4', 
            'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8', 'O1', 'O2'
        ]
        ch_types = ['eeg'] * 16

    else:
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels

    # ==================== PART 1: MNE BRAIN TOPOMAP ====================
    try:
        # Create info with specific types to allow MNE to ignore eye/neck channels
        info = mne.create_info(ch_names=ch_names[:n_channels], sfreq=250, ch_types=ch_types)
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')

        # Pick only 'eeg' types to avoid the "overlapping positions" error
        eeg_picks = mne.pick_types(info, eeg=True, eog=False, misc=False)
        filtered_weights = mean_weights[eeg_picks]
        filtered_info = mne.pick_info(info, sel=eeg_picks)

        fig, ax = plt.subplots(figsize=(7, 7))
        im, _ = mne.viz.plot_topomap(
            filtered_weights, 
            filtered_info,
            axes=ax, 
            show=False, 
            cmap='YlGnBu', 
            contours=4
        )
        plt.colorbar(im, ax=ax, label='Mean Spectral Attention Weight')
        ax.set_title(f'Spatial Attention Topomap: {task}\n({len(eeg_picks)} Scalp Channels)')
        plt.savefig(f'{save_dir}/{task}_brain_topomap.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create brain topomap: {e}")

    # ==================== PART 2: NAMED GRID MAP ====================
    # Calculate grid size (e.g., 5x5 for 22ch, 6x6 for 32ch, 8x8 for 64ch)
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    
    # Pad weights and names to match rectangular grid
    padded_weights = np.zeros(rows * cols)
    padded_weights[:n_channels] = mean_weights
    
    padded_names = [""] * (rows * cols)
    for i in range(n_channels):
        padded_names[i] = ch_names[i]

    # Create annotation labels (Electrode Name + Weight Value)
    labels = np.array([f"{name}\n{val:.2f}" if name else "" 
                      for name, val in zip(padded_names, padded_weights)]).reshape(rows, cols)
    heatmap_data = padded_weights.reshape(rows, cols)

    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    sns.heatmap(heatmap_data, annot=labels, fmt="", cmap='YlGnBu', 
                cbar_kws={'label': 'Mean Attention Weight'},
                xticklabels=False, yticklabels=False)
    
    plt.title(f'Spatial Attention Channel Grid: {task}', fontsize=14, pad=20)
    plt.savefig(f'{save_dir}/{task}_spatial_grid_named.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved brain topomap and named grid map to {save_dir}")
    plt.close()


# ==================== Main Ablation Study ====================

def run_ablation_study(task: str, config: Dict = None):
    """
    Run fusion strategy ablation study for a given task
    
    Args:
        task: Task name (e.g., 'SSVEP', 'P300', 'MI', etc.)
        config: Configuration dictionary with training/model parameters
    """
    strategies = ['static', 'global_attention', 'spatial_attention', 'glu', 'multiplicative', 'bilinear', 'cross_attention']
    
    # Get task configuration
    task_config = TASK_CONFIGS.get(task, {})
    n_classes = task_config.get('num_classes', 26)
    is_binary = (n_classes == 2)
    
    if config is None:
        config = {
            # Training
            'num_epochs': 100,
            'batch_size': 64,
            'lr': 3e-4,
            'weight_decay': 5e-4,
            'seed': 44,
            'patience': 15,
            
            # Model architecture (matching AdaptiveSCALENet)
            'cnn_filters': 24,
            'hidden_dim': 256,
            'dropout': 0.3,
            'cnn_dropout': 0.25,
            'use_hidden_layer': True,
            'classifier_hidden_dim': 128,
            'fusion_temperature': 2.0,
        }

    device, n_gpus = setup_device()
    
    print(f"\n{'='*70}")
    print(f"FUSION ABLATION STUDY: {task}")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"Testing {len(strategies)} fusion strategies")
    print(f"Architecture: AdaptiveSCALENet (matching scale_net_adaptive_v2.py)")
    print(f"{'='*70}\n")

    print("STFT Configuration:")
    for k, v in config.items():
        if k.startswith('stft_'):
            print(f"  {k}: {v}")
    
    results_log = []
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        seed_everything(config['seed'], deterministic=True)
        
        # Load dataset
        datasets = load_dataset(
            task=task,
            data_dir=task_config.get('data_dir'),
            num_seen=task_config.get('num_seen'),
            seed=config['seed']
        )
        
        if not datasets:
            print(f"Failed to load dataset for {task}")
            continue

        # Auto-determine fusion rank based on dataset size
        n_train_samples = len(datasets['train'][0])
        if n_train_samples < 1000:
            fusion_rank = 8
        elif n_train_samples < 5000:
            fusion_rank = 16
        else:
            fusion_rank = 32
        
        # STFT config (task-specific)
        stft_config = {
            'fs': config.get('stft_fs', task_config.get('sampling_rate', 250)),
            'nperseg': config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
            'noverlap': config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
            'nfft': config.get('stft_nfft', task_config.get('stft_nfft', 512))
        }
        
        loaders = create_dataloaders(
            datasets, 
            stft_config, 
            batch_size=config['batch_size'],
            num_workers=4,
            augment_train=config.get('augment_train', True),
            seed=config['seed']
        )
        
        # Get dimensions
        sample_x, _ = next(iter(loaders['train']))
        _, n_channels, T_raw = sample_x[0].shape
        _, _, freq_bins, time_bins = sample_x[1].shape
        
        print(f"Data: {n_channels} channels, {n_classes} classes")
        print(f"Spectral: {freq_bins}x{time_bins}, Temporal: {T_raw} samples")
        
        # Create model (matching AdaptiveSCALENet architecture exactly)
        model = AdaptiveSCALENetAblation(
            freq_bins=freq_bins,
            time_bins=time_bins,
            n_channels=n_channels,
            n_classes=n_classes,
            T_raw=T_raw,
            cnn_filters=config['cnn_filters'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            cnn_dropout=config['cnn_dropout'],
            use_hidden_layer=config['use_hidden_layer'],
            classifier_hidden_dim=config['classifier_hidden_dim'],
            fusion_mode=strategy,
            fusion_temperature=config['fusion_temperature'],
            fusion_rank=fusion_rank
        ).to(device)
        
        # Wrap for multi-GPU
        model = wrap_model_multi_gpu(model, n_gpus)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Training setup (matching AdaptiveSCALENet)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 5e-4)
        )
        
        # Loss function
        if is_binary:
            train_labels = datasets['train'][1]
            class_counts = np.bincount(train_labels)
            class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
            if class_ratio > 1.5 or class_ratio < 0.67:
                pos_weight = torch.tensor([class_ratio], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Scheduler (matching AdaptiveSCALENet)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training loop
        best_score = -1.0
        best_val_acc = 0
        epochs_no_improve = 0
        patience = config.get('patience', 15)
        
        model_save_path = f'./ablation_{task}/models/best_{task}_{strategy}.pth'
        os.makedirs(f'./ablation_{task}/models', exist_ok=True)
        
        for epoch in range(config['num_epochs']):
            train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device, is_binary)
            val_loss, val_acc, val_metrics = evaluate(
                model, loaders['val'], device, criterion, is_binary, return_metrics=True
            )
            
            # Step scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {current_lr:.6f}")
            
            # Print additional metrics for binary classification
            if is_binary and val_metrics:
                print(f"  Val F1: {val_metrics.get('f1', 0):.2f}%, "
                      f"Recall: {val_metrics.get('recall', 0):.2f}%, "
                      f"AUC: {val_metrics.get('auc', 0):.2f}%, "
                      f"PR-AUC: {val_metrics.get('pr_auc', 0):.2f}%")
            
            # Save best model: Use PR-AUC for P300 tasks, val_acc for others
            if is_binary and task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']:
                # Use PR-AUC (preferred) or AUC as the saving criterion for imbalanced binary tasks
                score = val_metrics.get("pr_auc", -1)
                if score is None or score < 0:
                    score = val_metrics.get("auc", -1)
            else:
                # Multi-class or other binary tasks: use accuracy
                score = val_acc
            
            if score > best_score:
                best_score = score
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(unwrap_model(model).state_dict(), model_save_path)
                score_name = "PR-AUC" if (is_binary and task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']) else "Acc"
                print(f"  → Best model saved! {score_name}: {best_score:.2f}, Val Acc: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"  Patience: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f"  → Early stopping (patience={patience})")
                break
        
        # Load best model
        unwrap_model(model).load_state_dict(torch.load(model_save_path))
        
        # Evaluate
        log_entry = {'strategy': strategy, 'val_acc': best_val_acc}
        
        # Evaluate on test sets
        test_loader = loaders.get('test2') or loaders.get('test1')
        test_name = 'test2' if 'test2' in loaders else 'test1'
        
        if test_loader:
            test_metrics = evaluate_comprehensive(model, test_loader, device, is_binary)
            log_entry.update({f"{test_name}_{k}": v for k, v in test_metrics.items()})
            
            # Collect statistics
            stats = collect_feature_statistics(model, test_loader, device, strategy)
            analysis = analyze_fusion_statistics(stats, strategy, task)
            log_entry.update({
                'spectral_dominance': analysis['spectral_dominance'],
                'temporal_dominance': analysis['temporal_dominance']
            })
            
            # Plot spatial attention if applicable
            if strategy == 'spatial_attention':
                plot_spatial_attention_heatmap(task, stats, n_channels)
        
        results_log.append(log_entry)
        print(f"\n{strategy} Results:")
        print(f"  Val Acc: {log_entry['val_acc']:.2f}%")
        if f'{test_name}_acc' in log_entry:
            print(f"  {test_name.upper()} Acc: {log_entry[f'{test_name}_acc']:.2f}%")
            print(f"  Spectral Dom: {log_entry.get('spectral_dominance', 0):.3f}")
            print(f"  Temporal Dom: {log_entry.get('temporal_dominance', 0):.3f}")

    # Save results
    df = pd.DataFrame(results_log)
    os.makedirs(f'./ablation_{task}/results', exist_ok=True)
    csv_path = f"./ablation_{task}/results/ablation_results_{task}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {csv_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("ABLATION SUMMARY:")
    print(df.to_string(index=False))
    
    # Find best strategy
    test_col = 'test2_acc' if 'test2_acc' in df.columns else 'test1_acc'
    if test_col in df.columns:
        best_idx = df[test_col].idxmax()
        best_strategy = df.loc[best_idx, 'strategy']
        best_acc = df.loc[best_idx, test_col]
        print(f"\n🏆 BEST STRATEGY: {best_strategy} ({best_acc:.2f}%)")
    
    # Plot modality contributions
    plot_modality_contributions(task, df)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fusion ablation study (AdaptiveSCALENet architecture)')
    parser.add_argument('--task', type=str, required=True,
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech',
                                'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation (default: augmentation ON).')
    
    # STFT configuration (optional - uses task defaults if not specified)
    parser.add_argument('--stft_fs', type=int, default=None)
    parser.add_argument('--stft_nperseg', type=int, default=None)
    parser.add_argument('--stft_noverlap', type=int, default=None)
    parser.add_argument('--stft_nfft', type=int, default=None)
    
    args = parser.parse_args()
    
    config = {
        # Training
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 5e-4,
        'seed': args.seed,
        'patience': args.patience,
        
        # Model architecture (matching AdaptiveSCALENet exactly)
        'cnn_filters': 24,
        'hidden_dim': 256,
        'dropout': 0.3,
        'cnn_dropout': 0.25,
        'use_hidden_layer': True,
        'classifier_hidden_dim': 128,
        'fusion_temperature': 2.0,
        'augment_train': (not args.no_augment),
    }
    
    # Add STFT config if provided
    if args.stft_fs is not None:
        config['stft_fs'] = args.stft_fs
    if args.stft_nperseg is not None:
        config['stft_nperseg'] = args.stft_nperseg
    if args.stft_noverlap is not None:
        config['stft_noverlap'] = args.stft_noverlap
    if args.stft_nfft is not None:
        config['stft_nfft'] = args.stft_nfft
    
    results = run_ablation_study(args.task, config)