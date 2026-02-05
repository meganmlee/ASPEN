"""
ASPEN: Adaptive Spectral Encoder Network

Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech
Dual-stream architecture with multiplicative fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from typing import Optional, Dict, Tuple
from seed_utils import seed_everything
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# Import from dataset.py
from dataset import (
    load_dataset,
    TASK_CONFIGS,
    EEGDataset,
    create_dataloaders,
)

# ==================== SE Block ====================
class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpectralResidualBlock(nn.Module):
    """
    Residual block for spectral stream - inspired by ResNet
    Improves gradient flow and allows deeper networks
    """
    def __init__(self, channels, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, reduction=4)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        out = out + residual
        out = F.relu(out, inplace=True)
        
        return out

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

class ASPEN(nn.Module):
    """
    Combines spectral (STFT) and temporal (raw) streams
    Inputs:
        x_time: (B, C, T_raw) - Raw temporal EEG data
        x_spec: (B, C, F, T) - STFT features
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
                 fusion_temperature=2.0):
        super().__init__()
        
        self.n_channels = n_channels
        self.T_raw = T_raw
        
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
        
        # ====== Multiplicative FUSION ======
        self.fusion_layer = MultiplicativeFusion(
            dim=hidden_dim,
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
            return x.numel() # This is your exact time_out_dim

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
        
        # Flatten and project: (B*C, features) -> (B, C, features) -> average over channels -> (B, hidden_dim)
        x_s = x_s.view(B, C, -1)  # (B, C, flattened_features)
        x_s = self.proj_spec(x_s).mean(dim=1)  # Average over channels: (B, hidden_dim)
        
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
        
        # 3. Global Attention Fusion
        features, weights = self.fusion_layer(x_s, x_t)  # (B, hidden_dim), (B, 2)
        
        # 4. Classifier
        if self.use_hidden_layer:
            features = self.hidden_layer(features)
        
        return self.classifier(features), weights


# ==================== Multi-GPU Setup ====================
def setup_device():
    """Setup device and return device info"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"CUDA available: {n_gpus} GPU(s) detected")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return device, n_gpus
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0


def wrap_model_multi_gpu(model, n_gpus):
    """Wrap model with DataParallel if multiple GPUs available"""
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
    return model


def unwrap_model(model):
    """Get the underlying model from DataParallel wrapper"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ==================== Training Functions ====================
def train_epoch(model, loader, criterion, optimizer, device, is_binary=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100)
    for inputs, labels in pbar:
        x_time, x_spec = inputs
        x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
        
        # Convert labels for binary classification
        if is_binary:
            labels_float = labels.float().unsqueeze(1)
        else:
            labels_float = labels
        
        optimizer.zero_grad()
        outputs, _ = model(x_time, x_spec)
        loss = criterion(outputs, labels_float)
        loss.backward()
        
        # Gradient clipping
        actual_model = unwrap_model(model)
        torch.nn.utils.clip_grad_norm_(actual_model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Prediction
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, criterion=None, is_binary=False, return_metrics=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        criterion: Loss function (optional)
        is_binary: Whether this is binary classification
        return_metrics: If True, return additional metrics (f1, recall, auc)
    
    Returns:
        If return_metrics=False: (avg_loss, acc)
        If return_metrics=True: (avg_loss, acc, metrics_dict)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            x_time, x_spec = inputs
            x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
            
            if is_binary:
                labels_float = labels.float().unsqueeze(1)
            else:
                labels_float = labels
            
            outputs, _ = model(x_time, x_spec)
            
            if criterion is not None:
                loss = criterion(outputs, labels_float)
                total_loss += loss.item()
            
            # Prediction
            if is_binary:
                probs = torch.sigmoid(outputs).squeeze(1)
                pred = (probs > 0.5).long()
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            else:
                probs = F.softmax(outputs, dim=1)
                _, pred = outputs.max(1)
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            
            if return_metrics:
                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    
    if not return_metrics:
        return avg_loss, acc
    
    # Calculate additional metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    metrics = {}
    
    # F1 score
    if is_binary:
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['f1'] = f1 * 100
    
    # Recall
    if is_binary:
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall'] = recall * 100
    
    # AUC
    try:
        if is_binary:
            auc = roc_auc_score(all_labels, all_probs)
            metrics['auc'] = auc * 100
        else:
            n_classes = all_probs.shape[1]
            if n_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
                metrics['auc'] = auc * 100
            else:
                try:
                    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    metrics['auc'] = auc * 100
                except ValueError:
                    try:
                        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                        metrics['auc'] = auc * 100
                    except ValueError:
                        metrics['auc'] = None
    except (ValueError, IndexError):
        metrics['auc'] = None
    
    return avg_loss, acc, metrics


# ==================== Multiplicative stats Collection ====================
def collect_fusion_stats(model, loader, device) -> Optional[Dict[str, np.ndarray]]:
    """
    Collects feature statistics for multiplicative fusion analysis.
    
    For multiplicative fusion: fused = proj_s(x_s) * proj_t(x_t)
    
    We analyze:
    1. Feature magnitudes (L2 norms) of each stream's projected features
    2. Mean activations per stream
    3. Contribution ratio: |proj_s| / (|proj_s| + |proj_t|)
    
    Returns:
        Dictionary containing:
        - 'spectral_magnitude': L2 norm of projected spectral features (N,)
        - 'temporal_magnitude': L2 norm of projected temporal features (N,)
        - 'spectral_contribution': Relative contribution ratio (N,)
        - 'spectral_mean_activation': Mean activation of spectral features (N,)
        - 'temporal_mean_activation': Mean activation of temporal features (N,)
        - 'feature_correlation': Correlation between streams per trial (N,)
    """
    model.eval()
    
    # Storage for statistics
    spectral_magnitudes = []
    temporal_magnitudes = []
    spectral_mean_acts = []
    temporal_mean_acts = []
    feature_correlations = []
    
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Register hooks to capture intermediate features
    spectral_features = []
    temporal_features = []
    
    def hook_spectral(module, input, output):
        # Move to CPU immediately to avoid GPU device conflicts
        spectral_features.append(output.detach().cpu())
    
    def hook_temporal(module, input, output):
        # Move to CPU immediately to avoid GPU device conflicts
        temporal_features.append(output.detach().cpu())
    
    # Register hooks on projection layers
    hook_s = actual_model.proj_spec.register_forward_hook(hook_spectral)
    hook_t = actual_model.proj_time.register_forward_hook(hook_temporal)
    
    try:
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc='Collecting Fusion Stats', ncols=100):
                x_time, x_spec = inputs
                x_time, x_spec = x_time.to(device), x_spec.to(device)
                
                # Clear feature lists
                spectral_features.clear()
                temporal_features.clear()
                
                # Forward pass (hooks will capture features)
                _ = model(x_time, x_spec)
                
                if spectral_features and temporal_features:
                    # With DataParallel, hooks may be called multiple times (once per GPU)
                    # Concatenate all chunks along batch dimension
                    if len(spectral_features) > 1:
                        x_s = torch.cat(spectral_features, dim=0)
                        x_t = torch.cat(temporal_features, dim=0)
                    else:
                        x_s = spectral_features[0]
                        x_t = temporal_features[0]
                    
                    # Features are already on CPU from hooks
                    
                    # Handle case where spectral still has channel dimension
                    if x_s.dim() == 3:
                        x_s = x_s.mean(dim=1)  # Average over channels
                    
                    # 1. L2 magnitudes
                    spec_mag = torch.norm(x_s, p=2, dim=-1)  # (B,)
                    temp_mag = torch.norm(x_t, p=2, dim=-1)  # (B,)
                    
                    spectral_magnitudes.append(spec_mag.numpy())
                    temporal_magnitudes.append(temp_mag.numpy())
                    
                    # 2. Mean activations
                    spec_mean = x_s.mean(dim=-1)  # (B,)
                    temp_mean = x_t.mean(dim=-1)  # (B,)
                    
                    spectral_mean_acts.append(spec_mean.numpy())
                    temporal_mean_acts.append(temp_mean.numpy())
                    
                    # 3. Feature correlation per trial
                    # Normalize features for correlation
                    x_s_norm = F.normalize(x_s, p=2, dim=-1)
                    x_t_norm = F.normalize(x_t, p=2, dim=-1)
                    correlation = (x_s_norm * x_t_norm).sum(dim=-1)  # Cosine similarity
                    
                    feature_correlations.append(correlation.numpy())
    
    finally:
        # Remove hooks
        hook_s.remove()
        hook_t.remove()
    
    if not spectral_magnitudes:
        return None
    
    # Concatenate all batches
    spectral_magnitudes = np.concatenate(spectral_magnitudes)
    temporal_magnitudes = np.concatenate(temporal_magnitudes)
    spectral_mean_acts = np.concatenate(spectral_mean_acts)
    temporal_mean_acts = np.concatenate(temporal_mean_acts)
    feature_correlations = np.concatenate(feature_correlations)
    
    # Calculate contribution ratio: spectral / (spectral + temporal)
    total_magnitude = spectral_magnitudes + temporal_magnitudes + 1e-8
    spectral_contribution = spectral_magnitudes / total_magnitude
    
    return {
        'spectral_magnitude': spectral_magnitudes,
        'temporal_magnitude': temporal_magnitudes,
        'spectral_contribution': spectral_contribution,
        'temporal_contribution': 1 - spectral_contribution,
        'spectral_mean_activation': spectral_mean_acts,
        'temporal_mean_activation': temporal_mean_acts,
        'feature_correlation': feature_correlations,
    }


def save_fusion_stats(stats: Dict[str, np.ndarray], task: str, split: str = 'test') -> str:
    """
    Save fusion statistics to CSV file.
    
    Args:
        stats: Dictionary from collect_fusion_stats
        task: Task name (e.g., 'SSVEP', 'P300')
        split: Data split name (e.g., 'test', 'val', 'unseen')
    
    Returns:
        Filename of saved CSV
    """
    df = pd.DataFrame({
        'Spectral_Magnitude': stats['spectral_magnitude'],
        'Temporal_Magnitude': stats['temporal_magnitude'],
        'Spectral_Contribution': stats['spectral_contribution'],
        'Temporal_Contribution': stats['temporal_contribution'],
        'Spectral_Mean_Activation': stats['spectral_mean_activation'],
        'Temporal_Mean_Activation': stats['temporal_mean_activation'],
        'Feature_Correlation': stats['feature_correlation'],
    })
    
    filename = f'{task.lower()}_fusion_stats_{split}.csv'
    df.to_csv(filename, index=False)
    
    return filename


def print_fusion_summary(stats: Dict[str, np.ndarray], task: str):
    """Print summary statistics for fusion analysis."""
    print(f"\n{'='*60}")
    print(f"Fusion Statistics Summary - {task}")
    print(f"{'='*60}")
    print(f"Number of trials: {len(stats['spectral_magnitude'])}")
    print(f"\nFeature Magnitudes (L2 Norm):")
    print(f"  Spectral: {stats['spectral_magnitude'].mean():.4f} ± {stats['spectral_magnitude'].std():.4f}")
    print(f"  Temporal: {stats['temporal_magnitude'].mean():.4f} ± {stats['temporal_magnitude'].std():.4f}")
    print(f"\nRelative Contributions:")
    print(f"  Spectral: {stats['spectral_contribution'].mean()*100:.1f}% ± {stats['spectral_contribution'].std()*100:.1f}%")
    print(f"  Temporal: {stats['temporal_contribution'].mean()*100:.1f}% ± {stats['temporal_contribution'].std()*100:.1f}%")
    print(f"\nMean Activations:")
    print(f"  Spectral: {stats['spectral_mean_activation'].mean():.4f} ± {stats['spectral_mean_activation'].std():.4f}")
    print(f"  Temporal: {stats['temporal_mean_activation'].mean():.4f} ± {stats['temporal_mean_activation'].std():.4f}")
    print(f"\nFeature Correlation (Cosine Similarity):")
    print(f"  Mean: {stats['feature_correlation'].mean():.4f} ± {stats['feature_correlation'].std():.4f}")
    print(f"{'='*60}")


# ==================== Main Training ====================
def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """
    Train model for a specific EEG task
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech'
        config: Training configuration (uses defaults if None)
        model_path: Path to save best model
    
    Returns:
        (model, results_dict)
    """
    # Get task-specific defaults
    task_config = TASK_CONFIGS.get(task, {})
    
    if config is None:
        config = {
            'data_dir': task_config.get('data_dir', '/ocean/projects/cis250213p/shared/ssvep'),
            'num_seen': task_config.get('num_seen', 33),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 26),
            
            # Model - FIXED ACROSS ALL TASKS FOR GENERALIZABILITY
            'cnn_filters': 24,  # Increased from 16 for better capacity
            'hidden_dim': 256,  # Increased from 128 for better capacity
            'dropout': 0.3,  # Slightly higher for regularization
            'cnn_dropout': 0.25,  # Balanced regularization
            'use_hidden_layer': True,
            'classifier_hidden_dim': 128,  # Increased from 64
            'fusion_temperature': 2.0,
            
            # STFT - Use task-specific parameters
            'stft_fs': task_config.get('sampling_rate', 250),
            'stft_nperseg': task_config.get('stft_nperseg', 128),
            'stft_noverlap': task_config.get('stft_noverlap', 112),
            'stft_nfft': task_config.get('stft_nfft', 512),
            
            # Training - FIXED ACROSS ALL TASKS
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 3e-4,  # Conservative learning rate
            'weight_decay': 5e-4,  # Stronger regularization
            'patience': 15,
            'scheduler': 'ReduceLROnPlateau',
        }
    else:
        # Fill in missing keys with task-specific defaults
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('stft_fs', task_config.get('sampling_rate', 250))
        config.setdefault('stft_nperseg', task_config.get('stft_nperseg', 128))
        config.setdefault('stft_noverlap', task_config.get('stft_noverlap', 112))
        config.setdefault('stft_nfft', task_config.get('stft_nfft', 512))
        config.setdefault('cnn_filters', 24)
        config.setdefault('hidden_dim', 256)
        config.setdefault('dropout', 0.3)
        config.setdefault('cnn_dropout', 0.25)
        config.setdefault('use_hidden_layer', True)
        config.setdefault('classifier_hidden_dim', 128)
        config.setdefault('scheduler', 'ReduceLROnPlateau')
        config.setdefault('fusion_temperature', 2.0)
        config.setdefault('lr', 3e-4)
        config.setdefault('weight_decay', 5e-4)
    
    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    # Setup device and multi-GPU
    device, n_gpus = setup_device()
    
    print(f"\n{'='*70}")
    print(f"{task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"Fusion: Multiplicative")
    
    # ====== Load Data ======
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir'),
        num_seen=config.get('num_seen'),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # STFT config
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    # Print STFT parameters
    print(f"\nSTFT Parameters (task-specific):")
    print(f"  Sampling Rate: {stft_config['fs']} Hz")
    print(f"  nperseg: {stft_config['nperseg']} samples ({stft_config['nperseg']/stft_config['fs']:.3f} sec)")
    print(f"  noverlap: {stft_config['noverlap']} samples ({100*stft_config['noverlap']/stft_config['nperseg']:.1f}% overlap)")
    print(f"  nfft: {stft_config['nfft']}")
    print(f"  Frequency resolution: {stft_config['fs']/stft_config['nfft']:.2f} Hz/bin")
    
    # ====== Create Data Loaders ======
    loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=config.get('augment_train', True),
        seed=seed,
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions from a sample
    sample_x, _ = next(iter(train_loader))
    sample_x_time, sample_x_spec = sample_x
    _, n_channels, T_raw = sample_x_time.shape
    _, _, freq_bins, time_bins = sample_x_spec.shape
    
    print(f"STFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = ASPEN(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        T_raw=T_raw,
        cnn_filters=config['cnn_filters'],
        hidden_dim=config['hidden_dim'],
        dropout=config.get('dropout', 0.3),
        cnn_dropout=config.get('cnn_dropout', 0.25),
        use_hidden_layer=config.get('use_hidden_layer', True),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        fusion_temperature=config.get('fusion_temperature', 2.0)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    # Wrap model for multi-GPU training
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # ====== Loss & Optimizer ======
    train_labels = datasets['train'][1]
    if n_classes == 2:
        # Calculate class imbalance
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
        # Only use pos_weight if imbalance ratio > 1.5
        if class_ratio > 1.5 or class_ratio < 0.67:
            pos_weight = torch.tensor([class_ratio], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCEWithLogitsLoss with pos_weight={class_ratio:.2f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print(f"Using BCEWithLogitsLoss without pos_weight (balanced)")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {n_classes}-class classification")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler_type = config.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] // 2,
            eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        seed = config.get('seed', 44)
        model_path = f'best_{task.lower()}_simplified_model_{seed}.pth'
    
    is_binary = (n_classes == 2)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_binary=is_binary)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, is_binary=is_binary)
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
            }, model_path)
            print(f"✓ Best model saved! ({val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print("\nEarly stopping triggered!")
            break
    
    # ====== Final Evaluation ======
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation...")
    print(f"Best model path: {model_path}")
    
    checkpoint = torch.load(model_path)
    unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate validation set with metrics
    val_loss, val_acc, val_metrics = evaluate(model, val_loader, device, criterion, is_binary=is_binary, return_metrics=True)
    
    results = {
        'val': best_val_acc,
        'val_f1': val_metrics.get('f1'),
        'val_recall': val_metrics.get('recall'),
        'val_auc': val_metrics.get('auc'),
    }
    
    if test1_loader:
        test1_loss, test1_acc, test1_metrics = evaluate(model, test1_loader, device, criterion, is_binary=is_binary, return_metrics=True)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
        results['test1_f1'] = test1_metrics.get('f1')
        results['test1_recall'] = test1_metrics.get('recall')
        results['test1_auc'] = test1_metrics.get('auc')
    
    if test2_loader:
        test2_loss, test2_acc, test2_metrics = evaluate(model, test2_loader, device, criterion, is_binary=is_binary, return_metrics=True)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
        results['test2_f1'] = test2_metrics.get('f1')
        results['test2_recall'] = test2_metrics.get('recall')
        results['test2_auc'] = test2_metrics.get('auc')
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task} (ASPEN)")
    print(f"{'='*70}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    if results.get('val_f1') is not None:
        print(f"Val F1: {results['val_f1']:.2f}%")
        print(f"Val Recall: {results['val_recall']:.2f}%")
        if results.get('val_auc') is not None:
            print(f"Val AUC: {results['val_auc']:.2f}%")
    
    if 'test1' in results:
        print(f"Test1 (Seen): {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
        if results.get('test1_f1') is not None:
            print(f"  Test1 F1: {results['test1_f1']:.2f}%")
            print(f"  Test1 Recall: {results['test1_recall']:.2f}%")
            if results.get('test1_auc') is not None:
                print(f"  Test1 AUC: {results['test1_auc']:.2f}%")
    
    if 'test2' in results:
        print(f"Test2 (Unseen): {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
        if results.get('test2_f1') is not None:
            print(f"  Test2 F1: {results['test2_f1']:.2f}%")
            print(f"  Test2 Recall: {results['test2_recall']:.2f}%")
            if results.get('test2_auc') is not None:
                print(f"  Test2 AUC: {results['test2_auc']:.2f}%")
    
    print(f"{'='*70}")
    
    # ====== Collect and Save Attention Weights ======
    print(f"Collecting fusion statistics for {task}...")
    
    stats_loader = test2_loader if test2_loader else val_loader
    stats_type = 'unseen' if test2_loader else 'val'
    
    if stats_loader:
        fusion_stats = collect_fusion_stats(model, stats_loader, device)
        
        if fusion_stats is not None:
            # Print summary
            print_fusion_summary(fusion_stats, task)
            
            # Save to CSV
            csv_filename = save_fusion_stats(fusion_stats, task, stats_type)
            print(f"Fusion statistics saved to: {csv_filename}")
        else:
            print(f"No fusion statistics collected")
    
    print(f"{'='*70}")
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints'):
    """
    Train models for all specified tasks
    
    Args:
        tasks: List of task names (default: all tasks)
        save_dir: Directory to save model checkpoints
    
    Returns:
        Dictionary of results for each task
    """
    if tasks is None:
        tasks = ['SSVEP', 'Lee2019_SSVEP', 'BI2014b_P300', 'BNCI2014_P300', 'MI', 'Lee2019_MI', 'Imagined_speech']
    
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}
    
    print("=" * 80)
    print("MULTI-TASK EEG CLASSIFICATION - ASPEN")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_{task.lower()}_simplified_model_44.pth')
            model, results = train_task(task, model_path=model_path)
            all_results[task] = results
            
            print(f"\n{task} completed!")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc: {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc: {results['test2']:.2f}%")
        
        except Exception as e:
            print(f"Error training {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS - ASPEN")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        if 'error' in results:
            print(f"\n{task}: FAILED - {results['error']}")
        else:
            print(f"\n{task}:")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc: {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc: {results['test2']:.2f}%")
    
    print(f"\n{'='*80}")
    print("MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


# Legacy function for backward compatibility
def train_model(config=None, model_path=None):
    """Legacy training function for SSVEP (backward compatibility)"""
    return train_task('SSVEP', config, model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ASPEN on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                       choices=['SSVEP', 'Lee2019_SSVEP', 'BI2014b_P300', 'BNCI2014_P300', 'MI', 'Lee2019_MI', 'Imagined_speech', 'all'],
                       help='Task to train on (default: SSVEP)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=44,
                       help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable train-time augmentation (default: augmentation ON).')
    
    args = parser.parse_args()
    

    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'patience': args.patience,
        
        'cnn_filters': 24,
        'hidden_dim': 256,
        'dropout': 0.3,
        'cnn_dropout': 0.25,
        'use_hidden_layer': True,
        'classifier_hidden_dim': 128,
        'lr': 3e-4,
        'weight_decay': 5e-4,
        'scheduler': 'ReduceLROnPlateau',
        'fusion_temperature': 2.0,
        'augment_train': (not args.no_augment),
    }
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir)
    else:
        seed = config.get('seed', 44)
        model_path = os.path.join(args.save_dir, f'best_{args.task.lower()}_aspennet_model_{seed}.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)