"""
AdaptiveSCALENet - Adaptive SCALE-Net Model for Multi-Task EEG Classification

Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech
Dual-stream architecture with global attention fusion
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


class TransformerCrossFusion(nn.Module):
    """
    Dual-branch fusion combining:
    1. Cross-Attention: Rich feature interaction between spectral and temporal
    2. Global Weighting: Interpretable stream importance weights
    
    Returns both fused features and (B, 2) shaped weights for CSV compatibility
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, temperature=2.0):
        super().__init__()
        
        # === Cross-Attention Branch ===
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # === Global Weighting Branch (for interpretability) ===
        self.temperature = temperature
        self.global_attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, 2)
        )

    def forward(self, x_s, x_t):
        """
        Args:
            x_s: Spectral features (B, C, Dim)
            x_t: Temporal features (B, C, Dim)
        
        Returns:
            output: Fused features (B, C, Dim)
            weights: Stream importance weights (B, 2) - [Spectral, Temporal]
        """
        # === 1. Cross-Attention Fusion ===
        attn_out, _ = self.mha(query=x_s, key=x_t, value=x_t)
        x_cross = self.norm1(x_s + self.dropout(attn_out))
        x_fused = self.norm2(x_cross + self.dropout(self.ffn(x_cross)))
        
        # === 2. Global Stream Weights (for interpretability) ===
        ctx = torch.cat([
            x_s.mean(dim=1),  # (B, Dim)
            x_t.mean(dim=1)   # (B, Dim)
        ], dim=-1)  # (B, Dim*2)
        
        logits = self.global_attn(ctx) / self.temperature  # (B, 2)
        w = torch.softmax(logits, dim=-1)  # (B, 2)
        
        # === 3. Combine with Global Weights ===
        output = (
            w[:, 0].view(-1, 1, 1) * x_fused +   # Weight for processed spectral
            w[:, 1].view(-1, 1, 1) * x_t          # Weight for original temporal
        )
        
        return output, w  # w is (B, 2): [Spectral_Weight, Temporal_Weight]


class AdaptiveSCALENet(nn.Module):
    """
    Adaptive SCALE-Net: Dual-stream architecture with global attention fusion
    
    Combines spectral (STFT) and temporal (raw) streams with adaptive trial-level weighting.
    
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
                 lstm_hidden=128,
                 pos_dim=16,
                 dropout=0.4,
                 cnn_dropout=0.25,
                 use_hidden_layer=False,
                 hidden_dim=64,
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
        F1 = 8   # F1 filters per temporal kernel
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
            nn.AvgPool2d((1, 8)),
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate temporal stream output dimension
        final_time_dim = (T_raw // 4) // 8
        self.time_out_dim = F2 * final_time_dim
        
        # ====== FEATURE PROJECTION (for fusion) ======
        self.proj_spec = nn.Sequential(
            nn.Linear(self.spec_out_dim, lstm_hidden),
            nn.Dropout(0.2)
        )
        self.proj_time = nn.Sequential(
            nn.Linear(self.time_out_dim, lstm_hidden),
            nn.Dropout(0.2)
        )

        # Global Attention Fusion
        self.fusion_layer = TransformerCrossFusion(
            dim=lstm_hidden,
            num_heads=4,
            dropout=dropout,
            temperature=fusion_temperature
        )
        
        # Channel Position Embedding
        self.chan_emb = nn.Embedding(n_channels, lstm_hidden)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Classifier
        self.use_hidden_layer = use_hidden_layer
        if use_hidden_layer:
            self.hidden_layer = nn.Sequential(
                nn.Linear(lstm_hidden, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            classifier_input = hidden_dim
        else:
            classifier_input = lstm_hidden
        
        self.classifier = nn.Linear(classifier_input, 1 if n_classes == 2 else n_classes)
        
        self._init_weights()

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
        
        # Project to common dimension
        x_s = self.proj_spec(x_s.view(B, C, -1))
        
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
        x_t = self.proj_time(x_t.view(B, -1)).unsqueeze(1).expand(-1, C, -1)
        
        # 3. Global Attention Fusion
        features, weights = self.fusion_layer(x_s, x_t)
        
        # 4. Channel Position Embedding
        if chan_ids is None:
            chan_ids = torch.arange(C, device=features.device).unsqueeze(0).expand(B, C)
        features = features + self.chan_emb(chan_ids)
        
        # 5. LSTM
        _, (h, _) = self.lstm(features)
        h = self.dropout_lstm(h.squeeze(0))
        
        # 6. Classifier
        if self.use_hidden_layer:
            h = self.hidden_layer(h)
        
        return self.classifier(h), weights


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


# ==================== Attention Weight Collection ====================

def collect_weights(model, loader, device):
    """
    Collects attention weights for every trial.
    Works with dual-branch TransformerCrossFusion that returns (B, 2) weights.
    
    Returns:
        weights_array: Shape (N, 2) where N is number of trials
                      Column 0: Spectral weight
                      Column 1: Temporal weight
    """
    model.eval()
    all_weights = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc='Collecting Weights', ncols=100):
            x_time, x_spec = inputs
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            
            _, weights = model(x_time, x_spec)
            
            if weights is not None:
                weights_np = weights.cpu().numpy()
                
                # Should be (B, 2) from dual-branch fusion
                if weights_np.ndim == 2 and weights_np.shape[1] == 2:
                    all_weights.append(weights_np)
                else:
                    print(f"Warning: Unexpected weight shape {weights_np.shape}, expected (B, 2)")
    
    if len(all_weights) == 0:
        return None
    
    return np.concatenate(all_weights, axis=0)



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
            
            # Model
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.4,
            'cnn_dropout': 0.25,
            'use_hidden_layer': False,
            'hidden_dim': 64,
            'fusion_temperature': 2.0,
            
            # STFT - Use task-specific parameters
            'stft_fs': task_config.get('sampling_rate', 250),
            'stft_nperseg': task_config.get('stft_nperseg', 128),
            'stft_noverlap': task_config.get('stft_noverlap', 112),
            'stft_nfft': task_config.get('stft_nfft', 512),
            
            # Training
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'scheduler': 'ReduceLROnPlateau',
        }
    else:
        # Fill in missing keys with task-specific defaults
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('stft_fs', task_config.get('sampling_rate', 250))
        config.setdefault('stft_nperseg', task_config.get('stft_nperseg', 128))
        config.setdefault('stft_noverlap', task_config.get('stft_noverlap', 112))
        config.setdefault('stft_nfft', task_config.get('stft_nfft', 512))
        config.setdefault('dropout', 0.4)
        config.setdefault('cnn_dropout', 0.25)
        config.setdefault('use_hidden_layer', False)
        config.setdefault('hidden_dim', 64)
        config.setdefault('scheduler', 'ReduceLROnPlateau')
        config.setdefault('fusion_temperature', 2.0)
    
    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    # Setup device and multi-GPU
    device, n_gpus = setup_device()
    
    print(f"\n{'='*70}")
    print(f"{task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"Fusion: Global Attention (temperature={config.get('fusion_temperature', 2.0)})")
    
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
    model = AdaptiveSCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        T_raw=T_raw,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config.get('dropout', 0.4),
        cnn_dropout=config.get('cnn_dropout', 0.25),
        use_hidden_layer=config.get('use_hidden_layer', False),
        hidden_dim=config.get('hidden_dim', 64),
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
        model_path = f'best_{task.lower()}_model_{seed}.pth'
    
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
    print(f"FINAL RESULTS - {task}")
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
    print(f"\n{'='*70}")
    print(f"Collecting attention weights for {task}...")
    
    weights_loader = test2_loader if test2_loader else val_loader
    weights_type = 'unseen' if test2_loader else 'val'
    
    if weights_loader:
        all_weights_array = collect_weights(model, weights_loader, device)
        
        if all_weights_array is not None and len(all_weights_array) > 0:
            # Ensure 2D shape (N, 2)
            if all_weights_array.ndim == 3:
                all_weights_array = all_weights_array.mean(axis=1)
            
            # Convert to DataFrame and save
            weights_df = pd.DataFrame(
                all_weights_array,
                columns=['Spectral_Weight', 'Temporal_Weight']
            )
            
            # Generate filename
            if model_path:
                base_name = os.path.basename(model_path)
                if '_stft_' in base_name and '_model.pth' in base_name:
                    config_part = base_name.split('_stft_')[1].replace('_model.pth', '')
                    csv_filename = f'{task.lower()}_attention_weights_{config_part}_{weights_type}.csv'
                else:
                    csv_filename = f'{task.lower()}_attention_weights_{weights_type}.csv'
            else:
                csv_filename = f'{task.lower()}_attention_weights_{weights_type}.csv'
            
            weights_df.to_csv(csv_filename, index=False)
            
            # Print statistics
            mean_spec = weights_df['Spectral_Weight'].mean()
            mean_time = weights_df['Temporal_Weight'].mean()
            print(f"✓ Attention weights (N={len(weights_df)}) saved to: {csv_filename}")
            print(f"  Mean Spectral Weight: {mean_spec:.4f}")
            print(f"  Mean Temporal Weight: {mean_time:.4f}")
        else:
            print(f"⚠ No attention weights collected")
    
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
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}
    
    print("=" * 80)
    print("MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_{task.lower()}_model_44.pth')
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
    print("SUMMARY RESULTS")
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
    
    parser = argparse.ArgumentParser(description='Train AdaptiveSCALENet on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                       choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300', 'all'],
                       help='Task to train on (default: SSVEP)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=44,
                       help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # STFT parameters (optional - uses task defaults if not specified)
    parser.add_argument('--stft_fs', type=int, default=None,
                       help='STFT sampling frequency (Hz). If not specified, uses task default.')
    parser.add_argument('--stft_nperseg', type=int, default=None,
                       help='STFT window length (nperseg). If not specified, uses task default.')
    parser.add_argument('--stft_noverlap', type=int, default=None,
                       help='STFT overlap length (noverlap). If not specified, uses task default.')
    parser.add_argument('--stft_nfft', type=int, default=None,
                       help='STFT FFT length (nfft). If not specified, uses task default.')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable train-time augmentation (default: augmentation ON).')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'cnn_filters': 16,
        'lstm_hidden': 128,
        'pos_dim': 16,
        'dropout': 0.3,
        'cnn_dropout': 0.2,
        'use_hidden_layer': True,
        'hidden_dim': 64,
        'weight_decay': 1e-4,
        'patience': args.patience,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
        'fusion_temperature': 2.0,
        'augment_train': (not args.no_augment)
    }
    
    # Add STFT config if provided (None values will use task defaults)
    if args.stft_fs is not None:
        config['stft_fs'] = args.stft_fs
    if args.stft_nperseg is not None:
        config['stft_nperseg'] = args.stft_nperseg
    if args.stft_noverlap is not None:
        config['stft_noverlap'] = args.stft_noverlap
    if args.stft_nfft is not None:
        config['stft_nfft'] = args.stft_nfft
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir)
    else:
        seed = config.get('seed', 44)
        model_path = os.path.join(args.save_dir, f'best_{args.task.lower()}_model_{seed}.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)