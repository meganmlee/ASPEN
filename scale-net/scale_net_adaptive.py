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
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_recall_curve, precision_score
from torch.utils.data import WeightedRandomSampler

# Import from dataset.py
from dataset import (
    load_dataset,
    TASK_CONFIGS,
    EEGDataset,
    create_dataloaders,
)


# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.5 when using WeightedRandomSampler)
        gamma: Focusing parameter (default: 1.5)
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, alpha=0.5, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, 1) or (N,) - logits from model
            targets: (N, 1) or (N,) - binary labels (0 or 1)
        """
        # Ensure inputs and targets are the right shape
        if inputs.dim() > 1:
            inputs = inputs.squeeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze(1)
        
        # Convert to float
        targets = targets.float()
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-bce_loss)  # p_t when target=1, 1-p_t when target=0
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


class GlobalAttentionFusion(nn.Module):
    """
    Trial-level global attention fusion.
    Dynamically weights spectral and temporal streams per trial.
    """
    def __init__(self, dim, temperature=2.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        # Simple MLP to determine weights for [Spectral, Temporal]
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, x_s, x_t):
        # Average across channels to get trial-level context
        ctx = torch.cat([x_s.mean(1), x_t.mean(1)], dim=-1)  # (B, dim*2)
        
        # Calculate attention weights
        attn_logits = self.attn(ctx) / self.temperature
        w = torch.softmax(attn_logits, dim=-1)  # (B, 2)
        
        # Apply weights to Spectral (0) and Temporal (1) streams
        res = w[:, 0].view(-1, 1, 1) * x_s + w[:, 1].view(-1, 1, 1) * x_t
        return res, w


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
                 dropout=0.5,
                 cnn_dropout=0.3,
                 use_hidden_layer=False,
                 hidden_dim=64,
                 fusion_temperature=2.0):
        super().__init__()
        
        self.n_channels = n_channels
        self.T_raw = T_raw
        
        # ====== SPECTRAL STREAM (2D CNN) ======
        self.spec_cnn_filters = cnn_filters * 2
        
        # Stage 1: Conv(1→16) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv1 = nn.Conv2d(1, cnn_filters, kernel_size=7, padding=3, bias=False)
        self.spec_bn1 = nn.BatchNorm2d(cnn_filters)
        self.spec_se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.spec_dropout_cnn1 = nn.Dropout2d(cnn_dropout)
        self.spec_pool1 = nn.MaxPool2d(2)  # (F, T) → (F/2, T/2)
        
        # Stage 2: Conv(16→32) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv2 = nn.Conv2d(cnn_filters, self.spec_cnn_filters, kernel_size=5, padding=2, bias=False)
        self.spec_bn2 = nn.BatchNorm2d(self.spec_cnn_filters)
        self.spec_se2 = SqueezeExcitation(self.spec_cnn_filters, reduction=4)
        self.spec_dropout_cnn2 = nn.Dropout2d(cnn_dropout)
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
            nn.Dropout(0.3)
        )
        self.proj_time = nn.Sequential(
            nn.Linear(self.time_out_dim, lstm_hidden),
            nn.Dropout(0.3)
        )
        
        # Global Attention Fusion
        self.fusion_layer = GlobalAttentionFusion(dim=lstm_hidden, temperature=fusion_temperature)
        
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
        
        # 1. Spectral Stream
        x_s = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
        x_s = self.spec_pool1(self.spec_se1(F.relu(self.spec_bn1(self.spec_conv1(x_s)))))
        x_s = self.spec_pool2(self.spec_se2(F.relu(self.spec_bn2(self.spec_conv2(x_s)))))
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

def find_best_threshold(y_true, y_prob, mode="f1"):
    """
    Find optimal threshold for binary classification
    
    Args:
        y_true: True labels (numpy array)
        y_prob: Predicted probabilities (numpy array)
        mode: Optimization mode
            - "f1": F1 score maximization
            - "youden": TPR-FPR maximization (Youden's J statistic)
            - "recall_at_precision": Maximize recall with precision >= 0.5
    
    Returns:
        best_threshold, best_score
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if mode == "f1":
            score = f1_score(y_true, y_pred, average="binary", zero_division=0)
        elif mode == "youden":
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr
        elif mode == "recall_at_precision":
            prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
            if prec >= 0.5:
                score = recall_score(y_true, y_pred, average="binary", zero_division=0)
            else:
                score = -1
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t), float(best_score)


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


def evaluate(model, loader, device, criterion=None, is_binary=False, return_metrics=False,
             threshold=0.5, optimize_threshold=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        criterion: Loss function (optional)
        is_binary: Whether this is binary classification
        return_metrics: If True, return additional metrics (f1, recall, auc)
        threshold: Threshold for binary classification (default: 0.5)
        optimize_threshold: If True, find optimal threshold on validation set (only for binary)
    
    Returns:
        If return_metrics=False: (avg_loss, acc)
        If return_metrics=True: (avg_loss, acc, metrics_dict)
            where metrics_dict contains 'f1', 'recall', 'auc', 'pr_auc', 'threshold' (if applicable)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation and threshold optimization
    
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
            
            # Prediction: binary uses sigmoid threshold, multi-class uses argmax
            if is_binary:
                probs = torch.sigmoid(outputs).squeeze(1)  # (B,)
                if return_metrics:
                    all_probs.append(probs.detach().cpu().numpy())
                # Use threshold for prediction (will be re-computed if optimize_threshold=True)
                pred = (probs >= threshold).long()
            else:
                probs = F.softmax(outputs, dim=1)  # (B, n_classes)
                pred = outputs.argmax(1)
                if return_metrics:
                    all_probs.append(probs.detach().cpu().numpy())
            
            if return_metrics:
                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    
    if not return_metrics:
        return avg_loss, acc
    
    # Concatenate all predictions and probabilities
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Optimize threshold for binary classification if requested
    best_thr = threshold
    if is_binary and optimize_threshold:
        best_thr, best_f1 = find_best_threshold(all_labels, all_probs, mode="f1")
        # Recompute predictions with optimal threshold
        all_preds = (all_probs >= best_thr).astype(int)
    else:
        all_preds = np.concatenate(all_preds)
    
    metrics = {}
    
    # F1 score (macro average for multi-class, binary for binary)
    if is_binary:
        metrics["threshold"] = best_thr
        metrics["f1"] = f1_score(all_labels, all_preds, average="binary", zero_division=0) * 100
        metrics["recall"] = recall_score(all_labels, all_preds, average="binary", zero_division=0) * 100
        metrics["precision"] = precision_score(all_labels, all_preds, average="binary", zero_division=0) * 100
        metrics["auc"] = roc_auc_score(all_labels, all_probs) * 100
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs) * 100
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


# ==================== Attention Weight Collection ====================
def collect_weights(model, loader, device):
    """
    Runs an evaluation pass and collects the attention weights for every trial.
    
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
                
                # Handle different weight shapes
                if weights_np.ndim == 2 and weights_np.shape[1] == 2:
                    # Global attention: (B, 2) - already in correct format
                    all_weights.append(weights_np)
                elif weights_np.ndim == 3 and weights_np.shape[2] == 2:
                    # Average over middle dimension if needed
                    weights_avg = weights_np.mean(axis=1)
                    all_weights.append(weights_avg)
    
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
            'dropout': 0.5,
            'cnn_dropout': 0.3,
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
        config.setdefault('dropout', 0.5)
        config.setdefault('cnn_dropout', 0.3)
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
    
    # Get n_classes for sampler creation
    n_classes = config['n_classes']
    
    # ====== Create Data Loaders ======
    # For P300 tasks, use WeightedRandomSampler for balanced batches
    train_sampler = None
    if task in ['P300', 'BNCI2014_P300', 'BI2014b_P300'] and n_classes == 2:
        train_labels = datasets['train'][1]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        print(f"\nUsing WeightedRandomSampler for balanced batches:")
        print(f"  Class 0: {class_counts[0]} samples, weight: {class_weights[0]:.4f}")
        print(f"  Class 1: {class_counts[1]} samples, weight: {class_weights[1]:.4f}")
    
    loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=config.get('augment_train', True),
        seed=seed,
        train_sampler=train_sampler
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
    # n_classes is already defined above
    model = AdaptiveSCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        T_raw=T_raw,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config.get('dropout', 0.5),
        cnn_dropout=config.get('cnn_dropout', 0.3),
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
    # Use Focal Loss for P300 tasks (binary classification with class imbalance)
    # Use Binary Cross Entropy for other binary classification tasks
    # Use Cross Entropy for multi-class classification (n_classes > 2)
    train_labels = datasets['train'][1]
    if n_classes == 2:
        # Calculate class imbalance
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
        # Use Focal Loss for P300 tasks
        if task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']:
            # Focal Loss parameters: alpha for class weighting, gamma for focusing
            # Since we use WeightedRandomSampler, use balanced alpha=0.5 to avoid double reweighting
            # alpha > 0.5 emphasizes positive class, alpha < 0.5 emphasizes negative class
            # With sampler, we want balanced focal loss: alpha=0.5, gamma=1.5
            focal_alpha = config.get('focal_alpha', 0.5)  # Fixed: balanced when using sampler
            focal_gamma = config.get('focal_gamma', 1.5)  # Slightly lower gamma with sampler
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) for P300 task")
            print(f"  Note: Using balanced alpha=0.5 since WeightedRandomSampler handles class imbalance")
        else:
            # For other binary tasks, use BCE with pos_weight if imbalanced
            if class_ratio > 1.5 or class_ratio < 0.67:
                pos_weight = torch.tensor([class_ratio], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"Using BCEWithLogitsLoss with pos_weight={class_ratio:.2f} (imbalanced)")
            else:
                criterion = nn.BCEWithLogitsLoss()
                print(f"Using BCEWithLogitsLoss without pos_weight (balanced dataset)")
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
    best_score = -1.0
    best_thr_for_ckpt = 0.5
    patience_counter = 0
    
    if model_path is None:
        seed = config.get('seed', 44)
        model_path = f'best_{task.lower()}_model_{seed}.pth'
    
    is_binary = (n_classes == 2)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_binary=is_binary)
        
        # Evaluate with threshold optimization for binary classification
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, device, criterion,
            is_binary=is_binary, return_metrics=True,
            threshold=0.5, optimize_threshold=is_binary  # Binary: optimize threshold on val
        )
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Print additional metrics for binary classification
        if is_binary and val_metrics:
            print(f"Val F1: {val_metrics.get('f1', 0):.2f}%, "
                  f"Recall: {val_metrics.get('recall', 0):.2f}%, "
                  f"AUC: {val_metrics.get('auc', 0):.2f}%, "
                  f"PR-AUC: {val_metrics.get('pr_auc', 0):.2f}%, "
                  f"Threshold: {val_metrics.get('threshold', 0.5):.3f}")
        
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
            best_thr_for_ckpt = val_metrics.get("threshold", 0.5) if is_binary else 0.5
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'best_val_acc': val_acc,
                'best_threshold': best_thr_for_ckpt,  # Save optimal threshold
                'task': task,
                'config': config,
            }, model_path)
            score_name = "PR-AUC" if (is_binary and task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']) else "Acc"
            print(f"✓ Best model saved! ({score_name}={best_score:.2f}, thr={best_thr_for_ckpt:.3f})")
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
    
    # Get best threshold from checkpoint (or re-optimize on val)
    best_thr = checkpoint.get("best_threshold", 0.5)
    
    # Re-optimize threshold on validation set for final evaluation
    val_loss, val_acc, val_metrics = evaluate(
        model, val_loader, device, criterion,
        is_binary=is_binary, return_metrics=True,
        threshold=best_thr, optimize_threshold=is_binary  # Re-optimize on val
    )
    best_thr = val_metrics.get("threshold", best_thr) if is_binary else 0.5
    
    results = {
        'val': checkpoint.get('best_val_acc', val_acc),
        'val_f1': val_metrics.get('f1'),
        'val_recall': val_metrics.get('recall'),
        'val_auc': val_metrics.get('auc'),
        'val_pr_auc': val_metrics.get('pr_auc') if is_binary else None,
    }
    
    if test1_loader:
        # Test1: Use best threshold from validation (no optimization)
        test1_loss, test1_acc, test1_metrics = evaluate(
            model, test1_loader, device, criterion,
            is_binary=is_binary, return_metrics=True,
            threshold=best_thr, optimize_threshold=False
        )
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
        results['test1_f1'] = test1_metrics.get('f1')
        results['test1_recall'] = test1_metrics.get('recall')
        results['test1_auc'] = test1_metrics.get('auc')
        results['test1_pr_auc'] = test1_metrics.get('pr_auc') if is_binary else None
    
    if test2_loader:
        # Test2: Use best threshold from validation (no optimization)
        test2_loss, test2_acc, test2_metrics = evaluate(
            model, test2_loader, device, criterion,
            is_binary=is_binary, return_metrics=True,
            threshold=best_thr, optimize_threshold=False
        )
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
        results['test2_f1'] = test2_metrics.get('f1')
        results['test2_recall'] = test2_metrics.get('recall')
        results['test2_auc'] = test2_metrics.get('auc')
        results['test2_pr_auc'] = test2_metrics.get('pr_auc') if is_binary else None
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task}")
    print(f"{'='*70}")
    if is_binary:
        print(f"Best Threshold:  {best_thr:.3f}")
        print(f"Val Acc:         {val_acc:.2f}%")
        if results.get('val_f1') is not None:
            print(f"Val F1:          {results['val_f1']:.2f}%")
            print(f"Val Recall:      {results['val_recall']:.2f}%")
            print(f"Val Precision:   {val_metrics.get('precision', 0):.2f}%")
            if results.get('val_auc') is not None:
                print(f"Val AUC:         {results['val_auc']:.2f}%")
            if results.get('val_pr_auc') is not None:
                print(f"Val PR-AUC:      {results['val_pr_auc']:.2f}%")
    else:
        print(f"Best Val Acc:    {checkpoint.get('best_val_acc', val_acc):.2f}%")
        if results.get('val_f1') is not None:
            print(f"Val F1:          {results['val_f1']:.2f}%")
            print(f"Val Recall:      {results['val_recall']:.2f}%")
            if results.get('val_auc') is not None and results['val_auc'] > 0:
                print(f"Val AUC:         {results['val_auc']:.2f}%")
    if 'test1' in results:
        print(f"\nTest1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
        if results.get('test1_f1') is not None:
            print(f"  Test1 F1:      {results['test1_f1']:.2f}%")
            print(f"  Test1 Recall:  {results['test1_recall']:.2f}%")
            if results.get('test1_auc') is not None:
                print(f"  Test1 AUC:     {results['test1_auc']:.2f}%")
            if results.get('test1_pr_auc') is not None:
                print(f"  Test1 PR-AUC:  {results['test1_pr_auc']:.2f}%")
    if 'test2' in results:
        print(f"\nTest2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
        if results.get('test2_f1') is not None:
            print(f"  Test2 F1:      {results['test2_f1']:.2f}%")
            print(f"  Test2 Recall:  {results['test2_recall']:.2f}%")
            if results.get('test2_auc') is not None:
                print(f"  Test2 AUC:     {results['test2_auc']:.2f}%")
            if results.get('test2_pr_auc') is not None:
                print(f"  Test2 PR-AUC:  {results['test2_pr_auc']:.2f}%")
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
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=44,
                       help='Random seed for reproducibility')
    parser.add_argument('--fusion_temperature', type=float, default=2.0,
                       help='Temperature for attention fusion (default: 2.0)')
    
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
        'dropout': 0.5,
        'cnn_dropout': 0.3,
        'use_hidden_layer': True,
        'hidden_dim': 64,
        'weight_decay': 1e-4,
        'patience': 15,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
        'fusion_temperature': args.fusion_temperature,
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