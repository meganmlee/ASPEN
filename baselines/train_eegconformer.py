"""
EEGConformer Baseline Model for Multi-Task EEG Classification using braindecode
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech, Lee2019_MI, Lee2019_SSVEP, BNCI2014_P300

Reference: Song et al. (2022) - EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization
https://ieeexplore.ieee.org/document/9991178
Uses braindecode library: https://braindecode.org/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import math
from typing import Optional, Dict, Tuple
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_recall_curve, precision_score

# Add model directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from seed_utils import seed_everything, worker_init_fn, get_generator
from dataset import load_dataset, TASK_CONFIGS

# Import braindecode models
try:
    from braindecode.models import EEGConformer as BraindecodeEEGConformer
    BRAINDECODE_AVAILABLE = True
except ImportError:
    print("Warning: braindecode not installed. Install with: pip install braindecode")
    BRAINDECODE_AVAILABLE = False
    BraindecodeEEGConformer = None


# ==================== EEGConformer Wrapper (using braindecode) ====================

class EEGConformer(nn.Module):
    """
    EEG Conformer wrapper using braindecode's EEGConformer
    
    Reference: Song et al. (2022) - EEG Conformer: Convolutional Transformer for EEG Decoding
    https://ieeexplore.ieee.org/document/9991178
    
    Uses braindecode.models.EEGConformer for the implementation
    """
    
    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 embed_dim: int = 40, n_heads: int = 10, n_layers: int = 6,
                 dim_ff: int = 256, kernel_size: int = 25,
                 pool_size: int = 75, pool_stride: int = 15,
                 dropout: float = 0.5, emb_dropout: float = 0.5):
        """
        Args:
            n_channels: Number of EEG channels
            n_samples: Number of time samples
            n_classes: Number of output classes
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of Transformer layers
            dim_ff: Feed-forward dimension
            kernel_size: Temporal convolution kernel size
            pool_size: Pooling size
            pool_stride: Pooling stride
            dropout: Dropout rate for Transformer
            emb_dropout: Dropout rate for patch embedding
        """
        super().__init__()
        
        if not BRAINDECODE_AVAILABLE:
            raise ImportError("braindecode is required. Install with: pip install braindecode")
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.is_binary = (n_classes == 2)
        
        # Use braindecode's EEGConformer
        # Map our parameters to braindecode's parameters
        # braindecode EEGConformer parameters vary by version:
        # - Newer versions (1.3+): num_layers, num_heads
        # - Older versions (0.8-1.0): att_depth, att_heads
        # Try different parameter name combinations
        try:
            # Try newest version first (num_layers, num_heads)
            self.model = BraindecodeEEGConformer(
                n_chans=n_channels,
                n_outputs=n_classes if not self.is_binary else 1,
                n_times=n_samples,
                n_filters_time=embed_dim,
                filter_time_length=kernel_size,
                pool_time_length=pool_size,
                pool_time_stride=pool_stride,
                drop_prob=dropout,
                num_layers=n_layers,
                num_heads=n_heads,
                att_drop_prob=emb_dropout,
                final_fc_length='auto',
            )
        except TypeError:
            try:
                # Try intermediate version (att_depth, num_heads)
                self.model = BraindecodeEEGConformer(
                    n_chans=n_channels,
                    n_outputs=n_classes if not self.is_binary else 1,
                    n_times=n_samples,
                    n_filters_time=embed_dim,
                    filter_time_length=kernel_size,
                    pool_time_length=pool_size,
                    pool_time_stride=pool_stride,
                    drop_prob=dropout,
                    att_depth=n_layers,
                    num_heads=n_heads,
                    att_drop_prob=emb_dropout,
                    final_fc_length='auto',
                )
            except TypeError:
                # Try oldest version (att_depth, att_heads)
                self.model = BraindecodeEEGConformer(
                    n_chans=n_channels,
                    n_outputs=n_classes if not self.is_binary else 1,
                    n_times=n_samples,
                    n_filters_time=embed_dim,
                    filter_time_length=kernel_size,
                    pool_time_length=pool_size,
                    pool_time_stride=pool_stride,
                    drop_prob=dropout,
                    att_depth=n_layers,
                    att_heads=n_heads,  # Oldest version parameter name
                    att_drop_prob=emb_dropout,
                    final_fc_length='auto',
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, T) or (B, 1, C, T)
            
        Returns:
            Logits of shape (B, n_classes) or (B, 1) for binary
        """
        # braindecode expects (B, C, T) format
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, C, T) -> (B, C, T)
        
        # braindecode models return (B, n_outputs)
        return self.model(x)


# ==================== Raw EEG Dataset ====================

class RawEEGDataset(Dataset):
    """
    Dataset for raw EEG data (no STFT transform)
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 normalize: bool = True, augment: bool = False):
        self.data = data.astype(np.float32)
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation on raw EEG"""
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05 * np.std(x)
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        if self.augment:
            x = self._augment(x)
        
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mean) / std
        
        return torch.FloatTensor(x), y


# ==================== Data Loader Creation ====================

def create_raw_dataloaders(datasets: Dict, batch_size: int = 32, 
                           num_workers: int = 4, augment_train: bool = True, 
                           seed: int = 44) -> Dict:
    """Create DataLoaders for raw EEG data"""
    loaders = {}
    
    for split, (X, y) in datasets.items():
        augment = augment_train if split == 'train' else False
        shuffle = (split == 'train')
        
        ds = RawEEGDataset(X, y, normalize=True, augment=augment)
        loaders[split] = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed),
            generator=get_generator(seed) if shuffle else None
        )
    
    return loaders


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
            # TPR - FPR = Recall - (1 - Specificity)
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
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_binary:
            labels_loss = labels.float().unsqueeze(1)
        else:
            labels_loss = labels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
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
            where metrics_dict contains 'f1', 'recall', 'auc', 'threshold' (if applicable)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation and threshold optimization
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_binary:
                labels_loss = labels.float().unsqueeze(1)
            else:
                labels_loss = labels
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels_loss)
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
                # Multi-class: use macro average of one-vs-rest AUC
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


# ==================== EEGConformer Configuration per Task ====================

def get_conformer_config(task: str, n_channels: int, n_samples: int, sampling_rate: int) -> Dict:
    """
    Get EEGConformer hyperparameters optimized for each task
    """
    # Default configuration
    config = {
        'embed_dim': 40,
        'n_heads': 10,
        'n_layers': 6,
        'dim_ff': 256,
        'kernel_size': max(sampling_rate // 10, 15),  # ~100ms window
        'pool_size': max(sampling_rate // 4, 25),
        'pool_stride': max(sampling_rate // 16, 10),
        'dropout': 0.5,
        'emb_dropout': 0.5,
    }
    
    # Task-specific adjustments
    if task == 'SSVEP' or task == 'Lee2019_SSVEP':
        config['kernel_size'] = max(sampling_rate // 4, 32)
        config['n_layers'] = 4
        config['embed_dim'] = 40
    elif task == 'P300' or task == 'BNCI2014_P300':
        config['kernel_size'] = sampling_rate // 8
        config['n_layers'] = 4
        config['embed_dim'] = 32
    elif task == 'MI' or task == 'Lee2019_MI':
        config['kernel_size'] = 25
        config['n_layers'] = 6
        config['embed_dim'] = 40
    elif task == 'Imagined_speech':
        config['kernel_size'] = 50
        config['pool_size'] = 150
        config['pool_stride'] = 30
        config['n_layers'] = 4
        config['embed_dim'] = 64
    
    # Ensure embed_dim is divisible by n_heads
    if config['embed_dim'] % config['n_heads'] != 0:
        config['n_heads'] = max(1, config['embed_dim'] // 4)
    
    # Ensure pooling doesn't make sequence too short
    min_seq_len = 2
    seq_len = (n_samples - config['pool_size']) // config['pool_stride'] + 1
    while seq_len < min_seq_len and config['pool_stride'] > 1:
        config['pool_stride'] = max(1, config['pool_stride'] // 2)
        config['pool_size'] = max(config['pool_stride'], config['pool_size'] // 2)
        seq_len = (n_samples - config['pool_size']) // config['pool_stride'] + 1
    
    return config


# ==================== Main Training ====================

def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """
    Train EEGConformer for a specific EEG task
    """
    task_config = TASK_CONFIGS.get(task, {})
    
    if config is None:
        config = {
            'data_dir': task_config.get('data_dir', '/ocean/projects/cis250213p/shared/ssvep'),
            'num_seen': task_config.get('num_seen', 33),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 26),
            'sampling_rate': task_config.get('sampling_rate', 250),
            
            # Training
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 1e-4,  # Lower LR for Transformer
            'weight_decay': 1e-4,
            'patience': 15,
            'scheduler': 'CosineAnnealingLR',
        }
    else:
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('sampling_rate', task_config.get('sampling_rate', 250))
        config.setdefault('scheduler', 'CosineAnnealingLR')
        config.setdefault('data_dir', task_config.get('data_dir'))
        config.setdefault('num_seen', task_config.get('num_seen'))

    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    device, n_gpus = setup_device()
    print(f"\n{'='*70}")
    print(f"EEGConformer - {task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    
    # ====== Load Data ======
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir'),
        num_seen=config.get('num_seen'),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # ====== Create Data Loaders ======
    loaders = create_raw_dataloaders(
        datasets, 
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=False,
        seed=seed
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions
    sample_x, _ = next(iter(train_loader))
    n_channels, n_samples = sample_x.shape[1], sample_x.shape[2]
    print(f"Input shape: ({n_channels} channels, {n_samples} samples)")
    
    # ====== Get Conformer Configuration ======
    conformer_config = get_conformer_config(
        task, n_channels, n_samples, config['sampling_rate']
    )
    print(f"\nEEGConformer Configuration:")
    for k, v in conformer_config.items():
        print(f"  {k}: {v}")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = EEGConformer(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **conformer_config
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # ====== Loss & Optimizer ======
    is_binary = (n_classes == 2)
    train_labels = datasets['train'][1]
    if is_binary:
        # Calculate class imbalance
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
        # Only use pos_weight if imbalance ratio > 1.5
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
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler_type = config.get('scheduler', 'CosineAnnealingLR')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'], eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # ====== Training Loop ======
    best_score = -1.0
    best_thr_for_ckpt = 0.5
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_conformer_{task.lower()}_model.pth'
    
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
                'conformer_config': conformer_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
            }, model_path)
            score_name = "PR-AUC" if (is_binary and task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']) else "Acc"
            print(f"✓ Best model saved! ({score_name}={best_score:.2f}, thr={best_thr_for_ckpt:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
        
        if patience_counter >= config['patience']:
            print("\nEarly stopping triggered!")
            break
    
    # ====== Final Evaluation ======
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation...")
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
    print(f"FINAL RESULTS - {task} (EEGConformer)")
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
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints', config: Optional[Dict] = None):
    """Train EEGConformer models for all specified tasks"""
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("EEGConformer - MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_conformer_{task.lower()}_model.pth')
            model, results = train_task(task, config=config, model_path=model_path)
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
    print("SUMMARY RESULTS (EEGConformer)")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        if 'error' in results:
            print(f"\n{task}: FAILED - {results['error']}")
        else:
            print(f"\n{task}:")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc:    {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc:    {results['test2']:.2f}%")
    
    print(f"\n{'='*80}")
    print("EEGConformer MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EEGConformer on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300','all'],
                        help='Task to train on (default: SSVEP)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 15,
        'scheduler': 'CosineAnnealingLR',
        'seed': args.seed,
    }
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir, config=config)
    else:
        model_path = os.path.join(args.save_dir, f'best_conformer_{args.task.lower()}_model.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)
