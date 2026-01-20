"""
EEGNet Baseline Model for Multi-Task EEG Classification
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech, Lee2019_MI, Lee2019_SSVEP, BNCI2014_P300

Reference: Lawhern et al. (2018) - EEGNet: A Compact Convolutional Network for EEG-based BCIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import random
import sys
from typing import Optional, Dict, Tuple

# Add scale-net directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scale-net'))

from seed_utils import seed_everything, worker_init_fn, get_generator
from dataset import load_dataset, TASK_CONFIGS


# ==================== EEGNet Model ====================

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
    
    Original paper: Lawhern et al. (2018)
    https://arxiv.org/abs/1611.08024
    
    Architecture:
    1. Temporal Convolution (captures frequency information)
    2. Depthwise Convolution (spatial filtering per channel)
    3. Separable Convolution (combines temporal and spatial features)
    4. Classification layer
    """
    
    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, dropout_rate: float = 0.5,
                 pool1: int = 4, pool2: int = 8):
        """
        Args:
            n_channels: Number of EEG channels
            n_samples: Number of time samples
            n_classes: Number of output classes
            F1: Number of temporal filters
            D: Depth multiplier (number of spatial filters per temporal filter)
            F2: Number of pointwise filters (typically F1 * D)
            kernel_length: Length of temporal convolution kernel (half of sampling rate recommended)
            dropout_rate: Dropout probability
            pool1: Pooling size after depthwise convolution
            pool2: Pooling size after separable convolution
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2 if F2 is not None else F1 * D
        
        # ====== Block 1: Temporal Convolution ======
        # Conv2D: (1, C, T) -> (F1, C, T)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # ====== Block 2: Depthwise Convolution (Spatial Filtering) ======
        # DepthwiseConv2D: (F1, C, T) -> (F1*D, 1, T)
        self.conv2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # Depthwise convolution
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # ====== Block 3: Separable Convolution ======
        # SeparableConv2D: (F1*D, 1, T/pool1) -> (F2, 1, T/pool1)
        sep_kernel = 16  # Separable convolution kernel length
        self.conv3_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, sep_kernel),
            padding=(0, sep_kernel // 2),
            groups=F1 * D,  # Depthwise
            bias=False
        )
        self.conv3_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # ====== Calculate Flattened Size ======
        # After pool1: T / pool1
        # After pool2: T / pool1 / pool2
        self._flat_size = self._get_flat_size(n_channels, n_samples)
        
        # ====== Classifier ======
        self.is_binary = (n_classes == 2)
        if self.is_binary:
            self.classifier = nn.Linear(self._flat_size, 1)
        else:
            self.classifier = nn.Linear(self._flat_size, n_classes)
        
        self._init_weights()
    
    def _get_flat_size(self, n_channels, n_samples):
        """Calculate the flattened feature size"""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.conv3_depthwise(x)
            x = self.conv3_pointwise(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            return x.numel()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, T) or (B, 1, C, T)
            
        Returns:
            Logits of shape (B, n_classes) or (B, 1) for binary
        """
        # Ensure input is (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)
        
        # Block 1: Temporal Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Block 2: Depthwise Convolution (Spatial)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = x.flatten(1)
        return self.classifier(x)


# ==================== Raw EEG Dataset ====================

class RawEEGDataset(Dataset):
    """
    Dataset for raw EEG data (no STFT transform)
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 normalize: bool = True, augment: bool = False):
        """
        Args:
            data: (N, C, T) raw EEG data
            labels: (N,) integer labels
            normalize: Whether to z-score normalize
            augment: Whether to apply data augmentation
        """
        self.data = data.astype(np.float32)
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.augment = augment
        
        # Precompute normalization statistics
        if normalize:
            # Per-channel normalization across all samples
            self.mean = np.mean(data, axis=(0, 2), keepdims=True)
            self.std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
    
    def __len__(self):
        return len(self.data)
    
    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation on raw EEG"""
        # Gaussian noise injection
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05 * np.std(x)
        # Amplitude scaling
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        # Time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        # Augmentation
        if self.augment:
            x = self._augment(x)
        
        # Normalize (per-sample, per-channel)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mean) / std
        
        return torch.FloatTensor(x), y


# ==================== Data Loader Creation ====================

def create_raw_dataloaders(datasets: Dict, batch_size: int = 32, 
                           num_workers: int = 4, augment_train: bool = True, 
                           seed: int = 44) -> Dict:
    """
    Create DataLoaders for raw EEG data
    
    Args:
        datasets: Dictionary from load_dataset() with 'train', 'val', 'test1', 'test2'
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        augment_train: Whether to augment training data
        seed: seed for shuffling
        
    Returns:
        Dictionary of DataLoaders
    """
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

def train_epoch(model, loader, criterion, optimizer, device, is_binary=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Convert labels for binary classification
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
        
        # Prediction
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, criterion=None, is_binary=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels for binary classification
            if is_binary:
                labels_loss = labels.float().unsqueeze(1)
            else:
                labels_loss = labels
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels_loss)
                total_loss += loss.item()
            
            # Prediction
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    return avg_loss, acc


# ==================== EEGNet Configuration per Task ====================

def get_eegnet_config(task: str, n_channels: int, n_samples: int, sampling_rate: int) -> Dict:
    """
    Get EEGNet hyperparameters optimized for each task
    
    Args:
        task: Task name
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of EEGNet configuration
    """
    # Default configuration (works well for most tasks)
    config = {
        'F1': 8,
        'D': 2,
        'F2': 16,
        'kernel_length': max(sampling_rate // 2, 32),  # Half of sampling rate
        'dropout_rate': 0.5,
        'pool1': 4,
        'pool2': 8,
    }
    
    # Task-specific adjustments
    if task == 'SSVEP' or task == 'Lee2019_SSVEP':
        # SSVEP benefits from longer temporal kernels for frequency detection
        config['kernel_length'] = max(sampling_rate // 2, 64)
        config['F1'] = 8
        config['D'] = 2
    elif task == 'P300' or task == 'BNCI2014_P300':
        # P300 has specific temporal patterns
        config['kernel_length'] = sampling_rate // 4
        config['F1'] = 8
        config['D'] = 2
    elif task == 'MI' or task == 'Lee2019_MI':
        # Motor imagery benefits from more spatial filters
        config['F1'] = 8
        config['D'] = 2
        config['kernel_length'] = 64
    elif task == 'Imagined_speech':
        # Imagined speech has high sampling rate
        config['kernel_length'] = min(256, n_samples // 4)
        config['pool1'] = 8
        config['pool2'] = 16
    
    # Adjust pooling based on signal length to avoid size issues
    total_pool = config['pool1'] * config['pool2']
    while n_samples // total_pool < 1:
        if config['pool2'] > 2:
            config['pool2'] //= 2
        elif config['pool1'] > 2:
            config['pool1'] //= 2
        else:
            break
        total_pool = config['pool1'] * config['pool2']
    
    return config


# ==================== Main Training ====================

def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """
    Train EEGNet for a specific EEG task
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech', etc.
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
            'sampling_rate': task_config.get('sampling_rate', 250),
            
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
        config.setdefault('sampling_rate', task_config.get('sampling_rate', 250))
        config.setdefault('scheduler', 'ReduceLROnPlateau')
        config.setdefault('data_dir', task_config.get('data_dir'))
        config.setdefault('num_seen', task_config.get('num_seen'))

    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    # Setup device and multi-GPU
    device, n_gpus = setup_device()
    print(f"\n{'='*70}")
    print(f"EEGNet Baseline - {task} Classification")
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
    
    # ====== Create Data Loaders (Raw EEG) ======
    loaders = create_raw_dataloaders(
        datasets, 
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=True,
        seed=seed
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions from a sample
    sample_x, _ = next(iter(train_loader))
    n_channels, n_samples = sample_x.shape[1], sample_x.shape[2]
    print(f"Input shape: ({n_channels} channels, {n_samples} samples)")
    
    # ====== Get EEGNet Configuration ======
    eegnet_config = get_eegnet_config(
        task, n_channels, n_samples, config['sampling_rate']
    )
    print(f"\nEEGNet Configuration:")
    for k, v in eegnet_config.items():
        print(f"  {k}: {v}")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **eegnet_config
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    # Wrap model for multi-GPU training
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
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler_type = config.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'] // 2, eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_eegnet_{task.lower()}_model.pth'
    
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
                'eegnet_config': eegnet_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
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
    
    results = {'val': best_val_acc}
    
    if test1_loader:
        test1_loss, test1_acc = evaluate(model, test1_loader, device, criterion, is_binary=is_binary)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
    
    if test2_loader:
        test2_loss, test2_acc = evaluate(model, test2_loader, device, criterion, is_binary=is_binary)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task} (EEGNet)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
    print(f"{'='*70}")
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints', config: Optional[Dict] = None):
    """
    Train EEGNet models for all specified tasks
    
    Args:
        tasks: List of task names (default: all tasks)
        save_dir: Directory to save model checkpoints
        config: Training configuration (uses defaults if None)
        
    Returns:
        Dictionary of results for each task
    """
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("EEGNet BASELINE - MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_eegnet_{task.lower()}_model.pth')
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
    print("SUMMARY RESULTS (EEGNet)")
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
    print("EEGNet MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EEGNet baseline on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'EPFL_P300', 'BI2014b_P300', 'all'],
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
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 15,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
    }
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir, config=config)
    else:
        model_path = os.path.join(args.save_dir, f'best_eegnet_{args.task.lower()}_model.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)
