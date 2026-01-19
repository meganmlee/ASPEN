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

# Add scale-net directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scale-net'))

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


def evaluate(model, loader, device, criterion=None, is_binary=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
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
            
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    return avg_loss, acc


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
        augment_train=True,
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
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_conformer_{task.lower()}_model.pth'
    
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
                'conformer_config': conformer_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
            }, model_path)
            print(f"✓ Best model saved! ({val_acc:.2f}%)")
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
    print(f"FINAL RESULTS - {task} (EEGConformer)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
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
