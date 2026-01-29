"""
SCALENet with ResNet - Baseline Model for Multi-Task EEG Classification
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech

Single-stream spectral (STFT) architecture with ResNet-style residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from tqdm import tqdm
import os
import random
from typing import Optional, Dict, Tuple
from seed_utils import seed_everything

# Import from dataset.py
from dataset import (
    load_dataset, 
    TASK_CONFIGS, 
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


class SCALENet(nn.Module):
    """
    SCALE-Net with ResNet: Channel-wise Spectral CNN-LSTM-DNN model
    Single-stream architecture using STFT features with residual blocks
    """
    
    def __init__(self, freq_bins, time_bins, n_channels, n_classes, 
                 cnn_filters=16, lstm_hidden=128, pos_dim=16,
                 dropout=0.3, cnn_dropout=0.2,
                 use_hidden_layer=False, hidden_dim=64):
        """
        Args:
            dropout: Dropout rate for LSTM output and classifier (default: 0.3)
            cnn_dropout: Dropout rate for CNN layers (default: 0.2)
            use_hidden_layer: Whether to add a hidden dense layer after LSTM (default: False)
            hidden_dim: Dimension of hidden layer if use_hidden_layer=True (default: 64)
        """
        super().__init__()
        print(f"[SCALENet-ResNet init] freq_bins={freq_bins}, time_bins={time_bins}, n_channels={n_channels}")
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        
        # ====== Per-channel CNN with ResNet Blocks ======
        
        # Stage 1: Initial Conv + ResNet Blocks
        self.conv1 = nn.Conv2d(1, cnn_filters, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cnn_filters)
        self.se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.dropout_cnn1 = nn.Dropout2d(cnn_dropout)
        
        # Add residual blocks for Stage 1
        self.res1 = SpectralResidualBlock(cnn_filters, dropout=cnn_dropout)
        self.res2 = SpectralResidualBlock(cnn_filters, dropout=cnn_dropout)
        
        self.pool1 = nn.MaxPool2d(2)  # (F, T) → (F/2, T/2)
        
        # Stage 2: Conv + ResNet Blocks
        self.conv2 = nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(cnn_filters * 2)
        self.se2 = SqueezeExcitation(cnn_filters * 2, reduction=4)
        self.dropout_cnn2 = nn.Dropout2d(cnn_dropout)
        
        # Add residual blocks for Stage 2
        self.res3 = SpectralResidualBlock(cnn_filters * 2, dropout=cnn_dropout)
        self.res4 = SpectralResidualBlock(cnn_filters * 2, dropout=cnn_dropout)
        
        self.pool2 = nn.MaxPool2d(2)  # (F/2, T/2) → (F/4, T/4)
        
        # CNN output dimension
        self.cnn_out_dim = (freq_bins // 4) * (time_bins // 4) * (cnn_filters * 2)
        
        # ====== Channel Position Embedding ======
        self.chan_emb = nn.Embedding(n_channels, pos_dim)
        self.pos_projection = nn.Linear(pos_dim, self.cnn_out_dim)
        
        # ====== LSTM across channels ======
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        
        # Dropout after LSTM output
        self.dropout_lstm = nn.Dropout(dropout)
        
        # ====== Optional Hidden Layer ======
        self.use_hidden_layer = use_hidden_layer
        if use_hidden_layer:
            self.hidden_layer = nn.Sequential(
                nn.Linear(lstm_hidden, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_dim
        else:
            classifier_input_dim = lstm_hidden
        
        # ====== Classifier ======
        self.is_binary = (n_classes == 2)
        if self.is_binary:
            self.classifier = nn.Linear(classifier_input_dim, 1)
        else:
            self.classifier = nn.Linear(classifier_input_dim, n_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, chan_ids: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, C, F, T) - Batch, Channels, Freq, Time
            chan_ids: Optional channel indices (default: 0 to C-1)
            
        Returns:
            (B, n_classes) - Classification logits
        """
        B, C, Fr, T = x.shape
        
        if C != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {C}")
        
        # ====== Step 1: Per-channel CNN with ResNet Blocks ======
        x = x.view(B * C, 1, Fr, T)
        
        # Stage 1: Conv + BN + ReLU + SE + Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.se1(x)
        x = self.dropout_cnn1(x)
        
        # Stage 1: Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        
        # Stage 1: Pool
        x = self.pool1(x)
        
        # Stage 2: Conv + BN + ReLU + SE + Dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.se2(x)
        x = self.dropout_cnn2(x)
        
        # Stage 2: Residual blocks
        x = self.res3(x)
        x = self.res4(x)
        
        # Stage 2: Pool
        x = self.pool2(x)
        
        # Flatten
        x = x.view(B, C, -1)  # (B, C, cnn_out_dim)
        
        # ====== Step 2: Add channel position embedding ======
        if chan_ids is None:
            chan_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, C)
        
        pos = self.chan_emb(chan_ids)
        pos = self.pos_projection(pos)
        x = x + pos
        
        # ====== Step 3: LSTM across channels ======
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)
        h = self.dropout_lstm(h)
        
        # ====== Step 4: Optional Hidden Layer ======
        if self.use_hidden_layer:
            h = self.hidden_layer(h)
        
        # ====== Step 5: Classify ======
        return self.classifier(h)

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
        if isinstance(inputs, list):
            inputs = [x.to(device) for x in inputs]
            inputs = inputs[1]  # Use STFT
        else:
            inputs = inputs.to(device)
            
        labels = labels.to(device)

        if is_binary:
            labels_float = labels.float().unsqueeze(1)
        else:
            labels_float = labels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_float)
        
        loss.backward()
        actual_model = unwrap_model(model)
        torch.nn.utils.clip_grad_norm_(actual_model.parameters(), 1.0)
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
            if isinstance(inputs, list):
                inputs = [x.to(device) for x in inputs]
                inputs = inputs[1]  # Use STFT
            else:
                inputs = inputs.to(device)
                
            labels = labels.to(device)

            if is_binary:
                labels_float = labels.float().unsqueeze(1)
            else:
                labels_float = labels
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels_float)
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
            'dropout': 0.3,
            'cnn_dropout': 0.2,
            'use_hidden_layer': False,
            'hidden_dim': 64,
            
            # STFT
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
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('stft_fs', task_config.get('sampling_rate', 250))
        config.setdefault('stft_nperseg', task_config.get('stft_nperseg', 128))
        config.setdefault('stft_noverlap', task_config.get('stft_noverlap', 112))
        config.setdefault('stft_nfft', task_config.get('stft_nfft', 512))
        config.setdefault('dropout', 0.3)
        config.setdefault('cnn_dropout', 0.2)
        config.setdefault('use_hidden_layer', False)
        config.setdefault('hidden_dim', 64)
        config.setdefault('scheduler', 'ReduceLROnPlateau')

    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    device, n_gpus = setup_device()
    print(f"\n{'='*70}")
    print(f"{task} Classification (SCALENet-ResNet)")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    
    # Load Data
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
    
    print(f"\nSTFT Parameters (task-specific):")
    print(f"  Sampling Rate: {stft_config['fs']} Hz")
    print(f"  nperseg: {stft_config['nperseg']} samples ({stft_config['nperseg']/stft_config['fs']:.3f} sec)")
    print(f"  noverlap: {stft_config['noverlap']} samples ({100*stft_config['noverlap']/stft_config['nperseg']:.1f}% overlap)")
    print(f"  nfft: {stft_config['nfft']}")
    print(f"  Frequency resolution: {stft_config['fs']/stft_config['nfft']:.2f} Hz/bin")
    
    # Create Data Loaders
    loaders = create_dataloaders(
        datasets, 
        stft_config, 
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
    _, n_channels, freq_bins, time_bins = sample_x[1].shape
    print(f"STFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    
    # Create Model
    n_classes = config['n_classes']
    model = SCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config.get('dropout', 0.3),
        cnn_dropout=config.get('cnn_dropout', 0.2),
        use_hidden_layer=config.get('use_hidden_layer', False),
        hidden_dim=config.get('hidden_dim', 64)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    # Wrap for multi-GPU
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # Loss & Optimizer
    train_labels = datasets['train'][1]
    if n_classes == 2:
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
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
            optimizer, T_max=config['num_epochs'] // 2, eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # Training Loop
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_{task.lower()}_resnet_model.pth'
    
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
    
    # Final Evaluation
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
    print(f"FINAL RESULTS - {task} (SCALENet-ResNet)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
    print(f"{'='*70}")
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SCALENet-ResNet on EEG tasks')
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
    
    args = parser.parse_args()
    
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
        'patience': 15,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
    }
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir)
    else:
        model_path = os.path.join(args.save_dir, f'best_{args.task.lower()}_resnet_model.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)