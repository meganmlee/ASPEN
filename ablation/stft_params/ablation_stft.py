"""
STFT 27-Config Ablation Study for Adaptive SCALE-Net

This script performs systematic ablation study by testing 27 settings
of STFT parameters (nperseg, noverlap, nfft) to find optimal configurations.

Each parameter has 3 values:
- nperseg: [min, default, max]
- overlap_ratio: [0.5, 0.75, 0.9375]
- nfft: [small, default, large]

Usage:
    python ablation_stft.py --task SSVEP
    python ablation_stft.py --task all
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_recall_curve, precision_score
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

# Add scale-net directory to path
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale_net')
sys.path.insert(0, scale_net_path)

# Import from scale-net
from train_scale_net_v2 import train_task, SCALENet
from dataset import TASK_CONFIGS, load_dataset, create_dataloaders


# ==================== Evaluation with Metrics ====================

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


def evaluate_with_metrics(model, loader, device, is_binary=False, threshold=0.5, optimize_threshold=False):
    """
    Evaluate model and calculate f1, recall, auc metrics
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation
        device: Device to run on
        is_binary: Whether this is binary classification
        threshold: Threshold for binary classification (default: 0.5)
        optimize_threshold: If True, find optimal threshold (only for binary)
        
    Returns:
        Dictionary with 'acc', 'f1', 'recall', 'auc', 'pr_auc', 'threshold' (if applicable)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval Metrics', ncols=100, leave=False):
            if isinstance(inputs, (list, tuple)):
                # Handle (x_time, x_spec) tuple
                x_time, x_spec = inputs
                x_time, x_spec = x_time.to(device), x_spec.to(device)
                outputs = model(x_spec)  # SCALENet only takes spectral input
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            labels = labels.to(device)
            
            # Get predictions and probabilities
            if is_binary:
                probs = torch.sigmoid(outputs).squeeze(1) if outputs.dim() > 1 else torch.sigmoid(outputs)
                all_probs.append(probs.detach().cpu().numpy())
                # Use threshold for prediction (will be re-computed if optimize_threshold=True)
                pred = (probs >= threshold).long()
            else:
                probs = F.softmax(outputs, dim=1)
                _, pred = outputs.max(1)
                all_probs.append(probs.detach().cpu().numpy())
            
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    # Concatenate all predictions and probabilities
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Optimize threshold for binary classification if requested
    best_thr = threshold
    if is_binary and optimize_threshold:
        best_thr, best_f1 = find_best_threshold(all_labels, all_probs, mode="f1")
        # Recompute predictions with optimal threshold
        all_preds = (all_probs >= best_thr).astype(int)
    
    # Calculate accuracy
    acc = 100.0 * correct / total
    
    metrics = {'acc': acc}
    
    # Calculate F1 score
    if is_binary:
        metrics['threshold'] = best_thr
        metrics['f1'] = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100
        metrics['recall'] = recall_score(all_labels, all_preds, average='binary', zero_division=0) * 100
        metrics['precision'] = precision_score(all_labels, all_preds, average='binary', zero_division=0) * 100
    else:
        metrics['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        metrics['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    # Calculate AUC
    try:
        if is_binary:
            metrics['auc'] = roc_auc_score(all_labels, all_probs) * 100
            metrics['pr_auc'] = average_precision_score(all_labels, all_probs) * 100
        else:
            # Multi-class: use one-vs-rest
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
            metrics['pr_auc'] = None
    except Exception as e:
        # If AUC calculation fails (e.g., only one class present), set to None
        metrics['auc'] = None
        metrics['pr_auc'] = None
    
    return metrics


# ==================== STFT 27-Config Generation ====================

def get_stft_param_settings(task: str) -> List[Dict]:
    """
    Generate 27 STFT parameter settings for a given task
    
    Args:
        task: Task name
        
    Returns:
        List of 27 STFT parameter configurations to test
    """
    task_config = TASK_CONFIGS.get(task, {})
    sampling_rate = task_config.get('sampling_rate', 250)
    
    # Estimate typical input length based on task
    typical_input_lengths = {
        'SSVEP': 250,  # 250 Hz: 250 samples = 1 sec
        'Lee2019_SSVEP': 1000,  # 1000 Hz: 1000 samples = 1 sec
        'P300': 256,  # 256 Hz: 256 samples = 1 sec
        'BNCI2014_P300': 256,  # 256 Hz: Fixed - no resampling, use original 256 samples
        'BI2014b_P300': 512,  # 512 Hz: 512 samples = 1 sec
        'MI': 1000,  # 250 Hz: 1000 samples = 4 sec
        'Lee2019_MI': 1000,  # 1000 Hz: 1000 samples = 1 sec
        'Imagined_speech': 1000,  # 1000 Hz: 1000 samples = 1 sec
    }
    max_input_length = typical_input_lengths.get(task, 250)
    
    # Default parameters for this task
    default_nperseg = task_config.get('stft_nperseg', 128)
    default_noverlap = task_config.get('stft_noverlap', 112)
    default_nfft = task_config.get('stft_nfft', 512)
    
    # Define parameter ranges
    if sampling_rate <= 256:
        nperseg_candidates = [64, 128, 256]
        if 512 <= max_input_length:
            nperseg_candidates.append(512)
    elif sampling_rate <= 512:
        nperseg_candidates = [64, 128, 256, 512]
        if 1024 <= max_input_length:
            nperseg_candidates.append(1024)
    else:  # 1000 Hz
        nperseg_candidates = [256, 512, 1024]
        if 2048 <= max_input_length:
            nperseg_candidates.append(2048)
    
    # Filter by max_input_length and ensure default is included
    nperseg_options = [n for n in nperseg_candidates if n <= max_input_length]
    
    # Ensure default_nperseg is included if it's valid
    if default_nperseg <= max_input_length and default_nperseg not in nperseg_options:
        nperseg_options.append(default_nperseg)
        nperseg_options = sorted(nperseg_options)
    
    # Select 3 nperseg values: min, default (or closest), max
    nperseg_values = []
    if len(nperseg_options) == 1:
        nperseg_values = nperseg_options
    elif len(nperseg_options) == 2:
        nperseg_values = nperseg_options
    else:
        # Always include default if possible, otherwise use middle value
        min_val = min(nperseg_options)
        max_val = max([n for n in nperseg_options if n <= max_input_length])
        
        if default_nperseg in nperseg_options:
            mid_val = default_nperseg
        else:
            # Find closest to default
            mid_val = min(nperseg_options, key=lambda x: abs(x - default_nperseg))
        
        nperseg_values = [min_val, mid_val, max_val]
        # Remove duplicates and ensure exactly 3
        nperseg_values = sorted(list(set(nperseg_values)))
        if len(nperseg_values) > 3:
            nperseg_values = [nperseg_values[0], nperseg_values[len(nperseg_values)//2], nperseg_values[-1]]
        elif len(nperseg_values) < 3:
            # Fill missing values
            if len(nperseg_values) == 2:
                # Add middle value
                mid = (nperseg_values[0] + nperseg_values[1]) // 2
                # Find closest in options
                closest = min(nperseg_options, key=lambda x: abs(x - mid))
                nperseg_values.insert(1, closest)
                nperseg_values = sorted(list(set(nperseg_values)))
    
    # Select 3 overlap ratios: low (0.5), medium (0.75), high (0.9375)
    overlap_values = [0.5, 0.75, 0.9375]
    
    # Select 3 nfft values: ensure all are >= max(nperseg_values) to guarantee 27 combinations
    nfft_options = [256, 512, 1024, 2048]
    max_nperseg = max(nperseg_values)
    
    # CRITICAL: Only select nfft values >= max_nperseg to ensure all combinations are valid
    # This guarantees that for all nperseg values, all nfft values will be valid (nfft >= nperseg)
    valid_nfft_options = [n for n in nfft_options if n >= max_nperseg]
    
    # If we don't have enough valid options (>= max_nperseg), we need to expand nfft_options
    if len(valid_nfft_options) < 3:
        # Try to find more options by checking larger values
        larger_options = [n for n in [4096, 8192] if n >= max_nperseg]
        valid_nfft_options.extend(larger_options)
        valid_nfft_options = sorted(list(set(valid_nfft_options)))
    
    # Select 3 nfft values around default, but ensure they're all >= max_nperseg
    nfft_values = []
    if len(valid_nfft_options) >= 3:
        # We have enough options >= max_nperseg
        if default_nfft >= max_nperseg and default_nfft in valid_nfft_options:
            # Default is valid, use it as center
            default_idx = valid_nfft_options.index(default_nfft)
            if default_idx == 0:
                nfft_values = valid_nfft_options[:3]
            elif default_idx == len(valid_nfft_options) - 1:
                nfft_values = valid_nfft_options[-3:]
            else:
                start = max(0, default_idx - 1)
                end = min(len(valid_nfft_options), start + 3)
                nfft_values = valid_nfft_options[start:end]
                if len(nfft_values) < 3:
                    nfft_values = valid_nfft_options[:3]
        else:
            # Default not valid (too small), use first 3 valid options
            nfft_values = valid_nfft_options[:3]
    else:
        # Not enough options >= max_nperseg, use what we have
        nfft_values = valid_nfft_options
    
    # Ensure we have exactly 3 nfft values, all >= max_nperseg
    if len(nfft_values) < 3:
        # This should not happen if max_nperseg is reasonable, but handle it
        # Try to use larger nfft values
        min_needed = max_nperseg
        larger_nfft = [n for n in nfft_options if n >= min_needed and n not in nfft_values]
        nfft_values.extend(larger_nfft[:3 - len(nfft_values)])
        nfft_values = sorted(nfft_values)
        
        # If still not enough, we have a problem - but try to proceed
        if len(nfft_values) < 3:
            print(f"WARNING: Only {len(nfft_values)} valid nfft values found for max_nperseg={max_nperseg}")
            print(f"  This will result in fewer than 27 combinations!")
    
    # Generate all 3×3×3 = 27 settings
    selected = []
    for nperseg in nperseg_values:
        for overlap_ratio in overlap_values:
            noverlap = int(nperseg * overlap_ratio)
            # CRITICAL: Ensure noverlap < nperseg (strict inequality required by scipy)
            if noverlap >= nperseg:
                # Adjust to maximum valid value
                noverlap = max(1, nperseg - 1)
                # Recalculate actual overlap ratio
                actual_overlap_ratio = noverlap / nperseg
                print(f"WARNING: Generated noverlap ({int(nperseg * overlap_ratio)}) >= nperseg ({nperseg})")
                print(f"  Adjusting to noverlap={noverlap}, actual overlap={actual_overlap_ratio*100:.1f}%")
            else:
                actual_overlap_ratio = overlap_ratio
            
            for nfft in nfft_values:
                if nfft < nperseg:
                    continue  # Skip invalid (nfft must be >= nperseg)
                
                selected.append({
                    'nperseg': nperseg,
                    'noverlap': noverlap,  # Use validated noverlap
                    'nfft': nfft,
                    'overlap_ratio': actual_overlap_ratio,  # Use actual overlap ratio
                })
    
    # Add names to configurations
    for idx, cfg in enumerate(selected):
        cfg['name'] = f"nperseg{cfg['nperseg']}_overlap{int(cfg['overlap_ratio']*100)}pct_nfft{cfg['nfft']}"
    
    print(f"Generated {len(selected)} settings (3×3×3 grid)")
    
    return selected


# ==================== Ablation Study ====================

def run_ablation(task: str, save_dir: str = './ablation_results',
                             epochs: int = 40, batch_size: int = 64,
                             seed: int = 44, verbose: bool = True) -> Dict:
    """
    Run STFT parameter ablation study for a task (27 settings)
    
    Args:
        task: Task name
        save_dir: Directory to save results
        epochs: Number of training epochs per configuration
        batch_size: Batch size
        seed: Random seed
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with results for all parameter configurations
    """
    print(f"\n{'='*80}")
    print(f"STFT 27-Config Ablation Study: {task}")
    print(f"{'='*80}")
    
    task_config = TASK_CONFIGS.get(task, {})
    if not task_config:
        raise ValueError(f"Unknown task: {task}")
    
    sampling_rate = task_config.get('sampling_rate', 250)
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    # Get parameter settings (always 27)
    param_settings = get_stft_param_settings(task)
    print(f"\nTesting {len(param_settings)} STFT parameter settings...")
    
    # Print all settings that will be tested
    print(f"\n{'='*80}")
    print(f"STFT PARAMETER SETTINGS TO TEST (27 settings):")
    print(f"{'='*80}")
    for idx, cfg in enumerate(param_settings, 1):
        print(f"{idx:3d}. {cfg['name']}")
        print(f"     nperseg={cfg['nperseg']:4d}, noverlap={cfg['noverlap']:4d}, nfft={cfg['nfft']:4d}, "
              f"overlap={cfg['overlap_ratio']*100:5.1f}%")
    print(f"{'='*80}")
    
    # Estimate time
    estimated_hours = len(param_settings) * epochs / 60  # Rough estimate
    print(f"Estimated time: ~{estimated_hours:.1f} hours (assuming {epochs} epochs per config)")
    print(f"{'='*80}")
    
    results_file = os.path.join(save_dir, f'{task.lower()}_stft_27_results.csv')
    results = []
    completed_names = set()

    if os.path.exists(results_file):
        try:
            existing_df = pd.read_csv(results_file)
            results = existing_df.to_dict('records')
            # Only consider it "complete" if there was no error
            completed_names = {r['config_name'] for r in results if 'error' not in str(r.get('error', ''))}
            print(f"Found existing results. Resuming from {len(completed_names)} completed configs.")
        except Exception as e:
            print(f"Could not load existing CSV, starting fresh. Error: {e}")

    best_acc = -1.0  # Changed to -1.0 to handle PR-AUC which can be 0-100
    best_config = None
    
    # Determine if this is a P300 task for best config selection
    n_classes = task_config.get('num_classes', 26)
    is_binary_p300 = (n_classes == 2) and (task in ['P300', 'BNCI2014_P300', 'BI2014b_P300'])
    
    for idx, stft_params in enumerate(param_settings, 1):
        nperseg = stft_params['nperseg']
        noverlap = stft_params['noverlap']
        nfft = stft_params['nfft']
        name = stft_params['name']
        overlap_ratio = stft_params['overlap_ratio']

        if name in completed_names:
            print(f"Skipping {name} (already completed).")
            continue
        
        # Validate parameters BEFORE using them
        # Fix noverlap if it's >= nperseg (should not happen, but safety check)
        original_noverlap = noverlap
        if noverlap >= nperseg:
            noverlap = max(1, nperseg - 1)
            print(f"WARNING: noverlap ({original_noverlap}) >= nperseg ({nperseg}), adjusting to {noverlap}")
        
        # Fix nfft if it's < nperseg
        original_nfft = nfft
        if nfft < nperseg:
            nfft = 2 ** (nperseg.bit_length() - 1)
            if nfft < nperseg:
                nfft = 2 ** nperseg.bit_length()
            print(f"WARNING: nfft ({original_nfft}) < nperseg ({nperseg}), adjusting to {nfft}")
        
        # Recalculate overlap_ratio based on corrected noverlap
        corrected_overlap_ratio = noverlap / nperseg if nperseg > 0 else 0.5
        
        print(f"\n{'-'*80}")
        print(f"Configuration {idx}/{len(param_settings)}: {name}")
        print(f"  nperseg: {nperseg} ({nperseg/sampling_rate:.3f} sec)")
        print(f"  noverlap: {noverlap} (original: {original_noverlap}, overlap: {corrected_overlap_ratio*100:.1f}%)")
        print(f"  nfft: {nfft} (original: {original_nfft})")
        print(f"  Frequency resolution: {sampling_rate/nfft:.2f} Hz/bin")
        print(f"  Time resolution: {(nperseg-noverlap)/sampling_rate:.3f} sec/step")
        print(f"{'-'*80}")
        
        # Create config for training (use VALIDATED parameters)
        config = {
            'seed': seed,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'stft_fs': sampling_rate,
            'stft_nperseg': nperseg,
            'stft_noverlap': noverlap,  # Use validated noverlap
            'stft_nfft': nfft,  # Use validated nfft
            # Model parameters
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.3,
            'cnn_dropout': 0.2,
            'use_hidden_layer': False,
            'hidden_dim': 64,
            # Training parameters
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'patience': 10,
            'scheduler': 'ReduceLROnPlateau',
        }
        
        # Create unique model path
        model_path = os.path.join(
            save_dir,
            f'{task.lower()}_stft_{name}_model.pth'
        )
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Train model
            model, train_results = train_task(
                task=task,
                config=config,
                model_path=model_path
            )
            
            # Load data and create loaders for metric calculation
            datasets = load_dataset(
                task=task,
                data_dir=task_config.get('data_dir'),
                num_seen=task_config.get('num_seen'),
                seed=seed
            )
            
            # Get n_classes for sampler creation
            n_classes = task_config.get('num_classes', 26)
            is_binary = (n_classes == 2)
            
            # Use VALIDATED parameters for stft_config (same as config above)
            stft_config = {
                'fs': sampling_rate,
                'nperseg': nperseg,
                'noverlap': noverlap,  # Use validated noverlap, not original from stft_params
                'nfft': nfft  # Use validated nfft, not original from stft_params
            }
            
            # For P300 tasks, use WeightedRandomSampler for balanced batches
            train_sampler = None
            if task in ['P300', 'BNCI2014_P300', 'BI2014b_P300'] and is_binary:
                train_labels = datasets['train'][1]
                class_counts = np.bincount(train_labels)
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[train_labels]
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_labels),
                    replacement=True
                )
            
            loaders = create_dataloaders(
                datasets,
                stft_config,
                batch_size=batch_size,
                num_workers=0,  # Use 0 to avoid multiprocessing issues
                augment_train=False,
                seed=seed,
                train_sampler=train_sampler
            )
            
            # Get model dimensions
            sample_x, _ = next(iter(loaders['train']))
            if isinstance(sample_x, (list, tuple)):
                _, n_channels, freq_bins, time_bins = sample_x[1].shape
            else:
                _, n_channels, freq_bins, time_bins = sample_x.shape
            
            # Reload best model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model and load weights
            eval_model = SCALENet(
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
            
            eval_model.load_state_dict(checkpoint['model_state_dict'])
            eval_model.eval()
            
            # Get best threshold from checkpoint (or use 0.5)
            best_thr = checkpoint.get("best_threshold", 0.5)
            
            # Calculate metrics for each split with threshold optimization on val
            val_metrics = evaluate_with_metrics(
                eval_model, loaders['val'], device, 
                is_binary=is_binary, 
                threshold=best_thr, 
                optimize_threshold=is_binary  # Re-optimize on val
            )
            best_thr = val_metrics.get("threshold", best_thr) if is_binary else 0.5
            
            test1_metrics = None
            test2_metrics = None
            
            if 'test1' in loaders:
                # Test1: Use best threshold from validation (no optimization)
                test1_metrics = evaluate_with_metrics(
                    eval_model, loaders['test1'], device, 
                    is_binary=is_binary,
                    threshold=best_thr,
                    optimize_threshold=False
                )
            
            if 'test2' in loaders:
                # Test2: Use best threshold from validation (no optimization)
                test2_metrics = evaluate_with_metrics(
                    eval_model, loaders['test2'], device, 
                    is_binary=is_binary,
                    threshold=best_thr,
                    optimize_threshold=False
                )
            
            # Extract results
            result = {
                'task': task,
                'config_name': name,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'nfft': nfft,
                'overlap_ratio': overlap_ratio,
                'time_resolution_sec': nperseg / sampling_rate,
                'time_step_sec': (nperseg - noverlap) / sampling_rate,
                'freq_resolution_hz': sampling_rate / nfft,
                'val_acc': val_metrics['acc'],
                'val_f1': val_metrics.get('f1'),
                'val_recall': val_metrics.get('recall'),
                'val_auc': val_metrics.get('auc'),
                'val_pr_auc': val_metrics.get('pr_auc'),
                'val_threshold': val_metrics.get('threshold') if is_binary else None,
                'test1_acc': test1_metrics['acc'] if test1_metrics else None,
                'test1_f1': test1_metrics.get('f1') if test1_metrics else None,
                'test1_recall': test1_metrics.get('recall') if test1_metrics else None,
                'test1_auc': test1_metrics.get('auc') if test1_metrics else None,
                'test1_pr_auc': test1_metrics.get('pr_auc') if test1_metrics else None,
                'test1_loss': train_results.get('test1_loss', None),
                'test2_acc': test2_metrics['acc'] if test2_metrics else None,
                'test2_f1': test2_metrics.get('f1') if test2_metrics else None,
                'test2_recall': test2_metrics.get('recall') if test2_metrics else None,
                'test2_auc': test2_metrics.get('auc') if test2_metrics else None,
                'test2_pr_auc': test2_metrics.get('pr_auc') if test2_metrics else None,
                'test2_loss': train_results.get('test2_loss', None),
            }
            
            results.append(result)
            pd.DataFrame(results).to_csv(results_file, index=False)
            
            # Track best configuration: Use PR-AUC for P300 tasks, val_acc for others
            if is_binary and task in ['P300', 'BNCI2014_P300', 'BI2014b_P300']:
                # Use PR-AUC (preferred) or AUC as the selection criterion
                current_score = result.get('val_pr_auc', -1)
                if current_score is None or current_score < 0:
                    current_score = result.get('val_auc', -1)
            else:
                current_score = result['val_acc']
            
            if current_score > best_acc:
                best_acc = current_score
                best_config = result.copy()
            
            if verbose:
                print(f"\n✓ Configuration {name} completed:")
                print(f"  Val Acc: {result['val_acc']:.2f}%")
                if result.get('val_f1') is not None:
                    print(f"  Val F1: {result['val_f1']:.2f}%, Recall: {result['val_recall']:.2f}%", end="")
                    if result.get('val_auc') is not None:
                        print(f", AUC: {result['val_auc']:.2f}%", end="")
                    if result.get('val_pr_auc') is not None:
                        print(f", PR-AUC: {result['val_pr_auc']:.2f}%", end="")
                    if result.get('val_threshold') is not None:
                        print(f", Threshold: {result['val_threshold']:.3f}", end="")
                    print()
                if result['test1_acc']:
                    print(f"  Test1 Acc: {result['test1_acc']:.2f}%", end="")
                    if result.get('test1_f1') is not None:
                        print(f" (F1: {result['test1_f1']:.2f}%, Recall: {result['test1_recall']:.2f}%", end="")
                        if result.get('test1_auc') is not None:
                            print(f", AUC: {result['test1_auc']:.2f}%", end="")
                        if result.get('test1_pr_auc') is not None:
                            print(f", PR-AUC: {result['test1_pr_auc']:.2f}%", end="")
                        print(")")
                    else:
                        print()
                if result['test2_acc']:
                    print(f"  Test2 Acc: {result['test2_acc']:.2f}%", end="")
                    if result.get('test2_f1') is not None:
                        print(f" (F1: {result['test2_f1']:.2f}%, Recall: {result['test2_recall']:.2f}%", end="")
                        if result.get('test2_auc') is not None:
                            print(f", AUC: {result['test2_auc']:.2f}%", end="")
                        if result.get('test2_pr_auc') is not None:
                            print(f", PR-AUC: {result['test2_pr_auc']:.2f}%", end="")
                        print(")")
                    else:
                        print()
        
        except Exception as e:
            print(f"\n✗ Configuration {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'task': task,
                'config_name': name,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'nfft': nfft,
                'error': str(e)
            })
            pd.DataFrame(results).to_csv(results_file, index=False)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Sort by validation accuracy or PR-AUC (for P300 tasks)
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        if is_binary_p300:
            # For P300 tasks, sort by PR-AUC (preferred) or AUC
            def get_score(x):
                score = x.get('val_pr_auc', -1)
                if score is None or score < 0:
                    score = x.get('val_auc', -1)
                return score if score is not None else -1
            successful_results.sort(key=get_score, reverse=True)
        else:
            successful_results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        # Save top 10 configurations
        top10_file = os.path.join(save_dir, f'{task.lower()}_stft_27_top10.csv')
        top10_df = pd.DataFrame(successful_results[:10])
        top10_df.to_csv(top10_file, index=False)
        print(f"✓ Top 10 configurations saved to: {top10_file}")
    
    # Save summary
    summary = {
        'task': task,
        'sampling_rate': sampling_rate,
        'total_configs': len(param_settings),
        'successful_configs': len(successful_results),
        'best_config': best_config,
        'top10_configs': successful_results[:10] if successful_results else [],
        'all_results': results
    }
    
    summary_file = os.path.join(save_dir, f'{task.lower()}_stft_27_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Print best configurations
    if successful_results:
        print(f"\n{'='*80}")
        print(f"TOP 5 STFT CONFIGURATIONS for {task}:")
        print(f"{'='*80}")
        for rank, result in enumerate(successful_results[:5], 1):
            print(f"\nRank {rank}: {result['config_name']}")
            print(f"  nperseg: {result['nperseg']}, noverlap: {result['noverlap']}, nfft: {result['nfft']}")
            print(f"  Overlap: {result['overlap_ratio']*100:.1f}%, Time step: {result['time_step_sec']:.3f}s")
            print(f"  Val Acc: {result['val_acc']:.2f}%", end="")
            if result.get('val_f1') is not None:
                print(f" | F1: {result['val_f1']:.2f}%, Recall: {result['val_recall']:.2f}%", end="")
                if result.get('val_auc') is not None:
                    print(f", AUC: {result['val_auc']:.2f}%", end="")
                if result.get('val_pr_auc') is not None:
                    print(f", PR-AUC: {result['val_pr_auc']:.2f}%", end="")
                if result.get('val_threshold') is not None:
                    print(f", Threshold: {result['val_threshold']:.3f}", end="")
                print()
            else:
                print()
        else:
            print()
        if result['test1_acc']:
            print(f"  Test1 Acc: {result['test1_acc']:.2f}%", end="")
            if result.get('test1_f1') is not None:
                print(f" | F1: {result['test1_f1']:.2f}%, Recall: {result['test1_recall']:.2f}%", end="")
                if result.get('test1_auc') is not None:
                    print(f", AUC: {result['test1_auc']:.2f}%")
                else:
                    print()
            else:
                print()
        if result['test2_acc']:
            print(f"  Test2 Acc: {result['test2_acc']:.2f}%", end="")
            if result.get('test2_f1') is not None:
                print(f" | F1: {result['test2_f1']:.2f}%, Recall: {result['test2_recall']:.2f}%", end="")
                if result.get('test2_auc') is not None:
                    print(f", AUC: {result['test2_auc']:.2f}%")
                else:
                    print()
            else:
                print()
        print(f"{'='*80}")
    
    return summary


def run_all_tasks_ablation(tasks: Optional[List[str]] = None,
                               save_dir: str = './ablation_results',
                               epochs: int = 50, batch_size: int = 64,
                               seed: int = 44) -> Dict:
    """
    Run STFT ablation study for all tasks (27 settings per task)
    
    Args:
        tasks: List of tasks (default: all tasks in TASK_CONFIGS)
        save_dir: Directory to save results
        epochs: Number of training epochs per configuration
        batch_size: Batch size
        seed: Random seed
        
    Returns:
        Dictionary with results for all tasks
    """
    if tasks is None:
        tasks = list(TASK_CONFIGS.keys())
    
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"STFT 27-Config Ablation Study - All Tasks")
    print(f"{'='*80}")
    print(f"Tasks: {tasks}")
    print(f"Configurations per task: 27")
    print(f"Estimated time: ~{len(tasks) * 27 * epochs / 60:.1f} hours")
    print(f"{'='*80}")
    
    for task in tasks:
        try:
            result = run_ablation(
                task=task,
                save_dir=save_dir,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed
            )
            all_results[task] = result
        except Exception as e:
            print(f"\n✗ Failed to run ablation for {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {'error': str(e)}
    
    # Create overall summary
    overall_summary = {
        'timestamp': datetime.now().isoformat(),
        'tasks': tasks,
        'epochs_per_config': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'configs_per_task': 27,
        'results': all_results
    }
    
    summary_file = os.path.join(save_dir, 'all_tasks_stft_27_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\n✓ Overall summary saved to: {summary_file}")
    
    # Print best configurations for each task
    print(f"\n{'='*80}")
    print("BEST STFT CONFIGURATIONS BY TASK:")
    print(f"{'='*80}")
    for task, result in all_results.items():
        if 'error' not in result and 'best_config' in result:
            best = result['best_config']
            print(f"\n{task}:")
            print(f"  Config: {best['config_name']}")
            print(f"  Params: nperseg={best['nperseg']}, noverlap={best['noverlap']}, nfft={best['nfft']}")
            print(f"  Val Acc: {best['val_acc']:.2f}%", end="")
            if best.get('val_f1') is not None:
                print(f" | F1: {best['val_f1']:.2f}%, Recall: {best['val_recall']:.2f}%", end="")
                if best.get('val_auc') is not None:
                    print(f", AUC: {best['val_auc']:.2f}%", end="")
                if best.get('val_pr_auc') is not None:
                    print(f", PR-AUC: {best['val_pr_auc']:.2f}%", end="")
                if best.get('val_threshold') is not None:
                    print(f", Threshold: {best['val_threshold']:.3f}", end="")
                print()
            else:
                print()
            if best.get('test1_acc'):
                print(f"  Test1 Acc: {best['test1_acc']:.2f}%", end="")
                if best.get('test1_f1') is not None:
                    print(f" | F1: {best['test1_f1']:.2f}%, Recall: {best['test1_recall']:.2f}%", end="")
                    if best.get('test1_auc') is not None:
                        print(f", AUC: {best['test1_auc']:.2f}%", end="")
                    if best.get('test1_pr_auc') is not None:
                        print(f", PR-AUC: {best['test1_pr_auc']:.2f}%", end="")
                    print()
                else:
                    print()
            if best.get('test2_acc'):
                print(f"  Test2 Acc: {best['test2_acc']:.2f}%", end="")
                if best.get('test2_f1') is not None:
                    print(f" | F1: {best['test2_f1']:.2f}%, Recall: {best['test2_recall']:.2f}%", end="")
                    if best.get('test2_auc') is not None:
                        print(f", AUC: {best['test2_auc']:.2f}%", end="")
                    if best.get('test2_pr_auc') is not None:
                        print(f", PR-AUC: {best['test2_pr_auc']:.2f}%", end="")
                    print()
                else:
                    print()
            print(f"\n{task}: FAILED")
    print(f"{'='*80}")
    
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='STFT 27-Config Ablation Study for Adaptive SCALE-Net (27 settings)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='SSVEP',
        choices=list(TASK_CONFIGS.keys()) + ['all'],
        help='Task to run ablation on (default: SSVEP)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./ablation_results',
        help='Directory to save results (default: ./ablation_results)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help='Number of training epochs per configuration (default: 40)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=44,
        help='Random seed (default: 44)'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.task == 'all':
        run_all_tasks_ablation(
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed
        )
    else:
        run_ablation(
            task=args.task,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed
        )
