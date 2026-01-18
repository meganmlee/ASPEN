"""
STFT Parameters Ablation Study for Adaptive SCALE-Net

This script performs ablation study by varying STFT parameters (nperseg, noverlap, nfft)
and evaluates the impact on model accuracy for each task.

Usage:
    python ablation_stft.py --task SSVEP --save_dir ./ablation_results
    python ablation_stft.py --task all --save_dir ./ablation_results
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Add scale-net directory to path
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale-net')
sys.path.insert(0, scale_net_path)

# Import from scale-net
from scale_net_adaptive import train_task
from dataset import TASK_CONFIGS


# ==================== STFT Parameter Configurations ====================

def get_stft_param_grid(task: str) -> List[Dict]:
    """
    Generate STFT parameter grid for a given task
    
    Args:
        task: Task name
        
    Returns:
        List of STFT parameter configurations to test
    """
    task_config = TASK_CONFIGS.get(task, {})
    sampling_rate = task_config.get('sampling_rate', 250)
    
    # Estimate typical input length based on task
    # This is approximate - actual length may vary
    typical_input_lengths = {
        'SSVEP': 250,
        'Lee2019_SSVEP': 1000,  # From preprocessing
        'P300': 256,
        'BNCI2014_P300': 512,
        'MI': 1000,  # 4 seconds at 250 Hz
        'Lee2019_MI': 1000,
        'Imagined_speech': 1000,
    }
    max_input_length = typical_input_lengths.get(task, 250)
    
    # Default parameters for this task
    default_nperseg = task_config.get('stft_nperseg', 128)
    default_noverlap = task_config.get('stft_noverlap', 112)
    default_nfft = task_config.get('stft_nfft', 512)
    
    # Define parameter variations
    # nperseg: window size in samples (affects time resolution)
    # Smaller nperseg = better time resolution, worse frequency resolution
    # Larger nperseg = worse time resolution, better frequency resolution
    nperseg_options = []
    
    # For different sampling rates, adjust nperseg options
    # But ensure nperseg <= max_input_length
    if sampling_rate <= 256:
        nperseg_options = [64, 128, 256]
        nperseg_options = [n for n in nperseg_options if n <= max_input_length]
        if 512 <= max_input_length:
            nperseg_options.append(512)
    elif sampling_rate <= 512:
        nperseg_options = [128, 256, 512]
        nperseg_options = [n for n in nperseg_options if n <= max_input_length]
        if 1024 <= max_input_length:
            nperseg_options.append(1024)
    else:  # 1000 Hz
        nperseg_options = [256, 512, 1024]
        nperseg_options = [n for n in nperseg_options if n <= max_input_length]
        if 2048 <= max_input_length:
            nperseg_options.append(2048)
    
    # noverlap: overlap in samples (affects time resolution)
    # Higher overlap = smoother time resolution, more computation
    # Typical values: 50%, 75%, 87.5%, 93.75% of nperseg
    overlap_ratios = [0.5, 0.75, 0.875, 0.9375]
    
    # nfft: FFT size (affects frequency resolution)
    # Larger nfft = better frequency resolution, more computation
    # Should be >= nperseg, typically power of 2
    nfft_options = [256, 512, 1024, 2048]
    
    # Generate parameter combinations
    param_grid = []
    
    # Add default configuration first
    param_grid.append({
        'nperseg': default_nperseg,
        'noverlap': default_noverlap,
        'nfft': default_nfft,
        'name': 'default'
    })
    
    # Vary nperseg (keep default overlap ratio and nfft)
    default_overlap_ratio = default_noverlap / default_nperseg if default_nperseg > 0 else 0.875
    for nperseg in nperseg_options:
        if nperseg != default_nperseg:
            noverlap = int(nperseg * default_overlap_ratio)
            nfft = max(nperseg, default_nfft)  # nfft should be >= nperseg
            # Round nfft to nearest power of 2 >= nperseg
            nfft = 2 ** (int(nfft).bit_length() - 1) if nfft > nperseg else 2 ** (nperseg.bit_length() - 1)
            param_grid.append({
                'nperseg': nperseg,
                'noverlap': noverlap,
                'nfft': nfft,
                'name': f'nperseg_{nperseg}'
            })
    
    # Vary overlap ratio (keep default nperseg and nfft)
    for overlap_ratio in overlap_ratios:
        if abs(overlap_ratio - default_overlap_ratio) > 0.01:
            noverlap = int(default_nperseg * overlap_ratio)
            param_grid.append({
                'nperseg': default_nperseg,
                'noverlap': noverlap,
                'nfft': default_nfft,
                'name': f'overlap_{int(overlap_ratio*100)}pct'
            })
    
    # Vary nfft (keep default nperseg and overlap)
    for nfft in nfft_options:
        if nfft != default_nfft and nfft >= default_nperseg:
            param_grid.append({
                'nperseg': default_nperseg,
                'noverlap': default_noverlap,
                'nfft': nfft,
                'name': f'nfft_{nfft}'
            })
    
    # Add a few combined variations
    # High time resolution (small nperseg, high overlap)
    small_nperseg = min(nperseg_options)
    param_grid.append({
        'nperseg': small_nperseg,
        'noverlap': int(small_nperseg * 0.875),
        'nfft': max(512, 2 ** (small_nperseg.bit_length() - 1)),
        'name': 'high_time_res'
    })
    
    # High frequency resolution (large nperseg, large nfft)
    large_nperseg = max(nperseg_options)
    large_nfft = max(nfft_options)
    param_grid.append({
        'nperseg': large_nperseg,
        'noverlap': int(large_nperseg * 0.875),
        'nfft': large_nfft,
        'name': 'high_freq_res'
    })
    
    # Balanced (medium nperseg, medium overlap, medium nfft)
    mid_nperseg = nperseg_options[len(nperseg_options) // 2]
    mid_nfft = nfft_options[len(nfft_options) // 2]
    param_grid.append({
        'nperseg': mid_nperseg,
        'noverlap': int(mid_nperseg * 0.75),
        'nfft': mid_nfft,
        'name': 'balanced'
    })
    
    return param_grid


# ==================== Ablation Study ====================

def run_ablation(task: str, save_dir: str = './ablation_results', 
                 epochs: int = 50, batch_size: int = 16, 
                 seed: int = 44, verbose: bool = True) -> Dict:
    """
    Run STFT parameter ablation study for a task
    
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
    print(f"STFT Parameter Ablation Study: {task}")
    print(f"{'='*80}")
    
    task_config = TASK_CONFIGS.get(task, {})
    if not task_config:
        raise ValueError(f"Unknown task: {task}")
    
    sampling_rate = task_config.get('sampling_rate', 250)
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    # Get parameter grid
    param_grid = get_stft_param_grid(task)
    print(f"\nTesting {len(param_grid)} STFT parameter configurations...")
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for idx, stft_params in enumerate(param_grid, 1):
        nperseg = stft_params['nperseg']
        noverlap = stft_params['noverlap']
        nfft = stft_params['nfft']
        name = stft_params['name']
        
        # Estimate typical input length based on task
        typical_input_lengths = {
            'SSVEP': 250,
            'Lee2019_SSVEP': 1000,
            'P300': 256,
            'BNCI2014_P300': 512,
            'MI': 1000,
            'Lee2019_MI': 1000,
            'Imagined_speech': 1000,
        }
        max_input_length = typical_input_lengths.get(task, 250)
        
        # Ensure nperseg doesn't exceed typical input length
        # If nperseg is too large, scipy will auto-adjust, but we want to avoid that
        if nperseg > max_input_length:
            print(f"  Warning: nperseg {nperseg} may be too large for input length ({max_input_length}), skipping...")
            results.append({
                'task': task,
                'config_name': name,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'nfft': nfft,
                'error': f'nperseg {nperseg} too large for typical input length {max_input_length}'
            })
            continue
        
        # Ensure nfft >= nperseg
        if nfft < nperseg:
            nfft = 2 ** (nperseg.bit_length() - 1)
            if nfft < nperseg:
                nfft = 2 ** nperseg.bit_length()
        
        # Ensure noverlap < nperseg (strictly less)
        if noverlap >= nperseg:
            noverlap = max(1, int(nperseg * 0.875))
        
        # Final validation
        if noverlap >= nperseg:
            noverlap = nperseg - 1
        
        print(f"\n{'-'*80}")
        print(f"Configuration {idx}/{len(param_grid)}: {name}")
        print(f"  nperseg: {nperseg} ({nperseg/sampling_rate:.3f} sec)")
        print(f"  noverlap: {noverlap} ({100*noverlap/nperseg:.1f}% overlap)")
        print(f"  nfft: {nfft}")
        print(f"  Frequency resolution: {sampling_rate/nfft:.2f} Hz/bin")
        print(f"{'-'*80}")
        
        # Create config for training with all required parameters
        config = {
            'seed': seed,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'stft_fs': sampling_rate,
            'stft_nperseg': nperseg,
            'stft_noverlap': noverlap,
            'stft_nfft': nfft,
            # Model parameters (using defaults from scale_net_adaptive.py)
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.3,
            'cnn_dropout': 0.2,
            'use_hidden_layer': False,
            'hidden_dim': 64,
            # Training parameters (using defaults from scale_net_adaptive.py)
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'scheduler': 'ReduceLROnPlateau',
        }
        
        # Create unique model path for this configuration
        model_path = os.path.join(
            save_dir, 
            f'{task.lower()}_stft_{name}_model.pth'
        )
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Train model with this STFT configuration
            model, train_results = train_task(
                task=task,
                config=config,
                model_path=model_path
            )
            
            # Extract results
            result = {
                'task': task,
                'config_name': name,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'nfft': nfft,
                'overlap_ratio': noverlap / nperseg,
                'time_resolution_sec': nperseg / sampling_rate,
                'freq_resolution_hz': sampling_rate / nfft,
                'val_acc': train_results.get('val', 0.0),
                'test1_acc': train_results.get('test1', 0.0),
                'test2_acc': train_results.get('test2', 0.0),
                'test1_loss': train_results.get('test1_loss', None),
                'test2_loss': train_results.get('test2_loss', None),
            }
            
            results.append(result)
            
            # Track best configuration
            current_acc = result['val_acc']
            if current_acc > best_acc:
                best_acc = current_acc
                best_config = result.copy()
            
            if verbose:
                print(f"\n✓ Configuration {name} completed:")
                print(f"  Val Acc: {result['val_acc']:.2f}%")
                if result['test1_acc']:
                    print(f"  Test1 Acc: {result['test1_acc']:.2f}%")
                if result['test2_acc']:
                    print(f"  Test2 Acc: {result['test2_acc']:.2f}%")
        
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
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(save_dir, f'{task.lower()}_stft_ablation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save summary
    summary = {
        'task': task,
        'sampling_rate': sampling_rate,
        'total_configs': len(param_grid),
        'successful_configs': len([r for r in results if 'error' not in r]),
        'best_config': best_config,
        'all_results': results
    }
    
    summary_file = os.path.join(save_dir, f'{task.lower()}_stft_ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Print best configuration
    if best_config:
        print(f"\n{'='*80}")
        print(f"BEST STFT CONFIGURATION for {task}:")
        print(f"{'='*80}")
        print(f"  Name: {best_config['config_name']}")
        print(f"  nperseg: {best_config['nperseg']} ({best_config['time_resolution_sec']:.3f} sec)")
        print(f"  noverlap: {best_config['noverlap']} ({best_config['overlap_ratio']*100:.1f}% overlap)")
        print(f"  nfft: {best_config['nfft']}")
        print(f"  Frequency resolution: {best_config['freq_resolution_hz']:.2f} Hz/bin")
        print(f"  Validation Accuracy: {best_config['val_acc']:.2f}%")
        if best_config['test1_acc']:
            print(f"  Test1 Accuracy: {best_config['test1_acc']:.2f}%")
        if best_config['test2_acc']:
            print(f"  Test2 Accuracy: {best_config['test2_acc']:.2f}%")
        print(f"{'='*80}")
    
    return summary


def run_all_tasks_ablation(tasks: Optional[List[str]] = None, 
                          save_dir: str = './ablation_results',
                          epochs: int = 50, batch_size: int = 16,
                          seed: int = 44) -> Dict:
    """
    Run STFT ablation study for all tasks
    
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
    print(f"STFT Parameter Ablation Study - All Tasks")
    print(f"{'='*80}")
    print(f"Tasks: {tasks}")
    print(f"Total configurations per task: ~15-20")
    print(f"Estimated time: ~{len(tasks) * 20 * epochs / 60:.1f} minutes (assuming {epochs} epochs per config)")
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
        'results': all_results
    }
    
    summary_file = os.path.join(save_dir, 'all_tasks_stft_ablation_summary.json')
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
            print(f"  Val Acc: {best['val_acc']:.2f}%")
            if best.get('test1_acc'):
                print(f"  Test1 Acc: {best['test1_acc']:.2f}%")
            if best.get('test2_acc'):
                print(f"  Test2 Acc: {best['test2_acc']:.2f}%")
        else:
            print(f"\n{task}: FAILED")
    print(f"{'='*80}")
    
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='STFT Parameter Ablation Study for Adaptive SCALE-Net'
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
        default=50,
        help='Number of training epochs per configuration (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
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
