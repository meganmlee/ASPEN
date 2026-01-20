"""
STFT Parameter Combinations Ablation Study for Adaptive SCALE-Net

This script performs comprehensive ablation study by testing combinations of 
STFT parameters (nperseg, noverlap, nfft) to find optimal configurations.

Since testing all combinations can be very expensive, this script provides:
1. Full grid search (all combinations) - expensive but thorough
2. Random search (sample combinations) - faster, good coverage
3. Smart search (prioritize promising combinations) - balanced

Usage:
    # Full grid search (all combinations)
    python ablation_stft_combinations.py --task SSVEP --mode grid --max_configs 100
    
    # Random search (sample 50 random combinations)
    python ablation_stft_combinations.py --task SSVEP --mode random --max_configs 50
    
    # Smart search (prioritize promising combinations)
    python ablation_stft_combinations.py --task SSVEP --mode smart --max_configs 30
"""

import os
import sys
import json
import argparse
import random
import itertools
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Add scale-net directory to path
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale-net')
sys.path.insert(0, scale_net_path)

# Import from scale-net
from scale_net_adaptive import train_task
from dataset import TASK_CONFIGS


# ==================== STFT Parameter Grid Generation ====================

def get_stft_param_combinations(task: str, mode: str = 'smart', max_configs: int = 50) -> List[Dict]:
    """
    Generate STFT parameter combinations for a given task
    
    Args:
        task: Task name
        mode: 'grid' (all combinations), 'random' (random sample), 'smart' (prioritized)
        max_configs: Maximum number of configurations to test
        
    Returns:
        List of STFT parameter configurations to test
    """
    task_config = TASK_CONFIGS.get(task, {})
    sampling_rate = task_config.get('sampling_rate', 250)
    
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
    
    # Default parameters for this task
    default_nperseg = task_config.get('stft_nperseg', 128)
    default_noverlap = task_config.get('stft_noverlap', 112)
    default_nfft = task_config.get('stft_nfft', 512)
    
    # Define parameter ranges
    if sampling_rate <= 256:
        nperseg_options = [n for n in [64, 128, 256] if n <= max_input_length]
        if 512 <= max_input_length:
            nperseg_options.append(512)
    elif sampling_rate <= 512:
        nperseg_options = [n for n in [128, 256, 512] if n <= max_input_length]
        if 1024 <= max_input_length:
            nperseg_options.append(1024)
    else:  # 1000 Hz
        nperseg_options = [n for n in [256, 512, 1024] if n <= max_input_length]
        if 2048 <= max_input_length:
            nperseg_options.append(2048)
    
    # Overlap ratios
    overlap_ratios = [0.5, 0.625, 0.75, 0.875, 0.9375]
    
    # FFT sizes (must be >= nperseg, power of 2)
    nfft_options = [256, 512, 1024, 2048]
    
    # Generate all valid combinations
    all_combinations = []
    
    for nperseg in nperseg_options:
        for overlap_ratio in overlap_ratios:
            noverlap = int(nperseg * overlap_ratio)
            if noverlap >= nperseg:
                continue  # Skip invalid combinations
            
            for nfft in nfft_options:
                if nfft < nperseg:
                    continue  # nfft must be >= nperseg
                
                # Round nfft to nearest power of 2 >= nperseg if needed
                if nfft < nperseg:
                    nfft = 2 ** (nperseg.bit_length() - 1)
                    if nfft < nperseg:
                        nfft = 2 ** nperseg.bit_length()
                
                all_combinations.append({
                    'nperseg': nperseg,
                    'noverlap': noverlap,
                    'nfft': nfft,
                    'overlap_ratio': overlap_ratio,
                })
    
    print(f"Total valid combinations: {len(all_combinations)}")
    
    # Filter combinations based on mode
    if mode == 'grid':
        # Use all combinations (up to max_configs)
        selected = all_combinations[:max_configs]
        print(f"Grid search: Testing {len(selected)} combinations")
        
    elif mode == 'random':
        # Random sample
        random.shuffle(all_combinations)
        selected = all_combinations[:max_configs]
        print(f"Random search: Sampling {len(selected)} random combinations")
        
    elif mode == 'smart':
        # Prioritize promising combinations (target ~10 configurations)
        selected = []
        
        # 1. Add default configuration
        default_cfg = {
            'nperseg': default_nperseg,
            'noverlap': default_noverlap,
            'nfft': default_nfft,
            'overlap_ratio': default_noverlap / default_nperseg if default_nperseg > 0 else 0.875
        }
        if default_cfg in all_combinations:
            selected.append(default_cfg)
        
        # 2. Add key variations around default (limited set)
        # Small nperseg variation with default overlap
        if len(nperseg_options) > 1:
            small_nperseg = min(nperseg_options)
            if small_nperseg != default_nperseg:
                cfg = {
                    'nperseg': small_nperseg,
                    'noverlap': int(small_nperseg * 0.875),
                    'nfft': default_nfft,
                    'overlap_ratio': 0.875
                }
                if cfg not in selected and cfg in all_combinations:
                    selected.append(cfg)
        
        # Large nperseg variation with default overlap
        if len(nperseg_options) > 1:
            large_nperseg = max([n for n in nperseg_options if n <= max_input_length])
            if large_nperseg != default_nperseg:
                cfg = {
                    'nperseg': large_nperseg,
                    'noverlap': int(large_nperseg * 0.875),
                    'nfft': default_nfft,
                    'overlap_ratio': 0.875
                }
                if cfg not in selected and cfg in all_combinations:
                    selected.append(cfg)
        
        # 3. Overlap variations with default nperseg
        for overlap_ratio in [0.5, 0.75, 0.9375]:  # Skip 0.875 (default)
            noverlap = int(default_nperseg * overlap_ratio)
            if noverlap < default_nperseg:
                cfg = {
                    'nperseg': default_nperseg,
                    'noverlap': noverlap,
                    'nfft': default_nfft,
                    'overlap_ratio': overlap_ratio
                }
                if cfg not in selected and cfg in all_combinations:
                    selected.append(cfg)
        
        # 4. nfft variations with default nperseg and overlap
        for nfft in [256, 1024, 2048]:  # Skip 512 (default)
            if nfft >= default_nperseg:
                cfg = {
                    'nperseg': default_nperseg,
                    'noverlap': default_noverlap,
                    'nfft': nfft,
                    'overlap_ratio': default_noverlap / default_nperseg if default_nperseg > 0 else 0.875
                }
                if cfg not in selected and cfg in all_combinations:
                    selected.append(cfg)
        
        # 5. Add a few strategic combinations
        # High frequency resolution (large nfft) with default nperseg
        if 2048 >= default_nperseg:
            cfg = {
                'nperseg': default_nperseg,
                'noverlap': int(default_nperseg * 0.875),
                'nfft': 2048,
                'overlap_ratio': 0.875
            }
            if cfg not in selected and cfg in all_combinations:
                selected.append(cfg)
        
        # Low overlap with high nfft
        if 2048 >= default_nperseg:
            cfg = {
                'nperseg': default_nperseg,
                'noverlap': int(default_nperseg * 0.5),
                'nfft': 2048,
                'overlap_ratio': 0.5
            }
            if cfg not in selected and cfg in all_combinations:
                selected.append(cfg)
        
        # Limit to max_configs (but aim for ~10)
        selected = selected[:max_configs]
        print(f"Smart search: Selected {len(selected)} prioritized combinations")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'grid', 'random', or 'smart'")
    
    # Add names to configurations
    for idx, cfg in enumerate(selected):
        cfg['name'] = f"nperseg{cfg['nperseg']}_overlap{int(cfg['overlap_ratio']*100)}pct_nfft{cfg['nfft']}"
    
    return selected


# ==================== Ablation Study ====================

def run_combination_ablation(task: str, save_dir: str = './ablation_results_combinations',
                             epochs: int = 30, batch_size: int = 16,
                             seed: int = 44, mode: str = 'smart',
                             max_configs: int = 50, verbose: bool = True) -> Dict:
    """
    Run STFT parameter combination ablation study for a task
    
    Args:
        task: Task name
        save_dir: Directory to save results
        epochs: Number of training epochs per configuration
        batch_size: Batch size
        seed: Random seed
        mode: 'grid', 'random', or 'smart'
        max_configs: Maximum number of configurations to test
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with results for all parameter configurations
    """
    print(f"\n{'='*80}")
    print(f"STFT Parameter Combinations Ablation Study: {task}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*80}")
    
    task_config = TASK_CONFIGS.get(task, {})
    if not task_config:
        raise ValueError(f"Unknown task: {task}")
    
    sampling_rate = task_config.get('sampling_rate', 250)
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    # Get parameter combinations
    param_combinations = get_stft_param_combinations(task, mode=mode, max_configs=max_configs)
    print(f"\nTesting {len(param_combinations)} STFT parameter combinations...")
    
    # Print all combinations that will be tested
    print(f"\n{'='*80}")
    print(f"STFT PARAMETER COMBINATIONS TO TEST ({mode.upper()} mode):")
    print(f"{'='*80}")
    for idx, cfg in enumerate(param_combinations, 1):
        print(f"{idx:3d}. {cfg['name']}")
        print(f"     nperseg={cfg['nperseg']:4d}, noverlap={cfg['noverlap']:4d}, nfft={cfg['nfft']:4d}, "
              f"overlap={cfg['overlap_ratio']*100:5.1f}%")
    print(f"{'='*80}")
    
    # Estimate time
    estimated_hours = len(param_combinations) * epochs / 60  # Rough estimate
    print(f"Estimated time: ~{estimated_hours:.1f} hours (assuming {epochs} epochs per config)")
    print(f"{'='*80}")
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for idx, stft_params in enumerate(param_combinations, 1):
        nperseg = stft_params['nperseg']
        noverlap = stft_params['noverlap']
        nfft = stft_params['nfft']
        name = stft_params['name']
        overlap_ratio = stft_params['overlap_ratio']
        
        # Validate parameters
        if noverlap >= nperseg:
            noverlap = max(1, nperseg - 1)
        
        if nfft < nperseg:
            nfft = 2 ** (nperseg.bit_length() - 1)
            if nfft < nperseg:
                nfft = 2 ** nperseg.bit_length()
        
        print(f"\n{'-'*80}")
        print(f"Configuration {idx}/{len(param_combinations)}: {name}")
        print(f"  nperseg: {nperseg} ({nperseg/sampling_rate:.3f} sec)")
        print(f"  noverlap: {noverlap} ({overlap_ratio*100:.1f}% overlap)")
        print(f"  nfft: {nfft}")
        print(f"  Frequency resolution: {sampling_rate/nfft:.2f} Hz/bin")
        print(f"  Time resolution: {(nperseg-noverlap)/sampling_rate:.3f} sec/step")
        print(f"{'-'*80}")
        
        # Create config for training
        config = {
            'seed': seed,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'stft_fs': sampling_rate,
            'stft_nperseg': nperseg,
            'stft_noverlap': noverlap,
            'stft_nfft': nfft,
            # Model parameters
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.3,
            'cnn_dropout': 0.2,
            'use_hidden_layer': False,
            'hidden_dim': 64,
            # Training parameters
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 5,
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
                'val_acc': train_results.get('val', 0.0),
                'val_f1': train_results.get('val_f1', None),
                'val_recall': train_results.get('val_recall', None),
                'val_auc': train_results.get('val_auc', None),
                'test1_acc': train_results.get('test1', 0.0),
                'test1_f1': train_results.get('test1_f1', None),
                'test1_recall': train_results.get('test1_recall', None),
                'test1_auc': train_results.get('test1_auc', None),
                'test1_loss': train_results.get('test1_loss', None),
                'test2_acc': train_results.get('test2', 0.0),
                'test2_f1': train_results.get('test2_f1', None),
                'test2_recall': train_results.get('test2_recall', None),
                'test2_auc': train_results.get('test2_auc', None),
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
                if result.get('val_f1') is not None:
                    print(f"  Val F1: {result['val_f1']:.2f}%, Recall: {result['val_recall']:.2f}%", end="")
                    if result.get('val_auc') is not None:
                        print(f", AUC: {result['val_auc']:.2f}%")
                    else:
                        print()
                if result['test1_acc']:
                    print(f"  Test1 Acc: {result['test1_acc']:.2f}%", end="")
                    if result.get('test1_f1') is not None:
                        print(f" (F1: {result['test1_f1']:.2f}%, Recall: {result['test1_recall']:.2f}%", end="")
                        if result.get('test1_auc') is not None:
                            print(f", AUC: {result['test1_auc']:.2f}%)")
                        else:
                            print(")")
                    else:
                        print()
                if result['test2_acc']:
                    print(f"  Test2 Acc: {result['test2_acc']:.2f}%", end="")
                    if result.get('test2_f1') is not None:
                        print(f" (F1: {result['test2_f1']:.2f}%, Recall: {result['test2_recall']:.2f}%", end="")
                        if result.get('test2_auc') is not None:
                            print(f", AUC: {result['test2_auc']:.2f}%)")
                        else:
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
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(save_dir, f'{task.lower()}_stft_combinations_{mode}_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Sort by validation accuracy
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        successful_results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        # Save top 10 configurations
        top10_file = os.path.join(save_dir, f'{task.lower()}_stft_combinations_{mode}_top10.csv')
        top10_df = pd.DataFrame(successful_results[:10])
        top10_df.to_csv(top10_file, index=False)
        print(f"✓ Top 10 configurations saved to: {top10_file}")
    
    # Save summary
    summary = {
        'task': task,
        'mode': mode,
        'sampling_rate': sampling_rate,
        'total_configs': len(param_combinations),
        'successful_configs': len(successful_results),
        'max_configs': max_configs,
        'best_config': best_config,
        'top10_configs': successful_results[:10] if successful_results else [],
        'all_results': results
    }
    
    summary_file = os.path.join(save_dir, f'{task.lower()}_stft_combinations_{mode}_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")
    
    # Print best configurations
    if successful_results:
        print(f"\n{'='*80}")
        print(f"TOP 5 STFT CONFIGURATIONS for {task} ({mode.upper()} mode):")
        print(f"{'='*80}")
        for rank, result in enumerate(successful_results[:5], 1):
            print(f"\nRank {rank}: {result['config_name']}")
            print(f"  nperseg: {result['nperseg']}, noverlap: {result['noverlap']}, nfft: {result['nfft']}")
            print(f"  Overlap: {result['overlap_ratio']*100:.1f}%, Time step: {result['time_step_sec']:.3f}s")
            print(f"  Val Acc: {result['val_acc']:.2f}%", end="")
        if result.get('val_f1') is not None:
            print(f" | F1: {result['val_f1']:.2f}%, Recall: {result['val_recall']:.2f}%", end="")
            if result.get('val_auc') is not None:
                print(f", AUC: {result['val_auc']:.2f}%")
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


def run_all_tasks_combinations(tasks: Optional[List[str]] = None,
                               save_dir: str = './ablation_results_combinations',
                               epochs: int = 50, batch_size: int = 16,
                               seed: int = 44, mode: str = 'smart',
                               max_configs: int = 10) -> Dict:
    """
    Run STFT combination ablation study for all tasks
    
    Args:
        tasks: List of tasks (default: all tasks in TASK_CONFIGS)
        save_dir: Directory to save results
        epochs: Number of training epochs per configuration
        batch_size: Batch size
        seed: Random seed
        mode: 'grid', 'random', or 'smart'
        max_configs: Maximum number of configurations per task
        
    Returns:
        Dictionary with results for all tasks
    """
    if tasks is None:
        tasks = list(TASK_CONFIGS.keys())
    
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"STFT Parameter Combinations Ablation Study - All Tasks")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*80}")
    print(f"Tasks: {tasks}")
    print(f"Configurations per task: {max_configs}")
    print(f"Estimated time: ~{len(tasks) * max_configs * epochs / 60:.1f} hours")
    print(f"{'='*80}")
    
    for task in tasks:
        try:
            result = run_combination_ablation(
                task=task,
                save_dir=save_dir,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                mode=mode,
                max_configs=max_configs
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
        'mode': mode,
        'tasks': tasks,
        'epochs_per_config': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'max_configs_per_task': max_configs,
        'results': all_results
    }
    
    summary_file = os.path.join(save_dir, f'all_tasks_stft_combinations_{mode}_summary.json')
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
                print(f", AUC: {best['val_auc']:.2f}%")
            else:
                print()
        else:
            print()
        if best.get('test1_acc'):
            print(f"  Test1 Acc: {best['test1_acc']:.2f}%", end="")
            if best.get('test1_f1') is not None:
                print(f" | F1: {best['test1_f1']:.2f}%, Recall: {best['test1_recall']:.2f}%", end="")
                if best.get('test1_auc') is not None:
                    print(f", AUC: {best['test1_auc']:.2f}%")
                else:
                    print()
            else:
                print()
        if best.get('test2_acc'):
            print(f"  Test2 Acc: {best['test2_acc']:.2f}%", end="")
            if best.get('test2_f1') is not None:
                print(f" | F1: {best['test2_f1']:.2f}%, Recall: {best['test2_recall']:.2f}%", end="")
                if best.get('test2_auc') is not None:
                    print(f", AUC: {best['test2_auc']:.2f}%")
                else:
                    print()
            else:
                print()
        else:
            print(f"\n{task}: FAILED")
    print(f"{'='*80}")
    
    return overall_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='STFT Parameter Combinations Ablation Study for Adaptive SCALE-Net'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='SSVEP',
        choices=list(TASK_CONFIGS.keys()) + ['all'],
        help='Task to run ablation on (default: SSVEP)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='smart',
        choices=['grid', 'random', 'smart'],
        help='Search mode: grid (all combinations), random (random sample), smart (prioritized) (default: smart)'
    )
    parser.add_argument(
        '--max_configs',
        type=int,
        default=10,
        help='Maximum number of configurations to test (default: 10)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./ablation_results_combinations',
        help='Directory to save results (default: ./ablation_results_combinations)'
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
        run_all_tasks_combinations(
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            mode=args.mode,
            max_configs=args.max_configs
        )
    else:
        run_combination_ablation(
            task=args.task,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            mode=args.mode,
            max_configs=args.max_configs
        )
