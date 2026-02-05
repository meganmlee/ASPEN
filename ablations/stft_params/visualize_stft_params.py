"""
Visualize STFT Parameters - Generate Spectrogram Images

This script generates spectrogram images for different STFT parameter configurations
to visually compare how different settings affect the frequency-time representation.

Usage:
    python visualize_stft_params.py --task SSVEP --save_dir ./stft_visualizations
    python visualize_stft_params.py --task SSVEP --save_dir ./stft_visualizations --sample_idx 0 --channel_idx 0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import argparse
from typing import Dict, List, Tuple

# Add model directory to path
model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model')
sys.path.insert(0, model_path)

# Import from model
from dataset import TASK_CONFIGS, load_dataset, apply_stft_transform
from ablation_stft import get_stft_param_grid


def visualize_stft_config(data_sample: np.ndarray, stft_config: Dict, 
                         task: str, config_name: str, save_path: str,
                         channel_idx: int = 0):
    """
    Visualize STFT for a single configuration
    
    Args:
        data_sample: Raw EEG sample (C, T)
        stft_config: STFT parameters dict with 'fs', 'nperseg', 'noverlap', 'nfft'
        task: Task name
        config_name: Configuration name
        save_path: Path to save the image
        channel_idx: Which channel to visualize (default: 0)
    """
    # Apply STFT transform
    stft_result = apply_stft_transform(
        data_sample[np.newaxis, :, :],  # Add batch dimension
        fs=stft_config['fs'],
        nperseg=stft_config['nperseg'],
        noverlap=stft_config['noverlap'],
        nfft=stft_config['nfft']
    )
    
    # Remove batch dimension if present
    if stft_result.ndim == 4:
        stft_result = stft_result[0]  # (C, F, T)
    
    # Get spectrogram for selected channel
    spectrogram = stft_result[channel_idx]  # (F, T)
    
    # Get frequency and time axes
    sampling_rate = stft_config['fs']
    nperseg = stft_config['nperseg']
    noverlap = stft_config['noverlap']
    nfft = stft_config['nfft']
    
    # Frequency bins (Hz)
    freqs = np.fft.rfftfreq(nfft, 1/sampling_rate)
    
    # Time bins (seconds)
    n_samples = data_sample.shape[1]
    time_step = (nperseg - noverlap) / sampling_rate
    time_bins = np.arange(spectrogram.shape[1]) * time_step
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Spectrogram (log scale)
    im1 = axes[0].imshow(
        20 * np.log10(np.abs(spectrogram) + 1e-10),  # Convert to dB
        aspect='auto',
        origin='lower',
        extent=[time_bins[0], time_bins[-1], freqs[0], freqs[-1]],
        cmap='viridis',
        interpolation='nearest'
    )
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title(
        f'Spectrogram (Log Scale) - {config_name}\n'
        f'nperseg={nperseg} ({nperseg/sampling_rate:.3f}s), '
        f'noverlap={noverlap} ({100*noverlap/nperseg:.1f}%), '
        f'nfft={nfft}, '
        f'freq_res={sampling_rate/nfft:.2f} Hz/bin',
        fontsize=11
    )
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')
    
    # Plot 2: Spectrogram (linear scale)
    im2 = axes[1].imshow(
        np.abs(spectrogram),
        aspect='auto',
        origin='lower',
        extent=[time_bins[0], time_bins[-1], freqs[0], freqs[-1]],
        cmap='hot',
        interpolation='nearest'
    )
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_title(
        f'Spectrogram (Linear Scale) - {config_name}\n'
        f'Shape: ({spectrogram.shape[0]} freq bins, {spectrogram.shape[1]} time bins)',
        fontsize=11
    )
    plt.colorbar(im2, ax=axes[1], label='Power')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path} (shape: {spectrogram.shape})")


def compare_stft_configs(data_sample: np.ndarray, configs: List[Dict],
                         task: str, save_path: str, channel_idx: int = 0):
    """
    Compare multiple STFT configurations side by side
    
    Args:
        data_sample: Raw EEG sample (C, T)
        configs: List of STFT configuration dicts with 'name', 'fs', 'nperseg', 'noverlap', 'nfft'
        task: Task name
        save_path: Path to save the comparison image
        channel_idx: Which channel to visualize
    """
    n_configs = len(configs)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_configs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, stft_config in enumerate(configs):
        config_name = stft_config.get('name', f'config_{idx}')
        
        # Apply STFT
        stft_result = apply_stft_transform(
            data_sample[np.newaxis, :, :],
            fs=stft_config['fs'],
            nperseg=stft_config['nperseg'],
            noverlap=stft_config['noverlap'],
            nfft=stft_config['nfft']
        )
        
        if stft_result.ndim == 4:
            stft_result = stft_result[0]
        
        spectrogram = stft_result[channel_idx]
        
        # Get axes
        sampling_rate = stft_config['fs']
        nperseg = stft_config['nperseg']
        noverlap = stft_config['noverlap']
        nfft = stft_config['nfft']
        
        freqs = np.fft.rfftfreq(nfft, 1/sampling_rate)
        n_samples = data_sample.shape[1]
        time_step = (nperseg - noverlap) / sampling_rate
        time_bins = np.arange(spectrogram.shape[1]) * time_step
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(
            20 * np.log10(np.abs(spectrogram) + 1e-10),
            aspect='auto',
            origin='lower',
            extent=[time_bins[0], time_bins[-1], freqs[0], freqs[-1]],
            cmap='viridis',
            interpolation='nearest'
        )
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(
            f'{config_name}\n'
            f'nperseg={nperseg}, noverlap={noverlap}\n'
            f'nfft={nfft}, shape={spectrogram.shape}',
            fontsize=9
        )
        plt.colorbar(im, ax=ax, label='dB')
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'STFT Parameter Comparison - {task} (Channel {channel_idx})', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison: {save_path}")


def visualize_all_configs(task: str, save_dir: str = './stft_visualizations',
                          sample_idx: int = 0, channel_idx: int = 0):
    """
    Visualize all STFT parameter configurations for a task
    
    Args:
        task: Task name
        save_dir: Directory to save visualizations
        sample_idx: Which sample to use for visualization
        channel_idx: Which channel to visualize
    """
    print(f"\n{'='*80}")
    print(f"STFT Parameter Visualization: {task}")
    print(f"{'='*80}")
    
    task_config = TASK_CONFIGS.get(task, {})
    if not task_config:
        raise ValueError(f"Unknown task: {task}")
    
    sampling_rate = task_config.get('sampling_rate', 250)
    print(f"Sampling Rate: {sampling_rate} Hz")
    
    # Load a sample data
    print(f"\nLoading data for {task}...")
    datasets = load_dataset(
        task=task,
        data_dir=task_config.get('data_dir'),
        num_seen=task_config.get('num_seen'),
        seed=44
    )
    
    if not datasets or 'train' not in datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # Get a sample
    X_train, y_train = datasets['train']
    if sample_idx >= len(X_train):
        sample_idx = 0
        print(f"Warning: sample_idx too large, using 0")
    
    data_sample = X_train[sample_idx]  # (C, T)
    label = y_train[sample_idx]
    
    print(f"Sample shape: {data_sample.shape}")
    print(f"Label: {label}")
    print(f"Visualizing channel {channel_idx} of {data_sample.shape[0]} channels")
    
    # Get parameter grid
    param_grid = get_stft_param_grid(task)
    print(f"\nGenerating visualizations for {len(param_grid)} configurations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create STFT configs for visualization
    stft_configs = []
    for params in param_grid:
        stft_configs.append({
            'name': params['name'],
            'fs': sampling_rate,
            'nperseg': params['nperseg'],
            'noverlap': params['noverlap'],
            'nfft': params['nfft']
        })
    
    # Generate individual visualizations
    print("\nGenerating individual visualizations...")
    for stft_config in stft_configs:
        config_name = stft_config['name']
        save_path = os.path.join(
            save_dir,
            f'{task.lower()}_stft_{config_name}_ch{channel_idx}_sample{sample_idx}.png'
        )
        
        try:
            visualize_stft_config(
                data_sample=data_sample,
                stft_config=stft_config,
                task=task,
                config_name=config_name,
                save_path=save_path,
                channel_idx=channel_idx
            )
        except Exception as e:
            print(f"  ✗ Failed to visualize {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison visualization
    print("\nGenerating comparison visualization...")
    comparison_path = os.path.join(
        save_dir,
        f'{task.lower()}_stft_comparison_ch{channel_idx}_sample{sample_idx}.png'
    )
    
    try:
        compare_stft_configs(
            data_sample=data_sample,
            configs=stft_configs,
            task=task,
            save_path=comparison_path,
            channel_idx=channel_idx
        )
    except Exception as e:
        print(f"  ✗ Failed to create comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate summary with key configurations
    print("\nGenerating key configurations comparison...")
    key_configs = [
        {'name': 'default', 'fs': sampling_rate, 'nperseg': task_config.get('stft_nperseg', 128),
         'noverlap': task_config.get('stft_noverlap', 112), 'nfft': task_config.get('stft_nfft', 512)},
        {'name': 'high_time_res', 'fs': sampling_rate, 'nperseg': 64, 'noverlap': 56, 'nfft': 512},
        {'name': 'high_freq_res', 'fs': sampling_rate, 'nperseg': 128, 'noverlap': 112, 'nfft': 2048},
        {'name': 'low_overlap', 'fs': sampling_rate, 'nperseg': 128, 'noverlap': 64, 'nfft': 512},
    ]
    
    # Filter to only include configs that exist in param_grid
    key_configs_filtered = []
    for key_cfg in key_configs:
        for cfg in stft_configs:
            if (cfg['nperseg'] == key_cfg['nperseg'] and 
                cfg['noverlap'] == key_cfg['noverlap'] and 
                cfg['nfft'] == key_cfg['nfft']):
                key_configs_filtered.append(cfg)
                break
    
    if key_configs_filtered:
        key_comparison_path = os.path.join(
            save_dir,
            f'{task.lower()}_stft_key_comparison_ch{channel_idx}_sample{sample_idx}.png'
        )
        try:
            compare_stft_configs(
                data_sample=data_sample,
                configs=key_configs_filtered,
                task=task,
                save_path=key_comparison_path,
                channel_idx=channel_idx
            )
        except Exception as e:
            print(f"  ✗ Failed to create key comparison: {e}")
    
    print(f"\n{'='*80}")
    print(f"Visualization complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize STFT Parameter Configurations'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='SSVEP',
        choices=list(TASK_CONFIGS.keys()),
        help='Task to visualize (default: SSVEP)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./stft_visualizations',
        help='Directory to save visualizations (default: ./stft_visualizations)'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Which sample to use for visualization (default: 0)'
    )
    parser.add_argument(
        '--channel_idx',
        type=int,
        default=0,
        help='Which channel to visualize (default: 0)'
    )
    
    args = parser.parse_args()
    
    visualize_all_configs(
        task=args.task,
        save_dir=args.save_dir,
        sample_idx=args.sample_idx,
        channel_idx=args.channel_idx
    )
