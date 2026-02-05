"""
Cross-Trial Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import helpers

def compute_channel_consistency_temporal(trials, sfreq, use_alignment):
    """
    Modified version of your function to work with numpy arrays directly.
    
    Args:
        trials: numpy array (n_trials, n_channels, n_times)
        sfreq: sampling frequency
        use_alignment: whether to align trials (only for MI)
    """
    n_trials, n_channels, n_times = trials.shape
    
    if n_trials < 2:
        return np.zeros(n_channels)
    
    baseline_end_idx = int(1.0 * sfreq)  # Assuming 1s baseline for MI
    max_shift_samples = int(0.2 * sfreq)
    
    channel_consistencies = []
    
    for ch_idx in range(n_channels):
        channel_data = trials[:, ch_idx, :]  # (n_trials, n_times)
        
        # Auto-align if requested (only for MI)
        if use_alignment:
            aligned_data = helpers.align_trials_to_mi_onset(channel_data, sfreq, baseline_end_idx)
        else:
            aligned_data = channel_data
        
        # Compute pairwise correlations (YOUR EXISTING LOGIC)
        correlations = []
        for i in range(n_trials):
            for j in range(i+1, n_trials):
                trial1 = aligned_data[i]
                trial2 = aligned_data[j]
                
                # Normalize
                trial1 = (trial1 - np.mean(trial1)) / (np.std(trial1) + 1e-10)
                trial2 = (trial2 - np.mean(trial2)) / (np.std(trial2) + 1e-10)
                
                # Cross-correlation (YOUR EXISTING LOGIC)
                xcorr = np.correlate(trial1, trial2, mode='same')
                xcorr = xcorr / len(trial1)
                
                center = len(xcorr) // 2
                start = max(0, center - max_shift_samples)
                end = min(len(xcorr), center + max_shift_samples)
                max_corr = np.max(xcorr[start:end])
                
                correlations.append(max_corr)
        
        channel_consistencies.append(np.mean(correlations))
    
    return np.array(channel_consistencies)


def compute_channel_consistency_frequency(trials, sfreq, fmin, fmax):
    """
    Modified to work with numpy arrays instead of MNE Epochs.
    
    Args:
        trials: numpy array (n_trials, n_channels, n_times)
        sfreq: sampling frequency
        fmin, fmax: frequency band
    """
    from scipy import signal as sp_signal
    
    n_trials, n_channels, n_times = trials.shape
    
    if n_trials < 2:
        return np.zeros(n_channels)
    
    channel_consistencies = []
    
    for ch_idx in range(n_channels):
        channel_data = trials[:, ch_idx, :]  # (n_trials, n_times)
        
        # Compute PSD for each trial
        psds = []
        for trial in channel_data:
            freqs, psd = sp_signal.welch(trial, fs=sfreq, nperseg=min(512, n_times))
            # Select frequency band
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            psds.append(psd[freq_mask])
        
        psds = np.array(psds)  # (n_trials, n_freqs)
        
        # Compute pairwise correlations (YOUR EXISTING LOGIC)
        corr_matrix = np.corrcoef(psds)
        triu_idx = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[triu_idx]
        
        channel_consistencies.append(np.mean(pairwise_corrs))
    
    return np.array(channel_consistencies)


def analyze_subject(subject_data, subject_id, task_type):
    """
    Modified to work with numpy arrays from DataLoader.
    
    Args:
        subject_data: {condition_id: numpy_array(n_trials, n_channels, n_times)}
        subject_id: subject identifier
        task_type: task name
    """
    print(f"\nAnalyzing Subject {subject_id}...")
    
    config = TASK_CONFIG[task_type]
    sfreq = config['sfreq']
    use_alignment = config['use_alignment']
    
    # Get channel names (generic for now)
    first_condition = list(subject_data.values())[0]
    n_channels = first_condition.shape[1]
    channel_names = [f'Ch{i}' for i in range(n_channels)]
    
    results = {
        'subject': subject_id,
        'task': task_type,
        'channel_names': channel_names,
        'conditions': {}
    }
    
    for condition_id, trials in subject_data.items():
        n_trials = trials.shape[0]
        
        if n_trials < 2:
            print(f"  Skipping condition {condition_id}: only {n_trials} trial(s)")
            continue
        
        print(f"  Condition {condition_id}: {n_trials} trials")
        
        # Temporal consistency
        temp_consistency = compute_channel_consistency_temporal(trials, sfreq, use_alignment)
        
        # Frequency consistency per band
        freq_consistency = {}
        for band_name, (fmin, fmax) in freq_bands.items():
            freq_consistency[band_name] = compute_channel_consistency_frequency(
                trials, sfreq, fmin, fmax
            )
        
        results['conditions'][condition_id] = {
            'n_trials': n_trials,
            'temporal': temp_consistency,
            'frequency': freq_consistency
        }
    
    return results

# ============================================================================
# Plot
# ============================================================================

def plot_subject_channel_consistency(results, save_path):
    """YOUR EXISTING FUNCTION - just adjusted for condition_id instead of names"""
    subject_id = results['subject']
    channel_names = results['channel_names']
    conditions = list(results['conditions'].keys())
    
    if not conditions:
        print(f"No conditions to plot for subject {subject_id}")
        return
    
    n_conds = min(len(conditions), 4)
    n_bands = len(freq_bands)
    n_freq_rows = (n_bands + 1) // 2  # 2 bands per row
    total_rows = 2 + n_freq_rows      # 2 rows for temporal + freq rows

    fig = plt.figure(figsize=(24, 6 * total_rows))
    gs = fig.add_gridspec(total_rows, 4, hspace=0.4, wspace=0.4)
    
    # Plot temporal consistency per channel for each condition
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, condition_id in enumerate(conditions[:4]):
        row, col = subplot_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        temp_scores = results['conditions'][condition_id]['temporal']
        
        x = np.arange(len(channel_names))
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green'
                  for score in temp_scores]
        bars = ax.bar(x, temp_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High (>0.7)')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (>0.5)')
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Mean Trial Correlation', fontsize=10)
        ax.set_title(f'Condition {condition_id} - Temporal Consistency', fontweight='bold', fontsize=11)
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Frequency band comparison - ONE SUBPLOT PER BAND (bottom 2 rows)
    if len(conditions) > 0:
        band_names = list(freq_bands.keys())
        
        for band_idx, band_name in enumerate(band_names):
            ax = fig.add_subplot(gs[2 + band_idx // 2, (band_idx % 2) * 2:(band_idx % 2) * 2 + 2])
            
            # Collect scores for this band across all conditions
            x = np.arange(len(conditions))
            band_scores = []
            
            for condition_id in conditions:
                freq_scores = results['conditions'][condition_id]['frequency'][band_name]
                band_scores.append(np.mean(freq_scores))
            
            # Color bars by consistency level
            colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in band_scores]
            bars = ax.bar(x, band_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, linewidth=2)
            ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_xlabel('Condition', fontsize=10)
            ax.set_ylabel('Mean Correlation', fontsize=10)
            ax.set_title(f'{band_name} Band ({freq_bands[band_name][0]}-{freq_bands[band_name][1]} Hz)', 
                        fontweight='bold', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f'C{c}' for c in conditions])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Subject {subject_id} - {results["task"]} Task', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def create_summary_dataframe(all_results):
    """YOUR EXISTING FUNCTION - minimal changes"""
    summary_data = []
    
    for result in all_results:
        subject = result['subject']
        task = result['task']
        
        for condition_id, cond_data in result['conditions'].items():
            temp_consistency = cond_data['temporal']
            
            row = {
                'Subject': subject,
                'Task': task,
                'Condition': condition_id,
                'N_Trials': cond_data['n_trials'],
                'Temporal_Mean': np.mean(temp_consistency),
                'Temporal_Std': np.std(temp_consistency),
            }
            
            # Add frequency bands
            for band_name in freq_bands.keys():
                freq_scores = cond_data['frequency'][band_name]
                row[f'{band_name}_Mean'] = np.mean(freq_scores)
                row[f'{band_name}_Std'] = np.std(freq_scores)
            
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def plot_cross_subject_summary(df, save_path):
    """YOUR EXISTING FUNCTION - keep as-is"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    conditions = df['Condition'].unique()
    
    # Plot 1: Temporal consistency by subject
    ax = axes[0, 0]
    for condition in conditions:
        data = df[df['Condition'] == condition]
        ax.plot(data['Subject'], data['Temporal_Mean'],
                marker='o', label=f'Cond {condition}', linewidth=2, markersize=8)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Mean Consistency')
    ax.set_title('Temporal Consistency Across Subjects', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Distribution by condition
    ax = axes[0, 1]
    temp_data = [df[df['Condition'] == c]['Temporal_Mean'].values for c in conditions]
    bp = ax.boxplot(temp_data, labels=[f'C{c}' for c in conditions], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylabel('Consistency')
    ax.set_title('Distribution by Condition', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Subject ranking
    ax = axes[1, 0]
    subject_avg = df.groupby('Subject')['Temporal_Mean'].mean().sort_values(ascending=False)
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' 
              for v in subject_avg.values]
    ax.bar(range(len(subject_avg)), subject_avg.values, color=colors, alpha=0.7)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subject (ranked)')
    ax.set_ylabel('Mean Consistency')
    ax.set_title('Subject Ranking', fontweight='bold')
    ax.set_xticks(range(len(subject_avg)))
    ax.set_xticklabels([f'S{s}' for s in subject_avg.index], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Frequency bands heatmap
    ax = axes[1, 1]
    band_cols = [col for col in df.columns if col.endswith('_Mean') and col != 'Temporal_Mean']
    band_data = df.groupby('Condition')[band_cols].mean()
    band_data.columns = [col.replace('_Mean', '') for col in band_data.columns]
    
    im = ax.imshow(band_data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(band_data.columns)))
    ax.set_xticklabels(band_data.columns)
    ax.set_yticks(range(len(band_data.index)))
    ax.set_yticklabels([f'C{i}' for i in band_data.index])
    ax.set_title('Frequency Band Consistency', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean Correlation')
    
    for i in range(len(band_data.index)):
        for j in range(len(band_data.columns)):
            ax.text(j, i, f'{band_data.values[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=9)
    
    plt.suptitle(f'Cross-Subject Summary - {TASK} Task', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved cross-subject summary")


def print_report(df):
    """YOUR EXISTING FUNCTION - keep as-is"""
    print("\n" + "="*80)
    print(f"CROSS-TRIAL CONSISTENCY ANALYSIS - {TASK} TASK")
    print("="*80)
    print(f"\nTotal subjects: {df['Subject'].nunique()}")
    print(f"Total trials: {df['N_Trials'].sum()}")
    print(f"Total conditions: {df['Condition'].nunique()}")
    
    print("\n" + "-"*80)
    print("OVERALL CONSISTENCY:")
    print(f"  Mean: {df['Temporal_Mean'].mean():.3f} ± {df['Temporal_Mean'].std():.3f}")
    print(f"  Range: [{df['Temporal_Mean'].min():.3f}, {df['Temporal_Mean'].max():.3f}]")
    
    print("\n" + "-"*80)
    print("PER-CONDITION BREAKDOWN:")
    for condition in df['Condition'].unique():
        cond_data = df[df['Condition'] == condition]
        print(f"\n  Condition {condition}:")
        print(f"    Temporal: {cond_data['Temporal_Mean'].mean():.3f} ± {cond_data['Temporal_Mean'].std():.3f}")
        
        # Best frequency band
        band_cols = [col for col in df.columns if col.endswith('_Mean') and col != 'Temporal_Mean']
        band_means = {col.replace('_Mean', ''): cond_data[col].mean() for col in band_cols}
        best_band = max(band_means, key=band_means.get)
        print(f"    Best freq band: {best_band} ({band_means[best_band]:.3f})")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Cross-Trial Consistency Analysis - {TASK} Task")
    print(f"{'='*80}\n")
    
    # Load ALL data for all subjects (not split - this is for analysis, not training)
    root_dir = DATA_PATHS[TASK]
    print(f"Loading ALL {TASK} data from: {root_dir}")
    
    kwargs = {}
    
    subject_data = load_all_subjects_for_task(TASK, root_dir, **kwargs)
    
    print(f"\n{'='*40}")
    print(f"Found {len(subject_data)} subjects")
    for sid, conds in subject_data.items():
        print(f"  Subject {sid}: {len(conds)} conditions, ", end='')
        total_trials = sum(trials.shape[0] for trials in conds.values())
        print(f"{total_trials} total trials")
    print(f"{'='*40}\n")
    
    # Analyze each subject
    all_results = []
    
    for subject_id, conditions in subject_data.items():
        try:
            result = analyze_subject(conditions, subject_id, TASK)
            all_results.append(result)
            
            # Plot individual subject
            plot_path = OUTPUT_DIR / f'subject_{subject_id}_consistency.png'
            plot_subject_channel_consistency(result, plot_path)
        
        except Exception as e:
            print(f"Error with subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary
    if all_results:
        df = create_summary_dataframe(all_results)
        df.to_csv(OUTPUT_DIR / 'consistency_summary.csv', index=False)
        print("\nSaved CSV summary")
        
        summary_plot_path = OUTPUT_DIR / 'cross_trial_summary.png'
        plot_cross_subject_summary(df, summary_plot_path)
        
        # Print report
        print_report(df)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
        print(f"{'='*80}\n")
    else:
        print("\nNo results to summarize!")