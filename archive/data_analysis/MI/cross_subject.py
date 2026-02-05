from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import helpers
import pandas as pd

# Across Subjects: Does the same channel for the same label look similar across subjects?
# Are label representations consistent across subjects?

# Output results
output_dir = Path('/home/megan/Downloads/cleaneddata/cross_subject')
output_dir.mkdir(exist_ok=True)

def compute_subject_channel_average_temporal(epochs, condition):
    """
    Compute average temporal pattern per channel for a subject-condition pair.
    Returns: array of shape (n_channels, n_times_aligned)
    """
    data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
    n_channels = data.shape[1]
    sfreq = epochs.info['sfreq']
    baseline_end_idx = int(1.0 * sfreq)

    channel_averages = []

    for ch_idx in range(n_channels):
        channel_data = data[:, ch_idx, :]
        aligned_data = helpers.align_trials_to_mi_onset(channel_data, sfreq, baseline_end_idx)

        # Average across trials
        avg_pattern = np.mean(aligned_data, axis=0)
        # Normalize
        avg_pattern = (avg_pattern - np.mean(avg_pattern)) / (np.std(avg_pattern) + 1e-10)
        channel_averages.append(avg_pattern)

    return np.array(channel_averages)

def compute_subject_channel_average_frequency(epochs, condition, fmin, fmax):
    """
    Compute average frequency pattern per channel for a subject-condition pair.
    Returns: array of shape (n_channels, n_freqs)
    """
    spectrum = epochs[condition].compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                             n_fft=512, verbose=False)
    psds = spectrum.get_data()  # (n_trials, n_channels, n_freqs)

    # Average across trials
    avg_psds = np.mean(psds, axis=0)  # (n_channels, n_freqs)

    # Normalize each channel
    for ch_idx in range(avg_psds.shape[0]):
        avg_psds[ch_idx] = (avg_psds[ch_idx] - np.mean(avg_psds[ch_idx])) / (np.std(avg_psds[ch_idx]) + 1e-10)

    return avg_psds

def analyze_q3_channel_consistency(all_subject_data):
    """
    Does the same channel for the same label look similar across subjects?

    For each condition and each channel, compare patterns across all subject pairs.
    """
    results = {
        'temporal': {},
        'frequency': {band: {} for band in helpers.freq_bands.keys()}
    }

    conditions = list(all_subject_data[list(all_subject_data.keys())[0]].keys())
    channel_names = all_subject_data[list(all_subject_data.keys())[0]][conditions[0]]['channel_names']
    n_channels = len(channel_names)

    print("\n" + "="*80)
    print("ANALYZING CHANNEL CONSISTENCY ACROSS SUBJECTS")
    print("="*80)

    for condition in conditions:

        # Temporal domain
        temporal_correlations = np.zeros((n_channels, len(all_subject_data), len(all_subject_data)))

        subject_list = list(all_subject_data.keys())
        for i, subj1 in enumerate(subject_list):
            for j, subj2 in enumerate(subject_list):
                pattern1 = all_subject_data[subj1][condition]['temporal']
                pattern2 = all_subject_data[subj2][condition]['temporal']

                # Compare each channel
                for ch_idx in range(n_channels):
                    corr = np.corrcoef(pattern1[ch_idx], pattern2[ch_idx])[0, 1]
                    temporal_correlations[ch_idx, i, j] = corr

        # Extract upper triangle (unique pairs) for each channel
        channel_consistency_scores = []
        for ch_idx in range(n_channels):
            corr_matrix = temporal_correlations[ch_idx]
            triu_idx = np.triu_indices_from(corr_matrix, k=1)
            channel_consistency_scores.append(np.mean(corr_matrix[triu_idx]))

        results['temporal'][condition] = {
            'channel_scores': np.array(channel_consistency_scores),
            'channel_names': channel_names,
            'mean': np.mean(channel_consistency_scores),
            'std': np.std(channel_consistency_scores)
        }

        # Frequency domain
        for band_name, (fmin, fmax) in helpers.freq_bands.items():
            freq_correlations = np.zeros((n_channels, len(all_subject_data), len(all_subject_data)))

            for i, subj1 in enumerate(subject_list):
                for j, subj2 in enumerate(subject_list):
                    pattern1 = all_subject_data[subj1][condition]['frequency'][band_name]
                    pattern2 = all_subject_data[subj2][condition]['frequency'][band_name]

                    for ch_idx in range(n_channels):
                        corr = np.corrcoef(pattern1[ch_idx], pattern2[ch_idx])[0, 1]
                        freq_correlations[ch_idx, i, j] = corr

            channel_consistency_scores = []
            for ch_idx in range(n_channels):
                corr_matrix = freq_correlations[ch_idx]
                triu_idx = np.triu_indices_from(corr_matrix, k=1)
                channel_consistency_scores.append(np.mean(corr_matrix[triu_idx]))

            results['frequency'][band_name][condition] = {
                'channel_scores': np.array(channel_consistency_scores),
                'mean': np.mean(channel_consistency_scores),
                'std': np.std(channel_consistency_scores)
            }

    return results

def analyze_q4_label_consistency(all_subject_data):
    """
    Q4: Are label representations consistent across subjects?

    For each subject pair, compare how similar the same label is.
    """
    results = {
        'temporal': {},
        'frequency': {band: {} for band in helpers.freq_bands.keys()}
    }

    conditions = list(all_subject_data[list(all_subject_data.keys())[0]].keys())
    subject_list = list(all_subject_data.keys())

    print("\n" + "="*80)
    print("ANALYZING LABEL CONSISTENCY ACROSS SUBJECTS")
    print("="*80)

    for condition in conditions:

        # Temporal domain: average across all channels for whole-brain representation
        pairwise_correlations = []

        for subj1, subj2 in combinations(subject_list, 2):
            pattern1 = all_subject_data[subj1][condition]['temporal']
            pattern2 = all_subject_data[subj2][condition]['temporal']

            # Flatten all channels and compute correlation
            pattern1_flat = pattern1.flatten()
            pattern2_flat = pattern2.flatten()

            corr = np.corrcoef(pattern1_flat, pattern2_flat)[0, 1]
            pairwise_correlations.append(corr)

        results['temporal'][condition] = {
            'correlations': np.array(pairwise_correlations),
            'mean': np.mean(pairwise_correlations),
            'std': np.std(pairwise_correlations),
            'median': np.median(pairwise_correlations)
        }

        # Frequency domain
        for band_name, (fmin, fmax) in helpers.freq_bands.items():
            pairwise_correlations = []

            for subj1, subj2 in combinations(subject_list, 2):
                pattern1 = all_subject_data[subj1][condition]['frequency'][band_name]
                pattern2 = all_subject_data[subj2][condition]['frequency'][band_name]

                pattern1_flat = pattern1.flatten()
                pattern2_flat = pattern2.flatten()

                corr = np.corrcoef(pattern1_flat, pattern2_flat)[0, 1]
                pairwise_correlations.append(corr)

            results['frequency'][band_name][condition] = {
                'correlations': np.array(pairwise_correlations),
                'mean': np.mean(pairwise_correlations),
                'std': np.std(pairwise_correlations),
                'median': np.median(pairwise_correlations)
            }

    return results

def plot_q3_results(q3_results, save_path):
    """Visualize Q3: channel consistency across subjects."""
    conditions = list(q3_results['temporal'].keys())

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Plot 1-4: Temporal consistency per channel for each condition
    for idx, condition in enumerate(conditions[:4]):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        channel_scores = q3_results['temporal'][condition]['channel_scores']
        channel_names = q3_results['temporal'][condition]['channel_names']

        x = np.arange(len(channel_names))
        colors = ['red' if score < 0.3 else 'orange' if score < 0.6 else 'green'
                  for score in channel_scores]
        ax.bar(x, channel_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.axhline(0.6, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High (>0.6)')
        ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (>0.3)')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([ch.replace('EEG-', '') for ch in channel_names[::2]],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Cross-Subject Correlation', fontsize=10)
        ax.set_title(f'{condition} - Per-Channel Consistency', fontweight='bold', fontsize=11)
        ax.set_ylim([-0.2, 1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight motor channels
        motor_indices = [i for i, ch in enumerate(channel_names)
                        if any(m in ch for m in ['C3', 'Cz', 'C4'])]
        for mi in motor_indices:
            ax.axvspan(mi-0.5, mi+0.5, alpha=0.2, color='blue')

    # Plot 5: Frequency band comparison across conditions
    ax = axes[2, 0]
    band_names = list(helpers.freq_bands.keys())
    x = np.arange(len(band_names))
    width = 0.2

    for cond_idx, condition in enumerate(conditions):
        band_scores = [q3_results['frequency'][band][condition]['mean']
                      for band in band_names]
        ax.bar(x + cond_idx*width, band_scores, width, label=condition, alpha=0.7)

    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Mean Cross-Subject Correlation', fontsize=10)
    ax.set_title('Frequency Band Consistency (All Channels)', fontweight='bold', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Heatmap of channel consistency
    ax = axes[2, 1]
    heatmap_data = []
    for condition in conditions:
        heatmap_data.append(q3_results['temporal'][condition]['channel_scores'])
    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='auto')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    channel_names = q3_results['temporal'][conditions[0]]['channel_names']
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels([ch.replace('EEG-', '') for ch in channel_names],
                       rotation=90, fontsize=7)
    ax.set_xlabel('Channel', fontsize=10)
    ax.set_ylabel('Condition', fontsize=10)
    ax.set_title('Cross-Subject Channel Consistency Heatmap', fontweight='bold', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Correlation', fontsize=10)

    plt.suptitle('Q3: Channel Consistency Across Subjects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Q3 plot: {save_path}")

def plot_q4_results(q4_results, save_path):
    """Visualize Q4: label consistency across subjects."""
    conditions = list(q4_results['temporal'].keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Temporal consistency distribution by condition
    ax = axes[0, 0]
    temporal_data = [q4_results['temporal'][c]['correlations'] for c in conditions]
    bp = ax.boxplot(temporal_data, labels=conditions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylabel('Cross-Subject Correlation', fontsize=10)
    ax.set_title('Temporal: Label Consistency Distribution', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Mean consistency by condition (temporal)
    ax = axes[0, 1]
    means = [q4_results['temporal'][c]['mean'] for c in conditions]
    stds = [q4_results['temporal'][c]['std'] for c in conditions]
    x = np.arange(len(conditions))
    colors = ['red' if m < 0.3 else 'orange' if m < 0.6 else 'green' for m in means]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Mean Correlation', fontsize=10)
    ax.set_title('Temporal: Mean Label Consistency', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Frequency band heatmap
    ax = axes[0, 2]
    band_names = list(helpers.freq_bands.keys())
    heatmap_data = []
    for condition in conditions:
        row = [q4_results['frequency'][band][condition]['mean'] for band in band_names]
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='auto')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names)
    ax.set_title('Frequency: Label Consistency', fontweight='bold', fontsize=11)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Correlation', fontsize=9)

    for i in range(len(conditions)):
        for j in range(len(band_names)):
            ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=9)

    # Plot 4: Comparison across frequency bands
    ax = axes[1, 0]
    x = np.arange(len(band_names))
    width = 0.2
    for cond_idx, condition in enumerate(conditions):
        band_means = [q4_results['frequency'][band][condition]['mean'] for band in band_names]
        ax.bar(x + cond_idx*width, band_means, width, label=condition, alpha=0.7)
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Mean Correlation', fontsize=10)
    ax.set_title('Frequency Band Label Consistency', fontweight='bold', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Temporal vs best frequency band
    ax = axes[1, 1]
    temporal_means = [q4_results['temporal'][c]['mean'] for c in conditions]

    # Find best band for each condition
    best_freq_means = []
    for condition in conditions:
        band_scores = {band: q4_results['frequency'][band][condition]['mean']
                      for band in band_names}
        best_freq_means.append(max(band_scores.values()))

    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, temporal_means, width, label='Temporal', alpha=0.7)
    ax.bar(x + width/2, best_freq_means, width, label='Best Frequency', alpha=0.7)
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Mean Correlation', fontsize=10)
    ax.set_title('Temporal vs Frequency Consistency', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Statistical summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = "STATISTICAL SUMMARY\n" + "="*40 + "\n\n"
    summary_text += "Temporal Domain:\n"
    for condition in conditions:
        mean = q4_results['temporal'][condition]['mean']
        median = q4_results['temporal'][condition]['median']
        summary_text += f"  {condition}: μ={mean:.3f}, med={median:.3f}\n"

    summary_text += "\nBest Frequency Band:\n"
    for condition in conditions:
        band_scores = {band: q4_results['frequency'][band][condition]['mean']
                      for band in band_names}
        best_band = max(band_scores, key=band_scores.get)
        best_score = band_scores[best_band]
        summary_text += f"  {condition}: {best_band} ({best_score:.3f})\n"

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle('Q4: Label Consistency Across Subjects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Q4 plot: {save_path}")

def create_summary_report(q3_results, q4_results):
    """Generate comprehensive text report."""
    print("\n" + "="*80)
    print("CROSS-SUBJECT CONSISTENCY ANALYSIS SUMMARY")
    print("="*80)

    # Q3 Summary
    print("\n" + "-"*80)
    print("Q3: CHANNEL CONSISTENCY ACROSS SUBJECTS")
    print("-"*80)
    print("\nDoes the same channel for the same label look similar across subjects?")
    print("\nTemporal Domain (per condition):")

    conditions = list(q3_results['temporal'].keys())
    for condition in conditions:
        mean = q3_results['temporal'][condition]['mean']
        std = q3_results['temporal'][condition]['std']
        print(f"  {condition:12s}: {mean:.3f} ± {std:.3f}")

    overall_temporal_mean = np.mean([q3_results['temporal'][c]['mean'] for c in conditions])
    print(f"\n  Overall:     {overall_temporal_mean:.3f}")

    print("\nFrequency Domain (averaged across conditions):")
    for band in helpers.freq_bands.keys():
        band_means = [q3_results['frequency'][band][c]['mean'] for c in conditions]
        print(f"  {band:8s}: {np.mean(band_means):.3f}")

    # Q4 Summary
    print("\n" + "-"*80)
    print("Q4: LABEL CONSISTENCY ACROSS SUBJECTS")
    print("-"*80)
    print("\nAre label representations consistent across subjects?")
    print("\nTemporal Domain (per condition):")

    for condition in conditions:
        mean = q4_results['temporal'][condition]['mean']
        median = q4_results['temporal'][condition]['median']
        std = q4_results['temporal'][condition]['std']
        print(f"  {condition:12s}: μ={mean:.3f}, med={median:.3f}, σ={std:.3f}")

    overall_q4_temporal = np.mean([q4_results['temporal'][c]['mean'] for c in conditions])
    print(f"\n  Overall:     {overall_q4_temporal:.3f}")

    print("\nFrequency Domain (best band per condition):")
    for condition in conditions:
        band_scores = {band: q4_results['frequency'][band][condition]['mean']
                      for band in helpers.freq_bands.keys()}
        best_band = max(band_scores, key=band_scores.get)
        best_score = band_scores[best_band]
        print(f"  {condition:12s}: {best_band} ({best_score:.3f})")

def load_all_subjects():
    """Load and process data for all subjects."""
    all_data = {}

    print("Loading subjects...")
    for subject in helpers.subjects:
        try:
            print(f"  Loading {subject}...")
            epochs = helpers.load_epochs(f'{subject}.fif')

            subject_data = {}
            for condition in epochs.event_id.keys():
                if len(epochs[condition]) < 2:
                    continue

                # Compute temporal patterns
                temporal_patterns = compute_subject_channel_average_temporal(epochs, condition)

                # Compute frequency patterns
                freq_patterns = {}
                for band_name, (fmin, fmax) in helpers.freq_bands.items():
                    freq_patterns[band_name] = compute_subject_channel_average_frequency(
                        epochs, condition, fmin, fmax
                    )

                subject_data[condition] = {
                    'temporal': temporal_patterns,
                    'frequency': freq_patterns,
                    'channel_names': epochs.ch_names
                }

            all_data[subject.replace('.fif', '')] = subject_data

        except Exception as e:
            print(f"  Error with {subject}: {e}")

    return all_data

# Main execution
if __name__ == "__main__":
    # Load all subjects
    all_subject_data = load_all_subjects()

    # Analyze Q3
    q3_results = analyze_q3_channel_consistency(all_subject_data)
    plot_q3_results(q3_results, output_dir / 'q3_channel_consistency.png')

    # Analyze Q4
    q4_results = analyze_q4_label_consistency(all_subject_data)
    plot_q4_results(q4_results, output_dir / 'q4_label_consistency.png')

    # Generate report
    create_summary_report(q3_results, q4_results)

    # Save results to CSV
    conditions = list(q3_results['temporal'].keys())

    # Q3 CSV - Channel consistency
    q3_rows = []
    for condition in conditions:
        channel_names = q3_results['temporal'][condition]['channel_names']
        channel_scores = q3_results['temporal'][condition]['channel_scores']
        for ch_name, score in zip(channel_names, channel_scores):
            q3_rows.append({
                'condition': condition,
                'channel': ch_name,
                'temporal_correlation': score
            })

    q3_df = pd.DataFrame(q3_rows)
    q3_df.to_csv(output_dir / 'q3_channel_consistency.csv', index=False)
    print(f"\nSaved Q3 results to: {output_dir / 'q3_channel_consistency.csv'}")

    # Q4 CSV - Label consistency
    q4_rows = []
    for condition in conditions:
        row = {
            'condition': condition,
            'temporal_mean': q4_results['temporal'][condition]['mean'],
            'temporal_std': q4_results['temporal'][condition]['std'],
            'temporal_median': q4_results['temporal'][condition]['median']
        }

        # Add frequency bands
        for band in helpers.freq_bands.keys():
            row[f'{band}_mean'] = q4_results['frequency'][band][condition]['mean']
            row[f'{band}_std'] = q4_results['frequency'][band][condition]['std']

        q4_rows.append(row)

    q4_df = pd.DataFrame(q4_rows)
    q4_df.to_csv(output_dir / 'q4_label_consistency.csv', index=False)
    print(f"Saved Q4 results to: {output_dir / 'q4_label_consistency.csv'}")
