import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helpers
from pathlib import Path

# Output directory (relative)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cross_trial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_channel_consistency_temporal(epochs, condition):
    """Per-channel temporal consistency across trials (with onset alignment)."""
    data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
    n_channels = data.shape[1]
    sfreq = epochs.info['sfreq']
    baseline_end_idx = int(1.0 * sfreq)

    channel_consistencies = []
    for ch_idx in range(n_channels):
        X = helpers.align_trials_to_mi_onset(data[:, ch_idx, :], sfreq, baseline_end_idx)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
        C = (X @ X.T) / X.shape[1]
        mask = ~np.eye(C.shape[0], dtype=bool)
        channel_consistencies.append(float(np.nanmean(C[mask])))
    return channel_consistencies


def plot_subject_channel_consistency(results, save_path):
    """Same layout as your friend's: per-channel temporal bars + band summary."""
    subject_id = results['subject']
    channel_names = results['channel_names']
    conditions = list(results['conditions'].keys())

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Temporal per-channel for up to 4 conditions
    for idx, condition in enumerate(conditions[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.bar(range(len(channel_names)), results['conditions'][condition]['temporal'])
        ax.set_title(f"{subject_id} — Temporal Consistency — {condition}")
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels([ch.replace('EEG-', '') for ch in channel_names], rotation=90, fontsize=8)
        ax.set_ylabel('Correlation')

    # Frequency band comparison across conditions
    ax = fig.add_subplot(gs[2, 0])
    band_names = list(helpers.freq_bands.keys())
    x = np.arange(len(band_names))
    width = 0.2
    for cond_idx, condition in enumerate(conditions):
        band_scores = [results['conditions'][condition]['frequency'][band]['mean'] for band in band_names]
        ax.bar(x + cond_idx * width, band_scores, width, label=condition, alpha=0.7)
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Mean Trial-Trial Correlation', fontsize=10)
    ax.set_title('Frequency Band Consistency Across Conditions', fontsize=12)
    ax.set_xticks(x + width * max(0, len(conditions)-1)/2.0)
    ax.set_xticklabels(band_names)
    ax.legend()

    # Temporal summary across conditions
    ax = fig.add_subplot(gs[2, 1])
    temporal_means = [results['conditions'][cond]['temporal_mean'] for cond in conditions]
    ax.bar(conditions, temporal_means, alpha=0.7)
    ax.set_ylabel('Mean Trial-Trial Corr')
    ax.set_title('Temporal Consistency Summary')

    fig.suptitle(f"Cross-Trial Consistency — {subject_id}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    q4_rows = []
    for subject_file in helpers.subjects:
        epochs = helpers.load_epochs(subject_file)
        subject_id = subject_file
        channel_names = epochs.ch_names
        conditions = list(epochs.event_id.keys())

        subject_results = {'subject': subject_id, 'channel_names': channel_names, 'conditions': {}}

        for condition in conditions:
            # Temporal per-channel
            temp_by_channel = compute_channel_consistency_temporal(epochs, condition)

            # Frequency: trial-trial summary per band
            freq_summary = {}
            # flatten trial-by-trial for temporal summary too
            data = epochs[condition].get_data()
            n_trials, n_channels, n_times = data.shape
            X_flat = data.reshape(n_trials, n_channels * n_times)
            X_flat = (X_flat - X_flat.mean(axis=1, keepdims=True)) / (X_flat.std(axis=1, keepdims=True) + 1e-12)
            C_flat = (X_flat @ X_flat.T) / X_flat.shape[1]
            temporal_mask = ~np.eye(n_trials, dtype=bool)
            temporal_mean = float(np.nanmean(C_flat[temporal_mask]))

            for band, (fmin, fmax) in helpers.freq_bands.items():
                spectrum = epochs[condition].compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                                         n_fft=512, verbose=False)
                psds = spectrum.get_data()  # (n_trials, n_channels, n_freqs)
                X = psds.reshape(psds.shape[0], -1)
                X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
                C = (X @ X.T) / X.shape[1]
                mask = ~np.eye(C.shape[0], dtype=bool)
                freq_summary[band] = {
                    'mean': float(np.nanmean(C[mask])),
                    'std': float(np.nanstd(C[mask])),
                    'median': float(np.nanmedian(C[mask]))
                }

            subject_results['conditions'][condition] = {
                'temporal': temp_by_channel,
                'temporal_mean': temporal_mean,
                'frequency': freq_summary
            }

        # Plot per subject
        fig_path = OUTPUT_DIR / f"{subject_id}_cross_trial.png"
        plot_subject_channel_consistency(subject_results, fig_path)

        # Flatten to a Q4-like row
        row = {'subject': subject_id}
        for condition in conditions:
            row[f'{condition}_temporal_mean'] = subject_results['conditions'][condition]['temporal_mean']
            for band in helpers.freq_bands.keys():
                row[f'{condition}_{band}_mean'] = subject_results['conditions'][condition]['frequency'][band]['mean']
        q4_rows.append(row)

    pd.DataFrame(q4_rows).to_csv(OUTPUT_DIR / 'cross_trial_summary.csv', index=False)
    print(f"Saved Cross-Trial outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()