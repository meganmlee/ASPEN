from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helpers

# Output directory (relative to this file)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "within_trial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_channel_consistency_temporal(epochs, condition):
    """Per-channel temporal consistency for a condition (mean off-diagonal pairwise trial corr)."""
    data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
    if data.shape[0] < 2:
        return None
    sfreq = epochs.info['sfreq']
    baseline_end_idx = int(1.0 * sfreq)  # 1 s baseline ends at t=0 per friend's logic

    scores = []
    for ch in range(data.shape[1]):
        X = helpers.align_trials_to_mi_onset(data[:, ch, :], sfreq, baseline_end_idx)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
        C = (X @ X.T) / X.shape[1]
        mask = ~np.eye(C.shape[0], dtype=bool)
        scores.append(float(np.nanmean(C[mask])))
    return scores


def compute_channel_consistency_frequency(epochs, condition, fmin, fmax):
    """Per-channel frequency-domain consistency using Welch PSD + pairwise trial correlations."""
    spectrum = epochs[condition].compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=512, verbose=False)
    psds = spectrum.get_data()  # (n_trials, n_channels, n_freqs)

    scores = []
    for ch in range(psds.shape[1]):
        X = psds[:, ch, :]
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
        C = (X @ X.T) / X.shape[1]
        mask = ~np.eye(C.shape[0], dtype=bool)
        scores.append(float(np.nanmean(C[mask])))
    return scores


def plot_channel_consistency(subject_id, channel_names, temporal_by_cond, frequency_by_cond, save_path):
    """Replicates your friend’s multi-panel style (temporal + frequency summaries)."""
    conditions = list(temporal_by_cond.keys())
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Temporal (up to 4 conditions for grid parity)
    for idx, condition in enumerate(conditions[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.bar(range(len(channel_names)), temporal_by_cond[condition])
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
        band_scores = [np.nanmean(frequency_by_cond[band][condition]) for band in band_names]
        ax.bar(x + cond_idx * width, band_scores, width, label=condition, alpha=0.7)
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Mean Trial-Trial Correlation', fontsize=10)
    ax.set_title('Frequency Band Consistency Across Conditions', fontsize=12)
    ax.set_xticks(x + width * max(0, len(conditions)-1)/2.0)
    ax.set_xticklabels(band_names)
    ax.legend()

    # Temporal summary across conditions
    ax = fig.add_subplot(gs[2, 1])
    temporal_means = [np.nanmean(temporal_by_cond[cond]) for cond in conditions]
    ax.bar(conditions, temporal_means, alpha=0.7)
    ax.set_ylabel('Mean Trial-Trial Corr')
    ax.set_title('Temporal Consistency Summary')

    fig.suptitle(f"Within-Trial Channel Consistency — {subject_id}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    summary_rows = []
    for subject_file in helpers.subjects:
        epochs = helpers.load_epochs(subject_file)
        subject_id = subject_file
        channel_names = epochs.ch_names

        temporal_scores = {}
        frequency_scores = {band: {} for band in helpers.freq_bands.keys()}

        for condition in epochs.event_id.keys():
            # Temporal per-channel
            t_scores = compute_channel_consistency_temporal(epochs, condition)
            if t_scores is None:
                continue
            temporal_scores[condition] = t_scores

            # Frequency per band
            for band, (fmin, fmax) in helpers.freq_bands.items():
                frequency_scores[band][condition] = compute_channel_consistency_frequency(epochs, condition, fmin, fmax)

        # Plot per subject
        fig_path = OUTPUT_DIR / f"{subject_id}_within_trial_consistency.png"
        plot_channel_consistency(subject_id, channel_names, temporal_scores, frequency_scores, fig_path)

        # Subject summary row
        row = {'subject': subject_id}
        for cond, vals in temporal_scores.items():
            row[f"{cond}_temporal_mean"] = float(np.nanmean(vals))
        for band in helpers.freq_bands.keys():
            for cond in frequency_scores[band].keys():
                row[f"{cond}_{band}_mean"] = float(np.nanmean(frequency_scores[band][cond]))
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "within_trial_summary.csv", index=False)
    print(f"Saved within-trial outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()