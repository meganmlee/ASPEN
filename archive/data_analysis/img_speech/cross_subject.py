from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helpers

# Output directory (relative)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cross_subject"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return x * 0.0  # constant vector -> zero vector to avoid /0 in corr
    return (x - m) / s


def compute_cross_subject_channel_temporal(all_subject_data, condition):
    """
    Returns: (n_channels, n_subjects, n_subjects) cross-subject correlation matrix per channel
    using the per-channel temporal pattern vectors.
    """
    subjects = list(all_subject_data.keys())
    channel_names = all_subject_data[subjects[0]]['channel_names']
    n_channels = len(channel_names)
    S = np.full((n_channels, len(subjects), len(subjects)), np.nan, dtype=float)

    for ch in range(n_channels):
        for i, si in enumerate(subjects):
            Xi = all_subject_data[si][condition]['temporal_pattern'][ch]
            for j, sj in enumerate(subjects):
                Xj = all_subject_data[sj][condition]['temporal_pattern'][ch]
                # Both Xi and Xj are 1D vectors (z-scored)
                if Xi.size == 0 or Xj.size == 0:
                    val = np.nan
                else:
                    # If either vector is constant, correlation is undefined -> nan (already zscored to zeros)
                    if np.allclose(Xi, 0) or np.allclose(Xj, 0):
                        val = np.nan
                    else:
                        c = np.corrcoef(Xi, Xj)
                        val = c[0, 1] if c.shape == (2, 2) else np.nan
                S[ch, i, j] = val
    return S


def compute_cross_subject_channel_frequency(all_subject_data, condition, band):
    """
    Returns: (n_channels, n_subjects, n_subjects) cross-subject correlation matrix per channel
    using the per-channel frequency pattern vectors for the specified band.
    """
    subjects = list(all_subject_data.keys())
    channel_names = all_subject_data[subjects[0]]['channel_names']
    n_channels = len(channel_names)
    S = np.full((n_channels, len(subjects), len(subjects)), np.nan, dtype=float)

    for ch in range(n_channels):
        for i, si in enumerate(subjects):
            Xi = all_subject_data[si][condition]['frequency_pattern'][band][ch]
            for j, sj in enumerate(subjects):
                Xj = all_subject_data[sj][condition]['frequency_pattern'][band][ch]
                if Xi.size == 0 or Xj.size == 0:
                    val = np.nan
                else:
                    if np.allclose(Xi, 0) or np.allclose(Xj, 0):
                        val = np.nan
                    else:
                        c = np.corrcoef(Xi, Xj)
                        val = c[0, 1] if c.shape == (2, 2) else np.nan
                S[ch, i, j] = val
    return S


def plot_cross_subject_recommendations(channel_summary, save_path):
    labels = [c['channel'] for c in channel_summary]
    scores = [c['score'] for c in channel_summary]
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(labels)), scores, alpha=0.7)
    plt.xticks(range(len(labels)), [l.replace('EEG-', '') for l in labels], rotation=90)
    plt.ylabel("Consistency Score")
    plt.title("Cross-Subject Channel Recommendations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Build per-subject store WITH PATTERNS (not scalars)
    # Structure: subject -> {
    #   condition -> {
    #       'temporal_pattern': List[np.ndarray(len=n_times)],
    #       'frequency_pattern': {band: List[np.ndarray(len=n_freqs_band)]}
    #   },
    #   'channel_names': [...]
    # }
    all_subject_data = {}
    for subject_file in helpers.subjects:
        epochs = helpers.load_epochs(subject_file)
        subject_id = subject_file
        channel_names = epochs.ch_names

        subject_results = {}
        conditions = list(epochs.event_id.keys())

        sfreq = epochs.info['sfreq']
        baseline_end_idx = int(1.0 * sfreq)  # 1s baseline ends at t=0

        for condition in conditions:
            data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
            n_trials, n_channels, n_times = data.shape

            # -------- Temporal pattern per channel (mean over aligned trials) --------
            temporal_patterns = []
            for ch in range(n_channels):
                X = data[:, ch, :]                                # (trials, time)
                X = helpers.align_trials_to_mi_onset(X, sfreq, baseline_end_idx)
                pattern = _zscore_1d(np.nanmean(X, axis=0))       # 1D vector
                temporal_patterns.append(pattern)

            # -------- Frequency pattern per channel per band (mean spectrum) --------
            frequency_patterns = {}
            for band, (fmin, fmax) in helpers.freq_bands.items():
                spectrum = epochs[condition].compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                                         n_fft=512, verbose=False)
                psds = spectrum.get_data()                        # (trials, ch, freqs)
                per_ch = []
                for ch in range(n_channels):
                    spec = _zscore_1d(np.nanmean(psds[:, ch, :], axis=0))  # 1D vector
                    per_ch.append(spec)
                frequency_patterns[band] = per_ch

            subject_results[condition] = {
                'temporal_pattern': temporal_patterns,
                'frequency_pattern': frequency_patterns
            }

        all_subject_data[subject_id] = {**subject_results, 'channel_names': channel_names}

    # ---------- Cross-subject correlation matrices ----------
    subjects = list(all_subject_data.keys())
    channel_names = all_subject_data[subjects[0]]['channel_names']
    n_channels = len(channel_names)

    # Rank channels across subjects/conditions/bands by mean cross-subject correlation
    channel_scores = np.zeros(n_channels, dtype=float)

    for condition in list(all_subject_data[subjects[0]].keys()):
        if condition == 'channel_names':
            continue

        # Temporal
        temp_corr = compute_cross_subject_channel_temporal(all_subject_data, condition)
        channel_scores += np.nanmean(temp_corr, axis=(1, 2))

        # Frequency bands
        for band in helpers.freq_bands.keys():
            freq_corr = compute_cross_subject_channel_frequency(all_subject_data, condition, band)
            channel_scores += np.nanmean(freq_corr, axis=(1, 2))

    # Normalize and export
    max_abs = np.nanmax(np.abs(channel_scores))
    if not np.isfinite(max_abs) or max_abs == 0:
        norm_scores = np.zeros_like(channel_scores)
    else:
        norm_scores = channel_scores / max_abs

    channel_summary = [{'channel': ch, 'score': float(sc)} for ch, sc in zip(channel_names, norm_scores)]
    channel_summary.sort(key=lambda x: x['score'], reverse=True)

    # CSV
    import csv
    summary_path = OUTPUT_DIR / 'cross_subject_channel_recommendations.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['channel', 'score'])
        writer.writeheader()
        for row in channel_summary:
            writer.writerow(row)
    print(f"Saved: {summary_path}")

    # Plot
    plot_path = OUTPUT_DIR / 'cross_subject_channel_recommendations.png'
    plot_cross_subject_recommendations(channel_summary, plot_path)
    print(f"Saved cross-subject outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()