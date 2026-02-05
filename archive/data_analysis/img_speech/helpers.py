from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional SciPy for PSD
try:
    from scipy.signal import welch  # type: ignore
except Exception:
    welch = None

# ------------------------------------------------------------------------------------
# Configuration (relative to this file)
# ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
base_path = BASE_DIR / "data_final"  # where mmXX.npy + mmXX_labels.npy live
SFREQ = 1000.0  # Hz (you specified 1000)

# Fixed class order you provided (strings)
CLASS_LIST = ['/diy/', '/iy/', '/m/', '/n/', '/piy/', '/tiy/', '/uw/', 'gnaw', 'knew', 'pat', 'pot']

# Frequency bands used by your friend’s analysis
freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}


# ------------------------------------------------------------------------------------
# Subject discovery + loader
# ------------------------------------------------------------------------------------
def _discover_subjects(data_dir: Path) -> List[str]:
    subs: List[str] = []
    for p in data_dir.glob("*.npy"):
        stem = p.stem
        if stem.endswith("_labels"):
            continue
        if (data_dir / f"{stem}_labels.npy").exists():
            subs.append(stem)
    subs = sorted(set(subs))
    if not subs:
        raise FileNotFoundError(f"No subject pairs found in {data_dir}. "
                                f"Expected files like mm15.npy and mm15_labels.npy.")
    return subs

subjects = _discover_subjects(base_path)


class _SpectrumLite:
    """Mimics MNE Epochs.compute_psd(...).get_data() -> (trials, channels, freqs)"""
    def __init__(self, psds: np.ndarray):
        self._psds = psds
    def get_data(self) -> np.ndarray:
        return self._psds


class _CondView:
    """epochs[condition] -> view with .get_data() and .compute_psd(...)"""
    def __init__(self, data: np.ndarray, sfreq: float):
        self._data = data  # (n_trials, n_channels, n_times)
        self._sfreq = sfreq
    def get_data(self) -> np.ndarray:
        return self._data
    def compute_psd(self, method='welch', fmin=0.0, fmax=None, n_fft=512, verbose=False) -> _SpectrumLite:
        X = self._data  # (trials, ch, time)
        n_trials, n_ch, n_t = X.shape
        if method != 'welch':
            raise NotImplementedError("Only Welch PSD is supported in this adapter")

        if welch is None:
            # Fallback: FFT-based PSD
            freqs = np.fft.rfftfreq(n_fft, d=1.0/self._sfreq)
            mask = (freqs >= fmin) & (freqs <= (fmax if fmax is not None else freqs[-1]))
            psds = []
            for tr in range(n_trials):
                rows = []
                for ch in range(n_ch):
                    x = X[tr, ch]
                    if len(x) < n_fft:
                        x = np.pad(x, (0, n_fft - len(x)))
                    else:
                        x = x[:n_fft]
                    Y = np.fft.rfft(x)
                    Pxx = (np.abs(Y) ** 2) / (self._sfreq * n_fft)
                    rows.append(Pxx[mask])
                psds.append(np.stack(rows, axis=0))
            return _SpectrumLite(np.stack(psds, axis=0))
        else:
            nperseg = min(n_fft, n_t)
            psds = []
            for tr in range(n_trials):
                rows = []
                for ch in range(n_ch):
                    f, Pxx = welch(X[tr, ch], fs=self._sfreq, nperseg=nperseg, noverlap=0, nfft=n_fft)
                    mask = (f >= fmin) & (f <= (fmax if fmax is not None else f[-1]))
                    rows.append(Pxx[mask])
                psds.append(np.stack(rows, axis=0))
            return _SpectrumLite(np.stack(psds, axis=0))


class _EpochsLite:
    """
    Minimal Epochs-like adapter exposing:
      - epochs.info['sfreq']
      - epochs.ch_names
      - epochs.event_id (dict[label -> int]) in fixed CLASS_LIST order where present
      - epochs[condition].get_data()
      - epochs[condition].compute_psd(...)
    """
    def __init__(self, full_data: np.ndarray, labels: np.ndarray, ch_names: List[str], sfreq: float):
        self._data = full_data
        self._labels = labels.astype(str)
        self.ch_names = list(ch_names)
        self.info = {'sfreq': sfreq}

        uniq = list(dict.fromkeys(self._labels.tolist()))
        # Use provided CLASS_LIST order if available; otherwise whatever appears
        order = [c for c in CLASS_LIST if c in uniq] + [c for c in uniq if c not in CLASS_LIST]
        self.event_id = {lab: i for i, lab in enumerate(order)}
        self._by_label: Dict[str, np.ndarray] = {
            lab: np.where(self._labels == lab)[0] for lab in order if np.any(self._labels == lab)
        }

    def __getitem__(self, label: str) -> _CondView:
        if label not in self._by_label:
            return _CondView(self._data[:0], self.info['sfreq'])
        idx = self._by_label[label]
        return _CondView(self._data[idx], self.info['sfreq'])


def load_epochs(subject_file: str) -> _EpochsLite:
    """Load one Imagine Speech subject as an Epochs-like object."""
    data_path = base_path / f"{subject_file}.npy"
    labels_path = base_path / f"{subject_file}_labels.npy"
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing: {data_path} and/or {labels_path}")

    data = np.load(data_path, allow_pickle=False)          # (trials, 62, 4900)
    labels = np.load(labels_path, allow_pickle=True).ravel()  # strings

    if data.ndim != 3 or data.shape[1] != 62 or data.shape[2] != 4900:
        raise ValueError(f"Expected shape (trials, 62, 4900), got {data.shape}")
    if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
        raise ValueError(f"Labels length {labels.shape[0]} != trials {data.shape[0]}")

    ch_names = [f"Ch{i+1}" for i in range(62)]
    return _EpochsLite(data, labels, ch_names, SFREQ)


# ------------------------------------------------------------------------------------
# Alignment helper used in your friend’s analysis
# ------------------------------------------------------------------------------------
def align_trials_to_mi_onset(channel_trials: np.ndarray, sfreq: float, baseline_end_idx: int) -> np.ndarray:
    """
    Given (n_trials, n_times) for a single channel, detect an onset per trial via
    max |d/dt| after baseline, align to the median onset, and pad/crop to common length.
    """
    X = np.asarray(channel_trials)
    n_trials, n_t = X.shape
    if n_trials == 0:
        return X.copy()

    # simple onset: argmax of first-derivative magnitude after baseline
    dX = np.abs(np.diff(X, axis=1, prepend=X[:, :1]))
    start = max(baseline_end_idx, 0)
    onset_idx = start + np.argmax(dX[:, start:], axis=1)
    ref = int(np.median(onset_idx))

    shifted = []
    for tr in range(n_trials):
        shift = ref - onset_idx[tr]
        if shift > 0:
            y = np.pad(X[tr], (shift, 0), mode='edge')[:n_t]
        elif shift < 0:
            y = np.pad(X[tr], (0, -shift), mode='edge')[-shift:n_t-shift]
            if y.shape[0] < n_t:
                y = np.pad(y, (0, n_t - y.shape[0]), mode='edge')
        else:
            y = X[tr]
        shifted.append(y)
    return np.vstack(shifted)