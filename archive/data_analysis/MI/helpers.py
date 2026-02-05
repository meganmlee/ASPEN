import os
import pandas as pd
import mne
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

TASK = 'SSVEP'  # Change to: 'MI', 'SSVEP', 'P300', 'Imagined_speech'

# Data paths (update these)
DATA_PATHS = {
    'MI': '/content/drive/MyDrive/IDL/IDL Project Team 5 F25/dataset/MI/cleaned_data',
    'SSVEP': '/home/megan/Downloads/cleaneddatas',
    'P300': '/home/megan/Downloads/p300',
    'Imagined_speech': '/content/drive/MyDrive/IDL/IDL Project Team 5 F25/dataset/speech_imagined/KARA_ONE/epochs/notched'
}

OUTPUT_DIR = Path(f'//home/megan/Downloads/p300/cross_trial_{TASK}')
OUTPUT_DIR.mkdir(exist_ok=True)

# Task-specific parameters
TASK_CONFIG = {
    'MI': {'sfreq': 250, 'n_classes': 4, 'use_alignment': True},
    'SSVEP': {'sfreq': 250, 'n_classes': 26, 'use_alignment': False},
    'P300': {'sfreq': 256, 'n_classes': 2, 'use_alignment': False},
    'Imagined_speech': {'sfreq': 128, 'n_classes': 11, 'use_alignment': False}
}

freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}

# ============================================================================
# Load data
# ============================================================================

def load_all_subjects_for_task(task, root_dir, **kwargs):
    """
    Load ALL data for all subjects (not split) - for data analysis purposes.
    Returns data organized by subject and condition.
    """
    if task == 'MI':
        return load_all_MI_data(root_dir, kwargs.get('label_dir'))
    elif task == 'SSVEP':
        return load_all_SSVEP_data(root_dir)
    elif task == 'P300':
        return load_all_P300_data(root_dir)
    elif task == 'Imagined_speech':
        return load_all_ImaginedSpeech_data(root_dir)


def load_all_MI_data(root_dir, label_dir=None):
    """Load all MI subjects without splitting - reuse dataset.py loading logic"""
    import numpy as np
    import os
    
    subject_data = {}
    
    # Load all 9 subjects (A01-A09)
    for subject_id in range(1, 10):
        try:
            # Load both sessions for each subject
            X, Y = MI_load_data_by_session(
                root_dir, subject_id, 
                ["first_session"], 
                label_dir
            )
            
            # Convert to numpy
            X = X.cpu().numpy()  # (n_trials, n_channels, n_times)
            Y = Y.cpu().numpy()
            
            # Group by condition
            subject_data[subject_id] = {}
            for label in np.unique(Y):
                mask = Y == label
                subject_data[subject_id][int(label)] = X[mask]
            
            print(f"Loaded Subject {subject_id}: {len(Y)} trials, {len(np.unique(Y))} conditions")
        
        except Exception as e:
            print(f"Could not load subject {subject_id}: {e}")
    
    return subject_data


def load_all_SSVEP_data(root_dir):
    """Load all SSVEP subjects without splitting"""
    import numpy as np
    import os
    
    subject_data = {}
    
    # Load all 35 subjects
    for subject_id in range(1, 36):
        try:
            # Load all 6 sessions
            X, Y = load_data_by_session(root_dir, subject_id, [0, 1, 2, 3, 4, 5])
            
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            
            # Group by condition (26 letters)
            subject_data[subject_id] = {}
            for label in np.unique(Y):
                mask = Y == label
                subject_data[subject_id][int(label)] = X[mask]
            
            print(f"Loaded Subject {subject_id}: {len(Y)} trials")
        
        except Exception as e:
            print(f"Could not load subject {subject_id}: {e}")
    
    return subject_data


def load_all_P300_data(root_dir):
    """Load all P300 subjects without splitting"""
    import numpy as np
    
    subject_data = {}
    
    # Load all subjects (typically 1-8)
    for subject_id in range(1, 44):
        try:
            data = P300_load_subject_data(subject_id, root_dir)
            
            X = data['X']  # Already numpy
            Y = data['Y']
            
            # Group by condition
            subject_data[subject_id] = {}
            for label in np.unique(Y):
                mask = Y == label
                subject_data[subject_id][int(label)] = X[mask]
            
            print(f"Loaded Subject {subject_id}: {len(Y)} trials")
        
        except Exception as e:
            print(f"Could not load subject {subject_id}: {e}")
    
    return subject_data


def load_all_ImaginedSpeech_data(root_dir):
    """Load all Imagined Speech subjects without splitting"""
    import numpy as np
    import os
    import re
    
    subject_data = {}
    
    # Find all available subjects from file names
    all_subjects = sorted([
        re.findall(r'epochs_(.*)\.npy', f)[0].replace("_notched", "")
        for f in os.listdir(root_dir)
        if f.startswith("epochs_") and f.endswith(".npy")
    ])
    
    for subject_id in all_subjects:
        try:
            data = ImaginedSpeech_load_subject_data(subject_id, root_dir)
            
            X = data['X']  # Already numpy
            Y = data['Y']
            
            # Group by condition
            subject_data[subject_id] = {}
            for label in np.unique(Y):
                mask = Y == label
                subject_data[subject_id][int(label)] = X[mask]
            
            print(f"Loaded Subject {subject_id}: {len(Y)} trials")
        
        except Exception as e:
            print(f"Could not load subject {subject_id}: {e}")
    
    return subject_data

def load_data_by_session(root_dir, subject_id, session_idx_list):
    data = np.load(os.path.join(root_dir, f"S{subject_id}_chars.npy"))
    data = data[:, session_idx_list]
    X = data.reshape(-1, 64, 250)
    Y = np.repeat(np.arange(26), len(session_idx_list))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

# --------- MI ---------
def MI_load_data_by_session(root_dir, subject_id, session_folders, label_dir):
    """
    root_dir/
      first_session/
        A01T_cleaned.fif … A09T_cleaned.fif
      second_session/
        A01E_cleaned.fif … A09E_cleaned.fif

    session_folders: list of folder names, e.g. ["first_session"] or ["second_session"]
    """
    X_list, Y_list = [], []

    prefix = "T"
    fname = f"A{subject_id:02d}{prefix}.fif"
    fpath = os.path.join(root_dir, "first_session", fname)
    raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

    # MI cue as '769'~'772'，mapping as 0–3 labels
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    motor_keys = ['769', '770', '771', '772']
    motor_event_id = {k: v for k, v in event_id.items() if k in motor_keys}
    if len(motor_event_id) < 4:
        raise ValueError(f"{fname} missing MI cues. Found: {event_id}")
    events = np.array([e for e in events if e[2] in motor_event_id.values()])
    label_map = {
        motor_event_id['769']: 0,
        motor_event_id['770']: 1,
        motor_event_id['771']: 2,
        motor_event_id['772']: 3,
    }
    labels = np.array([label_map[e[-1]] for e in events])

    epochs = mne.Epochs(
        raw, events,
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        verbose=False,
        event_repeated="drop"
    )
    data = epochs.get_data()
    X_list.append(torch.from_numpy(data).float())
    Y_list.append(torch.from_numpy(labels).long())

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    return X, Y

# --------- P300 ---------
def P300_load_subject_data(subject_id, root_dir):
    folder = os.path.join(root_dir, f"subject_{subject_id:02d}")
    X = np.load(os.path.join(folder, "X.npy"))                # shape: (n_trials, C, T)
    Y = np.load(os.path.join(folder, "y.npy"))                # shape: (n_trials,)
    Y = np.array([1 if label == 'Target' else 0 for label in Y])

    meta = pd.read_csv(os.path.join(folder, "metadata.csv"))  # contains at least 'session'

    trials_per_repetition = 12
    reps_per_level = 8
    trials_per_level = reps_per_level * trials_per_repetition  # 96
    levels_per_session = 9

    level_list = []
    repetition_list = []

    for sess in sorted(meta["session"].unique()):
        session_idxs = meta.index[meta["session"] == sess].tolist()
        for i, idx in enumerate(session_idxs):
            rep = i // trials_per_repetition
            level = rep // reps_per_level
            repetition = rep % reps_per_level
            level_list.append(level)
            repetition_list.append(repetition)

    meta["level"] = level_list
    meta["repetition"] = repetition_list

    return {
        "X": X,
        "Y": Y,
        "session": meta["session"].tolist(),
        "level": meta["level"].tolist(),
        "repetition": meta["repetition"].tolist()
    }

# --------- Imagined_speech ---------
def ImaginedSpeech_load_subject_data(subject_id, root_dir):
    x_path = os.path.join(root_dir, f"epochs_{subject_id}_notched.npy")
    y_path = os.path.join(root_dir, f"labels_{subject_id}.npy")

    X = np.load(x_path)  # shape: (n_trials, C, T)
    Y_raw = np.load(y_path, allow_pickle=True)  # shape: (n_trials,), string labels

    Y_raw = Y_raw.flatten()

    # Map labels to level indices
    label_set = sorted(set(Y_raw.tolist()))
    label2idx = {label: i for i, label in enumerate(label_set)}  # consistent across subjects

    level = [label2idx[label] for label in Y_raw]

    # Build repetition index for each stimulus
    counter = defaultdict(int)
    repetition = []
    for label in Y_raw:
        repetition.append(counter[label])
        counter[label] += 1

    # All trials are from a single session
    session = [0] * len(Y_raw)

    return {
        "X": X,
        "Y": np.array(level),
        "session": session,
        "level": level,
        "repetition": repetition
    }


# Configuration
base_path = Path('/home/megan/Downloads/cleaneddata/first_session')
subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']

event_mapping = {'Left hand': '769', 'Right hand': '770', 'Feet': '771', 'Tongue': '772'}
freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}


def load_epochs( subject_file):
    """Load epochs for a subject."""
    data = mne.io.read_raw_fif(base_path / subject_file, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(data)

    event_id = {label: event_dict[code] for label, code in event_mapping.items() if code in event_dict}

    epochs = mne.Epochs(data, events, event_id=event_id, tmin=-1, tmax=4,
                        baseline=(-1, 0), preload=True, picks='eeg', verbose=False)
    return epochs

def find_mi_window(trial_data, sfreq, baseline_end_idx):
    """
    Find the active MI period in a trial by detecting when power changes from baseline.
    Returns start and end indices of the MI window.
    """
    # Compute power (squared signal)
    power = trial_data ** 2

    # Get baseline power (before t=0)
    baseline_power = np.mean(power[:baseline_end_idx])
    baseline_std = np.std(power[:baseline_end_idx])

    # Find regions where power exceeds baseline + 2*std
    threshold = baseline_power + 2 * baseline_std
    active = power > threshold

    # Find the first sustained active period (at least 500ms)
    min_duration = int(0.5 * sfreq)
    active_start = None

    for i in range(baseline_end_idx, len(active) - min_duration):
        if np.sum(active[i:i+min_duration]) > 0.7 * min_duration:
            active_start = i
            break

    if active_start is None:
        # If no clear onset, use fixed window after cue
        active_start = baseline_end_idx + int(0.5 * sfreq)

    # MI window: 1.5 seconds from detected start
    active_end = min(active_start + int(1.5 * sfreq), len(trial_data))

    return active_start, active_end

def align_trials_to_mi_onset(data, sfreq, baseline_end_idx):
    """
    Align all trials to their detected MI onset.
    Returns aligned data.
    """
    n_trials, n_times = data.shape
    aligned_trials = []

    # Find MI windows for all trials
    windows = []
    for trial in data:
        start, end = find_mi_window(trial, sfreq, baseline_end_idx)
        windows.append((start, end))

    # Find common window length (use median)
    window_lengths = [end - start for start, end in windows]
    target_length = int(np.median(window_lengths))

    # Extract aligned windows
    for trial, (start, end) in zip(data, windows):
        if end - start >= target_length:
            aligned_trials.append(trial[start:start+target_length])
        else:
            # Pad if needed
            segment = trial[start:end]
            padded = np.pad(segment, (0, target_length - len(segment)), mode='edge')
            aligned_trials.append(padded)

    return np.array(aligned_trials)
