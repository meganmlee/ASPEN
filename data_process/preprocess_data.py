import os
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt

def preprocess_lee2019(raw_dir, save_dir, subjects=range(1, 55)):
    """
    Standardizes Lee2019 data to the MI paradigm: 22 channels, 250Hz, 4s epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    fs_new = 250
    
    # Specific indices for FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, 
    # CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz, and Fz
    chan_idx = [32, 8, 4, 9, 33, 34, 12, 35, 13, 36, 14, 37, 38, 18, 39, 19, 40, 41, 24, 42, 43, 44]

    for sid in subjects:
        sub_x, sub_y = [], []
        print(f"Processing Subject {sid}...")
        
        for sess in [1, 2]:
            file_path = os.path.join(raw_dir, f'session{sess}', f's{sid}', f'sess{sess:02d}_subj{sid:02d}_EEG_MI.mat')
            if not os.path.exists(file_path): continue
            
            mat = scipy.io.loadmat(file_path)
            for key in ['EEG_MI_train', 'EEG_MI_test']:
                if key not in mat: continue
                struct = mat[key][0, 0]
                raw_x = struct['x']          # Shape: (Time, 62)
                labels = struct['y_dec'].flatten() - 1 # 0=Right, 1=Left
                timestamps = struct['t'].flatten() # Trial start samples
                
                for i, start_sample in enumerate(timestamps):
                    start = int(start_sample) - 1
                    end = start + 4000 # 4s window at 1000Hz
                    
                    if end <= raw_x.shape[0]:
                        # Select only the 22 motor-relevant channels
                        trial = raw_x[start:end, chan_idx].T # Shape: (22, 4000)
                        
                        # 1. Downsample (1000Hz -> 250Hz)
                        trial = signal.resample(trial, 1000, axis=-1)
                        
                        # 2. Bandpass filter (4-40Hz) for MI mu/beta rhythms
                        b, a = signal.butter(4, [4, 40], btype='bandpass', fs=fs_new)
                        trial = signal.filtfilt(b, a, trial, axis=-1)
                        
                        # 3. Z-score normalization
                        trial = (trial - np.mean(trial)) / (np.std(trial) + 1e-8)
                        
                        sub_x.append(trial.astype(np.float32)) # Save as float32 to save space
                        sub_y.append(labels[i])
        
        if sub_x:
            save_path = os.path.join(save_dir, f"S{sid}_preprocessed.npz")
            np.savez(save_path, X=np.array(sub_x), y=np.array(sub_y))
            print(f"  Saved: {save_path} | Final Shape: {np.array(sub_x).shape}")

def preprocess_lee2019_ssvep(raw_dir, save_dir, subjects=range(1, 55), skip_existing=True):
    """
    Refactored Lee2019 Preprocessing for more rigor.
    
    Changes:
    1. Continuous Filtering: Filters the whole session to avoid trial-edge artifacts.
    2. Harder Window: Reduced from 4.0s to 1.0s (250 samples after downsampling).
    3. Proper Normalization: Z-score applied per trial.
    """
    os.makedirs(save_dir, exist_ok=True)
    fs_orig = 1000
    fs_new = 250
    # Window settings (Hard Mode: 1.0s)
    window_sec = 1.0 
    window_samples_orig = int(fs_orig * window_sec) # 1000
    chan_idx = list(range(62)) 

    # Design filter once (6-90 Hz)
    b, a = signal.butter(4, [6, 90], btype='bandpass', fs=fs_orig)

    for sid in subjects:
        save_path = os.path.join(save_dir, f"S{sid}_preprocessed.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping Subject {sid}: File already exists.")
            continue
            
        sub_x, sub_y = [], []
        print(f"Processing Subject {sid}...")
        
        for sess in [1, 2]:
            file_path = os.path.join(raw_dir, f'session{sess}', f's{sid}', f'sess{sess:02d}_subj{sid:02d}_EEG_SSVEP.mat')
            if not os.path.exists(file_path): 
                continue
            
            mat = scipy.io.loadmat(file_path)
            for key in ['EEG_SSVEP_train', 'EEG_SSVEP_test']:
                if key not in mat: 
                    continue
                    
                struct = mat[key][0, 0]
                raw_x = struct['x']          # (TotalSamples, Channels)
                labels = struct['y_dec'].flatten() - 1 
                timestamps = struct['t'].flatten() 
                
                # --- Filter CONTINUOUS data to prevent edge-artifact leakage ---
                # We filter along the time axis (axis=0)
                raw_filtered = signal.filtfilt(b, a, raw_x, axis=0)

                # --- Segment filtered data into trials ---
                for i, start_sample in enumerate(timestamps):
                    start = int(start_sample) - 1
                    end = start + window_samples_orig 
                    
                    if end <= raw_filtered.shape[0]:
                        # Extract channels and transpose to (Channels, Time)
                        trial = raw_filtered[start:end, chan_idx].T 
                        
                        # --- Downsample (1000Hz -> 250Hz) ---
                        num_samples_new = int(trial.shape[-1] * (fs_new / fs_orig))
                        trial = signal.resample(trial, num_samples_new, axis=-1)
                        
                        # --- Z-score Normalization ---
                        trial = (trial - np.mean(trial)) / (np.std(trial) + 1e-8)
                        
                        sub_x.append(trial.astype(np.float32))
                        sub_y.append(labels[i])
        
        if sub_x:
            X_arr = np.array(sub_x)
            y_arr = np.array(sub_y)
            np.savez(save_path, X=X_arr, y=y_arr)
            print(f"  Saved S{sid} | Shape: {X_arr.shape}")

def preprocess_epflp300(raw_dir, save_dir, subjects=[1, 2, 3, 4, 6, 7, 8, 9], skip_existing=True):
    os.makedirs(save_dir, exist_ok=True)
    
    fs_orig = 2048 
    fs_new = 256   
    t_start, t_end = -0.2, 0.8
    window_samples_orig = int((t_end - t_start) * fs_orig)
    offset_samples_orig = int(abs(t_start) * fs_orig) 
    
    chan_idx = list(range(32)) 
    
    sos = signal.butter(4, [0.1, 30], btype='bandpass', fs=fs_orig, output='sos')

    for sid in subjects:
        save_path = os.path.join(save_dir, f"S{sid}_preprocessed.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping Subject {sid}: Already exists.")
            continue
            
        sub_x, sub_y = [], []
        print(f"Processing Subject {sid}...")
        
        for sess in range(1, 5):
            sess_dir = os.path.join(raw_dir, f'subject{sid}', f'session{sess}')
            if not os.path.exists(sess_dir):
                continue
            
            mat_files = sorted([f for f in os.listdir(sess_dir) if f.endswith('.mat')])
            
            for mat_file in mat_files:
                file_full_path = os.path.join(sess_dir, mat_file)
                mat = scipy.io.loadmat(file_full_path)
                
                if 'data' not in mat or 'events' not in mat:
                    continue

                # Load and Transpose (Time, Channels)
                data = mat['data'].T 
                
                target_id = mat['target'][0, 0]
                stimuli_seq = mat['stimuli'].flatten()
                events = mat['events']

                raw_filtered = signal.sosfiltfilt(sos, data, axis=0)

                for i in range(len(stimuli_seq)):
                    stamp = int(events[i, 0]) - 1
                    label = 1 if stimuli_seq[i] == target_id else 0
                    
                    start = stamp - offset_samples_orig
                    end = start + window_samples_orig
                    
                    if start >= 0 and end <= raw_filtered.shape[0]:
                        trial = raw_filtered[start:end, chan_idx].T 
                        
                        # Baseline Correction
                        baseline = np.mean(trial[:, :offset_samples_orig], axis=1, keepdims=True)
                        trial = trial - baseline
                        
                        # Downsample
                        num_samples_new = int(trial.shape[-1] * (fs_new / fs_orig))
                        trial = signal.resample(trial, num_samples_new, axis=-1)
                        
                        # Z-score Normalization (Now safe from overflow)
                        std_val = np.std(trial)
                        if std_val > 1e-8:
                            trial = (trial - np.mean(trial)) / std_val
                        
                        sub_x.append(trial.astype(np.float32))
                        sub_y.append(label)

        if sub_x:
            X_arr = np.array(sub_x)
            y_arr = np.array(sub_y)
            np.savez(save_path, X=X_arr, y=y_arr)
            print(f"  >>> SAVED: S{sid} | Shape: {X_arr.shape} | Targets: {np.sum(y_arr)}")

def preprocess_bi2014b(raw_dir, save_dir, groups=range(1, 20), skip_existing=True):
    """
    Preprocessing for BrainInvaders 2014b.
    Uses search-based string extraction to handle Subject IDs reliably.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fs_orig = 512 
    fs_new = 256   
    t_start, t_end = -0.2, 0.8
    window_samples_orig = int((t_end - t_start) * fs_orig)
    offset_samples_orig = int(abs(t_start) * fs_orig) 
    
    chan_idx = list(range(1, 33)) # Subject A EEG channels (1-32)
    label_col = 66               # Target/Non-Target column
    
    # Using SOS for numerical stability at 512Hz
    sos = signal.butter(4, [0.1, 30], btype='bandpass', fs=fs_orig, output='sos')

    for gid in groups:
        # Construct group folder path
        group_path = os.path.join(raw_dir, f'group_{gid:02d}')
        if not os.path.exists(group_path):
            continue
            
        mat_files = sorted([f for f in os.listdir(group_path) if f.endswith('.mat') and 'sujet' in f])
        
        for mat_file in mat_files:
            # ROBUST PARSING: Extract Subject ID regardless of part index
            # Example: 'group_01_sujet_02.mat' -> G01_S02
            parts = mat_file.replace('.mat', '').split('_')
            
            try:
                # Find the numbers following 'group' and 'sujet'
                g_num = parts[parts.index('group') + 1]
                s_num = parts[parts.index('sujet') + 1]
                sid_str = f"G{g_num}_S{s_num}"
            except (ValueError, IndexError):
                print(f"Skipping malformed filename: {mat_file}")
                continue

            save_path = os.path.join(save_dir, f"{sid_str}_preprocessed.npz")
            if skip_existing and os.path.exists(save_path):
                continue

            print(f"Processing {mat_file}...")
            mat = scipy.io.loadmat(os.path.join(group_path, mat_file))
            
            # Extract matrix and continuous labels
            samples = mat['samples']
            eeg_data = samples[:, chan_idx]
            raw_filtered = signal.sosfiltfilt(sos, eeg_data, axis=0)
            
            trigger_stream = samples[:, label_col]
            # Flash onsets where trigger goes from 0 to 1 (Non-Target) or 2 (Target)
            onsets = np.where((trigger_stream[:-1] == 0) & (trigger_stream[1:] > 0))[0] + 1
            
            sub_x, sub_y = [], []
            for start in onsets:
                val = trigger_stream[start]
                if val not in [1, 2]: continue
                
                # Standardize: 1 for Target (val=2), 0 for Non-Target (val=1)
                label = 1 if val == 2 else 0
                
                t_start_idx = start - offset_samples_orig
                t_end_idx = start + (window_samples_orig - offset_samples_orig)
                
                if t_start_idx >= 0 and t_end_idx <= raw_filtered.shape[0]:
                    trial = raw_filtered[t_start_idx:t_end_idx, :].T 
                    
                    # Baseline Correction
                    trial = trial - np.mean(trial[:, :offset_samples_orig], axis=1, keepdims=True)
                    
                    # Downsample to 256Hz
                    num_samples_new = int(trial.shape[-1] * (fs_new / fs_orig))
                    trial = signal.resample(trial, num_samples_new, axis=-1)
                    
                    # Z-score Normalization
                    trial = (trial - np.mean(trial)) / (np.std(trial) + 1e-8)
                    
                    sub_x.append(trial.astype(np.float32))
                    sub_y.append(label)

            if sub_x:
                X_arr = np.array(sub_x)
                y_arr = np.array(sub_y)
                np.savez(save_path, X=X_arr, y=y_arr)
                print(f"  >>> SAVED: {sid_str} | Shape: {X_arr.shape} | Targets: {np.sum(y_arr)}")

# --- RUN EXECUTION ---

# RAW_SSVEP = '/ocean/projects/cis250213p/shared/mne_data/MNE-lee2019-ssvep-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542'
# PROCESSED_SSVEP = '/ocean/projects/cis250213p/shared/lee2019_ssvep_processed'

# preprocess_lee2019_ssvep(RAW_SSVEP, PROCESSED_SSVEP)

# RAW = '/ocean/projects/cis250213p/shared/lee2019_mi'
# PROCESSED = '/ocean/projects/cis250213p/shared/lee2019_processed'
# PLOTS = '/ocean/projects/cis250213p/shared/lee2019_plots'

# preprocess_lee2019(RAW, PROCESSED)

# RAW_EPFL = '/ocean/projects/cis250213p/shared/mne_data/MNE-epflp300-data/groups/m/mm/mmspg/www/BCI/p300'
# PROCESSED_EPFL = '/ocean/projects/cis250213p/shared/epfl_p300'

# # Subject 5 is missing; we only process 1-4 and 6-9
# EPFL_SUBJECTS = [1, 2, 3, 4, 6, 7, 8, 9]

# preprocess_epflp300(RAW_EPFL, PROCESSED_EPFL, subjects=EPFL_SUBJECTS)

RAW_SSVEP = '/ocean/projects/cis250213p/shared/mne_data/MNE-braininvaders2014b-data/record/3267302/files'
PROCESSED_SSVEP = '/ocean/projects/cis250213p/shared/bi2014b_processed'

preprocess_bi2014b(RAW_SSVEP, PROCESSED_SSVEP)