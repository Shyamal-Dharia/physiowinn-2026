#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import edfio
import csv
import json
import numpy as np
import os
from fractions import Fraction
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm

from helper_code import *

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

EEG_CHANNELS = ('f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1')
EEG_DERIVATIONS = {
    'f3-m2': ('f3', 'm2'),
    'f4-m1': ('f4', 'm1'),
    'c3-m2': ('c3', 'm2'),
    'c4-m1': ('c4', 'm1'),
    'o1-m2': ('o1', 'm2'),
    'o2-m1': ('o2', 'm1'),
}
TARGET_EEG_FS = 200.0
EEG_BANDPASS_HZ = (0.1, 45.0)
EEG_SEGMENT_SECONDS = 4.0
EEG_NUM_WINDOWS = 600
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 512
CNN_MODEL_FILE = 'cnn_model.pt'
JEPA_MSE_WEIGHT = 0.1
SIGREG_WEIGHT = 0.1
EEG_VIEW_NOISE_STD = 0.02
EEG_VIEW_CHANNEL_DROP = 0.1
DEFAULT_THRESHOLD_SCALE = 3.85
ENSEMBLE_SEEDS = (0, 1, 2)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################




################################################################################
#
# MODEL
#
################################################################################


































############################################################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    import torch

    os.makedirs(model_folder, exist_ok=True)
    cache_folder = os.path.join(model_folder, 'eeg_cache')

    if verbose:
        print('Building EEG cache...')
    meta = build_eeg_cache(data_folder, cache_folder, num_windows=EEG_NUM_WINDOWS, verbose=verbose)
    if meta['subjects_kept'] == 0:
        raise RuntimeError('No subjects with usable EEG windows.')

    model_state_dicts = []
    histories = []
    best_epochs = []
    for seed in ENSEMBLE_SEEDS:
        if verbose:
            print(f'Training EEG CNN seed {seed}...')
        model, history = train_eeg_baseline(
            cache_folder,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            data_folder=data_folder,
            num_workers=2,
            seed=seed,
        )
        model_state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        histories.append(history)
        best_epochs.append(max(history, key=auroc_selection_score))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = np.load(os.path.join(cache_folder, 'y.npy')).astype(np.int8)
    ages = load_cache_subject_ages(cache_folder, data_folder)
    checkpoint = {
        'model_state_dicts': model_state_dicts,
        'pos_weight': float(np.mean([row.get('pos_weight', 1.0) for row in best_epochs])),
        'threshold_scale': float(np.mean([row.get('reward_best_scale', DEFAULT_THRESHOLD_SCALE) for row in best_epochs])),
        'prevalence_labels': labels,
        'prevalence_ages': ages,
        'fallback_prevalence': float(labels.mean()),
        'channels': list(EEG_CHANNELS),
        'target_fs': TARGET_EEG_FS,
        'bandpass': EEG_BANDPASS_HZ,
        'num_windows': EEG_NUM_WINDOWS,
        'histories': histories,
    }
    torch.save(checkpoint, os.path.join(model_folder, CNN_MODEL_FILE))
    import shutil
    shutil.rmtree(cache_folder, ignore_errors=True)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    import torch

    checkpoint = torch.load(os.path.join(model_folder, CNN_MODEL_FILE), map_location='cpu', weights_only=False)
    state_dicts = checkpoint.get('model_state_dicts')
    if state_dicts is None:
        state_dicts = [checkpoint['model_state_dict']]
    checkpoint['models'] = []
    for state_dict in state_dicts:
        net = make_small_eeg_cnn()
        net.load_state_dict(state_dict)
        net.eval()
        checkpoint['models'].append(net)
    return checkpoint

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    import torch

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    age = load_age(patient_data)
    prevalence = estimate_age_prevalence(
        age,
        model['prevalence_labels'],
        model['prevalence_ages'],
        fallback=model['fallback_prevalence'],
    )
    if not np.isfinite(prevalence):
        prevalence = float(model['fallback_prevalence'])
    prevalence = float(np.clip(prevalence, 1e-6, 1 - 1e-6))

    probability_output = prevalence
    edf_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(edf_path):
        try:
            eeg_signals, fs = load_eeg_signals(
                edf_path,
                channels=tuple(model['channels']),
                target_fs=model['target_fs'],
                bandpass=tuple(model['bandpass']) if model['bandpass'] is not None else None,
            )
            windows = sample_eeg_windows(
                eeg_signals,
                fs,
                num_windows=int(model['num_windows']),
                channels=tuple(model['channels']),
                rng=0,
            )
            if windows.shape == (int(model['num_windows']), len(model['channels']), int(round(EEG_SEGMENT_SECONDS * model['target_fs']))):
                windows = (windows - windows.mean(axis=-1, keepdims=True)) / (windows.std(axis=-1, keepdims=True) + 1e-6)
                with torch.no_grad():
                    x = torch.from_numpy(windows.astype(np.float32, copy=False))
                    probability_output = predict_ensemble_probability(model['models'], x)
        except Exception as exc:
            if verbose:
                print(f'EEG inference failed for {patient_id}: {exc}')

    if not np.isfinite(probability_output):
        probability_output = prevalence
    probability_output = float(np.clip(probability_output, 0.0, 1.0))
    # ponytail: single CV-tuned scale; replace with held-out calibration if leaderboard drift persists.
    threshold = np.clip(prevalence * float(model.get('threshold_scale', DEFAULT_THRESHOLD_SCALE)), 1e-6, 1 - 1e-6)
    binary_output = bool(probability_output > threshold)
    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def _standardized_edf_signals(edf, csv_path=DEFAULT_CSV_PATH):
    raw_to_signal = {sig.label.lower().strip(): sig for sig in edf.signals}
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, dropped = standardize_channel_names_rename_only(list(raw_to_signal), rename_rules)

    signals = {}
    for raw_label, signal in raw_to_signal.items():
        if raw_label in dropped:
            continue
        signals.setdefault(rename_map.get(raw_label, raw_label), signal)
    return signals


def _resample_signal(signal, fs, target_fs):
    signal = np.asarray(signal, dtype=np.float32)
    if target_fs is None or abs(float(fs) - float(target_fs)) < 1e-6:
        return signal

    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(1000)
    return resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32, copy=False)


def _bandpass_signal(signal, fs, bandpass):
    if bandpass is None:
        return signal

    low, high = bandpass
    nyquist = 0.5 * float(fs)
    low = max(float(low), 0.0)
    high = min(float(high), 0.99 * nyquist)

    if low <= 0 and high >= nyquist:
        return signal
    if low > 0 and high > low:
        sos = butter(4, [low / nyquist, high / nyquist], btype='bandpass', output='sos')
    elif low > 0:
        sos = butter(4, low / nyquist, btype='highpass', output='sos')
    elif high > 0:
        sos = butter(4, high / nyquist, btype='lowpass', output='sos')
    else:
        return signal

    return sosfiltfilt(sos, signal).astype(np.float32, copy=False)


def _read_eeg_signal(signal, target_fs=TARGET_EEG_FS, bandpass=EEG_BANDPASS_HZ):
    fs = float(signal.sampling_frequency)
    out_fs = float(target_fs or fs)
    data = _resample_signal(signal.data, fs, target_fs)
    data = _bandpass_signal(data, out_fs, bandpass)
    return data


def load_eeg_signals(edf_path, csv_path=DEFAULT_CSV_PATH, channels=EEG_CHANNELS, target_fs=TARGET_EEG_FS, bandpass=EEG_BANDPASS_HZ):
    """
    Load only canonical bipolar EEG channels from an EDF file.

    Returns:
        eeg_signals: {channel_name: np.float32 signal}
        fs: target sampling frequency
    """
    edf = edfio.read_edf(edf_path, lazy_load_data=True)
    signal_map = _standardized_edf_signals(edf, csv_path=csv_path)
    out_fs = float(target_fs) if target_fs is not None else None
    eeg_signals = {}

    for channel in channels:
        if channel in signal_map:
            eeg_signals[channel] = _read_eeg_signal(signal_map[channel], target_fs, bandpass)
            continue

        if channel not in EEG_DERIVATIONS:
            continue

        pos, neg = EEG_DERIVATIONS[channel]
        if pos not in signal_map or neg not in signal_map:
            continue

        pos_fs = float(signal_map[pos].sampling_frequency)
        neg_fs = float(signal_map[neg].sampling_frequency)
        if target_fs is None and abs(pos_fs - neg_fs) >= 1e-6:
            continue

        pos_signal = _read_eeg_signal(signal_map[pos], target_fs, None)
        neg_signal = _read_eeg_signal(signal_map[neg], target_fs, None)
        n = min(len(pos_signal), len(neg_signal))
        eeg_signals[channel] = _bandpass_signal(pos_signal[:n] - neg_signal[:n], out_fs or pos_fs, bandpass)

    return eeg_signals, out_fs


def is_clean_eeg_segment(segment, min_std=1e-3, max_abs=1000.0):
    # ponytail: coarse artifact gate; replace with dataset-calibrated QC after plotting segment stats.
    return (
        len(segment) > 1
        and np.isfinite(segment).all()
        and np.std(segment) >= min_std
        and np.max(np.abs(segment)) <= max_abs
    )


def iter_eeg_segments(eeg_signals, fs, segment_seconds=EEG_SEGMENT_SECONDS, step_seconds=None):
    window = int(round(segment_seconds * fs))
    step = int(round((step_seconds or segment_seconds) * fs))
    for channel, signal in eeg_signals.items():
        for start in range(0, len(signal) - window + 1, step):
            segment = signal[start:start + window]
            if is_clean_eeg_segment(segment):
                yield channel, start / fs, segment


def _clean_multichannel_eeg_window(eeg_signals, channels, start, window):
    return all(is_clean_eeg_segment(eeg_signals[channel][start:start + window]) for channel in channels)


def sample_eeg_windows(eeg_signals, fs, num_windows=200, channels=EEG_CHANNELS, segment_seconds=EEG_SEGMENT_SECONDS, rng=None, max_attempts=None):
    """
    Return fixed 6-channel EEG windows for one subject.

    Shape: (N, C, T), usually (200, 6, 800).
    Pass rng for random windows; omit rng for evenly spaced windows.
    """
    if any(channel not in eeg_signals for channel in channels):
        return np.empty((0, len(channels), int(round(segment_seconds * fs))), dtype=np.float32)

    window = int(round(segment_seconds * fs))
    max_len = min(len(eeg_signals[channel]) for channel in channels)
    num_candidates = max(0, (max_len - window) // window + 1)
    if num_candidates == 0:
        return np.empty((0, len(channels), window), dtype=np.float32)

    if rng is not None:
        rng = np.random.default_rng(rng) if not hasattr(rng, 'integers') else rng
        starts = []
        seen = set()
        max_attempts = max_attempts or min(num_candidates, num_windows * 20)
        for _ in range(max_attempts):
            candidate = int(rng.integers(num_candidates))
            if candidate in seen:
                continue
            seen.add(candidate)
            start = candidate * window
            if _clean_multichannel_eeg_window(eeg_signals, channels, start, window):
                starts.append(start)
                if len(starts) == num_windows:
                    break
    else:
        starts = [
            start
            for start in range(0, max_len - window + 1, window)
            if _clean_multichannel_eeg_window(eeg_signals, channels, start, window)
        ]

    if not starts:
        return np.empty((0, len(channels), window), dtype=np.float32)

    chosen = range(len(starts)) if rng is not None else np.linspace(0, len(starts) - 1, min(num_windows, len(starts)), dtype=int)
    return np.stack([
        np.stack([eeg_signals[channel][starts[i]:starts[i] + window] for channel in channels])
        for i in chosen
    ]).astype(np.float32, copy=False)


def build_eeg_cache(data_folder, cache_folder, num_windows=200, seed=0, channels=EEG_CHANNELS, target_fs=TARGET_EEG_FS, bandpass=EEG_BANDPASS_HZ, verbose=True):
    """
    Build a subject-level EEG cache.

    Writes:
        X.dat: float32 memmap with shape from meta.json
        y.npy: labels for kept subjects
        records.csv: kept/skipped status
        meta.json: cache shape and preprocessing settings
    """
    os.makedirs(cache_folder, exist_ok=True)

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records = find_patients(patient_data_file)
    window = int(round(EEG_SEGMENT_SECONDS * target_fs))
    x_shape = (len(records), num_windows, len(channels), window)
    x_path = os.path.join(cache_folder, 'X.dat')
    y_path = os.path.join(cache_folder, 'y.npy')
    records_path = os.path.join(cache_folder, 'records.csv')
    meta_path = os.path.join(cache_folder, 'meta.json')

    X = np.memmap(x_path, dtype=np.float32, mode='w+', shape=x_shape)
    y = []
    rows = []
    kept = 0

    iterator = tqdm(records, desc='Building EEG cache', unit='record', disable=not verbose)
    for i, record in enumerate(iterator):
        patient_id = record[HEADERS['bids_folder']]
        site_id = record[HEADERS['site_id']]
        session_id = record[HEADERS['session_id']]
        row = {
            'index': i,
            'cache_index': '',
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
            'status': 'skipped',
            'reason': '',
        }

        try:
            label = load_diagnoses(patient_data_file, patient_id)
            edf_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
            if not os.path.exists(edf_path):
                row['reason'] = 'missing_edf'
            else:
                eeg_signals, fs = load_eeg_signals(edf_path, channels=channels, target_fs=target_fs, bandpass=bandpass)
                windows = sample_eeg_windows(eeg_signals, fs, num_windows=num_windows, channels=channels, rng=seed + i)
                if windows.shape != (num_windows, len(channels), window):
                    row['reason'] = f'windows_shape_{windows.shape}'
                else:
                    X[kept] = windows
                    y.append(label)
                    row['cache_index'] = kept
                    row['status'] = 'kept'
                    row['reason'] = ''
                    kept += 1
        except Exception as e:
            row['reason'] = str(e)

        rows.append(row)

    X.flush()
    del X
    os.truncate(x_path, kept * num_windows * len(channels) * window * np.dtype(np.float32).itemsize)
    np.save(y_path, np.asarray(y, dtype=np.int8))

    with open(records_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'cache_index', 'patient_id', 'site_id', 'session_id', 'status', 'reason'])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        'shape': [kept, num_windows, len(channels), window],
        'dtype': 'float32',
        'labels_shape': [kept],
        'fs': target_fs,
        'channels': list(channels),
        'segment_seconds': EEG_SEGMENT_SECONDS,
        'bandpass': list(bandpass) if bandpass is not None else None,
        'seed': seed,
        'subjects_total': len(records),
        'subjects_kept': kept,
        'subjects_skipped': len(records) - kept,
        'x_path': x_path,
        'y_path': y_path,
        'records_path': records_path,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"EEG cache: kept {kept}/{len(records)} subjects at {x_path}")

    return meta


def _cache_file(cache_folder, meta, key):
    path = meta[key]
    if os.path.exists(path):
        return path
    return os.path.join(cache_folder, os.path.basename(path))


class EEGCacheDataset:
    """
    PyTorch-compatible dataset for an EEG cache.

    level='window': returns x shape (6, 800), y
    level='subject': returns x shape (200, 6, 800), y
    """
    def __init__(self, cache_folder, level='window', normalize=True, subject_weights=None):
        import torch

        if level not in ('window', 'subject'):
            raise ValueError("level must be 'window' or 'subject'")

        with open(os.path.join(cache_folder, 'meta.json')) as f:
            self.meta = json.load(f)

        self.torch = torch
        self.cache_folder = cache_folder
        self.level = level
        self.normalize = normalize
        self.shape = tuple(self.meta['shape'])
        self.X = np.memmap(_cache_file(cache_folder, self.meta, 'x_path'), dtype=np.float32, mode='r', shape=self.shape)
        self.y = np.load(_cache_file(cache_folder, self.meta, 'y_path'))
        self.subject_weights = None if subject_weights is None else np.asarray(subject_weights, dtype=np.float32)

    def __len__(self):
        subjects, windows, _, _ = self.shape
        return subjects * windows if self.level == 'window' else subjects

    def __getitem__(self, index):
        if self.level == 'window':
            subject_index = index // self.shape[1]
            window_index = index % self.shape[1]
            x = np.array(self.X[subject_index, window_index], dtype=np.float32, copy=True)
        else:
            subject_index = index
            x = np.array(self.X[subject_index], dtype=np.float32, copy=True)

        if self.normalize:
            x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

        x = self.torch.from_numpy(x)
        y = self.torch.tensor(self.y[subject_index], dtype=self.torch.float32)
        if self.subject_weights is None:
            return x, y
        return x, y, self.torch.tensor(self.subject_weights[subject_index], dtype=self.torch.float32)


def make_eeg_dataloader(cache_folder, batch_size=64, level='window', shuffle=True, num_workers=0, normalize=True):
    from torch.utils.data import DataLoader

    return DataLoader(
        EEGCacheDataset(cache_folder, level=level, normalize=normalize),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def split_eeg_subjects(cache_folder, val_fraction=0.2, seed=0):
    from sklearn.model_selection import train_test_split

    y = np.load(os.path.join(cache_folder, 'y.npy'))
    subject_indices = np.arange(len(y))
    counts = np.bincount(y.astype(int), minlength=2)
    stratify = y if np.count_nonzero(counts) == 2 and counts.min() >= 2 else None
    train_idx, val_idx = train_test_split(
        subject_indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def _window_indices(subject_indices, num_windows):
    return np.concatenate([np.arange(i * num_windows, (i + 1) * num_windows) for i in subject_indices])


def make_eeg_supervised_loaders(cache_folder, batch_size=128, val_fraction=0.2, seed=0, num_workers=0, normalize=True, train_subjects=None, val_subjects=None, subject_weights=None):
    from torch.utils.data import DataLoader, Subset

    if train_subjects is None or val_subjects is None:
        train_subjects, val_subjects = split_eeg_subjects(cache_folder, val_fraction=val_fraction, seed=seed)

    window_dataset = EEGCacheDataset(cache_folder, level='window', normalize=normalize, subject_weights=subject_weights)
    subject_dataset = EEGCacheDataset(cache_folder, level='subject', normalize=normalize)
    num_windows = window_dataset.shape[1]

    train_loader = DataLoader(
        Subset(window_dataset, _window_indices(train_subjects, num_windows)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(subject_dataset, val_subjects),
        batch_size=max(1, min(8, batch_size // 8)),
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_subjects, val_subjects


def load_cache_subject_ages(cache_folder, data_folder):
    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    ages_by_record = {}
    with open(demographics_file, newline='') as f:
        for row in csv.DictReader(f):
            key = (row[HEADERS['bids_folder']], str(row[HEADERS['session_id']]))
            try:
                ages_by_record[key] = float(row[HEADERS['age']])
            except (TypeError, ValueError):
                ages_by_record[key] = float('nan')

    with open(os.path.join(cache_folder, 'meta.json')) as f:
        meta = json.load(f)
    ages = np.full(meta['shape'][0], float('nan'), dtype=np.float32)

    with open(os.path.join(cache_folder, 'records.csv'), newline='') as f:
        for row in csv.DictReader(f):
            if row['status'] != 'kept':
                continue
            key = (row['patient_id'], str(row['session_id']))
            ages[int(row['cache_index'])] = ages_by_record.get(key, float('nan'))

    return ages


def compute_reward_training_weights(labels, ages, train_subjects, clip=(0.25, 20.0)):
    from evaluate_model import compute_prevalence

    labels = np.asarray(labels, dtype=np.int8)
    ages = np.asarray(ages, dtype=np.float32)
    train_labels = labels[train_subjects]
    train_ages = ages[train_subjects]
    finite_train = np.isfinite(train_ages)
    fallback = float(train_labels[finite_train].mean()) if np.any(finite_train) else float(train_labels.mean())
    age_to_prevalence = compute_prevalence(ages, train_labels[finite_train], train_ages[finite_train], gap=2)

    prevalence = np.asarray([age_to_prevalence.get(age, fallback) for age in ages], dtype=np.float32)
    prevalence = np.clip(prevalence, 1e-3, 1 - 1e-3)

    counts = np.bincount(train_labels.astype(int), minlength=2)
    class_balance = np.where(labels == 1, counts[0] / max(1, counts[1]), 1.0)
    reward_weight = np.where(labels == 1, 1 / prevalence - 1, 1 / (1 - prevalence) - 1)
    weights = np.clip(class_balance * reward_weight, clip[0], clip[1]).astype(np.float32)
    weights /= max(float(weights[train_subjects].mean()), 1e-6)
    return weights


def make_small_eeg_cnn():
    import torch

    return torch.nn.Sequential(
        torch.nn.Conv1d(len(EEG_CHANNELS), 32, kernel_size=15, stride=2, padding=7),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 1),
    )


def predict_ensemble_probability(models, x):
    import torch

    if not models:
        raise ValueError('Ensemble must contain at least one model.')
    with torch.no_grad():
        logits = torch.stack([net(x).flatten() for net in models])
        return float(torch.sigmoid(logits.mean()).item())


def eeg_cnn_features(model, x):
    return model[:-1](x)


def sigreg_loss(x, sketch_dim=128):
    import torch

    n, c = x.size()
    A = torch.randn(c, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)
    proj = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1, -1)
    ecf = torch.exp(1j * args).mean(dim=0)
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.unsqueeze(0)
    return (torch.trapz(err, t, dim=1)).mean()


def make_eeg_view(x, noise_std=EEG_VIEW_NOISE_STD, channel_drop=EEG_VIEW_CHANNEL_DROP):
    import torch

    view = x + noise_std * torch.randn_like(x)
    if channel_drop <= 0:
        return view
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= channel_drop).to(x.dtype)
    return view * keep / max(1.0 - channel_drop, 1e-6)


def compute_joint_eeg_loss(model, loss_fn, x, y, sample_weight=None, return_parts=False):
    import torch

    x1 = make_eeg_view(x)
    x2 = make_eeg_view(x)
    z = eeg_cnn_features(model, torch.cat([x, x1, x2], dim=0))
    z_clean, z1, z2 = z.chunk(3, dim=0)
    logits = model[-1](z_clean).flatten()
    bce = loss_fn(logits, y)
    bce = (bce * sample_weight).mean() if sample_weight is not None else bce.mean()

    z_clean_target = torch.nn.functional.normalize(z_clean.detach(), dim=1)
    mse = 0.5 * (
        torch.nn.functional.mse_loss(torch.nn.functional.normalize(z1, dim=1), z_clean_target)
        + torch.nn.functional.mse_loss(torch.nn.functional.normalize(z2, dim=1), z_clean_target)
    )
    sigreg = sigreg_loss(z)
    loss = bce + JEPA_MSE_WEIGHT * mse + SIGREG_WEIGHT * sigreg
    if return_parts:
        return loss, {
            'bce': float(bce.detach().cpu()),
            'mse': float(mse.detach().cpu()),
            'sigreg': float(sigreg.detach().cpu()),
        }
    return loss


def compute_internal_reward_metrics(labels, probabilities, ages, prevalence_labels, prevalence_ages):
    from evaluate_model import compute_auroc_age, compute_auroc_weighted, compute_prevalence, compute_reward

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    ages = np.asarray(ages, dtype=np.float32)
    prevalence_labels = np.asarray(prevalence_labels, dtype=np.int8)
    prevalence_ages = np.asarray(prevalence_ages, dtype=np.float32)

    finite = np.isfinite(prevalence_ages)
    fallback = float(prevalence_labels[finite].mean()) if np.any(finite) else float(labels.mean())
    age_to_prevalence = compute_prevalence(ages, prevalence_labels[finite], prevalence_ages[finite], gap=2)
    for age in np.unique(ages[np.isfinite(ages)]):
        age_to_prevalence.setdefault(age, fallback)

    def reward_at_scale(scale):
        thresholds = np.asarray([age_to_prevalence.get(age, fallback) for age in ages], dtype=np.float32)
        thresholds = np.clip(thresholds * scale, 1e-6, 1 - 1e-6)
        binary = (probabilities > thresholds).astype(np.int8)
        return compute_reward(labels, binary, ages, age_to_prevalence), binary

    reward_prevalence, binary_prevalence = reward_at_scale(1.0)
    best = (reward_prevalence, 1.0, binary_prevalence)
    for scale in np.linspace(0.25, 4.0, 76):
        reward, binary = reward_at_scale(float(scale))
        if reward > best[0]:
            best = (reward, float(scale), binary)

    out = {
        'reward_prevalence': float(reward_prevalence),
        'reward_best': float(best[0]),
        'reward_best_scale': float(best[1]),
        'binary_positive_rate': float(best[2].mean()),
    }
    if len(np.unique(labels)) == 2:
        try:
            out['age_conditioned_auroc'] = float(compute_auroc_age(labels, probabilities, ages, gap=2))
        except ZeroDivisionError:
            out['age_conditioned_auroc'] = float('nan')
        out['age_weighted_auroc'] = float(compute_auroc_weighted(labels, probabilities, ages, gap=2))
    return out


def auroc_selection_score(metrics):
    for key in ('age_conditioned_auroc', 'auroc', 'average_precision'):
        value = metrics.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return -np.inf


def estimate_age_prevalence(age, prevalence_labels, prevalence_ages, gap=2, fallback=None):
    prevalence_labels = np.asarray(prevalence_labels, dtype=np.int8)
    prevalence_ages = np.asarray(prevalence_ages, dtype=np.float32)
    fallback = float(prevalence_labels.mean()) if fallback is None else float(fallback)
    if not np.isfinite(age):
        return fallback

    mask = np.isfinite(prevalence_ages) & (np.abs(prevalence_ages - float(age)) <= gap)
    if not np.any(mask):
        return fallback
    return max(float(prevalence_labels[mask].sum()), 0.5) / int(mask.sum())


def calibrate_pos_weighted_probabilities(probabilities, pos_weight):
    # ponytail: undo weighted-BCE class prior shift; use a held-out calibrator if this stops tracking reward.
    pos_weight = float(pos_weight)
    if not np.isfinite(pos_weight) or pos_weight <= 0:
        pos_weight = 1.0
    return probabilities / (probabilities + pos_weight * (1 - probabilities))


def predict_eeg_subject_probabilities(model, val_loader, device=None, pos_weight=1.0):
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in val_loader:
            batch_size, num_windows, channels, time = x.shape
            x = x.reshape(batch_size * num_windows, channels, time).to(device)
            subject_probs = torch.sigmoid(model(x).reshape(batch_size, num_windows).mean(dim=1))
            probs.extend(subject_probs.cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())

    return np.asarray(labels, dtype=np.int8), np.asarray(probs, dtype=np.float32)


def evaluate_eeg_baseline(model, val_loader, device=None, ages=None, prevalence_labels=None, prevalence_ages=None, pos_weight=1.0):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels, probs = predict_eeg_subject_probabilities(model, val_loader, device=device, pos_weight=pos_weight)
    metrics = {'positive_rate': float(labels.mean()), 'mean_probability': float(probs.mean())}
    if len(np.unique(labels)) == 2:
        metrics['auroc'] = float(roc_auc_score(labels, probs))
        metrics['average_precision'] = float(average_precision_score(labels, probs))
    else:
        metrics['auroc'] = float('nan')
        metrics['average_precision'] = float('nan')

    if ages is not None and prevalence_labels is not None and prevalence_ages is not None:
        metrics.update(compute_internal_reward_metrics(labels, probs, ages, prevalence_labels, prevalence_ages))
    return metrics


def cross_validate_eeg_baseline(cache_folder, data_folder, folds=5, epochs=5, batch_size=512, lr=1e-3, seed=0, num_workers=2, device=None, reward_weighted=False, weight_clip=(0.25, 20.0), results_path=None):
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    labels = np.load(os.path.join(cache_folder, 'y.npy')).astype(np.int8)
    ages = load_cache_subject_ages(cache_folder, data_folder)
    oof_probs = np.full(len(labels), np.nan, dtype=np.float32)
    fold_metrics = []

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_subjects, val_subjects) in enumerate(splitter.split(np.arange(len(labels)), labels), start=1):
        train_labels = labels[train_subjects]
        subject_weights = compute_reward_training_weights(labels, ages, train_subjects, clip=weight_clip) if reward_weighted else None
        train_loader, val_loader, _, _ = make_eeg_supervised_loaders(
            cache_folder,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            subject_weights=subject_weights,
        )

        counts = np.bincount(train_labels.astype(int), minlength=2)
        model = make_small_eeg_cnn().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        if reward_weighted:
            pos_weight_value = 1.0
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            pos_weight = torch.tensor([counts[0] / max(1, counts[1])], dtype=torch.float32, device=device)
            pos_weight_value = float(pos_weight.item())
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

        best_score = -np.inf
        best_metrics = None
        best_val_probs = None

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            parts = []
            for batch in train_loader:
                if reward_weighted:
                    x, yb, wb = batch
                    wb = wb.to(device)
                else:
                    x, yb = batch
                x = x.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, batch_parts = compute_joint_eeg_loss(model, loss_fn, x, yb, wb if reward_weighted else None, return_parts=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                parts.append(batch_parts)

            val_labels, val_probs = predict_eeg_subject_probabilities(model, val_loader, device=device, pos_weight=pos_weight_value)
            mean_parts = {key: float(np.mean([row[key] for row in parts])) for key in ('bce', 'mse', 'sigreg')}
            metrics = compute_internal_reward_metrics(
                val_labels,
                val_probs,
                ages[val_subjects],
                train_labels,
                ages[train_subjects],
            )
            metrics.update({
                'fold': fold,
                'epoch': epoch,
                'train_loss': float(np.mean(losses)),
                'train_bce_loss': mean_parts['bce'],
                'train_mse_loss': mean_parts['mse'],
                'train_sigreg_loss': mean_parts['sigreg'],
                'auroc': float(roc_auc_score(val_labels, val_probs)),
                'average_precision': float(average_precision_score(val_labels, val_probs)),
                'val_subjects': int(len(val_subjects)),
                'pos_weight': pos_weight_value,
                'lr': float(optimizer.param_groups[0]['lr']),
            })
            score = auroc_selection_score(metrics)
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_val_probs = val_probs.copy()
            scheduler.step(score if np.isfinite(score) else -1e9)
            print(
                f"fold={fold}/{folds} epoch={epoch}/{epochs} loss={metrics['train_loss']:.4f} "
                f"bce={metrics['train_bce_loss']:.4f} "
                f"mse={metrics['train_mse_loss']:.4f} "
                f"sigreg={metrics['train_sigreg_loss']:.4f} "
                f"lr={metrics['lr']:.2g} "
                f"ap={metrics['average_precision']:.3f} auroc={metrics['auroc']:.3f} "
                f"age_auc={metrics.get('age_conditioned_auroc', float('nan')):.3f} "
                f"reward={metrics['reward_prevalence']:.3f} "
                f"reward_best={metrics['reward_best']:.3f}"
            )

        oof_probs[val_subjects] = best_val_probs
        fold_metrics.append(best_metrics)
        print(f"fold={fold}/{folds} best_epoch={best_metrics['epoch']} best_age_auc={best_score:.3f}")

    valid = np.isfinite(oof_probs)
    weights = np.asarray([m['val_subjects'] for m in fold_metrics], dtype=np.float32)
    summary = {
        'folds': folds,
        'epochs': epochs,
        'subjects': int(valid.sum()),
        'oof_auroc': float(roc_auc_score(labels[valid], oof_probs[valid])),
        'oof_average_precision': float(average_precision_score(labels[valid], oof_probs[valid])),
        'oof_reward_prevalence': float(np.average([m['reward_prevalence'] for m in fold_metrics], weights=weights)),
        'oof_reward_best_optimistic': float(np.average([m['reward_best'] for m in fold_metrics], weights=weights)),
        'reward_weighted': reward_weighted,
        'fold_metrics': fold_metrics,
    }

    if results_path:
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print(
        f"OOF ap={summary['oof_average_precision']:.3f} "
        f"auroc={summary['oof_auroc']:.3f} "
        f"reward={summary['oof_reward_prevalence']:.3f} "
        f"reward_best_opt={summary['oof_reward_best_optimistic']:.3f}"
    )
    return summary


def train_eeg_baseline(cache_folder, model_path=None, epochs=3, batch_size=128, lr=1e-3, val_fraction=0.2, seed=0, num_workers=0, device=None, data_folder=None, reward_weighted=False, weight_clip=(0.25, 20.0)):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    all_labels = np.load(os.path.join(cache_folder, 'y.npy'))
    train_subjects, val_subjects = split_eeg_subjects(cache_folder, val_fraction=val_fraction, seed=seed)
    train_labels = all_labels[train_subjects]
    subject_ages = load_cache_subject_ages(cache_folder, data_folder) if data_folder is not None else None

    if reward_weighted and subject_ages is None:
        raise ValueError("reward_weighted=True requires data_folder for subject ages.")

    subject_weights = compute_reward_training_weights(all_labels, subject_ages, train_subjects, clip=weight_clip) if reward_weighted else None
    train_loader, val_loader, train_subjects, val_subjects = make_eeg_supervised_loaders(
        cache_folder,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        subject_weights=subject_weights,
    )

    counts = np.bincount(train_labels.astype(int), minlength=2)
    model = make_small_eeg_cnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    if reward_weighted:
        pos_weight_value = 1.0
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        pos_weight = torch.tensor([counts[0] / max(1, counts[1])], dtype=torch.float32, device=device)
        pos_weight_value = float(pos_weight.item())
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    history = []
    best_score = -np.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        parts = []
        for batch in train_loader:
            if reward_weighted:
                x, yb, wb = batch
                wb = wb.to(device)
            else:
                x, yb = batch
            x = x.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_parts = compute_joint_eeg_loss(model, loss_fn, x, yb, wb if reward_weighted else None, return_parts=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            parts.append(batch_parts)

        if subject_ages is None:
            metrics = evaluate_eeg_baseline(model, val_loader, device=device, pos_weight=pos_weight_value)
        else:
            metrics = evaluate_eeg_baseline(
                model,
                val_loader,
                device=device,
                ages=subject_ages[val_subjects],
                prevalence_labels=train_labels,
                prevalence_ages=subject_ages[train_subjects],
                pos_weight=pos_weight_value,
            )
        mean_parts = {key: float(np.mean([row[key] for row in parts])) for key in ('bce', 'mse', 'sigreg')}
        metrics.update({
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'train_bce_loss': mean_parts['bce'],
            'train_mse_loss': mean_parts['mse'],
            'train_sigreg_loss': mean_parts['sigreg'],
            'reward_weighted': reward_weighted,
            'pos_weight': pos_weight_value,
            'lr': float(optimizer.param_groups[0]['lr']),
        })
        history.append(metrics)
        score = auroc_selection_score(metrics)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(score if np.isfinite(score) else -1e9)
        print(
            f"epoch={epoch} loss={metrics['train_loss']:.4f} "
            f"bce={metrics['train_bce_loss']:.4f} "
            f"mse={metrics['train_mse_loss']:.4f} "
            f"sigreg={metrics['train_sigreg_loss']:.4f} "
            f"lr={metrics['lr']:.2g} "
            f"ap={metrics['average_precision']:.3f} auroc={metrics['auroc']:.3f} "
            f"reward={metrics.get('reward_prevalence', float('nan')):.3f} "
            f"reward_best={metrics.get('reward_best', float('nan')):.3f} "
            f"pred_pos={metrics.get('binary_positive_rate', float('nan')):.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if model_path:
        torch.save({'model_state_dict': model.state_dict(), 'history': history}, model_path)

    return model, history


def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age
    age = load_age(data)
    age = np.array([age])

    # 2. Sex feature (one-hot encoding for Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (5 dimensions)
    # Standardizes the raw text into one of five categories using the helper function
    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    if race == 'Asian':
        race_vec[0] = 1
    elif race == 'Black':
        race_vec[1] = 1
    elif race == 'Others':
        race_vec[2] = 1
    elif race == 'Unavailable':
        race_vec[3] = 1
    elif race == 'White':
        race_vec[4] = 1
    else:
        race_vec[2] = 1 # Default to 'Others' for any unrecognized

    # 4. Body mass index (BMI)
    bmi = load_bmi(data)
    bmi = np.array([bmi])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    
    # return np.concatenate([age, sex_vec, race_vec, bmi])
    return np.concatenate([age])


def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            # Report error and stop if no FS is found for a kept channel
            raise KeyError(f"Sampling frequency (fs) not found for channel '{old_label}' ")
        
    if 'physiological_data' in locals(): del physiological_data

    # Step 3: Construct Bipolar Derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        # 1. Skip if target already exists or pos channel missing
        if target in processed_channels or pos not in processed_channels:
            continue
        
        # 2. Check all neg channels exist
        if not all(n in processed_channels for n in neg_list):
            continue

        # 3. Check sampling rate consistency
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        
        if len(set(fs_values)) > 1:
            raise ValueError(f"Sampling rate mismatch for {target}: {dict(zip(all_involved, fs_values))}")

        # 4. Derive bipolar signal
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'] # Added sao2 as fallback for spo2
    }
    
    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        fs = None
        
        # Identify the first available candidate
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate)
                break 

        if sig is not None and len(sig) > 1:
            
            ### this is where we can get raw signals from EEG and EOG and chin and leg and ECG and resp and spo2, and then we can extract features from them.##
            # get only EEG signal raw
            if lead_type == 'eeg':

                eeg_sig = sig
                eeg_fs = fs

                final_features.extend([eeg_fs])  # Append EEG sampling frequency as a feature

        #     # --- Time Domain Features (Very Fast) ---
        #     std_val = np.std(sig)
        #     mav_val = np.mean(np.abs(sig))

        #     # Zero Crossing Rate (Proxy for frequency/slowing)
        #     zcr = np.mean(np.diff(np.sign(sig)) != 0)

        #     # Root Mean Square
        #     rms = np.sqrt(np.mean(sig**2))

        #     # Signal Activity (Variance)
        #     activity = np.var(sig)
            
        #     # Mobility (Hjorth Parameter) - Proxy for mean frequency
        #     # sqrt(var(diff(sig)) / var(sig))
        #     diff_sig = np.diff(sig)
        #     mobility = np.sqrt(np.var(diff_sig) / activity) if activity > 0 else 0.0

        #     # Complexity (Hjorth Parameter) - Proxy for bandwidth
        #     diff2_sig = np.diff(diff_sig)
        #     var_d2 = np.var(diff2_sig)
        #     var_d1 = np.var(diff_sig)
        #     complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0.0

        #     final_features.extend([std_val, mav_val, zcr, rms, activity, mobility, complexity])

        # else:
        #     # Padding: 7 features per lead type
        #     final_features.extend([float('nan')] * 7)

    if 'processed_channels' in locals(): del processed_channels

    return np.array(final_features)

def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 12
    """
    if not algo_data:
        return np.full(12, float('nan'))

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return float('nan')
        
        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)
        
        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)
        
        return num_events / total_hours
    
    ahi_auto = count_discrete_events('resp_caisr')      # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events('arousal_caisr') # Automated Arousal Index
    limb_auto = count_discrete_events('limb_caisr')    # Automated Limb Movement Index
    
    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 5=W, 4=R, 3=N1, 2=N2, 1=N3 (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        
        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [float('nan')]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [float('nan')]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [float('nan')]))
    
    # Standardize '9.0' or other filler values to NaN
    clean_prob = lambda x: x if x < 1.0 else float('nan')
    features.extend([clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    return np.array(features)

def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.full(12, float('nan'))

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0
    
    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return float('nan')
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')      # Human AHI
    arousal_human = count_discrete_events('arousal_expert') # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')       # Human PLMI
    
    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))
    
    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]
    
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = float('nan')

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else float('nan')
    else:
        transitions = waso_minutes = rem_latency = float('nan')

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)
