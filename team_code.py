#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# EEG-only raw-window pipeline.
#
# Training flow:
# 1. Extract 4-second raw EEG windows for every subject.
# 2. Preprocess each channel with bandpass 0.1-49 Hz, resample to 128 Hz,
#    and robustly clip into [-1, 1].
# 3. Pretrain a Conv1d + sinusoidal positional encoding + Transformer encoder
#    with a JEPA-style masked latent prediction objective.
# 4. Mean-pool each subject's window embeddings and fit a linear probe.
#
################################################################################

import copy
import hashlib
import joblib
import json
import math
import numpy as np
import os
import pandas as pd
import random
import shutil

from bisect import bisect_right
from fractions import Fraction
from functools import lru_cache
from scipy import signal
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from helper_code import *

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    TORCH_IMPORT_ERROR = e
else:
    TORCH_IMPORT_ERROR = None

TorchModuleBase = nn.Module if nn is not None else object
TorchDatasetBase = Dataset if Dataset is not None else object

################################################################################
# Path & Constant Configuration
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

EEG_CHANNEL_ORDER = ('f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1')
TARGET_EEG_FS = 128.0
WINDOW_SECONDS = 4
WINDOW_SIZE = int(TARGET_EEG_FS * WINDOW_SECONDS)

BANDPASS_LOW_HZ = 0.1
BANDPASS_HIGH_HZ = 49.0
FILTER_ORDER = 4
ROBUST_CLIP_PERCENTILE = 99.5

PATCH_SIZE = 16
EMBED_DIM = int(os.environ.get('PN2026_EMBED_DIM', '256'))
TRANSFORMER_DEPTH = int(os.environ.get('PN2026_TRANSFORMER_DEPTH', '8'))
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_DROPOUT = 0.1

MASK_RATIO = 0.5
PRETRAIN_EPOCHS = int(os.environ.get('PN2026_PRETRAIN_EPOCHS', '5'))
PRETRAIN_BATCH_SIZE = 64
PROBE_BATCH_SIZE = 128
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
TARGET_EMA = 0.996
MAX_GRAD_NORM = 1.0
PRETRAIN_NUM_WORKERS = 0

WINDOW_CACHE_DIRNAME = 'window_cache'
WINDOW_CACHE_ROOT = os.path.join(SCRIPT_DIR, '.window_cache')
WINDOW_CACHE_MANIFEST = 'manifest.json'
WINDOW_CACHE_VERSION = 1
REUSE_LOCAL_WINDOW_CACHE = True
KEEP_LOCAL_WINDOW_CACHE = True

RANDOM_STATE = 56

INTERNAL_EXPERIMENT_MODE = os.environ.get('PN2026_INTERNAL_EXPERIMENT', '0').strip().lower() in ('1', 'true', 'yes', 'y')
INTERNAL_TRAIN_FRACTION = 0.80
INTERNAL_VAL_FRACTION = 0.10
INTERNAL_TEST_FRACTION = 0.10
INTERNAL_SPLIT_SEED = RANDOM_STATE
INTERNAL_REFIT_ON_TRAIN_VAL = True
INTERNAL_METRICS_FILENAME = 'internal_metrics.json'
INTERNAL_SPLIT_FILENAME = 'internal_split.csv'

################################################################################
#
# Required functions
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    require_torch()
    set_random_seed(RANDOM_STATE)
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('Finding the Challenge data...')

    subject_records = collect_subject_records(data_folder)
    if not subject_records:
        raise FileNotFoundError('No usable physiological data were provided.')

    pretrain_records = [record for record in subject_records if record['has_eeg']]
    probe_records = [record for record in pretrain_records if record['label'] is not None]

    if not pretrain_records:
        raise RuntimeError('No usable EEG recordings were found for pretraining.')
    if not probe_records:
        raise RuntimeError('No labeled EEG recordings were found for the linear probe.')

    internal_split = None
    cache_source_records = pretrain_records
    probe_train_records = probe_records
    probe_val_records = None
    probe_test_records = None
    final_probe_train_records = probe_records

    if INTERNAL_EXPERIMENT_MODE:
        internal_split = create_internal_split(probe_records)
        cache_source_records = (
            internal_split['train']
            + internal_split['val']
            + internal_split['test']
        )
        probe_train_records = internal_split['train']
        probe_val_records = internal_split['val']
        probe_test_records = internal_split['test']
        final_probe_train_records = (
            probe_train_records + probe_val_records
            if INTERNAL_REFIT_ON_TRAIN_VAL
            else probe_train_records
        )

        if verbose:
            print_internal_split_summary(internal_split)

    cache_dir = get_window_cache_dir(
        subject_records=cache_source_records,
        data_folder=data_folder,
        csv_path=csv_path,
    )

    if verbose:
        print('Preparing preprocessed EEG window cache...')

    cached_records = maybe_load_window_cache(
        subject_records=cache_source_records,
        cache_dir=cache_dir,
        data_folder=data_folder,
        csv_path=csv_path,
        verbose=verbose,
    )

    if cached_records is None:
        cached_records = cache_subject_windows(
            subject_records=cache_source_records,
            cache_dir=cache_dir,
            data_folder=data_folder,
            csv_path=csv_path,
            verbose=verbose,
        )
    elif verbose:
        print(f'Reusing cached EEG windows from {cache_dir}')

    if internal_split is not None:
        cached_split = split_cached_records_by_split(cached_records)
        cached_pretrain_records = [record for record in cached_split['train'] if record['num_windows'] > 0]
        cached_probe_train_records = cached_pretrain_records
        cached_probe_val_records = [record for record in cached_split['val'] if record['num_windows'] > 0]
        cached_probe_test_records = [record for record in cached_split['test'] if record['num_windows'] > 0]
        cached_final_probe_train_records = (
            cached_probe_train_records + cached_probe_val_records
            if INTERNAL_REFIT_ON_TRAIN_VAL
            else cached_probe_train_records
        )
    else:
        cached_pretrain_records = [record for record in cached_records if record['num_windows'] > 0]
        cached_probe_train_records = [record for record in cached_pretrain_records if record['label'] is not None]
        cached_probe_val_records = None
        cached_probe_test_records = None
        cached_final_probe_train_records = cached_probe_train_records

    if not cached_pretrain_records:
        raise RuntimeError('No cached EEG windows were available for pretraining.')
    if not cached_probe_train_records:
        raise RuntimeError('No labeled cached EEG windows were available for the linear probe.')

    device = get_torch_device()

    if verbose:
        print(f'Using device: {device}')
        print('Pretraining the raw EEG encoder...')

    encoder = build_window_encoder().to(device)
    pretrain_window_encoder(
        encoder=encoder,
        subject_records=cached_pretrain_records,
        device=device,
        verbose=verbose,
        probe_train_records=cached_probe_train_records if internal_split is not None else None,
        probe_val_records=cached_probe_val_records if internal_split is not None else None,
    )

    if verbose:
        print('Building subject embeddings for the linear probe...')

    features, labels = build_probe_dataset(
        encoder=encoder,
        subject_records=cached_final_probe_train_records,
        device=device,
        verbose=verbose,
    )

    classifier = fit_linear_probe(features, labels)

    model = {
        'classifier': classifier,
        'csv_path': os.path.abspath(csv_path),
        'encoder_config': get_encoder_config(),
        'encoder_state_dict': {
            key: value.detach().cpu()
            for key, value in encoder.state_dict().items()
        },
    }

    if internal_split is not None:
        internal_metrics = evaluate_internal_split(
            encoder=encoder,
            classifier=classifier,
            selection_train_records=cached_probe_train_records,
            val_records=cached_probe_val_records,
            test_records=cached_probe_test_records,
            device=device,
            final_probe_train_size=len(cached_final_probe_train_records),
        )
        save_internal_split_artifacts(
            model_folder=model_folder,
            cached_records=cached_records,
            metrics=internal_metrics,
        )

        if verbose:
            print_internal_metrics(internal_metrics)

    save_model(model_folder, model)
    if not KEEP_LOCAL_WINDOW_CACHE:
        cleanup_window_cache(cache_dir)

    if verbose:
        print(f'Pretrained on {len(cached_pretrain_records)} subjects.')
        print(f'Linear probe fit on {len(labels)} labeled subjects.')
        print('Done.')
        print()


def load_model(model_folder, verbose):
    require_torch()

    model_filename = os.path.join(model_folder, 'model.sav')
    wrapped_model = joblib.load(model_filename)
    model = wrapped_model['model']

    encoder_state_filename = model.get('encoder_state_filename')
    if encoder_state_filename is None:
        raise RuntimeError('Saved model is missing the encoder state filename.')

    device = get_torch_device()
    encoder = build_window_encoder(model.get('encoder_config')).to(device)
    encoder_state_path = os.path.join(model_folder, encoder_state_filename)
    try:
        encoder_state = torch.load(encoder_state_path, map_location=device, weights_only=True)
    except TypeError:
        encoder_state = torch.load(encoder_state_path, map_location=device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    model['encoder'] = encoder
    model['device'] = device

    return {'model': model}


def run_model(model, record, data_folder, verbose):
    model = model['model']
    encoder = model['encoder']
    classifier = model['classifier']
    csv_path = model.get('csv_path', DEFAULT_CSV_PATH)
    device = model['device']

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    physiological_data_file = os.path.join(
        data_folder,
        PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id,
        f'{patient_id}_ses-{session_id}.edf',
    )

    subject_embedding = np.zeros(EMBED_DIM, dtype=np.float32)

    if os.path.exists(physiological_data_file):
        physiological_data, physiological_fs = load_signal_data(physiological_data_file)
        windows = extract_eeg_windows(
            physiological_data=physiological_data,
            physiological_fs=physiological_fs,
            csv_path=csv_path,
        )
        if windows.shape[0] > 0:
            subject_embedding = compute_subject_embedding(
                encoder=encoder,
                windows=windows,
                device=device,
                batch_size=PROBE_BATCH_SIZE,
            )

    subject_embedding = subject_embedding.reshape(1, -1)
    binary_output = int(classifier.predict(subject_embedding)[0])
    probability_output = predict_positive_probability(classifier, subject_embedding)

    return binary_output, probability_output

################################################################################
#
# Raw EEG training pipeline
#
################################################################################

def collect_subject_records(data_folder):
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_table = pd.read_csv(patient_data_file)

    unique_rows = patient_table.drop_duplicates(
        subset=[HEADERS['bids_folder'], HEADERS['site_id'], HEADERS['session_id']]
    )

    subject_records = []
    for _, row in unique_rows.iterrows():
        row_dict = row.to_dict()
        patient_id = row_dict[HEADERS['bids_folder']]
        site_id = row_dict[HEADERS['site_id']]
        session_id = row_dict[HEADERS['session_id']]

        physiological_data_file = os.path.join(
            data_folder,
            PHYSIOLOGICAL_DATA_SUBFOLDER,
            site_id,
            f'{patient_id}_ses-{session_id}.edf',
        )
        has_eeg = os.path.exists(physiological_data_file)

        raw_label = row_dict.get(HEADERS['label'])
        label = None if pd.isna(raw_label) else int(load_label(row_dict))

        subject_records.append({
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
            'physiological_data_file': physiological_data_file,
            'has_eeg': has_eeg,
            'label': label,
        })

    return subject_records


def pretrain_window_encoder(
    encoder,
    subject_records,
    device,
    verbose,
    probe_train_records=None,
    probe_val_records=None,
):
    predictor = JEPAPredictor(embed_dim=encoder.embed_dim).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.eval()
    for parameter in target_encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=PRETRAIN_LR,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )

    dataset = CachedEEGWindowDataset(subject_records)
    if len(dataset) == 0:
        raise RuntimeError('The cached pretraining dataset is empty.')

    track_validation = bool(probe_train_records) and bool(probe_val_records)
    best_val_auroc = None
    best_encoder_state = None

    for epoch in range(PRETRAIN_EPOCHS):
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_windows = 0

        dataloader = DataLoader(
            dataset,
            batch_size=PRETRAIN_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=PRETRAIN_NUM_WORKERS,
            pin_memory=(device.type == 'cuda'),
        )

        iterator = tqdm(
            dataloader,
            desc=f'JEPA Epoch {epoch + 1}/{PRETRAIN_EPOCHS}',
            unit='batch',
            disable=not verbose,
        )

        encoder.train()
        predictor.train()

        for batch in iterator:
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=(device.type == 'cuda'))

            loss_value = jepa_train_step(
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                optimizer=optimizer,
                batch=batch,
            )

            epoch_loss += loss_value
            epoch_batches += 1
            epoch_windows += int(batch.shape[0])

            if verbose and epoch_batches > 0:
                iterator.set_postfix({
                    'loss': f'{epoch_loss / epoch_batches:.4f}',
                    'windows': epoch_windows,
                })

        if verbose and epoch_batches > 0:
            print(f'Epoch {epoch + 1} JEPA loss: {epoch_loss / epoch_batches:.4f}')

        if track_validation:
            _, val_metrics = fit_probe_and_evaluate(
                encoder=encoder,
                train_records=probe_train_records,
                eval_records=probe_val_records,
                device=device,
            )

            current_val_auroc = val_metrics['auroc']
            if verbose:
                print(
                    f'Epoch {epoch + 1} val AUROC: {current_val_auroc:.4f} '
                    f'| val AUPRC: {val_metrics["auprc"]:.4f}'
                )

            if best_val_auroc is None or current_val_auroc > best_val_auroc:
                best_val_auroc = current_val_auroc
                best_encoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in encoder.state_dict().items()
                }

                if verbose:
                    print(f'New best validation AUROC at epoch {epoch + 1}: {best_val_auroc:.4f}')

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
    encoder.eval()


def jepa_train_step(encoder, target_encoder, predictor, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)

    token_mask = sample_token_mask(
        batch_size=batch.shape[0],
        token_count=encoder.num_tokens,
        mask_ratio=MASK_RATIO,
        device=batch.device,
    )

    student_tokens = encoder.forward_tokens(batch, token_mask=token_mask)
    predicted_tokens = predictor(student_tokens)

    with torch.no_grad():
        target_tokens = target_encoder.forward_tokens(batch)

    predicted_masked = predicted_tokens[token_mask]
    target_masked = target_tokens[token_mask].detach()

    predicted_masked = F.normalize(predicted_masked, dim=-1)
    target_masked = F.normalize(target_masked, dim=-1)
    loss = 2.0 - 2.0 * F.cosine_similarity(predicted_masked, target_masked, dim=-1).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), MAX_GRAD_NORM)
    optimizer.step()
    update_ema(target_encoder, encoder, TARGET_EMA)

    return float(loss.detach().cpu())


def build_probe_dataset(encoder, subject_records, device, verbose):
    features = []
    labels = []

    iterator = tqdm(
        subject_records,
        desc='Subject Embeddings',
        unit='subject',
        disable=not verbose,
    )

    for subject_record in iterator:
        if subject_record['label'] is None:
            continue

        windows = load_cached_subject_windows(subject_record)
        if windows.shape[0] == 0:
            continue

        subject_embedding = compute_subject_embedding(
            encoder=encoder,
            windows=windows,
            device=device,
            batch_size=PROBE_BATCH_SIZE,
        )

        features.append(subject_embedding)
        labels.append(int(subject_record['label']))

    if not features:
        raise RuntimeError('No subject embeddings were available for the linear probe.')

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def compute_subject_embedding(encoder, windows, device, batch_size):
    encoder.eval()

    running_sum = np.zeros(encoder.embed_dim, dtype=np.float64)
    total_windows = 0

    with torch.no_grad():
        for batch_start in range(0, windows.shape[0], batch_size):
            batch_np = windows[batch_start:batch_start + batch_size]
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            batch_embeddings = encoder.forward_embedding(batch)
            running_sum += batch_embeddings.sum(dim=0).cpu().numpy()
            total_windows += int(batch_embeddings.shape[0])

    if total_windows == 0:
        return np.zeros(encoder.embed_dim, dtype=np.float32)

    return (running_sum / total_windows).astype(np.float32)


def fit_linear_probe(features, labels):
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        classifier = DummyClassifier(strategy='constant', constant=int(unique_labels[0]))
        classifier.fit(features, labels)
        return classifier

    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver='liblinear',
        )),
    ])
    classifier.fit(features, labels)
    return classifier


def predict_positive_probability(classifier, features):
    if not hasattr(classifier, 'predict_proba'):
        return float(classifier.predict(features)[0])

    probabilities = classifier.predict_proba(features)[0]
    classes = np.asarray(getattr(classifier, 'classes_', np.arange(len(probabilities))))
    positive_index = np.where(classes == 1)[0]

    if positive_index.size == 0:
        return 0.0

    return float(probabilities[int(positive_index[0])])


def predict_positive_probabilities(classifier, features):
    if not hasattr(classifier, 'predict_proba'):
        return classifier.predict(features).astype(np.float32)

    probabilities = classifier.predict_proba(features)
    classes = np.asarray(getattr(classifier, 'classes_', np.arange(probabilities.shape[1])))
    positive_index = np.where(classes == 1)[0]

    if positive_index.size == 0:
        return np.zeros(features.shape[0], dtype=np.float32)

    return probabilities[:, int(positive_index[0])].astype(np.float32)


def create_internal_split(subject_records):
    validate_internal_split_fractions()

    labels = np.asarray([int(record['label']) for record in subject_records], dtype=np.int64)
    indices = np.arange(len(subject_records))

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=INTERNAL_TRAIN_FRACTION,
        stratify=labels,
        random_state=INTERNAL_SPLIT_SEED,
    )

    temp_labels = labels[temp_indices]
    val_fraction_of_temp = INTERNAL_VAL_FRACTION / (INTERNAL_VAL_FRACTION + INTERNAL_TEST_FRACTION)

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction_of_temp,
        stratify=temp_labels,
        random_state=INTERNAL_SPLIT_SEED,
    )

    split_map = {
        'train': sorted(train_indices.tolist()),
        'val': sorted(val_indices.tolist()),
        'test': sorted(test_indices.tolist()),
    }

    split_records = {}
    for split_name, split_indices in split_map.items():
        split_records[split_name] = [
            dict(subject_records[index], split=split_name)
            for index in split_indices
        ]

    return split_records


def validate_internal_split_fractions():
    fraction_sum = INTERNAL_TRAIN_FRACTION + INTERNAL_VAL_FRACTION + INTERNAL_TEST_FRACTION
    if not np.isclose(fraction_sum, 1.0):
        raise ValueError('Internal split fractions must sum to 1.0.')


def print_internal_split_summary(split_records):
    print('Using internal stratified split:')
    for split_name in ('train', 'val', 'test'):
        records = split_records[split_name]
        labels = np.asarray([int(record['label']) for record in records], dtype=np.int64)
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        print(
            f'  {split_name}: {len(records)} subjects '
            f'({negatives} healthy, {positives} impaired)'
        )


def split_cached_records_by_split(cached_records):
    split_records = {'train': [], 'val': [], 'test': []}
    for record in cached_records:
        split_name = record.get('split')
        if split_name in split_records:
            split_records[split_name].append(record)
    return split_records


def fit_probe_and_evaluate(encoder, train_records, eval_records, device):
    train_features, train_labels = build_probe_dataset(
        encoder=encoder,
        subject_records=train_records,
        device=device,
        verbose=False,
    )
    classifier = fit_linear_probe(train_features, train_labels)

    eval_features, eval_labels = build_probe_dataset(
        encoder=encoder,
        subject_records=eval_records,
        device=device,
        verbose=False,
    )

    metrics = score_probe(classifier, eval_features, eval_labels)
    return classifier, metrics


def score_probe(classifier, features, labels):
    probabilities = predict_positive_probabilities(classifier, features)
    binary_predictions = classifier.predict(features)
    unique_labels = np.unique(labels)

    if unique_labels.size < 2:
        auroc = float('nan')
        auprc = float('nan')
    else:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))

    metrics = {
        'num_subjects': int(len(labels)),
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': float(accuracy_score(labels, binary_predictions)),
        'f_measure': float(f1_score(labels, binary_predictions, pos_label=1, average='binary')),
    }
    return metrics


def evaluate_internal_split(
    encoder,
    classifier,
    selection_train_records,
    val_records,
    test_records,
    device,
    final_probe_train_size,
):
    metrics = {
        'mode': 'internal_experiment',
        'config': {
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'embed_dim': EMBED_DIM,
            'depth': TRANSFORMER_DEPTH,
            'heads': TRANSFORMER_HEADS,
        },
        'split_sizes': {
            'train': int(len(selection_train_records)),
            'val': int(len(val_records)) if val_records is not None else 0,
            'test': int(len(test_records)) if test_records is not None else 0,
        },
        'final_probe_train_size': int(final_probe_train_size),
    }

    if val_records:
        _, val_metrics = fit_probe_and_evaluate(
            encoder=encoder,
            train_records=selection_train_records,
            eval_records=val_records,
            device=device,
        )
        metrics['val'] = val_metrics

    if test_records:
        test_features, test_labels = build_probe_dataset(
            encoder=encoder,
            subject_records=test_records,
            device=device,
            verbose=False,
        )
        metrics['test'] = score_probe(classifier, test_features, test_labels)

    return metrics


def save_internal_split_artifacts(model_folder, cached_records, metrics):
    split_rows = []
    for record in cached_records:
        split_name = record.get('split')
        if split_name is None:
            continue
        split_rows.append({
            'patient_id': record['patient_id'],
            'site_id': record['site_id'],
            'session_id': record['session_id'],
            'label': record['label'],
            'split': split_name,
            'num_windows': int(record.get('num_windows', 0)),
        })

    if split_rows:
        split_df = pd.DataFrame(split_rows)
        split_df.to_csv(os.path.join(model_folder, INTERNAL_SPLIT_FILENAME), index=False)

    metrics_path = os.path.join(model_folder, INTERNAL_METRICS_FILENAME)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_internal_metrics(metrics):
    val_metrics = metrics.get('val')
    if val_metrics is not None:
        print(
            f'Internal val AUROC: {val_metrics["auroc"]:.4f} '
            f'| AUPRC: {val_metrics["auprc"]:.4f} '
            f'| Acc: {val_metrics["accuracy"]:.4f} '
            f'| F1: {val_metrics["f_measure"]:.4f}'
        )

    test_metrics = metrics.get('test')
    if test_metrics is not None:
        print(
            f'Internal test AUROC: {test_metrics["auroc"]:.4f} '
            f'| AUPRC: {test_metrics["auprc"]:.4f} '
            f'| Acc: {test_metrics["accuracy"]:.4f} '
            f'| F1: {test_metrics["f_measure"]:.4f}'
        )

################################################################################
#
# Raw EEG preprocessing and window extraction
#
################################################################################

@lru_cache(maxsize=4)
def get_rename_rules(csv_path):
    return load_rename_rules(os.path.abspath(csv_path))


def load_subject_windows(subject_record, csv_path):
    physiological_data_file = subject_record['physiological_data_file']
    if not os.path.exists(physiological_data_file):
        return np.zeros((0, len(EEG_CHANNEL_ORDER), WINDOW_SIZE), dtype=np.float32)

    physiological_data, physiological_fs = load_signal_data(physiological_data_file)
    return extract_eeg_windows(
        physiological_data=physiological_data,
        physiological_fs=physiological_fs,
        csv_path=csv_path,
    )


def get_window_cache_dir(subject_records, data_folder, csv_path):
    cache_descriptor = build_window_cache_descriptor(
        subject_records=subject_records,
        data_folder=data_folder,
        csv_path=csv_path,
    )
    cache_key = hashlib.sha1(
        json.dumps(cache_descriptor, sort_keys=True).encode('utf-8')
    ).hexdigest()[:16]
    return os.path.join(WINDOW_CACHE_ROOT, f'cache_{cache_key}')


def build_window_cache_descriptor(subject_records, data_folder, csv_path):
    return {
        'version': WINDOW_CACHE_VERSION,
        'data_folder': os.path.abspath(data_folder),
        'csv_path': os.path.abspath(csv_path),
        'channels': list(EEG_CHANNEL_ORDER),
        'target_fs': TARGET_EEG_FS,
        'window_seconds': WINDOW_SECONDS,
        'window_size': WINDOW_SIZE,
        'bandpass_low_hz': BANDPASS_LOW_HZ,
        'bandpass_high_hz': BANDPASS_HIGH_HZ,
        'filter_order': FILTER_ORDER,
        'clip_percentile': ROBUST_CLIP_PERCENTILE,
        'subjects': [
            {
                'patient_id': record['patient_id'],
                'site_id': record['site_id'],
                'session_id': record['session_id'],
                'physiological_data_file': os.path.abspath(record['physiological_data_file']),
                'label': record['label'],
            }
            for record in subject_records
        ],
    }


def get_window_cache_manifest_path(cache_dir):
    return os.path.join(cache_dir, WINDOW_CACHE_MANIFEST)


def maybe_load_window_cache(subject_records, cache_dir, data_folder, csv_path, verbose):
    if not REUSE_LOCAL_WINDOW_CACHE:
        return None

    manifest_path = get_window_cache_manifest_path(cache_dir)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    expected_descriptor = build_window_cache_descriptor(
        subject_records=subject_records,
        data_folder=data_folder,
        csv_path=csv_path,
    )

    if manifest.get('descriptor') != expected_descriptor:
        if verbose:
            print('Existing EEG cache does not match the current dataset/config. Rebuilding cache...')
        return None

    cached_records = manifest.get('cached_records', [])
    if len(cached_records) != len(subject_records):
        return None

    for cached_record in cached_records:
        num_windows = int(cached_record.get('num_windows', 0))
        cache_path = cached_record.get('cache_path', '')
        if num_windows > 0 and not os.path.isfile(cache_path):
            return None

    return cached_records


def write_window_cache_manifest(cache_dir, descriptor, cached_records):
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = get_window_cache_manifest_path(cache_dir)
    manifest = {
        'descriptor': descriptor,
        'cached_records': cached_records,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)


def cache_subject_windows(subject_records, cache_dir, data_folder, csv_path, verbose):
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cache_descriptor = build_window_cache_descriptor(
        subject_records=subject_records,
        data_folder=data_folder,
        csv_path=csv_path,
    )

    cached_records = []
    iterator = tqdm(
        enumerate(subject_records),
        total=len(subject_records),
        desc='Window Cache',
        unit='subject',
        disable=not verbose,
    )

    total_windows = 0

    for record_index, subject_record in iterator:
        windows = load_subject_windows(subject_record, csv_path)

        cached_record = dict(subject_record)
        cached_record['num_windows'] = int(windows.shape[0])
        cached_record['cache_path'] = ''

        if windows.shape[0] > 0:
            cache_path = os.path.join(
                cache_dir,
                f'{record_index:05d}_{sanitize_for_filename(subject_record["patient_id"])}_ses-{subject_record["session_id"]}.npy',
            )
            np.save(cache_path, windows.astype(np.float32, copy=False), allow_pickle=False)
            cached_record['cache_path'] = cache_path
            total_windows += int(windows.shape[0])

        cached_records.append(cached_record)

        if verbose:
            iterator.set_postfix({'windows': total_windows})

    write_window_cache_manifest(
        cache_dir=cache_dir,
        descriptor=cache_descriptor,
        cached_records=cached_records,
    )

    return cached_records


def load_cached_subject_windows(subject_record):
    cache_path = subject_record.get('cache_path', '')
    num_windows = int(subject_record.get('num_windows', 0))

    if not cache_path or num_windows <= 0 or not os.path.exists(cache_path):
        return np.zeros((0, len(EEG_CHANNEL_ORDER), WINDOW_SIZE), dtype=np.float32)

    return np.asarray(np.load(cache_path, mmap_mode='r'), dtype=np.float32)


@lru_cache(maxsize=16)
def load_cached_windows_memmap(cache_path):
    return np.load(cache_path, mmap_mode='r')


def cleanup_window_cache(cache_dir):
    load_cached_windows_memmap.cache_clear()
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def sanitize_for_filename(value):
    safe_value = str(value).replace(os.sep, '_')
    return ''.join(char if char.isalnum() or char in ('-', '_', '.') else '_' for char in safe_value)


def extract_eeg_windows(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    processed_channels, processed_fs = standardize_eeg_channels(
        physiological_data=physiological_data,
        physiological_fs=physiological_fs,
        csv_path=csv_path,
    )

    channel_signals = []
    usable_lengths = []

    for channel_name in EEG_CHANNEL_ORDER:
        if channel_name not in processed_channels or channel_name not in processed_fs:
            channel_signals.append(None)
            continue

        preprocessed_signal = preprocess_eeg_signal(
            signal_array=processed_channels[channel_name],
            sampling_frequency=processed_fs[channel_name],
        )

        if preprocessed_signal.size < WINDOW_SIZE:
            channel_signals.append(None)
            continue

        channel_signals.append(preprocessed_signal)
        usable_lengths.append(preprocessed_signal.size)

    if not usable_lengths:
        return np.zeros((0, len(EEG_CHANNEL_ORDER), WINDOW_SIZE), dtype=np.float32)

    usable_samples = (min(usable_lengths) // WINDOW_SIZE) * WINDOW_SIZE
    if usable_samples < WINDOW_SIZE:
        return np.zeros((0, len(EEG_CHANNEL_ORDER), WINDOW_SIZE), dtype=np.float32)

    eeg_tensor = np.zeros((len(EEG_CHANNEL_ORDER), usable_samples), dtype=np.float32)
    for channel_index, signal_array in enumerate(channel_signals):
        if signal_array is None:
            continue
        eeg_tensor[channel_index] = signal_array[:usable_samples]

    window_count = usable_samples // WINDOW_SIZE
    windows = eeg_tensor.reshape(len(EEG_CHANNEL_ORDER), window_count, WINDOW_SIZE).transpose(1, 0, 2)
    return windows.astype(np.float32, copy=False)


def standardize_eeg_channels(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    rename_rules = get_rename_rules(csv_path)
    original_labels = list(physiological_data.keys())
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}

    for old_label, values in physiological_data.items():
        if old_label in cols_to_drop:
            continue

        new_label = rename_map.get(old_label, old_label.lower())
        channel_fs = physiological_fs.get(old_label)
        if channel_fs is None:
            continue

        processed_channels[new_label] = np.asarray(values, dtype=np.float32)
        processed_fs[new_label] = float(channel_fs)

    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']),
        ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']),
        ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']),
        ('o2-m1', 'o2', ['m1']),
    ]

    for target, source, references in bipolar_configs:
        if target in processed_channels or source not in processed_channels:
            continue
        if not all(reference in processed_channels for reference in references):
            continue

        involved_channels = [source] + references
        fs_values = [processed_fs.get(channel_name) for channel_name in involved_channels]
        if any(fs_value is None for fs_value in fs_values):
            continue
        if len({round(fs_value, 6) for fs_value in fs_values}) != 1:
            continue

        reference_signal = (
            processed_channels[references[0]]
            if len(references) == 1
            else tuple(processed_channels[reference] for reference in references)
        )
        derived_signal = derive_bipolar_signal(processed_channels[source], reference_signal)
        if derived_signal is None:
            continue

        processed_channels[target] = np.asarray(derived_signal, dtype=np.float32)
        processed_fs[target] = processed_fs[source]

    return processed_channels, processed_fs


def preprocess_eeg_signal(signal_array, sampling_frequency):
    signal_array = np.asarray(signal_array, dtype=np.float32).reshape(-1)
    signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)

    if signal_array.size == 0:
        return signal_array

    signal_array = bandpass_filter(signal_array, float(sampling_frequency))
    signal_array = resample_signal(signal_array, float(sampling_frequency), TARGET_EEG_FS)
    signal_array = robust_unit_clip(signal_array)

    return signal_array.astype(np.float32, copy=False)


def bandpass_filter(signal_array, sampling_frequency):
    nyquist = 0.5 * sampling_frequency
    high_hz = min(BANDPASS_HIGH_HZ, 0.95 * nyquist)

    if nyquist <= BANDPASS_LOW_HZ or high_hz <= BANDPASS_LOW_HZ:
        return signal_array

    sos = signal.butter(
        FILTER_ORDER,
        [BANDPASS_LOW_HZ, high_hz],
        btype='bandpass',
        fs=sampling_frequency,
        output='sos',
    )

    try:
        filtered = signal.sosfiltfilt(sos, signal_array)
    except ValueError:
        filtered = signal.sosfilt(sos, signal_array)

    return np.asarray(filtered, dtype=np.float32)


def resample_signal(signal_array, original_fs, target_fs):
    if signal_array.size == 0 or np.isclose(original_fs, target_fs):
        return signal_array

    ratio = Fraction(target_fs / original_fs).limit_denominator(1024)
    resampled = signal.resample_poly(
        signal_array,
        ratio.numerator,
        ratio.denominator,
    )

    return np.asarray(resampled, dtype=np.float32)


def robust_unit_clip(signal_array):
    scale = np.percentile(np.abs(signal_array), ROBUST_CLIP_PERCENTILE)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0

    normalized = signal_array / scale
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)

################################################################################
#
# Raw-window encoder
#
################################################################################

def require_torch():
    if TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            'PyTorch is required for the raw EEG conv-transformer pipeline, '
            f'but importing torch failed: {TORCH_IMPORT_ERROR}'
        )


def get_torch_device():
    require_torch()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_encoder_config():
    return {
        'in_channels': len(EEG_CHANNEL_ORDER),
        'window_size': WINDOW_SIZE,
        'patch_size': PATCH_SIZE,
        'embed_dim': EMBED_DIM,
        'depth': TRANSFORMER_DEPTH,
        'num_heads': TRANSFORMER_HEADS,
        'mlp_ratio': TRANSFORMER_MLP_RATIO,
        'dropout': TRANSFORMER_DROPOUT,
    }


def build_window_encoder(config=None):
    require_torch()

    if config is None:
        config = get_encoder_config()

    return EEGWindowEncoder(
        in_channels=config['in_channels'],
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout'],
    )


def build_sinusoidal_encoding(num_positions, embed_dim):
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )

    encoding = torch.zeros(1, num_positions, embed_dim, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(position * div_term)
    encoding[0, :, 1::2] = torch.cos(position * div_term)
    return encoding


class EEGWindowEncoder(TorchModuleBase):
    def __init__(self, in_channels, window_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()

        if window_size % patch_size != 0:
            raise ValueError('window_size must be divisible by patch_size')

        self.in_channels = in_channels
        self.window_size = window_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_tokens = window_size // patch_size

        self.patch_embed = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            'positional_encoding',
            build_sinusoidal_encoding(self.num_tokens, embed_dim),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def tokenize(self, x):
        tokens = self.patch_embed(x).transpose(1, 2)
        pos = self.positional_encoding[:, :tokens.shape[1], :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + pos

    def encode_tokens(self, tokens):
        return self.norm(self.transformer(tokens))

    def forward_tokens(self, x, token_mask=None):
        tokens = self.tokenize(x)

        if token_mask is not None:
            pos = self.positional_encoding[:, :tokens.shape[1], :].to(device=tokens.device, dtype=tokens.dtype)
            mask_tokens = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1) + pos
            tokens = torch.where(token_mask.unsqueeze(-1), mask_tokens, tokens)

        return self.encode_tokens(tokens)

    def forward_embedding(self, x):
        token_embeddings = self.forward_tokens(x)
        return token_embeddings.mean(dim=1)

    def forward(self, x):
        return self.forward_embedding(x)


class JEPAPredictor(TorchModuleBase):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, token_embeddings):
        return self.net(token_embeddings)


def sample_token_mask(batch_size, token_count, mask_ratio, device):
    masked_token_count = max(1, int(round(token_count * mask_ratio)))
    ranking = torch.argsort(torch.rand(batch_size, token_count, device=device), dim=1)
    mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
    mask.scatter_(1, ranking[:, :masked_token_count], True)
    return mask


def update_ema(target_encoder, online_encoder, momentum):
    with torch.no_grad():
        for target_param, online_param in zip(target_encoder.parameters(), online_encoder.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


class CachedEEGWindowDataset(TorchDatasetBase):
    def __init__(self, subject_records):
        self.subject_records = [record for record in subject_records if int(record.get('num_windows', 0)) > 0]
        self.cumulative_windows = []

        running_total = 0
        for record in self.subject_records:
            running_total += int(record['num_windows'])
            self.cumulative_windows.append(running_total)

    def __len__(self):
        if not self.cumulative_windows:
            return 0
        return self.cumulative_windows[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        subject_index = bisect_right(self.cumulative_windows, index)
        window_start = 0 if subject_index == 0 else self.cumulative_windows[subject_index - 1]
        window_index = index - window_start

        record = self.subject_records[subject_index]
        windows = load_cached_windows_memmap(record['cache_path'])
        return np.array(windows[window_index], dtype=np.float32, copy=True)

################################################################################
#
# Save/load helpers
#
################################################################################

def save_model(model_folder, model):
    require_torch()

    model_to_save = dict(model)
    encoder_state_dict = model_to_save.pop('encoder_state_dict')

    encoder_state_filename = 'encoder_state.pt'
    encoder_state_path = os.path.join(model_folder, encoder_state_filename)
    torch.save(encoder_state_dict, encoder_state_path)

    model_to_save['encoder_state_filename'] = encoder_state_filename

    wrapped_model = {'model': model_to_save}
    model_filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(wrapped_model, model_filename, protocol=0)
