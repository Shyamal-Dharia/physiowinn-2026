#!/usr/bin/env python

"""Rebuild zero-filled subjects in an existing EEG cache.

A contiguous extent of eeg_cache/X.dat (cache indices 1-745) was lost after the
original build, leaving those subjects as all-zero windows. This script
regenerates exactly the windows the original build would have written — same
EDF loading, filtering, and per-subject rng (seed + original demographics
index) — writes them in place, and verifies the result.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from team_code import (
    EEG_SEGMENT_SECONDS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    load_eeg_signals,
    sample_eeg_windows,
)


def _rebuild_subject(task):
    cache_index, original_index, patient_id, site_id, session_id, data_folder, meta = task
    edf_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f'{patient_id}_ses-{session_id}.edf')
    try:
        eeg_signals, fs = load_eeg_signals(
            edf_path,
            channels=tuple(meta['channels']),
            target_fs=meta['fs'],
            bandpass=tuple(meta['bandpass']) if meta['bandpass'] is not None else None,
        )
        windows = sample_eeg_windows(
            eeg_signals,
            fs,
            num_windows=meta['shape'][1],
            channels=tuple(meta['channels']),
            rng=meta['seed'] + original_index,
        )
        expected = (meta['shape'][1], len(meta['channels']), int(round(EEG_SEGMENT_SECONDS * meta['fs'])))
        if windows.shape != expected:
            return cache_index, None, f'shape {windows.shape} != {expected}'
        return cache_index, windows, ''
    except Exception as exc:
        return cache_index, None, str(exc)


def find_zero_subjects(X):
    zero = []
    for i in range(X.shape[0]):
        if not X[i].any():
            zero.append(i)
        elif not X[i].std(axis=(1, 2)).all():
            zero.append(i)  # partially zero windows count as damaged
    return zero


def main(args):
    with open(os.path.join(args.cache_folder, 'meta.json')) as f:
        meta = json.load(f)
    shape = tuple(meta['shape'])
    X = np.memmap(os.path.join(args.cache_folder, 'X.dat'), dtype=np.float32, mode='r+', shape=shape)

    print('Scanning for damaged subjects...', flush=True)
    targets = find_zero_subjects(X)
    print(f'damaged subjects: {len(targets)} (min={min(targets)}, max={max(targets)})', flush=True)
    if not targets:
        print('Nothing to repair.')
        return

    # Untouched-subject guard: checksum a few intact rows before writing.
    intact = [i for i in (0, shape[0] // 2, shape[0] - 1) if i not in targets]
    guards = {i: float(np.abs(X[i]).sum()) for i in intact}

    by_cache_index = {}
    with open(os.path.join(args.cache_folder, 'records.csv'), newline='') as f:
        for row in csv.DictReader(f):
            if row['status'] == 'kept':
                by_cache_index[int(row['cache_index'])] = row

    tasks = []
    for cache_index in targets:
        row = by_cache_index[cache_index]
        tasks.append((
            cache_index, int(row['index']), row['patient_id'], row['site_id'],
            int(row['session_id']), args.data_folder, meta,
        ))

    repaired = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for cache_index, windows, reason in pool.map(_rebuild_subject, tasks, chunksize=1):
            if windows is None:
                failures.append((cache_index, reason))
                print(f'FAILED cache_index={cache_index}: {reason}', flush=True)
                continue
            X[cache_index] = windows
            repaired += 1
            if repaired % 50 == 0:
                X.flush()
                print(f'repaired {repaired}/{len(tasks)}', flush=True)
    X.flush()

    print('Verifying...', flush=True)
    for i, checksum in guards.items():
        assert abs(float(np.abs(X[i]).sum()) - checksum) < 1e-3, f'intact subject {i} changed!'
    still_zero = [i for i in targets if not X[i].std(axis=(1, 2)).all()]
    print(f'repaired={repaired} failed={len(failures)} still_damaged={len(still_zero)}', flush=True)
    if failures:
        print('failures:', failures, flush=True)
    if still_zero:
        raise SystemExit(f'still damaged after repair: {still_zero}')
    print('Cache repair complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-folder', default='eeg_cache')
    parser.add_argument('--data-folder', default='dataset')
    parser.add_argument('--num-workers', type=int, default=16)
    main(parser.parse_args())
