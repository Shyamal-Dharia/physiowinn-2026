#!/usr/bin/env python

"""Assemble robust-EEG benchmark runs into the decisions for the final entry.

Reads the per-run npz/json artifacts written by benchmark_robust_eeg.py and
reports, for each channel-dropout setting:
  1. single-model and 6-model-ensemble OOF age-conditioned AUROC,
     at best-epoch-per-fold and at every fixed epoch budget;
  2. channel-ablation stress scores (1 and 2 channels zeroed);
  3. window aggregation comparison (mean / trimmed mean / median) and
     window-count subsampling curves from window-level probabilities.
"""

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
from scipy.stats import rankdata

from team_code import load_cache_subject_ages


def age_pairs(labels, ages, indices, gap=2):
    indices = np.asarray(indices)
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    eligible = np.abs(ages[positive, None] - ages[negative]) <= gap
    pos_row, neg_row = np.nonzero(eligible)
    return positive[pos_row], negative[neg_row]


def age_auroc(scores, pairs):
    positive, negative = pairs
    difference = scores[positive] - scores[negative]
    return float((np.count_nonzero(difference > 0) + 0.5 * np.count_nonzero(difference == 0)) / len(difference))


def rank01(values):
    return (rankdata(values, method='average') - 0.5) / len(values)


def load_runs(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        name = os.path.basename(path)[:-5]
        match = re.match(r'(cnn|rescnn)_cd([\d.]+)_s(\d+)_f(\d+)$', name)
        if not match:
            continue
        model, cd, seed, fold = match.group(1), float(match.group(2)), int(match.group(3)), int(match.group(4))
        with open(path) as f:
            summary = json.load(f)
        data = np.load(os.path.join(results_dir, name + '.npz'))
        runs[(model, cd, seed, fold)] = {'summary': summary, 'data': data}
    return runs


def oof_scores(runs, members, cd, folds, n_subjects, epoch=None):
    """OOF scores for an ensemble of (model, seed) members at a fixed epoch or best-epoch."""
    scores = np.full(n_subjects, np.nan, dtype=np.float64)
    for fold in folds:
        member_scores = []
        for model, seed in members:
            run = runs.get((model, cd, seed, fold))
            if run is None:
                return None
            data = run['data']
            if epoch is None:
                probs = data['per_epoch_val_probs'][run['summary']['best_epoch'] - 1]
            else:
                probs = data['per_epoch_val_probs'][epoch - 1]
            member_scores.append(rank01(probs))
        scores[data['val_subjects']] = rank01(np.mean(member_scores, axis=0))
    return scores


def main(args):
    labels = np.load(os.path.join(args.cache_folder, 'y.npy')).astype(np.int8)
    ages = load_cache_subject_ages(args.cache_folder, args.data_folder)
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f'no runs found in {args.results_dir}')
    cds = sorted({k[1] for k in runs})
    folds = sorted({k[3] for k in runs})
    seeds = sorted({k[2] for k in runs})
    epochs = runs[next(iter(runs))]['data']['per_epoch_val_probs'].shape[0]
    all_pairs = age_pairs(labels, ages, np.arange(len(labels)))
    report = {'runs': len(runs), 'cds': cds, 'folds': folds, 'seeds': seeds}

    print(f'loaded {len(runs)} runs | cds={cds} seeds={seeds} folds={folds}')

    # --- 1. single-model and ensemble OOF, best-epoch vs fixed-epoch -------
    ensembles = {
        'cnn(1seed)': [('cnn', 0)],
        'rescnn(1seed)': [('rescnn', 0)],
        'cnn+rescnn(1seed)': [('cnn', 0), ('rescnn', 0)],
        'cnn x3seeds': [('cnn', s) for s in seeds],
        'rescnn x3seeds': [('rescnn', s) for s in seeds],
        'all6': [(m, s) for m, s in itertools.product(('cnn', 'rescnn'), seeds)],
    }
    fixed_epoch_grid = [e for e in range(5, epochs + 1, 5)]
    report['oof'] = {}
    for cd in cds:
        print(f'\n=== channel_dropout={cd} ===')
        print(f"{'ensemble':>20} {'best-epoch':>11}" + ''.join(f'  ep{e:<4}' for e in fixed_epoch_grid))
        for name, members in ensembles.items():
            best = oof_scores(runs, members, cd, folds, len(labels), epoch=None)
            if best is None:
                continue
            row = {'best_epoch': age_auroc(best, all_pairs)}
            line = f'{name:>20} {row["best_epoch"]:11.4f}'
            for e in fixed_epoch_grid:
                fixed = oof_scores(runs, members, cd, folds, len(labels), epoch=e)
                row[f'epoch_{e}'] = age_auroc(fixed, all_pairs)
                line += f'  {row[f"epoch_{e}"]:.4f}'
            report['oof'][f'cd{cd}|{name}'] = row
            print(line)

    # --- 2. stress: channel ablation degradation --------------------------
    print('\n=== channel-ablation stress (best-state, ensemble of all runs per cd) ===')
    report['stress'] = {}
    for cd in cds:
        for variant in ('clean', 'stress1', 'stress2'):
            scores = np.full(len(labels), np.nan)
            ok = True
            for fold in folds:
                member_scores = []
                for model, seed in ensembles['all6']:
                    run = runs.get((model, cd, seed, fold))
                    if run is None:
                        ok = False
                        break
                    data = run['data']
                    if variant == 'clean':
                        probs = data['best_window_probs'].astype(np.float32).mean(axis=1)
                    else:
                        probs = data[f'best_{variant}_probs']
                    member_scores.append(rank01(probs))
                if not ok:
                    break
                scores[data['val_subjects']] = rank01(np.mean(member_scores, axis=0))
            if ok:
                value = age_auroc(scores, all_pairs)
                report['stress'][f'cd{cd}|{variant}'] = value
                print(f'cd={cd} {variant:>8}: {value:.4f}')

    # --- 3. aggregation and window-count curves ----------------------------
    print('\n=== window aggregation (best-state, all6 ensemble per cd) ===')
    report['aggregation'] = {}

    def aggregate(window_probs, how):
        if how == 'mean':
            return window_probs.mean(axis=1)
        if how == 'median':
            return np.median(window_probs, axis=1)
        trim = int(how.split('_')[1]) / 100
        k = int(window_probs.shape[1] * trim)
        ordered = np.sort(window_probs, axis=1)
        return ordered[:, k:window_probs.shape[1] - k].mean(axis=1)

    for cd in cds:
        for how in ('mean', 'trimmed_10', 'trimmed_20', 'median'):
            scores = np.full(len(labels), np.nan)
            ok = True
            for fold in folds:
                member_scores = []
                for model, seed in ensembles['all6']:
                    run = runs.get((model, cd, seed, fold))
                    if run is None:
                        ok = False
                        break
                    wp = run['data']['best_window_probs'].astype(np.float32)
                    member_scores.append(rank01(aggregate(wp, how)))
                if not ok:
                    break
                scores[run['data']['val_subjects']] = rank01(np.mean(member_scores, axis=0))
            if ok:
                value = age_auroc(scores, all_pairs)
                report['aggregation'][f'cd{cd}|{how}'] = value
                print(f'cd={cd} {how:>11}: {value:.4f}')

    print('\n=== window-count subsampling (best-state, all6, mean agg) ===')
    rng = np.random.default_rng(0)
    for cd in cds:
        line = f'cd={cd}'
        for count in (50, 100, 200, 300):
            values = []
            for draw in range(5 if count < 300 else 1):
                scores = np.full(len(labels), np.nan)
                ok = True
                for fold in folds:
                    member_scores = []
                    for model, seed in ensembles['all6']:
                        run = runs.get((model, cd, seed, fold))
                        if run is None:
                            ok = False
                            break
                        wp = run['data']['best_window_probs'].astype(np.float32)
                        cols = rng.choice(wp.shape[1], size=count, replace=False) if count < wp.shape[1] else np.arange(wp.shape[1])
                        member_scores.append(rank01(wp[:, cols].mean(axis=1)))
                    if not ok:
                        break
                    scores[run['data']['val_subjects']] = rank01(np.mean(member_scores, axis=0))
                if ok:
                    values.append(age_auroc(scores, all_pairs))
            if values:
                report['aggregation'][f'cd{cd}|windows_{count}'] = float(np.mean(values))
                line += f'  w{count}={np.mean(values):.4f}'
        print(line)

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nreport written to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='/scratch/shyamal/physionet-2026/results/robust')
    parser.add_argument('--cache-folder', default='eeg_cache')
    parser.add_argument('--data-folder', default='dataset')
    parser.add_argument('--output', default='/scratch/shyamal/physionet-2026/results/robust_eeg_report.json')
    main(parser.parse_args())
