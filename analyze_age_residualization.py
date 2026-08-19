#!/usr/bin/env python

import argparse
import json

import numpy as np

from team_code import compute_eeg_oof_metrics, load_cache_subject_ages, rank_normalize_eeg_folds


def residualize_logits(probabilities, ages, fold_indices, alpha):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    ages = np.asarray(ages, dtype=np.float64)
    fold_indices = np.asarray(fold_indices, dtype=np.int16)
    logits = np.log(probabilities / (1 - probabilities))
    adjusted = logits.copy()
    slopes = {}

    for fold in np.unique(fold_indices):
        mask = (fold_indices == fold) & np.isfinite(ages)
        centered_ages = ages[mask] - ages[mask].mean()
        centered_logits = logits[mask] - logits[mask].mean()
        slope = float(centered_ages @ centered_logits / (centered_ages @ centered_ages))
        adjusted[mask] -= float(alpha) * slope * centered_ages
        slopes[int(fold)] = slope

    return adjusted, slopes


def evaluate_alpha(labels, probabilities, ages, fold_indices, alpha):
    adjusted, slopes = residualize_logits(probabilities, ages, fold_indices, alpha)
    normalized = rank_normalize_eeg_folds(adjusted, fold_indices)
    combined = compute_eeg_oof_metrics(labels, normalized, ages)
    per_fold = {
        int(fold): compute_eeg_oof_metrics(labels[fold_indices == fold], adjusted[fold_indices == fold], ages[fold_indices == fold])['age_conditioned_auroc']
        for fold in np.unique(fold_indices)
    }
    return {
        'alpha': float(alpha),
        'combined_age_conditioned_auroc': combined['age_conditioned_auroc'],
        'fold_age_conditioned_auroc': per_fold,
        'slopes_per_year': slopes,
    }


def main(args):
    with open(args.predictions) as f:
        oof = json.load(f)

    labels = np.load(args.labels).astype(np.int8)
    ages = load_cache_subject_ages(args.cache_folder, args.data_folder)
    probabilities = np.asarray(oof['oof_probabilities'], dtype=np.float64)
    fold_indices = np.asarray(oof['oof_fold_indices'], dtype=np.int16)
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    curve = [evaluate_alpha(labels, probabilities, ages, fold_indices, alpha) for alpha in alphas]
    best = max(curve, key=lambda row: row['combined_age_conditioned_auroc'])

    leave_one_fold_out = []
    folds = sorted(np.unique(fold_indices).tolist())
    for held_out in folds:
        training_folds = [fold for fold in folds if fold != held_out]
        selected = max(
            curve,
            key=lambda row: np.mean([row['fold_age_conditioned_auroc'][fold] for fold in training_folds]),
        )
        leave_one_fold_out.append({
            'held_out_fold': int(held_out),
            'selected_alpha': selected['alpha'],
            'held_out_age_conditioned_auroc': selected['fold_age_conditioned_auroc'][held_out],
            'baseline_age_conditioned_auroc': curve[0]['fold_age_conditioned_auroc'][held_out],
        })

    result = {
        'predictions': args.predictions,
        'baseline': curve[0],
        'best': best,
        'leave_one_fold_out': leave_one_fold_out,
        'leave_one_fold_out_mean': float(np.mean([row['held_out_age_conditioned_auroc'] for row in leave_one_fold_out])),
        'leave_one_fold_out_baseline_mean': float(np.mean([row['baseline_age_conditioned_auroc'] for row in leave_one_fold_out])),
        'curve': curve,
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"baseline={result['baseline']['combined_age_conditioned_auroc']:.6f}")
    print(f"best_alpha={best['alpha']:.2f} best={best['combined_age_conditioned_auroc']:.6f}")
    for row in leave_one_fold_out:
        print(
            f"held_out_fold={row['held_out_fold']} selected_alpha={row['selected_alpha']:.2f} "
            f"baseline={row['baseline_age_conditioned_auroc']:.6f} "
            f"residualized={row['held_out_age_conditioned_auroc']:.6f}"
        )
    print(
        f"leave_one_fold_out_baseline={result['leave_one_fold_out_baseline_mean']:.6f} "
        f"leave_one_fold_out_residualized={result['leave_one_fold_out_mean']:.6f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test linear age residualization on saved EEG OOF predictions.')
    parser.add_argument('--predictions', default='/scratch/shyamal/physionet-2026/results/rescnn_oof_3fold.json')
    parser.add_argument('--labels', default='eeg_cache/y.npy')
    parser.add_argument('--cache-folder', default='eeg_cache')
    parser.add_argument('--data-folder', default='dataset')
    parser.add_argument('--output', default='/scratch/shyamal/physionet-2026/results/rescnn_age_residualization.json')
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=1.5)
    parser.add_argument('--alpha-step', type=float, default=0.05)
    main(parser.parse_args())
