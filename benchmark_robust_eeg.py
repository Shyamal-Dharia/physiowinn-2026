#!/usr/bin/env python

"""Channel-dropout robustness benchmark for the EEG CNN/ResCNN branch.

Trains one (model, train-seed, channel-dropout, fold) cell on the shared
StratifiedKFold(3, random_state=0) split used by the existing OOF baselines,
recording per-epoch validation probabilities, best/final window-level
probabilities, and channel-ablation stress scores so every downstream
decision (fixed epoch budget, ensembling, aggregation, robustness) can be
made offline without retraining.
"""

import argparse
import json
import os
import time

import numpy as np

from team_code import (
    compute_eeg_oof_metrics,
    load_cache_subject_ages,
    make_eeg_model,
    make_eeg_supervised_loaders,
)


def apply_channel_dropout(x, rate):
    """Zero random channels per window; never drop every channel."""
    import torch

    keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= rate
    keep |= ~keep.any(dim=1, keepdim=True)
    return x * keep


def stress_masks(num_subjects, num_channels, dropped, seed=9000):
    """Deterministic per-subject channel ablation masks, shared across runs."""
    masks = np.ones((num_subjects, num_channels), dtype=np.float32)
    for i in range(num_subjects):
        rng = np.random.default_rng(seed + i)
        masks[i, rng.choice(num_channels, size=dropped, replace=False)] = 0.0
    return masks


def predict_val(model, val_loader, device, channel_masks=None):
    """Window-level sigmoid probabilities for the subject-level val loader."""
    import torch

    model.eval()
    window_probs = []
    labels = []
    position = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            if channel_masks is not None:
                mask = torch.from_numpy(channel_masks[position:position + x.shape[0]]).to(device)
                x = x * mask[:, None, :, None]
            batch, windows, channels, samples = x.shape
            probs = torch.sigmoid(model(x.reshape(batch * windows, channels, samples)).reshape(batch, windows))
            window_probs.append(probs.float().cpu().numpy())
            labels.extend(y.numpy().tolist())
            position += batch
    return np.asarray(labels, dtype=np.int8), np.concatenate(window_probs, axis=0)


def main(args):
    import torch
    from sklearn.model_selection import StratifiedKFold

    started = time.time()
    labels = np.load(os.path.join(args.cache_folder, 'y.npy')).astype(np.int8)
    ages = load_cache_subject_ages(args.cache_folder, args.data_folder)
    with open(os.path.join(args.cache_folder, 'meta.json')) as f:
        cache_shape = tuple(json.load(f)['shape'])
    num_channels = cache_shape[2]

    # Integrity gate: the cache once lost a contiguous extent to write-back
    # failure, silently zeroing 11.5% of subjects. First window per subject
    # (~125MB strided read) catches any hole wider than one subject stride.
    X = np.memmap(os.path.join(args.cache_folder, 'X.dat'), dtype=np.float32, mode='r', shape=cache_shape)
    bad = [i for i in range(cache_shape[0]) if not np.abs(X[i, 0]).max() > 0]
    del X
    if bad:
        raise SystemExit(f'cache integrity check failed: zero subjects {bad[:10]} (n={len(bad)})')
    splits = list(
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.fold_seed)
        .split(np.arange(len(labels)), labels)
    )
    train_subjects, val_subjects = splits[args.fold - 1]
    if args.smoke:
        val_subjects = val_subjects[:200]

    torch.manual_seed(args.train_seed + args.fold - 1)
    np.random.seed(args.train_seed + args.fold - 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _, _ = make_eeg_supervised_loaders(
        args.cache_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )
    val_ages = ages[val_subjects]

    counts = np.bincount(labels[train_subjects].astype(int), minlength=2)
    model = make_eeg_model(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([counts[0] / max(1, counts[1])], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    per_epoch_probs = np.zeros((args.epochs, len(val_subjects)), dtype=np.float32)
    best = {'score': -np.inf, 'epoch': 0, 'state': None}
    val_labels = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, (x, yb) in enumerate(train_loader):
            if args.smoke and step >= 20:
                break
            x = x.to(device)
            if args.channel_dropout:
                x = apply_channel_dropout(x, args.channel_dropout)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).flatten(), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_labels, window_probs = predict_val(model, val_loader, device)
        probs = window_probs.mean(axis=1)
        per_epoch_probs[epoch - 1] = probs
        metrics = compute_eeg_oof_metrics(val_labels, probs, val_ages)
        metrics.update({'epoch': epoch, 'train_loss': float(np.mean(losses))})
        history.append(metrics)
        if metrics['age_conditioned_auroc'] > best['score']:
            best = {
                'score': metrics['age_conditioned_auroc'],
                'epoch': epoch,
                'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        print(
            f"epoch={epoch}/{args.epochs} loss={metrics['train_loss']:.4f} "
            f"age_auroc={metrics['age_conditioned_auroc']:.4f} auroc={metrics['auroc']:.4f}",
            flush=True,
        )

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    outputs = {
        'val_subjects': val_subjects.astype(np.int32),
        'val_labels': val_labels.astype(np.int8),
        'val_ages': val_ages.astype(np.float32),
        'per_epoch_val_probs': per_epoch_probs,
    }
    stress = {}
    for tag, state in (('best', best['state']), ('final', final_state)):
        model.load_state_dict(state)
        _, window_probs = predict_val(model, val_loader, device)
        outputs[f'{tag}_window_probs'] = window_probs.astype(np.float16)
        for dropped in (1, 2):
            masks = stress_masks(len(val_subjects), num_channels, dropped)
            _, stressed = predict_val(model, val_loader, device, channel_masks=masks)
            probs = stressed.mean(axis=1)
            outputs[f'{tag}_stress{dropped}_probs'] = probs
            stress[f'{tag}_drop{dropped}_age_auroc'] = compute_eeg_oof_metrics(val_labels, probs, val_ages)['age_conditioned_auroc']

    np.savez_compressed(args.output_base + '.npz', **outputs)
    torch.save(
        {
            'model_name': args.model_name,
            'best_epoch': best['epoch'],
            'best_state_dict': best['state'],
            'final_state_dict': final_state,
        },
        args.output_base + '.pt',
    )
    summary = {
        'model_name': args.model_name,
        'train_seed': args.train_seed,
        'channel_dropout': args.channel_dropout,
        'fold': args.fold,
        'folds': args.folds,
        'fold_seed': args.fold_seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'smoke': args.smoke,
        'val_subjects': int(len(val_subjects)),
        'best_epoch': best['epoch'],
        'best_age_auroc': best['score'],
        'final_age_auroc': history[-1]['age_conditioned_auroc'],
        'stress': stress,
        'elapsed_seconds': time.time() - started,
        'history': history,
    }
    with open(args.output_base + '.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'history'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', choices=('cnn', 'rescnn'), required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--channel-dropout', type=float, default=0.0)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--fold-seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cache-folder', default='eeg_cache')
    parser.add_argument('--data-folder', default='dataset')
    parser.add_argument('--output-base', required=True)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
