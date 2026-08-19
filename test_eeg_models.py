import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from analyze_age_residualization import residualize_logits
from team_code import EEG_BINARY_THRESHOLD, EEG_CHANNEL_DROPOUT, EEG_EPOCHS, EEG_MIN_CHANNELS, EEG_MIN_WINDOWS, EEG_MODEL_FILE, EEG_MODEL_NAME, EEG_SEEDS, age_conditioned_pairwise_loss, apply_eeg_channel_dropout, compute_eeg_oof_metrics, eeg_checkpoint_score, empirical_cdf_score, load_model, make_eeg_model, rank_normalize_eeg_folds, sample_eeg_windows_tolerant, verify_eeg_cache


class TestEEGModels(unittest.TestCase):
    def test_checkpoint_score_prefers_age_conditioned_auroc(self):
        high_reward = {'age_conditioned_auroc': 0.70, 'auroc': 0.80, 'reward_best': 0.20}
        high_age_auroc = {'age_conditioned_auroc': 0.75, 'auroc': 0.76, 'reward_best': 0.01}
        self.assertIs(max((high_reward, high_age_auroc), key=eeg_checkpoint_score), high_age_auroc)

    def test_checkpoint_score_can_restore_original_reward_selection(self):
        high_reward = {'age_conditioned_auroc': 0.70, 'auroc': 0.80, 'reward_best': 0.20}
        high_age_auroc = {'age_conditioned_auroc': 0.75, 'auroc': 0.76, 'reward_best': 0.01}
        selected = max((high_reward, high_age_auroc), key=lambda row: eeg_checkpoint_score(row, 'reward_best'))
        self.assertIs(selected, high_reward)

    def test_age_conditioned_pairwise_loss_rewards_correct_local_ranking(self):
        labels = torch.tensor([1.0, 0.0])
        ages = torch.tensor([60.0, 61.0])
        correct = age_conditioned_pairwise_loss(torch.tensor([2.0, -2.0]), labels, ages)
        reversed_order = age_conditioned_pairwise_loss(torch.tensor([-2.0, 2.0]), labels, ages)
        self.assertLess(correct, reversed_order)

    def test_age_conditioned_pairwise_loss_ignores_distant_ages(self):
        logits = torch.tensor([2.0, -2.0], requires_grad=True)
        loss = age_conditioned_pairwise_loss(logits, torch.tensor([1.0, 0.0]), torch.tensor([40.0, 80.0]))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_oof_metrics_compute_age_conditioned_score_on_combined_predictions(self):
        metrics = compute_eeg_oof_metrics(
            labels=[1, 0, 1, 0],
            probabilities=[0.9, 0.1, 0.2, 0.8],
            ages=[50, 51, 80, 81],
        )
        self.assertEqual(metrics['subjects'], 4)
        self.assertEqual(metrics['age_conditioned_auroc'], 0.5)
        self.assertEqual(metrics['auroc'], 0.75)

    def test_fold_rank_normalization_removes_fold_scale_shift(self):
        probabilities = rank_normalize_eeg_folds(
            probabilities=[0.9, 0.8, 0.2, 0.1],
            fold_indices=[1, 1, 2, 2],
        )
        metrics = compute_eeg_oof_metrics(
            labels=[1, 0, 1, 0],
            probabilities=probabilities,
            ages=[50, 50, 50, 50],
        )
        self.assertEqual(metrics['age_conditioned_auroc'], 1.0)

    def test_full_age_residualization_removes_linear_logit_trend(self):
        probabilities = 1 / (1 + np.exp(-np.asarray([-1, 1, -2, 2])))
        adjusted, slopes = residualize_logits(
            probabilities=probabilities,
            ages=[50, 60, 50, 60],
            fold_indices=[1, 1, 2, 2],
            alpha=1,
        )
        self.assertAlmostEqual(adjusted[0], adjusted[1])
        self.assertAlmostEqual(adjusted[2], adjusted[3])
        self.assertEqual(set(slopes), {1, 2})

    def test_submission_model_shape(self):
        x = torch.zeros(2, 6, 800)
        self.assertEqual(EEG_MODEL_NAME, 'cnn')
        self.assertEqual(make_eeg_model(EEG_MODEL_NAME)(x).shape, (2, 1))

    def test_optional_conformer_has_no_dropout(self):
        model = make_eeg_model('eegconformer_depth4')
        dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
        self.assertTrue(dropouts)
        self.assertTrue(all(module.p == 0 for module in dropouts))

    def test_empirical_cdf_score_interpolates_validation_ranks(self):
        calibration = [0.1, 0.2, 0.4]
        self.assertAlmostEqual(empirical_cdf_score(0.1, calibration), 1 / 6)
        self.assertAlmostEqual(empirical_cdf_score(0.2, calibration), 1 / 2)
        self.assertAlmostEqual(empirical_cdf_score(0.3, calibration), 2 / 3)
        self.assertAlmostEqual(empirical_cdf_score(0.2, [0.1, 0.2, 0.2, 0.4]), 1 / 2)

    def test_snapshot_ensemble_checkpoint_round_trip(self):
        saved_models = [
            {
                'model_name': EEG_MODEL_NAME,
                'seed': seed,
                'snapshot_state_dicts': [make_eeg_model(EEG_MODEL_NAME).state_dict() for _ in range(3)],
                'calibration_probabilities': np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
            }
            for seed in EEG_SEEDS
        ]
        with TemporaryDirectory() as model_folder:
            torch.save(
                {
                    'models': saved_models,
                    'ensemble_calibration_probabilities': np.asarray([0.2, 0.5, 0.8], dtype=np.float32),
                    'binary_threshold': EEG_BINARY_THRESHOLD,
                },
                f'{model_folder}/{EEG_MODEL_FILE}',
            )
            loaded = load_model(model_folder, verbose=False)
        self.assertEqual(len(loaded['models']), len(EEG_SEEDS))
        for entry in loaded['models']:
            self.assertEqual(len(entry['snapshots']), 3)
            for net in entry['snapshots']:
                x = torch.zeros(2, 6, 800, device=next(net.parameters()).device)
                self.assertEqual(net(x).shape, (2, 1))

    def test_channel_dropout_zeros_channels_but_never_all(self):
        torch.manual_seed(0)
        x = torch.ones(512, 6, 800)
        out = apply_eeg_channel_dropout(x, 0.5)
        alive = (out.abs().sum(dim=-1) > 0).sum(dim=1)
        self.assertGreater(int((alive < 6).sum()), 0, 'no channel was ever dropped')
        self.assertEqual(int((alive == 0).sum()), 0, 'a window lost every channel')

    def test_channel_dropout_disabled_is_identity(self):
        x = torch.ones(8, 6, 800)
        self.assertTrue(torch.equal(apply_eeg_channel_dropout(x, 0.0), x))

    def test_tolerant_sampler_zero_fills_missing_channels(self):
        fs = 200.0
        channels = ('f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1')
        rng = np.random.default_rng(0)
        signals = {c: rng.normal(size=int(fs * 600)).astype(np.float32) for c in channels[:4]}
        windows = sample_eeg_windows_tolerant(
            signals, fs, num_windows=50, channels=channels,
            min_windows=EEG_MIN_WINDOWS, min_channels=EEG_MIN_CHANNELS,
        )
        self.assertEqual(windows.shape, (50, 6, 800))
        per_channel = np.abs(windows).sum(axis=(0, 2))
        self.assertTrue(all(per_channel[:4] > 0))
        self.assertTrue(all(per_channel[4:] == 0), 'missing channels must be zero-filled')

    def test_tolerant_sampler_rejects_too_few_channels_or_windows(self):
        fs = 200.0
        channels = ('f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1')
        rng = np.random.default_rng(0)
        one_channel = {channels[0]: rng.normal(size=int(fs * 600)).astype(np.float32)}
        self.assertIsNone(sample_eeg_windows_tolerant(
            one_channel, fs, 50, channels, EEG_MIN_WINDOWS, EEG_MIN_CHANNELS))
        too_short = {c: rng.normal(size=int(fs * 40)).astype(np.float32) for c in channels}
        self.assertIsNone(sample_eeg_windows_tolerant(
            too_short, fs, 50, channels, EEG_MIN_WINDOWS, EEG_MIN_CHANNELS))

    def test_verify_eeg_cache_rejects_zero_filled_subjects(self):
        import json

        with TemporaryDirectory() as cache_folder:
            shape = [4, 2, 6, 800]
            X = np.memmap(f'{cache_folder}/X.dat', dtype=np.float32, mode='w+', shape=tuple(shape))
            X[:] = 1.0
            X[2] = 0.0
            X.flush()
            del X
            with open(f'{cache_folder}/meta.json', 'w') as f:
                json.dump({'shape': shape}, f)
            with self.assertRaises(RuntimeError):
                verify_eeg_cache(cache_folder, verbose=False)

    def test_submission_config_matches_benchmarked_grid(self):
        self.assertEqual(EEG_MODEL_NAME, 'cnn')
        self.assertEqual(tuple(EEG_SEEDS), (0, 1, 2))
        self.assertEqual(EEG_EPOCHS, 10)
        self.assertAlmostEqual(EEG_CHANNEL_DROPOUT, 0.15)
        self.assertAlmostEqual(EEG_BINARY_THRESHOLD, 0.75)


if __name__ == '__main__':
    unittest.main()
