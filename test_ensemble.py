import unittest

import torch

from team_code import make_small_eeg_cnn, predict_ensemble_probability


class ConstantModel(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return x.new_full((len(x), 1), self.logit)


class EnsembleTest(unittest.TestCase):
    def test_averages_model_and_window_probabilities(self):
        models = [ConstantModel(-2), ConstantModel(0), ConstantModel(2)]
        self.assertAlmostEqual(predict_ensemble_probability(models, torch.zeros(4, 1)), 0.5)

    def test_averages_probabilities_like_submission_one(self):
        models = [ConstantModel(0), ConstantModel(2)]
        expected = (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))) / 2
        self.assertAlmostEqual(predict_ensemble_probability(models, torch.zeros(4, 1)), expected.item(), places=6)

    def test_ecg_cnn_accepts_one_channel(self):
        self.assertEqual(make_small_eeg_cnn(1)(torch.zeros(2, 1, 800)).shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
