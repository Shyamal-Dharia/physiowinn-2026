import unittest

import torch

from team_code import predict_ensemble_probability


class ConstantModel(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return x.new_full((len(x), 1), self.logit)


class EnsembleTest(unittest.TestCase):
    def test_averages_model_and_window_logits(self):
        models = [ConstantModel(-2), ConstantModel(0), ConstantModel(2)]
        self.assertAlmostEqual(predict_ensemble_probability(models, torch.zeros(4, 1)), 0.5)


if __name__ == '__main__':
    unittest.main()
