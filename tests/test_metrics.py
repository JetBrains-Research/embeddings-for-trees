import unittest

import torch

from utils.metrics import calculate_batch_statistics, calculate_metrics


class MetricsTest(unittest.TestCase):
    def test_collecting_stats(self):
        original_tokens = torch.tensor([
            [1, 2, 3, -1],
            [1, 2, 3, -1],
            [1, 0, -1, -1],
            [0, -1, -1, -1]
        ])
        predicted_tokens = torch.tensor([
            [2, 4, 1, 5],   # TP = 2, FP = 2, FN = 1
            [4, 5, 6, -1],  # TP = 0, FP = 3, FN = 3
            [1, 2, 3, -1],  # TP = 1, FP = 2, FN = 0
            [0, 0, 0, -1],  # TP = 0, FP = 0, FN = 0
        ])
        true_statistics = {
            'true_positive': 3,
            'false_positive': 7,
            'false_negative': 4
        }
        test_statistics = calculate_batch_statistics(original_tokens, predicted_tokens, [-1, 0])
        for statistic in ['true_positive', 'false_positive', 'false_negative']:
            self.assertEqual(true_statistics[statistic], test_statistics[statistic])

    def test_calculating_zero_metrics(self):
        statistics = {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0
        }
        metrics = calculate_metrics(statistics)
        for metric, value in metrics.items():
            self.assertEqual(0.0, value)


if __name__ == '__main__':
    unittest.main()
