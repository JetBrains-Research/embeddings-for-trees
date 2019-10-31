import sys
import unittest

import torch

sys.path.append('.')

from utils.metrics import calculate_per_subtoken_statistic, calculate_metrics


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
        tp, fp, fn = calculate_per_subtoken_statistic(original_tokens, predicted_tokens)
        self.assertEqual(tp, 3)
        self.assertEqual(fp, 7)
        self.assertEqual(fn, 4)

    def test_calculating_metrics(self):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        precision, recall, f1_score = calculate_metrics(true_positive, false_positive, false_negative)
        self.assertEqual(precision, 0)
        self.assertEqual(recall, 0)
        self.assertEqual(f1_score, 0)


if __name__ == '__main__':
    unittest.main()
