from torch import Tensor
from typing import Tuple, List


def calculate_per_subtoken_statistic(
        original_tokens: Tensor, predicted_tokens: Tensor, skipping_tokens: List
) -> Tuple[int, int, int]:
    """Calculate The TP, FP, and FN for given tensors with original and predicted tokens.
    Each tensor is a 2d matrix, where each row contain information about corresponding subtokens.

    :param skipping_tokens: list of tokens, which will be skipped
    :param original_tokens: tensor with original subtokens
    :param predicted_tokens: tensor with predicted subtokens
    :return: statistic for given tensors
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for original_token, predicted_token in zip(original_tokens, predicted_tokens):
        for subtoken in predicted_token:
            if subtoken in skipping_tokens:
                continue
            if subtoken in original_token:
                true_positive += 1
            else:
                false_positive += 1
        for subtoken in original_token:
            if subtoken in skipping_tokens:
                continue
            if subtoken not in predicted_token:
                false_negative += 1

    return true_positive, false_positive, false_negative


def calculate_metrics(true_positive: int, false_positive: int, false_negative: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, and f1 based on a TP, FP, FN

    :param true_positive:
    :param false_positive:
    :param false_negative:
    :return: metrics for given statistic
    """
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0.0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0.0
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    return precision, recall, f1_score
