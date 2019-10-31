from torch import Tensor
from typing import Tuple


def calculate_per_subtoken_statistic(
        original_tokens: Tensor, predicted_tokens: Tensor, unk_id: int = 0, nan_id: int = -1
) -> Tuple[int, int, int]:
    """Calculate The TP, FP, and FN for given tensors with original and predicted tokens.
    Each tensor is a 2d matrix, where each row contain information about corresponding subtokens.

    :param original_tokens: tensor with original subtokens
    :param predicted_tokens: tensor with predicted subtokens
    :param unk_id: id of UNK subtoken
    :param nan_id: id of NAN subtoken
    :return: statistic for given tensors
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for original_token, predicted_token in zip(original_tokens, predicted_tokens):
        for subtoken in predicted_token:
            if subtoken == unk_id or subtoken == nan_id:
                continue
            if subtoken in original_token:
                true_positive += 1
            else:
                false_positive += 1
        for subtoken in original_token:
            if subtoken == unk_id or subtoken == nan_id:
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
    eps = 1e10
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1_score
