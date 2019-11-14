from typing import List, Dict

import torch


def calculate_batch_statistics(
        original_tokens: torch.Tensor, predicted_tokens: torch.Tensor, skipping_tokens: List
) -> Dict:
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

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative
    }


def calculate_metrics(statistics: Dict) -> Dict:
    """Calculate precision, recall, and f1 based on a TP, FP, FN

    :param statistics: dict with statistics
    :return: metrics for given statistics
    """
    true_positive = statistics['true_positive']
    false_positive = statistics['false_positive']
    false_negative = statistics['false_negative']
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
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
