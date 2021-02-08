from typing import List, Tuple

import torch


def segment_sizes_to_slices(sizes: torch.Tensor) -> List:
    cum_sums = torch.cumsum(sizes, dim=0)
    slices = [slice(0, cum_sums[0])]
    slices += [slice(start, end) for start, end in zip(cum_sums[:-1], cum_sums[1:])]
    return slices


def cut_encoded_data(
    encoded_data: torch.Tensor, samples_per_label: torch.LongTensor, mask_value: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut encoded data into batches
    :param encoded_data: [n contexts; units]
    :param samples_per_label: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """
    batch_size = len(samples_per_label)
    max_context_len = max(samples_per_label)

    batched_contexts = encoded_data.new_zeros((batch_size, max_context_len, encoded_data.shape[-1]))
    attention_mask = encoded_data.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(samples_per_label)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, samples_per_label)):
        batched_contexts[i, :cur_size] = encoded_data[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    return batched_contexts, attention_mask
