from argparse import ArgumentParser
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import torch
import torch.nn as nn
from numpy import cumsum
from tqdm.auto import tqdm

from data_workers.dataset import JavaDataset
from model.tree2seq import ModelFactory, Tree2Seq
from utils.common import fix_seed, get_device, split_tokens_to_subtokens
from utils.metrics import calculate_per_subtoken_statistic, calculate_metrics


def train(params: Dict) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    training_set = JavaDataset(params['paths']['train_batches'], params['batch_size'], True)
    validation_set = JavaDataset(params['paths']['validation_batches'], params['batch_size'], True)

    with open(params['paths']['labels_path'], 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)
    sublabel_to_id, label_to_sublabel = split_tokens_to_subtokens(label_to_id, device=device)

    with open(params['paths']['vocabulary_path'], 'rb') as pkl_file:
        vocabulary = pkl_load(pkl_file)
        token_to_id = vocabulary['token_to_id']
        type_to_id = vocabulary['type_to_id']
    subtoken_to_id, token_to_subtoken = split_tokens_to_subtokens(
        token_to_id, required_tokens=['UNK', 'METHOD_NAME', 'NAN', 'PAD'], return_ids=True, device=device
    )

    # create models
    params['embedding']['params']['token_vocab_size'] = len(subtoken_to_id)
    params['embedding']['params']['token_to_subtoken'] = token_to_subtoken
    params['embedding']['params']['padding_index'] = subtoken_to_id['PAD']
    params['embedding']['params']['type_vocab_size'] = len(type_to_id)
    params['decoder']['params']['out_size'] = len(sublabel_to_id)
    model_factory = ModelFactory(params['embedding'], params['encoder'], params['decoder'])
    model: Tree2Seq = model_factory.construct_model(device)

    # create optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=sublabel_to_id['PAD']).to(device)

    # train loop
    print("starting train loop...")
    for epoch in range(params['n_epochs']):
        epoch_loss = 0.0
        true_positive = false_positive = false_negative = 0
        number_of_samples = 0
        for batch_id in tqdm(range(len(training_set))):
            model.train()
            graph, labels = training_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)

            # Create target tensor [max sequence length, batch size]
            # Each row starts with START token and ends with END token, after it padded with PAD token
            sublabels_length = torch.tensor([label_to_sublabel[label].shape[0] for label in labels])
            max_sublabel_length = sublabels_length.max()
            torch_labels = torch.full((max_sublabel_length.item() + 2, len(labels)), sublabel_to_id['PAD'], dtype=torch.long).to(device)
            torch_labels[0, :] = sublabel_to_id['START']
            torch_labels[sublabels_length + 1, torch.arange(0, len(labels))] = sublabel_to_id['END']
            for sample, label in enumerate(labels):
                torch_labels[1:sublabels_length[sample] + 1, sample] = label_to_sublabel[label]

            # Find indexes of roots in batched graph
            mask = torch.zeros(graph.number_of_nodes())
            idx_of_roots = cumsum([0] + graph.batch_num_nodes)[:-1]
            mask[idx_of_roots] = 1
            root_indexes = torch.nonzero(mask).squeeze(1).to(device)

            # Model step
            model.zero_grad()
            root_logits = model(graph, root_indexes, torch_labels, device)
            root_logits = root_logits[1:]
            torch_labels = torch_labels[1:]
            loss = criterion(
                root_logits.view(-1, root_logits.shape[-1]),
                torch_labels.view(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
            optimizer.step()

            # Calculate metrics
            prediction = model.predict(root_logits)
            epoch_loss += loss.item()
            number_of_samples += torch_labels.shape[0]
            cur_metrics = calculate_per_subtoken_statistic(
                torch_labels, prediction,
                [sublabel_to_id['UNK'], sublabel_to_id['PAD'], sublabel_to_id['END']]
            )
            true_positive += cur_metrics[0]
            false_positive += cur_metrics[1]
            false_negative += cur_metrics[2]
            print(loss.item(), calculate_metrics(*cur_metrics))

        precision, recall, f1_score = calculate_metrics(true_positive, false_positive, false_negative)
        print(f"epoch #{epoch} -- "
              f"loss: {epoch_loss / number_of_samples}, "
              f"precision: {precision}, recall: {recall}, f1: {f1_score}")


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file))
