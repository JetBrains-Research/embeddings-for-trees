from argparse import ArgumentParser
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from numpy import cumsum

from data_process.dataset import JavaDataset
from model.decoder import LinearDecoder
from model.embedding import TokenEmbedding
from model.treelstm import TreeLSTM
from utils.pytorch_utils import fix_seed, get_device


def train(params: Dict) -> None:
    fix_seed()
    device = get_device()

    training_set = JavaDataset(params['train_batches'])
    validation_set = JavaDataset(params['validation_batches'])

    with open(params['labels_path'], 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)

    with open(params['vocabulary_path'], 'rb') as pkl_file:
        vocabulary = pkl_load(pkl_file)
        token_to_id = vocabulary['token_to_id']
        type_to_id = vocabulary['type_to_id']

    # create models
    embedding = TokenEmbedding(len(token_to_id), params['x_size']).to(device)
    encoder = TreeLSTM(params['x_size'], params['h_size']).to(device)
    decoder = LinearDecoder(params['x_size'], len(label_to_id)).to(device)

    # create optimizers
    embedding_optimizer = torch.optim.RMSprop(
        embedding.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    encoder_optimizer = torch.optim.RMSprop(
        encoder.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    decoder_optimizer = torch.optim.RMSprop(
        decoder.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(params['n_epochs']):
        running_loss = 0.0
        for step, (graph, labels) in tqdm(enumerate(training_set)):
            torch_labels = torch.tensor([label_to_id.get(label, 0) for label in labels]).to(device)
            mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
            idx_of_roots = cumsum([0] + graph.batch_num_nodes)[:-1]
            mask[idx_of_roots] = 1

            embedding_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            graph_feature_embeds = embedding(graph)
            tree_embeds = encoder(graph_feature_embeds)
            logits = decoder(tree_embeds)
            root_logits = logits[mask]

            loss = criterion(root_logits, torch_labels)
            loss.backward()
            embedding_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()

            running_loss += loss.item()
        print(f"epoch #{epoch} -- loss: {running_loss}")


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file))

