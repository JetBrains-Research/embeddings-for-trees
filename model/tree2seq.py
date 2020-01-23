from typing import Dict, Union, Tuple, List

import torch
import torch.nn as nn
from dgl import BatchedDGLGraph

from model.attention import LuongConcatAttention
from model.attention_decoder import LSTMAttentionDecoder, _IAttentionDecoder
from model.decoder import _IDecoder, LinearDecoder, LSTMDecoder
from model.embedding import _IEmbedding, FullTokenEmbedding, SubTokenEmbedding, SubTokenTypeEmbedding
from model.encoder import _IEncoder
from model.treeLSTM import TokenTreeLSTM, TokenTypeTreeLSTM
from utils.common import PAD, EOS, SOS


class Tree2Seq(nn.Module):
    def __init__(self, embedding: _IEmbedding, encoder: _IEncoder,
                 decoder: Union[_IDecoder, _IAttentionDecoder], using_attention: bool) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.using_attention = using_attention

    def forward(self,
                graph: BatchedDGLGraph, root_indexes: torch.LongTensor, labels: List[str],
                teacher_force: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param graph: the batched graph with function's asts
        :param root_indexes: indexes of roots in the batched graph
        :param labels: [batch size]
        :param teacher_force: probability of teacher forcing, 0 means use previous output
        :param device: torch device
        :return: logits and ground truth in the same manner
        """
        embedded_graph = self.embedding(graph, device)
        # [number of nodes, hidden state]
        node_hidden_states, node_memory_cells = self.encoder(embedded_graph, device)
        # [1, batch size, hidden state] (LSTM input requires)
        root_hidden_states = node_hidden_states[root_indexes].unsqueeze(0)
        root_memory_cells = node_memory_cells[root_indexes].unsqueeze(0)

        ground_truth = self.decoder.convert_labels(labels, device)

        max_length, batch_size = ground_truth.shape
        # [length of the longest sequence, batch size, number of classes]
        outputs = torch.zeros(max_length, batch_size, self.decoder.out_size).to(device)

        tree_sizes = [(root_indexes[i] - root_indexes[i - 1]).item() for i in range(1, batch_size)]
        tree_sizes.append(node_hidden_states.shape[0] - root_indexes[-1].item())

        current_input = ground_truth[0]
        for step in range(1, max_length):

            if self.using_attention:
                output, root_hidden_states, root_memory_cells = \
                    self.decoder(current_input, root_hidden_states, root_memory_cells,
                                 node_hidden_states, tree_sizes)
            else:
                output, root_hidden_states, root_memory_cells = \
                    self.decoder(current_input, root_hidden_states, root_memory_cells)

            outputs[step] = output
            current_input = ground_truth[step] if torch.rand(1) < teacher_force else output.argmax(dim=1)

        return outputs, ground_truth

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        """Predict token for each step by given logits

        :param logits: [max length, batch size, number of classes] logits for each position in sequence
        :return: [max length, batch size] token's ids for each position in sequence
        """
        tokens_probas = nn.functional.softmax(logits, dim=-1)
        return tokens_probas.argmax(dim=-1)


class ModelFactory:
    _embeddings = {
        'FullTokenEmbedding': FullTokenEmbedding,
        'SubTokenEmbedding': SubTokenEmbedding,
        'SubTokenTypeEmbedding': SubTokenTypeEmbedding
    }
    _encoders = {
        'TokenTreeLSTM': TokenTreeLSTM,
        'TokenTypeTreeLSTM': TokenTypeTreeLSTM
    }
    _decoders = {
        'LinearDecoder': LinearDecoder,
        'LSTMDecoder': LSTMDecoder,
        'LSTMAttentionDecoder': LSTMAttentionDecoder
    }
    _attentions = {
        'LuongConcatAttention': LuongConcatAttention
    }

    def __init__(self, embedding_info: Dict, encoder_info: Dict, decoder_info: Dict,
                 hidden_states: Dict, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict):
        self.embedding_info = embedding_info
        self.encoder_info = encoder_info
        self.decoder_info = decoder_info

        self.hidden_states = hidden_states

        self.token_to_id = token_to_id
        self.type_to_id = type_to_id
        self.label_to_id = label_to_id

        self.embedding = self._get_module(self.embedding_info['name'], self._embeddings)
        self.encoder = self._get_module(self.encoder_info['name'], self._encoders)

        self.using_attention = 'attention' in self.decoder_info
        if self.using_attention:
            self.attention_info = self.decoder_info['attention']
            self.attention = self._get_module(self.attention_info['name'], self._attentions)
        self.decoder = self._get_module(self.decoder_info['name'], self._decoders)

    @staticmethod
    def _get_module(module_name: str, modules_dict: Dict) -> nn.Module:
        if module_name not in modules_dict:
            raise ModuleNotFoundError(f"Unknown module {module_name}, try one of {', '.join(modules_dict.keys())}")
        return modules_dict[module_name]

    def construct_model(self, device: torch.device) -> Tree2Seq:
        if self.using_attention:
            attention_part = self.attention(
                h_enc=self.hidden_states['encoder'], h_dec=self.hidden_states['decoder'],
                **self.attention_info['params']
            )
            decoder_part = self.decoder(
                h_enc=self.hidden_states['encoder'], h_dec=self.hidden_states['decoder'], attention=attention_part,
                label_to_id=self.label_to_id, **self.decoder_info['params'],
            )
        else:
            decoder_part = self.decoder(
                h_enc=self.hidden_states['encoder'], h_dec=self.hidden_states['decoder'],
                label_to_id=self.label_to_id, **self.decoder_info['params']
            )
        return Tree2Seq(
            self.embedding(
                h_emb=self.hidden_states['embedding'], token_to_id=self.token_to_id,
                type_to_id=self.type_to_id, **self.embedding_info['params']
            ),
            self.encoder(
                h_emb=self.hidden_states['embedding'], h_enc=self.hidden_states['encoder'],
                **self.encoder_info['params']
            ),
            decoder_part,
            self.using_attention
        ).to(device)

    def save_configuration(self) -> Dict:
        return {
            'embedding_info': self.embedding_info,
            'encoder_info': self.encoder_info,
            'decoder_info': self.decoder_info,
            'hidden_states': self.hidden_states,
            'token_to_id': self.token_to_id,
            'type_to_id': self.type_to_id,
            'label_to_id': self.label_to_id
        }


def load_model(path_to_model: str, device: torch.device) -> Tree2Seq:
    checkpoint = torch.load(path_to_model, map_location=device)
    configuration = checkpoint['configuration']
    model_factory = ModelFactory(**configuration)
    model: Tree2Seq = model_factory.construct_model(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
