import unittest

import torch

from model.encoder.transformer_encoder import TransformerEncoder
from tests.generator import generate_node_with_children
from utils.common import fix_seed


class TransformerEncoderTest(unittest.TestCase):

    def test_transformer_encoder_forward_pass(self):
        fix_seed()
        device = torch.device('cpu')

        number_of_children = [3, 5, 128, 256]
        hidden_state = [5, 10, 128, 256]
        n_heads = [1, 2, 16, 32]

        for n_children, h_emb, n_head in zip(number_of_children, hidden_state, n_heads):
            with self.subTest(f"test transformer encoder with params: {n_children}, {h_emb}, {n_head}"):
                g = generate_node_with_children(n_children)
                x = torch.rand(n_children + 1, h_emb, device=device)
                g.ndata['x'] = x

                my_model = TransformerEncoder(h_emb, h_emb, n_head)
                transformer_layer = torch.nn.TransformerEncoderLayer(h_emb, n_head)
                transformer = torch.nn.TransformerEncoder(transformer_layer, 1)
                my_model.eval()
                transformer.eval()

                state_dict = {}
                for layer_name in transformer.state_dict().keys():
                    state_dict[layer_name] = my_model.state_dict()[f'transformer.{layer_name}']
                transformer.load_state_dict(state_dict)

                my_result = my_model(g)

                transformer_result = torch.empty_like(my_result)
                transformer_result[1:] = my_model.norm(x[1:])
                h_root = transformer(transformer_result[1:].unsqueeze(1)).transpose(0, 1).sum(1)
                transformer_result[0] = my_model.norm(x[0] + h_root)

                self.assertTrue(transformer_result.allclose(my_result))


if __name__ == '__main__':
    unittest.main()
