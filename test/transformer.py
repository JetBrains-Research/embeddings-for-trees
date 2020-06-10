import unittest

import torch

from encoder.transformer_encoder import TransformerEncoder
from test_utils import gen_node_with_children
from utils.common import fix_seed


class TransformerEncoderTest(unittest.TestCase):

    def test_simple_forward_pass(self):
        fix_seed()
        device = torch.device('cpu')

        number_of_children = [3, 5, 128, 256]
        hidden_state = [5, 10, 128, 256]
        n_heads = [1, 2, 16, 32]

        for n_children, h_emb, n_head in zip(number_of_children, hidden_state, n_heads):
            with self.subTest(f"test transformer encoder with params: {n_children}, {h_emb}, {n_head}"):
                g = gen_node_with_children(n_children)
                x = torch.rand(n_children + 1, h_emb, device=device)
                g.ndata['x'] = x

                my_model = TransformerEncoder(h_emb, h_emb, n_head)
                transformer = torch.nn.TransformerEncoderLayer(h_emb, n_head)
                my_model.eval()
                transformer.eval()

                state_dict = {}
                prefix = 'transformer.layers.0'
                for layer_name in transformer.state_dict().keys():
                    state_dict[layer_name] = my_model.state_dict()[f'{prefix}.{layer_name}']
                transformer.load_state_dict(state_dict)

                h_model, c_model = my_model(g, device)

                x_trans = my_model.transformer_layer(x.unsqueeze(1)).squeeze(1)

                x_attn = torch.zeros_like(x)
                x_attn[0] = x_trans[0]
                x_attn[1:] = x[1:]

                h = torch.tanh(my_model.linear_h(x_attn))
                c = torch.tanh(my_model.linear_c(x_attn))

                self.assertTrue(h.allclose(h_model))
                self.assertTrue(c.allclose(c_model))


if __name__ == '__main__':
    unittest.main()
