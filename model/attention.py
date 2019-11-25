import torch
import torch.nn as nn


class _IAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_hidden_state: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """ Compute attention weights based on previous decoder state and encoder output

        :param prev_hidden_state: [batch size, hidden size]
        :param encoder_output: [batch size, max number of nodes, hidden size]
        :return: attention weights [batch size, max number of nodes]
        """
        raise NotImplementedError


class LuongConcatAttention(_IAttention):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(1, self.hidden_size), requires_grad=True)

    def forward(self, prev_hidden_state: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes_number = encoder_output.shape[:2]

        # [batch size, max number of nodes, hidden size]
        repeated_hidden_state = prev_hidden_state.unsqueeze(1).expand(-1, max_nodes_number, -1)

        # [batch size, max number of nodes, hidden size]
        energy = torch.tanh(self.linear(
            torch.cat((repeated_hidden_state, encoder_output), dim=2)
        ))

        # [batch size, hidden size, max number of nodes]
        energy = energy.permute(0, 2, 1)

        # [batch size, 1, hidden size]
        v = self.v.expand(batch_size, -1).unsqueeze(1)

        # [batch size, max number of nodes]
        attention = torch.bmm(v, energy).squeeze(1)

        return nn.functional.softmax(attention, dim=1)
