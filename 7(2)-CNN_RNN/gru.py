import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.hidden_size = hidden_size
        # update gate
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # reset gate
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # candidate
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        # 실제 GRU 수식에서는 히든레이어에 곱해는 가중치와 인풋값에 넣는
        # 가중치를 따로 뒀지만, 계산량을 줄이고 backward를 한번만 하는게 
        # 편하므로 걍 합쳐버리자.
        combined = torch.cat((x, h), dim=-1)

        u = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        combined_hidden = torch.cat((x, r*h), dim=-1)
        hidden_state = torch.tanh(self.candidate_gate(combined_hidden))

        hidden_next_state = (1-u) * h + u * hidden_state
        return hidden_next_state


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, sequence_length, d_model = inputs.size()
        
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for i in range(sequence_length):
            x_t = inputs[:, i, :]
            h = self.cell(x_t, h)

        return h