from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CausalLocalMixer(nn.Module):
    """Kausale, kanalweise Faltung mit begrenztem Streaming-Puffer."""

    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, bias=True
        )
        self.channel_mix = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)

    @property
    def state_length(self) -> int:
        return self.kernel_size - 1

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.state_length, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x muss die Form [batch, sequence, d_model] besitzen")
        batch = x.shape[0]
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)
        joined = torch.cat((state, x), dim=1)
        local = self.depthwise(joined.transpose(1, 2)).transpose(1, 2)
        mixed = self.channel_mix(local)
        output = x + torch.sigmoid(self.gate(x)) * F.silu(mixed)
        next_state = joined[:, -self.state_length :].contiguous() if self.state_length else joined[:, :0]
        return output, next_state

