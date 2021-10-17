
import torch
import torch.nn as nn


class FeedForward(torch.nn.Module):
    """
    position-wise feedforward network.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, hidden_size)
        )

    def forward(self, H_all: torch.Tensor):
        """
        :param H_all: (N, L, H)
        :return: H_all: (N, L, H)
        """
        return self.layers(H_all)
