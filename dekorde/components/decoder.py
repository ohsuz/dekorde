import torch
import torch.nn as nn
from typing import Tuple
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        max_length: int, 
        heads: int,
        dropout: float,
        lookahead_mask: torch.Tensor
    ):
        super().__init__()
        # masked, multi-head self-attention layer.
        self.masked_mha_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, lookahead_mask)
        self.norm_1 = nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer.
        self.mha_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_2 = nn.LayerNorm(hidden_size)
        # position-wise feedfowrard network.
        self.ffn = FeedForward(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_pair: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param H_pair: (H_x = (N, L, H), H_y = (N, L, H))
        :return: H_x (as-is), H_y (updated)
        """
        H_x, H_y = H_pair
        masked_attention = self.dropout(self.masked_mha_layer.forward(H_q=H_y, H_k=H_y, H_v=H_y))
        out_1 = self.norm_1(masked_attention + H_y)  # skip connection
        attention = self.dropout(self.mha_layer.forward(H_q=out_1, H_k=H_x, H_v=H_x))
        out_2 = self.norm_2(attention + out_1)  # skip connection
        forward = self.dropout(self.ffn(out_2))
        out_3 = self.norm_3(forward + out_2)  # skip connection
        return H_x, out_3


class Decoder(torch.nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        max_length: int, 
        heads: int, 
        num_layers: int, 
        dropout: float,
        lookahead_mask: torch.Tensor
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads, dropout, lookahead_mask) for _ in range(num_layers)]
        )

    def forward(self, H_x: torch.Tensor, Y_embed: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H)
        :param Y_embed: (N, L, H)
        :return: H_y: (N, L, H)
        """
        _, H_y = self.layers((H_x, Y_embed))
        return H_y
