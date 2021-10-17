import torch
import torch.nn as nn
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int, 
        max_length: int, 
        heads: int,
        dropout: float
    ):
        super().__init__()
        # any layers to optimise?
        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_x: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H), or (N, L, E) if this layer is the first layer.
        :return: H_x: (N, L, H)
        """
        attention = self.dropout(self.multi_head_self_attention_layer.forward(H_q=H_x, H_k=H_x, H_v=H_x))
        out_1 = self.norm_1(attention + H_x)
        forward = self.dropout(self.ffn(out_1))
        out_2 = self.norm_2(forward + out_1)
        return out_2


class Encoder(torch.nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        max_length: int, 
        heads: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(hidden_size, max_length, heads, dropout) for _ in range(num_layers)]
        )
        
    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: H_x: (N, L, H)
        """
        return self.encoder_layers(X_embed)
