from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """
    def __init__(
        self, 
        hidden_size: int, 
        max_length: int, 
        heads: int,
        lookahead_mask: Optional[torch.Tensor] = None
    ):
        """
        :param hidden_size:
        :param max_length:
        :param heads:
        :param lookahead_mask: (L, L)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.heads = heads
        self.lookahead_mask = lookahead_mask  # attention mask - (L, L)
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        self.head_dim = hidden_size // heads
        # any layers to optimise? - four linear layers in total.
        # TODO - define the shape of the weights.
        self.W_q = nn.Linear(self.head_dim, self.head_dim)
        self.W_k = nn.Linear(self.head_dim, self.head_dim)
        self.W_v = nn.Linear(self.head_dim, self.head_dim)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size)  # for aggregating the multi-head outputs.

    def forward(self, H_q: torch.Tensor, H_k: torch.Tensor, H_v: torch.Tensor) -> torch.Tensor:
        """
        :param H_q: (N, L, H)
        :param H_k: (N, L, H)
        :param H_v: (N, L, H)
        :return: H_all (N, L, H)
        """
        # Reshape all matrices
        # (N, L, E) -> (N, L, heads, head_dim)
        N = H_q.shape[0]
        q_len, k_len, v_len = H_q.shape[1], H_k.shape[1], H_v.shape[1]
        H_q = torch.reshape(N, q_len, self.heads, self.head_dim)
        H_k = torch.reshape(N, k_len, self.heads, self.head_dim)
        H_v = torch.reshape(N, v_len, self.heads, self.head_dim)
        
        # Pass h linear layers
        # (N, ?_len, heads, head_dim) -> (N, ?_len, heads, head_dim)
        Q = self.W_q(H_q)
        K = self.W_k(H_k)
        V = self.W_v(H_v)
        
        # Scaled Dot-Product Attention
        # (N, ?_len, heads, head_dim) -> (N, q_len, embed_size)
        attention = self.scaled_dot_product_attention(Q, K, V)
        
        # Pass a linear layer
        # (N, q_len, embed_size) -> (N, q_len, embed_size)
        H_all = self.W_o(attention)
        
        return H_all

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, L, heads, head_dim)
        :param K: (N, L, heads, head_dim)
        :param V: (N, L, heads, head_dim)
        :return: out (N, L, E)
        """
        N, L = Q.shape[0], Q.shape[1]
        
        # Matrix Multiplication : Q & K
        attention = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        
        # Scaled
        attention = attention / (self.embed_size ** (1/2))
        
        # Mask (Optional)
        if self.lookahead_mask:
            attention = attention.masked_fill(self.lookahead_mask.expand(N, self.heads, L, L) == 0, 
                                              float("-1e20"))
        
        # Softmax
        attention = torch.softmax(attention, dim=3)
        
        # Matrix Multiplication : attention & V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            N, L, self.heads*self.head_dim
        )
        
        return out
