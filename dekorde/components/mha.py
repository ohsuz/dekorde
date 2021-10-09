import torch
import torch.nn as nn


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        # any layers to optimise? - four linear layers in total.
        # TODO - define the shape of the weights.
        self.head_dim = embed_size // heads
        self.W_q = nn.Linear(self.head_dim, self.head_dim)
        self.W_k = nn.Linear(self.head_dim, self.head_dim)
        self.W_v = nn.Linear(self.head_dim, self.head_dim)
        self.W_o = nn.Linear(self.embed_size, self.embed_size)  # for aggregating the multi-head outputs.

    def forward(self, EP_q: torch.Tensor, EP_k: torch.Tensor, EP_v: torch.Tensor, M: torch.Tensor = None) -> torch.Tensor:
        """
        :param EP_q: (N, L, E)
        :param EP_k: (N, L, E)
        :param EP_v: (N, L, E)
        :param M: ???
        :return: out (N, L, E)
        """
        # Reshape all matrices
        # (N, L, E) -> (N, L, heads, head_dim)
        N = EP_q.shape[0]
        q_len, k_len, v_len = EP_q.shape[1], EP_k.shape[1], EP_v.shape[1]
        EP_q = torch.reshape(N, q_len, self.heads, self.head_dim)
        EP_k = torch.reshape(N, k_len, self.heads, self.head_dim)
        EP_v = torch.reshape(N, v_len, self.heads, self.head_dim)
        
        # Pass h linear layers
        # (N, ?_len, heads, head_dim) -> (N, ?_len, heads, head_dim)
        Q = self.W_q(EP_q)
        K = self.W_k(EP_k)
        V = self.W_v(EP_v)
        
        # Scaled Dot-Product Attention
        # (N, ?_len, heads, head_dim) -> (N, q_len, embed_size)
        attention = self.scaled_dot_product_attention(Q, K, V, M)
        
        # Pass a linear layer
        # (N, q_len, embed_size) -> (N, q_len, embed_size)
        out = self.W_o(attention)
        
        return out

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     M: torch.Tensor = None) -> torch.Tensor:
        """
        :param Q: (N, L, heads, head_dim)
        :param K: (N, L, heads, head_dim)
        :param V: (N, L, heads, head_dim)
        :param M: ??
        :return: out (N, L, E)
        """
        N, L = Q.shape[0], Q.shape[1]
        
        # Matrix Multiplication : Q & K
        attention = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        
        # Scaled
        attention = attention / (self.embed_size ** (1/2))
        
        # Mask (Optional)
        if M:
            attention = attention.masked_fill(M == 0, float("-1e20"))
        
        # Softmax
        attention = torch.softmax(attention, dim=3)
        
        # Matrix Multiplication : attention & V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            N, L, self.heads*self.head_dim
        )
        
        return out