"""Hand-written attention building blocks (Core Req. 5: "write at least one key component yourselves").

Nothing here calls torch.nn.MultiheadAttention or torch.nn.Transformer*: scaled dot-product
attention, the multi-head split/merge, the sinusoidal positional encoding, the patch embedding and
the pre-norm Transformer block are implemented from the equations in Vaswani et al. (2017) and
Nie et al. (2023, PatchTST).
"""
import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V      (Vaswani et al., 2017, Eq. 1)

    q: [B, h, Tq, d_k]   k, v: [B, h, Tk, d_k]   mask: broadcastable to [B, h, Tq, Tk], True = blocked
    returns (output [B, h, Tq, d_k], weights [B, h, Tq, Tk])
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W_O,  head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_weights = None          # kept for interpretation / error analysis

    def _split(self, x):                  # [B, T, d_model] -> [B, h, T, d_k]
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q, k, v = self._split(self.w_q(query)), self._split(self.w_k(key)), self._split(self.w_v(value))
        out, w = scaled_dot_product_attention(q, k, v, mask, self.drop)
        self.last_weights = w.detach()
        B, _, Tq, _ = out.shape
        out = out.transpose(1, 2).reshape(B, Tq, self.h * self.d_k)   # merge heads
        return self.w_o(out)


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x, offset: int = 0):          # x: [B, T, d]
        return x + self.pe[offset: offset + x.size(1)]


class PatchEmbedding(nn.Module):
    """Cut the [B, L, F] input into non-overlapping patches of P steps and project each to d_model.
    168 hourly steps / P = 24  ->  7 daily tokens (PatchTST idea, Nie et al., 2023)."""

    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.P = patch_len
        self.proj = nn.Linear(n_features * patch_len, d_model)

    def forward(self, x):
        B, L, F = x.shape
        assert L % self.P == 0, "lookback must be a multiple of patch_len"
        return self.proj(x.reshape(B, L // self.P, self.P * F))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm block: x + SelfAttn(LN(x)), then x + FF(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderBlock(nn.Module):
    """Self-attention over the 24 future-hour queries, cross-attention to the encoded past, FF.
    With use_cross_attention=False (ablation) the cross-attention is replaced by adding the
    mean-pooled encoder memory, i.e. the decoder no longer chooses WHICH past steps to look at."""

    def __init__(self, d_model, n_heads, d_ff, dropout, use_cross_attention=True):
        super().__init__()
        self.use_cross = use_cross_attention
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout) if use_cross_attention else None
        self.pool_proj = None if use_cross_attention else nn.Linear(d_model, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h))
        h = self.ln2(x)
        if self.use_cross:
            x = x + self.drop(self.cross_attn(h, memory, memory))
        else:
            x = x + self.drop(self.pool_proj(memory.mean(dim=1, keepdim=True)).expand_as(x))
        return x + self.drop(self.ff(self.ln3(x)))
