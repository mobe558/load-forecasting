"""Modern model: patch-based Transformer encoder-decoder, built only from our hand-written blocks
in attention.py (Vaswani et al., 2017; patching from PatchTST, Nie et al., 2023).

Encoder : 168 past hours -> 7 daily patches (24 h each) -> linear patch embedding + positional
          encoding -> N self-attention blocks.
Decoder : 24 query tokens, one per future hour = projection of its known calendar features +
          learned horizon embedding + positional encoding -> self-attention among the 24 hours ->
          cross-attention into the 7 encoded days (the model learns WHICH past days matter, e.g.
          the same weekday last week) -> linear head -> 24 loads.
"""
import torch
from torch import nn

from .attention import DecoderBlock, EncoderBlock, PatchEmbedding, SinusoidalPositionalEncoding


class TransformerForecaster(nn.Module):
    def __init__(self, n_past: int, n_fut: int, horizon: int = 24, d_model: int = 64, n_heads: int = 4,
                 enc_layers: int = 2, dec_layers: int = 2, d_ff: int = 128, dropout: float = 0.1,
                 patch_len: int = 24, use_cross_attention: bool = True):
        super().__init__()
        self.patch = PatchEmbedding(n_past, patch_len, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model)
        self.fut_proj = nn.Linear(n_fut, d_model) if n_fut > 0 else None
        self.step = nn.Embedding(horizon, d_model)
        self.enc = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(enc_layers)])
        self.dec = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout, use_cross_attention)
                                  for _ in range(dec_layers)])
        self.ln_enc, self.ln_dec = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_past, x_fut):
        B, H, _ = x_fut.shape
        mem = self.drop(self.pe(self.patch(x_past)))
        for blk in self.enc:
            mem = blk(mem)
        mem = self.ln_enc(mem)
        q = self.step(torch.arange(H, device=x_fut.device)).expand(B, H, -1)
        if self.fut_proj is not None:
            q = q + self.fut_proj(x_fut)
        q = self.drop(self.pe(q, offset=mem.size(1)))
        for blk in self.dec:
            q = blk(q, mem)
        return self.head(self.ln_dec(q)).squeeze(-1)          # [B, H]

    def cross_attention_maps(self):
        """Attention of each future hour over the 7 past days (last decoder layer, head-averaged)."""
        a = self.dec[-1].cross_attn
        return None if a is None or a.last_weights is None else a.last_weights.mean(1)
