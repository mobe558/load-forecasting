"""Classic model: LSTM encoder-decoder (Sutskever et al., 2014; Hochreiter & Schmidhuber, 1997).

Encoder LSTM reads the 168 past hours (load + calendar). Its final (h, c) state initialises a decoder
LSTM that reads the 24 KNOWN future calendar vectors (+ a learned horizon-step embedding) and emits one
load value per future hour. The decoder never consumes its own predictions (non-autoregressive), so
errors do not compound over the horizon.
"""
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_past: int, n_fut: int, horizon: int = 24, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1, step_emb: int = 4):
        super().__init__()
        self.step = nn.Embedding(horizon, step_emb)
        rd = dropout if layers > 1 else 0.0                     # recurrent dropout needs >1 layer
        self.encoder = nn.LSTM(n_past, hidden, layers, batch_first=True, dropout=rd)
        self.decoder = nn.LSTM(n_fut + step_emb, hidden, layers, batch_first=True, dropout=rd)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_past, x_fut):
        B, H, _ = x_fut.shape
        _, state = self.encoder(x_past)
        steps = self.step(torch.arange(H, device=x_fut.device)).expand(B, H, -1)
        out, _ = self.decoder(torch.cat([x_fut, steps], dim=-1), state)
        return self.head(out).squeeze(-1)                     # [B, H]
