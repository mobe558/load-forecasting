"""Advanced architecture: Temporal Fusion Transformer (TFT, Lim et al., 2021), compact re-implementation.

Written from the paper's equations (no pytorch-forecasting); the pieces are:
  * GLU / Gated Residual Network (GRN)                  - Eq. 2-5: gated skip connections that can
                                                          switch off unneeded non-linear processing
  * Variable Selection Networks (VSN) for past and
    known-future inputs                                 - Eq. 6-8: learns a softmax weight per input
                                                          variable at every time step (interpretable)
  * LSTM encoder (past) / decoder (known future)        - Sec. 4.5: local sequence processing
  * Interpretable multi-head attention                  - Eq. 13-16: heads share one value projection
                                                          and are averaged, so weights are readable
  * causal mask, position-wise GRN, gated skips, linear output for the 24 future hours.
Static covariates are omitted (single series, none exist).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """GRN(a) = LayerNorm(skip(a) + GLU(W1 ELU(W2 a + b2) + b1))"""

    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.w2, self.w1 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, a):
        h = self.w1(F.elu(self.w2(a)))
        return self.ln(self.skip(a) + self.glu(self.drop(h)))


class VariableSelection(nn.Module):
    """Each scalar input variable gets its own embedding and its own GRN (batched with einsum);
    a GRN over all embeddings produces softmax selection weights v_t (Eq. 6-8)."""

    def __init__(self, n_vars, d, dropout):
        super().__init__()
        self.V, self.d = n_vars, d
        self.emb_w = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.emb_b = nn.Parameter(torch.zeros(n_vars, d))
        self.select = GRN(n_vars * d, d, n_vars, dropout)
        # per-variable GRN, batched: weights [V, d, d]
        self.g_w2 = nn.Parameter(torch.randn(n_vars, d, d) / math.sqrt(d))
        self.g_b2 = nn.Parameter(torch.zeros(n_vars, d))
        self.g_w1 = nn.Parameter(torch.randn(n_vars, d, 2 * d) / math.sqrt(d))
        self.g_b1 = nn.Parameter(torch.zeros(n_vars, 2 * d))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                       # x: [B, T, V]
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b            # [B, T, V, d]
        w = torch.softmax(self.select(e.flatten(-2)), dim=-1)    # [B, T, V]
        h = F.elu(torch.einsum("btvd,vde->btve", e, self.g_w2) + self.g_b2)
        a, b = (torch.einsum("btvd,vde->btve", self.drop(h), self.g_w1) + self.g_b1).chunk(2, dim=-1)
        processed = self.ln(e + a * torch.sigmoid(b))           # [B, T, V, d]
        return (w.unsqueeze(-1) * processed).sum(-2), w


class InterpretableMultiHeadAttention(nn.Module):
    """Heads have their own Q/K projections but SHARE the value projection; head outputs are averaged
    (Eq. 14-16), so the averaged attention matrix is directly interpretable."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.wv = nn.Linear(d, self.dk)
        self.wo = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)
        self.last_weights = None

    def forward(self, q, k, v, mask):
        B, Tq, _ = q.shape
        Tk = k.size(1)
        Q = self.wq(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        K = self.wk(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        V = self.wv(v).unsqueeze(1)                              # shared across heads
        s = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        s = s.masked_fill(mask, float("-inf"))
        w = self.drop(torch.softmax(s, dim=-1))
        self.last_weights = w.detach().mean(1)
        return self.wo((w @ V).mean(1))                          # average heads -> [B, Tq, dk] -> d


class TFTForecaster(nn.Module):
    def __init__(self, n_past: int, n_fut: int, horizon: int = 24, d_model: int = 64, var_dim: int = 16,
                 n_heads: int = 4, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.H, self.use_attention = horizon, use_attention
        self.step = nn.Embedding(horizon, 1)                     # learned horizon index as a future input
        self.vsn_past = VariableSelection(n_past, var_dim, dropout)
        self.vsn_fut = VariableSelection(n_fut + 1, var_dim, dropout)
        self.up_past, self.up_fut = nn.Linear(var_dim, d_model), nn.Linear(var_dim, d_model)
        self.enc = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_lstm, self.ln_lstm = GLU(d_model, d_model), nn.LayerNorm(d_model)
        self.enrich = GRN(d_model, d_model, d_model, dropout)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.gate_attn, self.ln_attn = GLU(d_model, d_model), nn.LayerNorm(d_model)
        self.pos_ff = GRN(d_model, d_model, d_model, dropout)
        self.gate_out, self.ln_out = GLU(d_model, d_model), nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.var_weights = None

    def forward(self, x_past, x_fut):
        B, H, _ = x_fut.shape
        L = x_past.size(1)
        steps = self.step(torch.arange(H, device=x_fut.device)).expand(B, H, 1)
        ep, wp = self.vsn_past(x_past)
        ef, wf = self.vsn_fut(torch.cat([x_fut, steps], -1))
        self.var_weights = (wp.detach().mean((0, 1)), wf.detach().mean((0, 1)))
        ep, ef = self.up_past(ep), self.up_fut(ef)
        hp, state = self.enc(ep)
        hf, _ = self.dec(ef, state)
        phi = torch.cat([hp, hf], 1)                                       # [B, L+H, d]
        xi = torch.cat([ep, ef], 1)
        phi = self.ln_lstm(xi + self.gate_lstm(phi))                       # gated skip over LSTM
        theta = self.enrich(phi)
        if self.use_attention:
            q = theta[:, L:]                                               # only the 24 future positions
            T = L + H
            qpos = torch.arange(L, T, device=x_past.device)[:, None]
            kpos = torch.arange(T, device=x_past.device)[None, :]
            mask = (kpos > qpos)[None, None]                               # causal: no peeking ahead
            a = self.attn(q, theta, theta, mask)
            delta = self.ln_attn(theta[:, L:] + self.gate_attn(a))
        else:                                                              # ablation: no attention
            delta = theta[:, L:]
        psi = self.pos_ff(delta)
        out = self.ln_out(phi[:, L:] + self.gate_out(psi))
        return self.head(out).squeeze(-1)                                  # [B, H]
