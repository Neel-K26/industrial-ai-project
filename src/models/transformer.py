"""Shared Transformer RUL regressor architecture (E2's design).

Single source of truth for the architecture used across E2, E6, E7, and E8 —
defined once here and imported everywhere it's needed, rather than copied into
each consumer. (A previous notebook, E6, copy-pasted this architecture and
silently omitted the positional encoding, producing a materially worse model
under the same name; centralizing the definition removes that entire class of
mistake for every consumer after this one.)
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerRULRegressor(nn.Module):
    """E2's exact architecture: Transformer encoder regressor mapping a sensor-window to a scalar RUL."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 30,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :])


class QuantileTransformerRULRegressor(nn.Module):
    """Same backbone as TransformerRULRegressor, with an n-quantile-wide head instead of one scalar."""

    def __init__(
        self,
        input_size: int,
        n_quantiles: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 30,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :])


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """Mean pinball (quantile) loss across all quantile heads.

    Args:
        preds: (batch, n_quantiles) model output.
        target: (batch, 1) ground-truth RUL.
        quantiles: quantile level for each output column, e.g. [0.1, 0.5, 0.9].
    """
    errors = target - preds
    losses = [torch.max((q - 1) * errors[:, i], q * errors[:, i]) for i, q in enumerate(quantiles)]
    return torch.stack(losses, dim=1).mean()
