from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class TemporalInstanceNorm(nn.Module):
    """Instance normalization across the temporal axis with learnable affine params."""

    def __init__(self, feature_dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) with 1 for valid frames, optional
        """
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
        else:
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * mask).sum(dim=1, keepdim=True) / lengths
            var = ((x - mean) * mask).pow(2).sum(dim=1, keepdim=True) / lengths

        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.gamma + self.beta


class DilatedTCNEncoder(nn.Module):
    """Temporal convolutional encoder with configurable dilations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilations: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]
        if len(dilations) != num_layers:
            raise ValueError("Length of dilations must match num_layers")

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = dilations[i]
            padding = ((kernel_size - 1) * dilation) // 2
            conv = nn.Conv1d(
                in_channels,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
            layers.append(
                nn.Sequential(
                    conv,
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, hidden_dim)
        """
        x = x.transpose(1, 2)  # (B, D, T)
        out = self.net(x)
        return out.transpose(1, 2)


class XLSTMBlock(nn.Module):
    """
    A lightweight implementation of an xLSTM-inspired block.
    Supports 'mlstm' (multiplicative) or 'slstm' (scaled) preconditioning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        variant: str = "mlstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.precondition = None
        if self.variant == "mlstm":
            self.precondition = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
            )
            lstm_input_dim = hidden_dim
        elif self.variant == "slstm":
            self.scale = nn.Parameter(torch.ones(1, 1, input_dim))
            self.shift = nn.Parameter(torch.zeros(1, 1, input_dim))
            lstm_input_dim = input_dim
        else:
            raise ValueError(f"Unsupported xLSTM variant: {variant}")

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, hidden=None):
        residual = x
        if self.variant == "mlstm":
            gate = self.precondition(x)
            lstm_in = x * gate
        else:  # slstm
            lstm_in = x * self.scale + self.shift

        out, hidden = self.lstm(lstm_in, hidden)
        out = self.dropout(out)
        out = self.proj(out)
        out = self.norm(out + residual)
        return out, hidden


class XLSTMStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 2,
        variant: str = "mlstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                XLSTMBlock(input_dim=input_dim, hidden_dim=hidden_dim, variant=variant, dropout=dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = None
        for layer in self.layers:
            x, h = layer(x, h)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        return x


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(num_heads, input_dim))
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T), optional
        Returns:
            (B, D)
        """
        B, T, D = x.shape
        keys = self.key_proj(x)  # (B, T, D)
        values = self.value_proj(x)

        attn_outputs = []
        for head in range(self.num_heads):
            q = self.query[head].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            logits = (keys * q).sum(-1) / (D ** 0.5)  # (B, T)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, float("-inf"))
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
            pooled = (values * weights).sum(dim=1)
            attn_outputs.append(pooled)

        stacked = torch.stack(attn_outputs, dim=1)  # (B, num_heads, D)
        combined = stacked.mean(dim=1)
        return self.out_proj(combined)
