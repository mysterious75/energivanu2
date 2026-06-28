# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu PEB System — Model Architecture (Production-Grade)
==============================================================
TCN + Multi-Head Attention with Adaptive Domain Normalization.

Architecture:
  Input (B, seq_len, n_features)
    → Per-feature adaptive normalization
    → Input projection to attention_dim
    → TCN backbone (dilated causal convolutions)
    → Multi-Head self-attention
    → Dual-head output:
        1. Power regression: (B, pred_horizon)
        2. BESS signal classification: (B, 3) hold/discharge/charge
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with causal padding + residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.drop2(out)

        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""

    def __init__(self, in_ch: int, channels: List[int],
                 kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        for i, (out_ch, k) in enumerate(zip(channels, kernel_sizes)):
            layers.append(TemporalBlock(in_ch, out_ch, k, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head self-attention with residual + LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, weights = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out)), weights


class EnergivanuPEB(nn.Module):
    """
    Predictive Energy Buffer (PEB) Model.

    Features:
      - Adaptive Domain Normalization: per-feature-type normalization
      - TCN backbone with dilated causal convolutions
      - Multi-Head self-attention on temporal features
      - Dual-head: power regression + BESS signal classification
    """

    def __init__(
        self,
        n_features: int = 15,
        seq_len: int = 30,
        pred_horizon: int = 10,
        tcn_channels: List[int] = None,
        tcn_kernels: List[int] = None,
        attention_heads: int = 8,
        attention_dim: int = 128,
        hidden_dims: List[int] = None,
        n_signal_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64, 128]
        if tcn_kernels is None:
            tcn_kernels = [5, 3, 3]
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.attention_dim = attention_dim

        # LayerNorm for per-feature normalization (works with any batch size)
        # Dynamically compute feature group sizes from n_features
        n_power = min(7, n_features)                           # indices 0-6
        n_telemetry = min(7, max(0, n_features - 7))           # indices 7-13
        n_temporal = max(0, n_features - 14)                    # indices 14+
        # Ensure each group has at least 1 feature for LayerNorm
        self._n_power = n_power
        self._n_telemetry = n_telemetry
        self._n_temporal = n_temporal
        if n_power > 0:
            self.power_norm = nn.LayerNorm(n_power)
        else:
            self.power_norm = nn.Identity()
        if n_telemetry > 0:
            self.telemetry_norm = nn.LayerNorm(n_telemetry)
        else:
            self.telemetry_norm = nn.Identity()
        if n_temporal > 0:
            self.temporal_norm = nn.LayerNorm(n_temporal)
        else:
            self.temporal_norm = nn.Identity()

        self.input_proj = nn.Linear(n_features, attention_dim)
        self.tcn = TCNBackbone(attention_dim, tcn_channels, tcn_kernels, dropout)
        self.attention = MultiHeadAttention(tcn_channels[-1], attention_heads, dropout)
        self.last_step_weight = nn.Linear(tcn_channels[-1], 1)

        self.power_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[1], pred_horizon),
        )

        self.signal_head = nn.Sequential(
            nn.Linear(tcn_channels[-1] + pred_horizon, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[1], n_signal_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def _adaptive_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features by domain using LayerNorm."""
        B, T, F = x.shape
        out = torch.zeros_like(x)
        if self._n_power > 0:
            out[:, :, :self._n_power] = self.power_norm(x[:, :, :self._n_power])
        if self._n_telemetry > 0:
            start = self._n_power
            end = start + self._n_telemetry
            if end <= F:
                out[:, :, start:end] = self.telemetry_norm(x[:, :, start:end])
        if self._n_temporal > 0:
            start = self._n_power + self._n_telemetry
            if start < F:
                out[:, :, start:] = self.temporal_norm(x[:, :, start:])
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self._adaptive_normalize(x)
        x_proj = self.input_proj(x_norm)
        x_tcn = self.tcn(x_proj.transpose(1, 2))
        x_attn_in = x_tcn.transpose(1, 2)
        x_attn, attn_weights = self.attention(x_attn_in)

        last_step = x_attn[:, -1, :]
        mean_pool = x_attn.mean(dim=1)
        alpha = torch.sigmoid(self.last_step_weight(last_step))
        x_agg = alpha * last_step + (1 - alpha) * mean_pool

        power_pred = self.power_head(x_agg)
        signal_input = torch.cat([x_agg, power_pred], dim=1)
        signal_logits = self.signal_head(signal_input)

        return power_pred, signal_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(checkpoint_path: str) -> EnergivanuPEB:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = EnergivanuPEB(
        n_features=cfg.get("n_features", 15),
        seq_len=cfg.get("seq_len", 30),
        pred_horizon=cfg.get("pred_horizon", 10),
        tcn_channels=cfg.get("tcn_channels", [32, 64, 128]),
        tcn_kernels=cfg.get("tcn_kernels", [5, 3, 3]),
        attention_heads=cfg.get("attention_heads", 8),
        attention_dim=cfg.get("attention_dim", 128),
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        n_signal_classes=cfg.get("n_signal_classes", 3),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
