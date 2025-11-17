from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .modules import (
    DilatedTCNEncoder,
    TemporalAttentionPooling,
    TemporalInstanceNorm,
    XLSTMStack,
)


@dataclass
class SignModelConfig:
    num_classes: int
    pose_dim: int = 17 * 3
    hand_dim: int = 42 * 3
    face_dim: int = 478 * 3
    stream_hidden: int = 256
    tcn_layers: int = 3
    tcn_kernel: int = 3
    tcn_dilations: tuple = (1, 2, 4)
    fusion_dim: int = 512
    xlstm_hidden: int = 512
    xlstm_layers: int = 2
    xlstm_variant: str = "mlstm"
    attn_heads: int = 4
    dropout: float = 0.1


class SignLanguageXLSTM(nn.Module):
    """
    Multi-stream keypoint encoder with per-stream TCNs followed by an xLSTM stack.
    """

    def __init__(self, config: SignModelConfig):
        super().__init__()
        self.config = config

        # Normalization layers
        self.pose_norm = TemporalInstanceNorm(config.pose_dim)
        self.hand_norm = TemporalInstanceNorm(config.hand_dim)
        self.face_norm = TemporalInstanceNorm(config.face_dim)

        # TCN encoders
        self.pose_encoder = DilatedTCNEncoder(
            input_dim=config.pose_dim,
            hidden_dim=config.stream_hidden,
            num_layers=config.tcn_layers,
            kernel_size=config.tcn_kernel,
            dilations=config.tcn_dilations,
            dropout=config.dropout,
        )
        self.hand_encoder = DilatedTCNEncoder(
            input_dim=config.hand_dim,
            hidden_dim=config.stream_hidden,
            num_layers=config.tcn_layers,
            kernel_size=config.tcn_kernel,
            dilations=config.tcn_dilations,
            dropout=config.dropout,
        )
        self.face_encoder = DilatedTCNEncoder(
            input_dim=config.face_dim,
            hidden_dim=config.stream_hidden,
            num_layers=config.tcn_layers,
            kernel_size=config.tcn_kernel,
            dilations=config.tcn_dilations,
            dropout=config.dropout,
        )

        fusion_input = config.stream_hidden * 3
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.xlstm = XLSTMStack(
            input_dim=config.fusion_dim,
            hidden_dim=config.xlstm_hidden,
            depth=config.xlstm_layers,
            variant=config.xlstm_variant,
            dropout=config.dropout,
        )

        self.temporal_pool = TemporalAttentionPooling(config.fusion_dim, num_heads=config.attn_heads)

        head_dim = max(config.fusion_dim // 2, 128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_dim, config.num_classes),
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if mask is None and "mask" in inputs:
            mask = inputs["mask"]

        pose = self.pose_norm(inputs["pose"], mask)
        hand = self.hand_norm(inputs["hand"], mask)
        face = self.face_norm(inputs["face"], mask)

        pose_feat = self.pose_encoder(pose)
        hand_feat = self.hand_encoder(hand)
        face_feat = self.face_encoder(face)

        fused = torch.cat([pose_feat, hand_feat, face_feat], dim=-1)
        fused = self.fusion_proj(fused)

        contextual = self.xlstm(fused, mask)
        pooled = self.temporal_pool(contextual, mask)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "features": pooled,
        }
