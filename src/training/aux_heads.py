"""
src/training/aux_heads.py
Auxiliary prediction heads attached to intermediate Whisper encoder layers.

CTCPhonemeHead  — sequence-level CTC loss over 39 ARPAbet phoneme tokens
PhonFeatureHead — segment-level BCE loss over panphon articulatory feature vectors
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.phonology import NUM_PHONES, NUM_PHON_FEATURES


class CTCPhonemeHead(nn.Module):
    """
    Linear projection + CTC loss over the 39 ARPAbet phoneme classes.

    Input  : hidden states  (B, T, hidden_dim)
    Output : log-probs      (T, B, NUM_PHONES + 1)  [time-first, CTC convention]

    Blank token is appended as the last class index (NUM_PHONES).
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.blank_id = NUM_PHONES
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_PHONES + 1),
        )
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id, reduction="mean", zero_infinity=True
        )

    def forward(
        self,
        hidden:         torch.Tensor,
        targets:        Optional[torch.Tensor] = None,
        input_lengths:  Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        log_prob = self.proj(hidden).log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
        if targets is None:
            return log_prob, None
        loss = self.ctc_loss(log_prob, targets, input_lengths, target_lengths)
        return log_prob, loss


class PhonFeatureHead(nn.Module):
    """
    Two-layer MLP → NUM_PHON_FEATURES sigmoid outputs.
    Trained with BCE against panphon articulatory feature vectors.

    Input  : segment embeddings  (N, hidden_dim)  — mean-pooled per phone segment
    Output : feature predictions (N, NUM_PHON_FEATURES)
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_PHON_FEATURES),
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, segment_embeddings, targets=None):
        logits = self.head(segment_embeddings)
        if targets is None:
            return logits.sigmoid(), None
        
        targets_binary = targets.clone()
        mask = targets != 0   # mask out neutral features
        targets_binary = ((targets_binary + 1) / 2) # map to {0, 1}
        
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], targets_binary[mask].float()
        )
        return logits.sigmoid(), loss
