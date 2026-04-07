"""
Auxiliary prediction heads attached to intermediate Whisper encoder layers.

CTCPhonemeHead  - sequence-level CTC loss over phoneme tokens
PhonFeatureHead - segment-level BCE loss over 16-dim phonological feature vectors
"""

import torch
import torch.nn as nn
from typing import Optional


class CTCPhonemeHead(nn.Module):
    """
    Linear projection + CTC loss attached to an intermediate encoder layer.

    Input  : hidden states (B, T, hidden_dim)
    Output : log-probs    (T, B, num_phones + 1)   [CTC convention: time-first]

    The blank token is appended as the last class index (num_phones).
    """

    def __init__(self, hidden_dim: int = 512, num_phones: int = 39, dropout: float = 0.1):
        super().__init__()
        self.num_phones = num_phones
        self.blank_id   = num_phones          # blank = last index

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phones + 1),
        )
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)

    def forward(
        self,
        hidden: torch.Tensor,                 # (B, T, D)
        targets: Optional[torch.Tensor] = None,   # (sum_lengths,) packed phone ids
        input_lengths: Optional[torch.Tensor] = None,   # (B,) frames per item
        target_lengths: Optional[torch.Tensor] = None,  # (B,) phones per item
    ):
        logits   = self.proj(hidden)           # (B, T, num_phones+1)
        log_prob = logits.log_softmax(-1)      # (B, T, C)
        log_prob = log_prob.permute(1, 0, 2)   # (T, B, C) — CTC expects time-first

        if targets is None:
            return log_prob, None

        loss = self.ctc_loss(
            log_prob,
            targets,
            input_lengths,
            target_lengths,
        )
        return log_prob, loss


class PhonFeatureHead(nn.Module):
    """
    Two-layer MLP → 16-dim sigmoid output.  Trained with binary cross-entropy
    on phonological feature vectors (from PHON_FEATURE_MATRIX).

    Input  : segment embeddings (N, hidden_dim)  — already mean-pooled per segment
    Output : feature predictions (N, num_features)
    """

    def __init__(self, hidden_dim: int = 512, num_features: int = 16, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_features),
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        segment_embeddings: torch.Tensor,     # (N, D)
        targets: Optional[torch.Tensor] = None,   # (N, num_features)
    ):
        logits = self.head(segment_embeddings)    # (N, num_features)

        if targets is None:
            return logits.sigmoid(), None

        loss = self.bce_loss(logits, targets.float())
        return logits.sigmoid(), loss
