"""
src/training/whisper_aux.py
WhisperWithAuxHeads — wraps WhisperForConditionalGeneration and injects
auxiliary supervision into the encoder at two configurable layer taps:

    CTC_LAYER  (default 4) → CTCPhonemeHead   (mid-encoder phoneme alignment)
    FEAT_LAYER (default 12) → PhonFeatureHead  (top encoder articulatory features)

The main ASR cross-entropy loss is unchanged.
Set lambda_ctc=0 or lambda_feat=0 to disable the respective head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration

from src.training.aux_heads import CTCPhonemeHead, PhonFeatureHead

CTC_LAYER  = 4    # after 5th transformer block — mid-encoder phoneme signal
FEAT_LAYER = 12   # after 12th transformer block — top encoder (whisper-small)


@dataclass
class AuxLossOutput:
    loss:      Optional[torch.Tensor] = None   # total combined loss
    loss_asr:  Optional[torch.Tensor] = None
    loss_ctc:  Optional[torch.Tensor] = None
    loss_feat: Optional[torch.Tensor] = None
    logits:    Optional[torch.Tensor] = None   # ASR decoder logits


class WhisperWithAuxHeads(nn.Module):
    """
    Parameters
    ----------
    model_name   : HuggingFace model id
    lambda_ctc   : weight for CTC phoneme loss  (0 → head disabled)
    lambda_feat  : weight for feature BCE loss  (0 → head disabled)
    ctc_layer    : encoder layer index to tap for CTC  (0-indexed)
    feat_layer   : encoder layer index to tap for features (0-indexed)
    """

    def __init__(
        self,
        model_name:  str   = "openai/whisper-small",
        lambda_ctc:  float = 0.3,
        lambda_feat: float = 0.0,
        ctc_layer:   int   = CTC_LAYER,
        feat_layer:  int   = FEAT_LAYER,
    ):
        super().__init__()
        self.lambda_ctc  = lambda_ctc
        self.lambda_feat = lambda_feat
        self.ctc_layer   = ctc_layer
        self.feat_layer  = feat_layer

        self.whisper   = WhisperForConditionalGeneration.from_pretrained(model_name)
        hidden_dim     = self.whisper.config.d_model          # 512 for whisper-small

        self.ctc_head  = CTCPhonemeHead(hidden_dim=hidden_dim)
        self.feat_head = PhonFeatureHead(hidden_dim=hidden_dim)

        self._ctc_hidden:  Optional[torch.Tensor] = None
        self._feat_hidden: Optional[torch.Tensor] = None
        self._register_hooks()

        print(
            f"WhisperWithAuxHeads | lambda_ctc={lambda_ctc} @ layer {ctc_layer}"
            f" | lambda_feat={lambda_feat} @ layer {feat_layer}"
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        layers = self.whisper.model.encoder.layers

        def _hook_detach(store: str):
            def _fn(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                setattr(self, store, hs.detach())   # CTC: detached
            return _fn

        def _hook_live(store: str):
            def _fn(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                setattr(self, store, hs)            # Feat: live grad
            return _fn

        if self.lambda_ctc > 0:
            layers[self.ctc_layer].register_forward_hook(_hook_detach("_ctc_hidden"))

        if self.lambda_feat > 0 and self.feat_layer != self.ctc_layer:
            layers[self.feat_layer].register_forward_hook(_hook_live("_feat_hidden"))

    # ------------------------------------------------------------------
    # Optimizer param groups
    # ------------------------------------------------------------------

    def param_groups(
        self, base_lr: float = 1e-4, head_lr: float = 1e-3
    ) -> list[dict]:
        backbone = [p for p in self.whisper.parameters() if p.requires_grad]
        heads    = list(self.ctc_head.parameters()) + list(self.feat_head.parameters())
        return [
            {"params": backbone, "lr": base_lr},
            {"params": heads,    "lr": head_lr},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_features:      torch.Tensor,
        labels:              Optional[torch.Tensor] = None,
        ctc_targets:         Optional[torch.Tensor] = None,
        ctc_input_lengths:   Optional[torch.Tensor] = None,
        ctc_target_lengths:  Optional[torch.Tensor] = None,
        feat_targets:        Optional[torch.Tensor] = None,
        feat_frame_spans:    Optional[list]         = None,
    ) -> AuxLossOutput:

        self._ctc_hidden  = None
        self._feat_hidden = None

        whisper_out = self.whisper(
            input_features = input_features,
            labels         = labels,
            return_dict    = True,
        )
        loss_asr = whisper_out.loss
        logits   = whisper_out.logits

        loss_ctc  = None
        loss_feat = None

        # CTC head
        if self.lambda_ctc > 0 and self._ctc_hidden is not None:
            _, loss_ctc = self.ctc_head(
                hidden         = self._ctc_hidden,
                targets        = ctc_targets,
                input_lengths  = ctc_input_lengths,
                target_lengths = ctc_target_lengths,
            )

        # Feature head — pool segments on-the-fly from feat hidden states
        if (self.lambda_feat > 0
                and self._feat_hidden is not None
                and feat_frame_spans is not None
                and feat_targets is not None):
            hs = self._feat_hidden                 # (B, T, D)
            pooled = []
            for b_idx, spans in enumerate(feat_frame_spans):
                for s, e in spans:
                    e = min(e, hs.shape[1])
                    if e > s:
                        pooled.append(hs[b_idx, s:e].mean(0))
            if pooled:
                seg_embs = torch.stack(pooled)     # (N, D)
                _, loss_feat = self.feat_head(seg_embs, feat_targets)

        total = loss_asr
        if loss_ctc  is not None: total = total + self.lambda_ctc  * loss_ctc
        if loss_feat is not None: total = total + self.lambda_feat * loss_feat

        return AuxLossOutput(
            loss=total, loss_asr=loss_asr,
            loss_ctc=loss_ctc, loss_feat=loss_feat, logits=logits,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """Return top encoder hidden state (B, T, D), no grad."""
        with torch.no_grad():
            return self.whisper.model.encoder(
                input_features, output_hidden_states=False, return_dict=True,
            ).last_hidden_state

    @torch.no_grad()
    def transcribe(self, input_features: torch.Tensor, **gen_kwargs) -> torch.Tensor:
        return self.whisper.generate(input_features, **gen_kwargs)
