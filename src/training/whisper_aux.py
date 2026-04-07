"""
WhisperWithAuxHeads - wraps WhisperForConditionalGeneration and injects
two auxiliary supervision signals into the encoder:

  Layer CTC_LAYER  → CTCPhonemeHead   (default: layer index 4)
  Layer FEAT_LAYER → PhonFeatureHead  (default: layer index 6, i.e. top encoder)

The main ASR cross-entropy loss is unchanged.

Forward returns an AuxLossOutput dataclass so individual losses are accessible
for logging and ablation (pass lambda=0 to disable either head).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

from src.training.aux_heads import CTCPhonemeHead, PhonFeatureHead
from src.eval.probing.probe_utils import parse_textgrid, PHON_FEATURE_MATRIX


# Which encoder layer outputs to tap (0-indexed transformer layer outputs,
# so layer 4 = after the 5th transformer block)
CTC_LAYER  = 4   # mid-encoder for phoneme CTC
FEAT_LAYER = 6   # top encoder for feature prediction (=last layer for whisper-small)


class AuxCollator:
    """
    Collate a list of dataset items into a batch suitable for WhisperWithAuxHeads.

    Args:
        processor:       WhisperProcessor (handles audio→log-mel + text→tokens)
        max_label_len:   Maximum decoder label sequence length (tokens).  Items
                         longer than this are truncated to avoid OOM.
        sampling_rate:   Audio sampling rate expected (default 16000).
    """

    def __init__(
        self,
        processor:     WhisperProcessor,
        max_label_len: int = 448,
        sampling_rate: int = 16000,
    ):
        self.processor     = processor
        self.max_label_len = max_label_len
        self.sampling_rate = sampling_rate
        self.pad_token_id  = processor.tokenizer.pad_token_id

    # ------------------------------------------------------------------

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = []
        for it in items:
            if "audio" in it:
                audios.append(it["audio"])
            else:
                arr, sr = sf.read(it["wav_path"], dtype="float32", always_2d=False)
                # resample if needed (L2-ARCTIC is already 16kHz so this is a no-op)
                audios.append(arr)
        texts     = [it["text"]    for it in items]
        tg_paths  = [it["textgrid"] for it in items]

        # ---- Log-mel features ----------------------------------------
        feat_out = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",   # pad/truncate to 3000 frames
        )
        input_features = feat_out.input_features   # (B, 80, 3000)

        # ---- Decoder labels ------------------------------------------
        label_out = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_label_len,
        )
        labels = label_out.input_ids.clone()
        labels[labels == self.pad_token_id] = -100   # mask padding for CE loss

        # ---- Encoder frame counts (for CTC) --------------------------
        # Whisper log-mel has fixed 3000-frame window; actual speech may be shorter.
        # We approximate input length as min(3000, audio_frames * 2 / hop).
        # For simplicity, use 3000 for all items (CTC zero_infinity handles padding).
        B = len(items)
        T_enc = input_features.shape[-1] // 2   # 3000 / 2 = 1500 frames after conv
        ctc_input_lengths = torch.full((B,), T_enc, dtype=torch.long)

        # ---- Phoneme sequences (for CTC) -----------------------------
        ctc_seqs    = []
        ctc_lengths = []
        # feat_targets_list   = []
        # feat_frame_spans    = []   # one list of (s, e) tuples per batch item
        # feat_item_indices   = []   # which batch item each segment belongs to

        for b_idx, tg_path in enumerate(tg_paths):
            segments = parse_textgrid(str(tg_path))
            phone_ids  = []
            spans      = []
            feat_vecs  = []

            for seg in segments:
                pid = seg.phone_id
                if pid < 0:
                    continue   # unknown phone — skip for both heads
                phone_ids.append(pid)
                spans.append((seg.start_frame, seg.end_frame))
                # feat_vecs.append(PHON_FEATURE_MATRIX[pid])   # (16,)

            if not phone_ids:
                # Fallback: single silence token so CTC doesn't crash
                phone_ids = [0]
                spans     = [(0, 1)]
                # feat_vecs = [PHON_FEATURE_MATRIX[0]]

            ctc_seqs.append(torch.tensor(phone_ids, dtype=torch.long))
            ctc_lengths.append(len(phone_ids))
            # feat_targets_list.extend(feat_vecs)
            # feat_frame_spans.append(spans)
            # feat_item_indices.extend([b_idx] * len(feat_vecs))

        # Pack CTC targets (1-D, concatenated)
        ctc_targets        = torch.cat(ctc_seqs)
        ctc_target_lengths = torch.tensor(ctc_lengths, dtype=torch.long)
        # feat_targets       = torch.tensor(
        #     np.stack(feat_targets_list), dtype=torch.float32
        # )
        # feat_item_idx      = torch.tensor(feat_item_indices, dtype=torch.long)

        return {
            "input_features":      input_features,
            "labels":              labels,
            "ctc_targets":         ctc_targets,
            "ctc_input_lengths":   ctc_input_lengths,
            "ctc_target_lengths":  ctc_target_lengths,
            # "feat_targets":        feat_targets,
            # "feat_frame_spans":    feat_frame_spans,   # list of lists (not a tensor)
            # "feat_item_idx":       feat_item_idx,
        }


@dataclass
class AuxLossOutput:
    loss:       Optional[torch.Tensor] = None   # total combined loss
    loss_asr:   Optional[torch.Tensor] = None
    loss_ctc:   Optional[torch.Tensor] = None
    loss_feat:  Optional[torch.Tensor] = None
    logits:     Optional[torch.Tensor] = None   # ASR decoder logits


class WhisperWithAuxHeads(nn.Module):
    """
    Thin wrapper around WhisperForConditionalGeneration that:
      1. Hooks the encoder to capture hidden states at CTC_LAYER and FEAT_LAYER
      2. Runs CTCPhonemeHead and PhonFeatureHead on those hidden states
      3. Returns a combined loss

    Args:
        model_name:   HuggingFace model id, e.g. "openai/whisper-small"
        lambda_ctc:   weight for CTC phoneme loss  (0 → disabled)
        lambda_feat:  weight for feature BCE loss   (0 → disabled)
        ctc_layer:    encoder layer to tap for CTC  (default CTC_LAYER)
        feat_layer:   encoder layer to tap for features (default FEAT_LAYER)
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
        self._feat_hidden_shares_ctc = (feat_layer == ctc_layer)

        # Load base Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name)

        hidden_dim = self.whisper.config.d_model   # 512 for whisper-small

        # Aux heads
        self.ctc_head  = CTCPhonemeHead(hidden_dim=hidden_dim)
        self.feat_head = PhonFeatureHead(hidden_dim=hidden_dim)

        # Storage for hooked hidden states
        self._ctc_hidden:  Optional[torch.Tensor] = None
        self._feat_hidden: Optional[torch.Tensor] = None

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        encoder_layers = self.whisper.model.encoder.layers

        def make_hook(target: str):
            def hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                if target == "ctc":
                    self._ctc_hidden = hs
                else:
                    self._feat_hidden = hs
            return hook

        if self.lambda_ctc > 0:
            encoder_layers[self.ctc_layer].register_forward_hook(make_hook("ctc"))

        if self.lambda_feat > 0:
            if self.feat_layer == self.ctc_layer:
                self._feat_hidden_shares_ctc = True   # just alias in forward()
            else:
                encoder_layers[self.feat_layer].register_forward_hook(make_hook("feat"))

    # ------------------------------------------------------------------
    # Parameter groups for the optimizer
    # ------------------------------------------------------------------

    def param_groups(self, base_lr: float = 1e-4, head_lr: float = 1e-3):
        """
        Returns optimizer param groups:
          - Whisper backbone params  (LoRA layers or all if no LoRA) → base_lr
          - Aux head params → head_lr  (heads learn faster from scratch)
        """
        backbone_params = [p for p in self.whisper.parameters() if p.requires_grad]
        head_params     = list(self.ctc_head.parameters()) + list(self.feat_head.parameters())
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params,     "lr": head_lr},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        # Standard Whisper ASR inputs
        input_features:  torch.Tensor,            # (B, 80, 3000)
        labels:          Optional[torch.Tensor] = None,   # (B, S) token ids

        # CTC auxiliary inputs
        ctc_targets:         Optional[torch.Tensor] = None,  # packed phone ids
        ctc_input_lengths:   Optional[torch.Tensor] = None,  # (B,) frame counts
        ctc_target_lengths:  Optional[torch.Tensor] = None,  # (B,) phone counts

        # Feature auxiliary inputs
        feat_segment_embeddings: Optional[torch.Tensor] = None,   # (N, D) pre-pooled
        feat_targets:            Optional[torch.Tensor] = None,   # (N, 16)

        # If True, pool feat embeddings from encoder here (alternative path)
        feat_frame_spans: Optional[list] = None,  # [(start, end), ...] per segment
    ) -> AuxLossOutput:

        # Reset hook captures
        self._ctc_hidden  = None
        self._feat_hidden = None
        

        # --- Main Whisper forward (captures hooks as side-effect) ---
        whisper_out = self.whisper(
            input_features        = input_features,
            decoder_input_ids     = labels[:, :-1].clamp(min=0),  # shift right, no -100
            output_hidden_states  = False,
            return_dict           = True,
        )

        if self._feat_hidden_shares_ctc:
            self._feat_hidden = self._ctc_hidden

        logits       = whisper_out.logits          # (B, S, vocab)
        shift_labels = labels[:, 1:].contiguous() # shift left
        loss_asr = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index = -100,
        )

        loss_ctc  = None
        loss_feat = None

        # --- CTC Phoneme Head ---
        if self.lambda_ctc > 0 and self._ctc_hidden is not None:
            _, loss_ctc = self.ctc_head(
                hidden        = self._ctc_hidden,
                targets       = ctc_targets,
                input_lengths = ctc_input_lengths,
                target_lengths= ctc_target_lengths,
            )

        # --- Phonological Feature Head ---
        if self.lambda_feat > 0 and self._feat_hidden is not None:
            if feat_segment_embeddings is None and feat_frame_spans is not None:
                # Pool on-the-fly from feat hidden states
                # feat_frame_spans: list of (start_frame, end_frame) for each segment
                # Assume single item in batch for now (can be batched later)
                feat_hidden = self._feat_hidden  # (B, T, D)
                pooled = []
                for (s, e) in feat_frame_spans:
                    e = min(e, feat_hidden.shape[1])
                    if e > s:
                        pooled.append(feat_hidden[0, s:e].mean(0))
                if pooled:
                    feat_segment_embeddings = torch.stack(pooled)  # (N, D)

            if feat_segment_embeddings is not None and feat_targets is not None:
                _, loss_feat = self.feat_head(feat_segment_embeddings, feat_targets)

        # --- Combine ---
        total_loss = loss_asr if loss_asr is not None else torch.tensor(0.0)
        if loss_ctc is not None:
            total_loss = total_loss + self.lambda_ctc * loss_ctc
        if loss_feat is not None:
            total_loss = total_loss + self.lambda_feat * loss_feat

        return AuxLossOutput(
            loss      = total_loss,
            loss_asr  = loss_asr,
            loss_ctc  = loss_ctc,
            loss_feat = loss_feat,
            logits    = logits,
        )

    # ------------------------------------------------------------------
    # Convenience: run just the encoder (for probing / inference)
    # ------------------------------------------------------------------

    def encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """Return top encoder hidden state (B, T, D), no grad."""
        with torch.no_grad():
            enc = self.whisper.model.encoder(
                input_features,
                output_hidden_states=False,
                return_dict=True,
            )
        return enc.last_hidden_state

    # ------------------------------------------------------------------
    # Convenience: transcribe (greedy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def transcribe(self, input_features: torch.Tensor, **gen_kwargs) -> torch.Tensor:
        return self.whisper.generate(input_features, **gen_kwargs)