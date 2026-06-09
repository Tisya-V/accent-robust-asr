"""
Frozen-decoder teacher-forcing CE loss for bridge on-manifold regularization.

The bridge's x0 prediction (z_hat) is fed as encoder hidden states to the
frozen Whisper decoder with teacher-forced ground-truth tokens. Gradients flow
through z_hat into bridge weights only — Whisper weights are never updated.

See docs/superpowers/specs/2026-06-09-bridge-ce-loss-design.md for full design.
"""

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def compute_ce_loss(
    z_hat: torch.Tensor,
    texts: list[str],
    whisper: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    l2_speech_end: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Args:
        z_hat:         [B, max_l2, 768] bridge x0 prediction — must be in the
                       autograd graph (no detach); gradients flow back through
                       cross-attention into bridge weights.
        texts:         raw transcript strings, length B
        whisper:       frozen WhisperForConditionalGeneration (eval mode,
                       all params requires_grad=False)
        processor:     WhisperProcessor for tokenization
        l2_speech_end: [B] per-sample speech frame boundary (kept for API
                       consistency; z_hat is already cropped to max_l2 so no
                       additional encoder masking is needed)
        device:        compute device

    Returns:
        scalar CE loss (mean over unmasked token positions across the batch)
    """
    tokenizer = processor.tokenizer
    B, _, _ = z_hat.shape

    # Tokenize transcripts — no Whisper special tokens; we add the decoder
    # prompt prefix manually so we control exactly what goes into decoder_input_ids
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    text_ids  = encoding["input_ids"].to(device)       # [B, N]
    text_mask = encoding["attention_mask"].to(device)  # [B, N] 1=real 0=pad

    # Build Whisper decoder prefix: [SOT, lang, task, notimestamps]
    sot_id = whisper.config.decoder_start_token_id
    forced = tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")
    prefix_ids = torch.tensor(
        [sot_id] + [tok_id for _, tok_id in forced],
        device=device, dtype=torch.long,
    )  # [4]
    prefix = prefix_ids.unsqueeze(0).expand(B, -1)     # [B, 4]

    # decoder_input_ids = prefix + all text tokens (including padding)
    decoder_input_ids = torch.cat([prefix, text_ids], dim=1)    # [B, 4+N]

    # decoder_attention_mask: 1 for real positions, 0 for padding.
    # Prefix tokens are always real; text tokens use the tokenizer mask.
    # Passing this silences the pad==eos warning and correctly excludes
    # padding positions from causal self-attention.
    decoder_attention_mask = torch.cat([
        torch.ones(B, prefix.shape[1], device=device, dtype=torch.long),  # [B, 4]
        text_mask,                                                          # [B, N]
    ], dim=1)  # [B, 4+N]

    # Labels = decoder_input_ids shifted left by 1 (standard causal LM objective).
    # logits[b, i] predicts the token at position i+1 = decoder_input_ids[b, i+1].
    labels = decoder_input_ids[:, 1:].clone()                              # [B, 4+N-1]

    # Mask prefix prediction positions (0-2 predict lang/task/notimestamps —
    # always fixed, carry no useful speech-content gradient signal)
    labels[:, :3] = -100

    # Mask padding in the text part of labels via the tokenizer attention mask.
    # text_label_mask covers label positions 3..3+N-2 (all but the last text token).
    # Position 3+N-1 corresponds to the last real text token for the longest
    # sequence in the batch, which is always unpadded by construction.
    text_label_mask = text_mask[:, :-1]                                    # [B, N-1]
    labels[:, 3:3 + text_label_mask.shape[1]][text_label_mask == 0] = -100

    # Teacher-forced forward pass through frozen Whisper.
    # Pass z_hat as pre-computed encoder_outputs — this skips the Whisper encoder
    # and feeds z_hat directly into the decoder cross-attention.
    # z_hat is already [B, max_l2, 768] (cropped to speech frames by
    # _bridge_loss_dtw_fixed), so the decoder sees speech frames only.
    # z_hat cast to whisper.dtype (decoder fp32; bridge trains in bf16).
    # NOT wrapped in torch.no_grad() — gradient must flow through z_hat back to
    # bridge weights. Whisper weights have requires_grad=False so no grad
    # accumulates for them regardless.
    output = whisper.model(
        encoder_outputs=(z_hat.to(whisper.dtype),),
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )
    logits = whisper.proj_out(output.last_hidden_state)  # [B, 4+N, V]

    # CE over positions 0..4+N-2 (logits[:, :-1]) against shifted labels [B, 4+N-1]
    V = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
    )
