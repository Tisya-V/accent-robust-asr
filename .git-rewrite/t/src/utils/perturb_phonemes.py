"""
src/utils/phoneme_perturber.py

Builds and caches a token-level phonemic neighbour table for Whisfusion.
Step 3: cache builder only.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import nltk
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from src.utils.phonology import ARPABET_VOCAB, phones_to_feature_matrix, feature_edit_distance

logger = logging.getLogger(__name__)
PHONE_SET = set(ARPABET_VOCAB)


class PhonemePerturber:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        k: int = 10,
        cache_dir: str = "src/utils/cache",
        length_bucket_threshold: int = 3,
        min_word_length: int = 2,
    ):
        logger.info(f"Initializing PhonemePerturber with tokenizer={tokenizer} k={k} cache_dir={cache_dir} length_bucket_threshold={length_bucket_threshold} min_word_length={min_word_length}")
        self.tokenizer = tokenizer
        self.k = k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.length_bucket_threshold = length_bucket_threshold
        self.min_word_length = min_word_length
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_name = getattr(tokenizer, "name_or_path", "tokenizer")
        self.cache_path = self._cache_path()
        self.special_ids = {x for x in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id, getattr(tokenizer, "mask_token_id", None)] if x is not None}

        self.cmudict = self._load_cmudict()
        self.neighbour_table = None
        self.phonemisable_count = 0
        self.phonemisable_tokens = {}
        self.feature_matrices = {}

        t0 = time.time()
        if self.cache_path.exists():
            logger.info(f"Loading phoneme neighbour table from cache: {self.cache_path}")
            payload = torch.load(self.cache_path, map_location="cpu")
            self.neighbour_table = payload["neighbour_table"]
            self.phonemisable_count = payload.get("phonemisable_count", 0)
            self.phonemisable_tokens = payload.get("phonemisable_tokens", {})
            logger.info("Loaded phoneme neighbour table from %s", self.cache_path)
            return

        logger.info(f"No cache found at {self.cache_path}. Building phoneme neighbour table from scratch...")
        logger.info("Step 1/3: Building phonemisable token index...")
        self._build_phonemisable_index()
        logger.info("Step 2/3: Building phoneme feature matrices...")
        self._build_feature_matrices()
        logger.info("Step 3/3: Building phoneme neighbour table...")
        self.neighbour_table = self._build_neighbour_table()
        torch.save(
            {
                "neighbour_table": self.neighbour_table,
                "phonemisable_count": self.phonemisable_count,
                "phonemisable_tokens": self.phonemisable_tokens,
                "tokenizer_name": self.tokenizer_name,
                "k": self.k,
                "length_bucket_threshold": self.length_bucket_threshold,
                "min_word_length": self.min_word_length,
                "built_at": time.time(),
            },
            self.cache_path,
        )
        logger.info(
            "Built phoneme neighbour table: %d phonemisable tokens / %d vocab tokens in %.1fs",
            self.phonemisable_count,
            self.vocab_size,
            time.time() - t0,
        )
        logger.info(f"Cache saved to {self.cache_path}")

    def _cache_path(self) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", self.tokenizer_name)
        return self.cache_dir / f"phoneme_neighbours_{safe}_k{self.k}.pt"

    def _load_cmudict(self):
        try:
            return nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            return nltk.corpus.cmudict.dict()

    def _token_to_word(self, token_id: int):
        raw = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        if not raw.startswith("▁"):
            return None
        word = raw[1:].strip().lower()
        if len(word) < self.min_word_length:
            return None
        if not word.isalpha():
            return None
        return word

    def _build_phonemisable_index(self):
        phonemisable = {}
        for token_id in tqdm(range(self.vocab_size), desc="Scanning vocab", leave=False):
            if token_id in self.special_ids:
                continue
            word = self._token_to_word(token_id)
            if word is None:
                continue
            prons = self.cmudict.get(word)
            if not prons:
                continue
            phones = [p.rstrip("012") for p in prons[0] if p.rstrip("012") in PHONE_SET]
            if phones:
                phonemisable[token_id] = phones
        self.phonemisable_tokens = phonemisable
        self.phonemisable_count = len(phonemisable)

    def _build_feature_matrices(self):
        feats = {}
        for token_id, phones in tqdm(self.phonemisable_tokens.items(), desc="Building feature matrices", leave=False):
            feats[token_id] = phones_to_feature_matrix(phones)
        self.feature_matrices = feats

    def _build_neighbour_table(self):
        length_buckets = defaultdict(list)
        for token_id, phones in self.phonemisable_tokens.items():
            length_buckets[len(phones)].append(token_id)

        table = torch.full((self.vocab_size, self.k), -1, dtype=torch.int32)
        lengths = sorted(length_buckets)

        for len_i in tqdm(lengths, desc="Length buckets", leave=False):
            bucket_i = length_buckets[len_i]
            candidate_js = []
            for len_j in lengths:
                if abs(len_i - len_j) <= self.length_bucket_threshold:
                    candidate_js.extend(length_buckets[len_j])
            candidate_js = [j for j in candidate_js if j not in bucket_i]
            if not candidate_js:
                continue

            by_source = defaultdict(list)
            pairs = [(src, dst) for src in bucket_i for dst in candidate_js]
            for start in tqdm(range(0, len(pairs), 256), desc=f"Distances len={len_i}", leave=False):
                chunk = pairs[start:start + 256]
                for src, dst in chunk:
                    dist = feature_edit_distance(self.feature_matrices[src], self.feature_matrices[dst])
                    by_source[src].append((dst, float(dist)))

            for src in bucket_i:
                items = sorted(by_source.get(src, []), key=lambda x: x[1])
                if not items:
                    continue
                neigh = [dst for dst, _ in items[: self.k]]
                while len(neigh) < self.k:
                    neigh.append(neigh[-1])
                table[src] = torch.tensor(neigh[: self.k], dtype=torch.int32)

        return table

    def summary(self):
        loaded = self.neighbour_table is not None
        return {
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "phonemisable_count": self.phonemisable_count,
            "coverage_pct": round(100.0 * self.phonemisable_count / self.vocab_size, 2),
            "k": self.k,
            "cache_path": str(self.cache_path),
            "loaded_from_cache": loaded,
            "table_shape": tuple(self.neighbour_table.shape) if loaded else None,
        }
    
    def to(self, device):
        if self.neighbour_table is None:
            raise RuntimeError("Neighbour table is not built")
        self.neighbour_table = self.neighbour_table.to(device)
        return self

    def _ensure_table(self, device=None):
        if self.neighbour_table is None:
            raise RuntimeError("Neighbour table is not built")
        if device is not None and self.neighbour_table.device != device:
            raise RuntimeError(f"Neighbour table is on {self.neighbour_table.device}, expected {device}. Move it once with perturber.to(device).")
        return self.neighbour_table

    def perturb(self, token_ids: torch.Tensor, perturb_prob: float = 0.15, mask_token_id: int | None = None):
        """Perturb visible tokens in-place using phonemic neighbours.

        token_ids: (B, L) tensor. If mask_token_id is provided, those positions are never perturbed.
        Returns: perturbed_ids, perturb_mask
        """
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        device = token_ids.device
        table = self._ensure_table(device=device)

        valid = (token_ids >= 0) & (token_ids < self.vocab_size)
        if mask_token_id is not None:
            valid = valid & (token_ids != mask_token_id)

        if not valid.any():
            return token_ids, torch.zeros_like(token_ids, dtype=torch.bool)

        rows = token_ids.clamp(0, self.vocab_size - 1)
        neigh = table[rows]
        has_neigh = (neigh >= 0).any(dim=-1)
        can_perturb = valid & has_neigh
        perturb_mask = can_perturb & (torch.rand(token_ids.shape, device=device) < perturb_prob)
        if not perturb_mask.any():
            return token_ids, perturb_mask

        choice = torch.randint(0, self.k, token_ids.shape, device=device)
        replacement = neigh.gather(-1, choice.unsqueeze(-1)).squeeze(-1).to(token_ids.dtype)
        out = token_ids.clone()
        out[perturb_mask] = replacement[perturb_mask]
        return out, perturb_mask

def main():
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="src/utils/cache")
    parser.add_argument("--length_bucket_threshold", type=int, default=3)
    parser.add_argument("--min_word_length", type=int, default=2)
    parser.add_argument("--mask_prob", type=float, default=0.35)
    parser.add_argument("--perturb_prob", type=float, default=0.50)
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pert = PhonemePerturber(
        tok,
        k=args.k,
        cache_dir=args.cache_dir,
        length_bucket_threshold=args.length_bucket_threshold,
        min_word_length=args.min_word_length,
    )

    ids = tok(args.text, return_tensors="pt").input_ids
    mask = torch.rand(ids.shape, device=ids.device) < args.mask_prob
    masked = ids.clone()
    mask_token_id = tok.mask_token_id if tok.mask_token_id is not None else tok.eos_token_id
    masked[mask] = mask_token_id
    noisy, p_mask = pert.perturb(masked, perturb_prob=args.perturb_prob, mask_token_id=mask_token_id)

    def to_tokens(x):
        return tok.convert_ids_to_tokens(x[0].tolist())

    original = to_tokens(ids)
    masked_tokens = to_tokens(masked)
    noised = to_tokens(noisy)
    decoded_original = tok.decode(ids[0], skip_special_tokens=False)
    decoded_masked = tok.decode(masked[0], skip_special_tokens=False)
    decoded_noised = tok.decode(noisy[0], skip_special_tokens=False)

    print("\nOriginal tokens:")
    print(original)
    print("\nMasked tokens:")
    print(masked_tokens)
    print("\nNoised tokens:")
    print(noised)
    print("\nOriginal text:")
    print(decoded_original)
    print("\nMasked text:")
    print(decoded_masked)
    print("\nNoised text:")
    print(decoded_noised)
    print("\nSummary:")
    print({
        "seq_len": ids.shape[1],
        "masked": int(mask.sum().item()),
        "perturbed": int(p_mask.sum().item()),
        "mask_prob": args.mask_prob,
        "perturb_prob": args.perturb_prob,
        "summary": pert.summary(),
    })


if __name__ == "__main__":
    main()