"""
vocab_coverage_probe.py

Throwaway diagnostic script — not part of the final codebase.

Probes the TinyLlama tokenizer vocabulary to estimate how many tokens
are phonemisable (i.e. decode to a real English word present in CMUdict).

Run:
    python vocab_coverage_probe.py

Requires:
    pip install transformers nltk
    python -c "import nltk; nltk.download('cmudict')"
"""

import nltk
import re
from collections import Counter
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
N_EXAMPLES      = 10   # how many examples to print per category

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print(f"Vocab size: {tokenizer.vocab_size}")

print("Loading CMUdict...")
nltk.download("cmudict", quiet=True)
cmudict = nltk.corpus.cmudict.dict()  # word -> list of pronunciations
print(f"CMUdict entries: {len(cmudict)}")

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

WORD_BOUNDARY_MARKER = "\u2581"   # ▁  (SentencePiece word-boundary prefix)

categories = {
    "phonemisable":        [],   # word-boundary token, word in CMUdict
    "word_not_in_cmudict": [],   # word-boundary token, word NOT in CMUdict
    "subword_fragment":    [],   # no word-boundary marker (continuation piece)
    "special_token":       [],   # special tokens ([PAD], [MASK], etc.)
    "other":               [],   # digits, punctuation, empty, etc.
}

special_ids = {
    tokenizer.pad_token_id,
    tokenizer.eos_token_id,
    tokenizer.bos_token_id,
    tokenizer.unk_token_id,
    # mask token may not exist in a causal LM tokenizer — guard with getattr
    getattr(tokenizer, "mask_token_id", None),
}
special_ids.discard(None)

for token_id in range(tokenizer.vocab_size):

    # Get the raw SentencePiece token string (preserves ▁ marker)
    raw = tokenizer.convert_ids_to_tokens([token_id])[0]

    # --- Special tokens ---
    if token_id in special_ids or raw in ("<unk>", "<s>", "</s>", "<pad>"):
        categories["special_token"].append((token_id, raw))
        continue

    # --- Sub-word fragment (no word-boundary marker) ---
    if not raw.startswith(WORD_BOUNDARY_MARKER):
        categories["subword_fragment"].append((token_id, raw))
        continue

    # --- Word-boundary token: strip marker, lowercase, remove non-alpha ---
    word = raw[len(WORD_BOUNDARY_MARKER):]   # strip ▁
    word_clean = word.lower().strip()

    # Exclude if empty, purely numeric, or contains non-alpha characters
    # (punctuation tokens like ▁. or ▁, fall here)
    if not word_clean or not re.fullmatch(r"[a-z]+", word_clean) or len(word_clean) < 2:
        categories["other"].append((token_id, raw, word_clean))
        continue

    # --- CMUdict lookup ---
    if word_clean in cmudict:
        categories["phonemisable"].append((token_id, raw, word_clean))
    else:
        categories["word_not_in_cmudict"].append((token_id, raw, word_clean))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

total = tokenizer.vocab_size
print("\n" + "=" * 60)
print("VOCABULARY COVERAGE REPORT")
print("=" * 60)

for label, items in categories.items():
    pct = 100 * len(items) / total
    print(f"  {label:<25}  {len(items):>6}  ({pct:5.1f}%)")

print("-" * 60)
print(f"  {'TOTAL':<25}  {total:>6}")

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

print(f"\n--- {N_EXAMPLES} PHONEMISABLE examples ---")
for token_id, raw, word in categories["phonemisable"][:N_EXAMPLES]:
    pron = " ".join(cmudict[word][0])   # first pronunciation, with stress
    print(f"  id={token_id:>6}  raw={raw:<20}  cmudict={pron}")

print(f"\n--- {N_EXAMPLES} WORD_NOT_IN_CMUDICT examples ---")
for token_id, raw, word in categories["word_not_in_cmudict"][:N_EXAMPLES]:
    print(f"  id={token_id:>6}  raw={raw:<20}  word={word}")

print(f"\n--- {N_EXAMPLES} SUBWORD_FRAGMENT examples ---")
for token_id, raw in categories["subword_fragment"][:N_EXAMPLES]:
    print(f"  id={token_id:>6}  raw={raw}")

print(f"\n--- {N_EXAMPLES} OTHER examples (numeric / punctuation / mixed) ---")
for entry in categories["other"][:N_EXAMPLES]:
    token_id, raw, word_clean = entry
    print(f"  id={token_id:>6}  raw={raw:<20}  cleaned={word_clean!r}")

# ---------------------------------------------------------------------------
# Phoneme sequence length distribution for phonemisable tokens
# ---------------------------------------------------------------------------

print("\n--- Phoneme sequence length distribution (phonemisable tokens) ---")
length_counts = Counter()
for token_id, raw, word in categories["phonemisable"]:
    # Strip stress digits from CMUdict pronunciation
    phones = [p.rstrip("012") for p in cmudict[word][0]]
    length_counts[len(phones)] += 1

for length in sorted(length_counts):
    bar = "#" * (length_counts[length] // 20)
    print(f"  len={length:>2}  count={length_counts[length]:>5}  {bar}")

print("\nDone.")