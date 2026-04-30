from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pronouncing
from transformers import AutoTokenizer

from src.config import RANDOM_SEED


EXAMPLE_TEXTS = [
    "Author of the danger trail Philip Steels etc",
    "Lord but I'm glad to see you again Phil",
    "You used to joy ride like the very devil.",
]

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEFAULT_MAX_CHAR_DELTA = 2
DEFAULT_MAX_PHONEME_DISTANCE = 3
DEFAULT_TOP_K_NEIGHBORS = 5
DEFAULT_MAX_PERTURBED_WORDS_PER_SAMPLE = 2


class FastTokenizerAwarePhonemePerturber:
    def __init__(
        self,
        tokenizer_name: str,
        confusion_map: Optional[Dict[str, Sequence[str]]] = None,
        seed: int = RANDOM_SEED,
        max_char_delta: int = DEFAULT_MAX_CHAR_DELTA,
        max_phoneme_distance: int = DEFAULT_MAX_PHONEME_DISTANCE,
        top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        print("Initializing FastTokenizerAwarePhonemePerturber...")
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        pronouncing.init_cmu()
        self.rng = random.Random(seed)
        self.confusion_map = {
            k: tuple(v) for k, v in (confusion_map or self.default_confusion_map()).items()
        }
        self.max_char_delta = max_char_delta
        self.max_phoneme_distance = max_phoneme_distance
        self.top_k_neighbors = top_k_neighbors
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.lexicon = self._build_fast_lexicon()
        self.by_token_len = self._group_by_token_len()
        self.neighbor_graph = self._load_or_build_neighbor_graph()

        print(
            f"Built lexicon with {len(self.lexicon)} entries, "
            f"grouped into {len(self.by_token_len)} token length buckets."
        )
        print(
            f"Built perturb graph for {len(self.neighbor_graph)} words "
            f"(top_k={self.top_k_neighbors}, max_phoneme_distance={self.max_phoneme_distance}, "
            f"max_char_delta={self.max_char_delta})."
        )

    @staticmethod
    def default_confusion_map() -> Dict[str, Sequence[str]]:
        return {
            "TH": ["S", "T", "F"],
            "DH": ["D", "Z", "V"],
            "V": ["B", "F", "W"],
            "B": ["V", "P"],
            "R": ["L", "W"],
            "L": ["R"],
            "SH": ["S", "CH"],
            "S": ["TH", "SH", "Z"],
            "Z": ["S", "DH"],
            "F": ["TH", "V"],
            "T": ["TH", "D"],
            "D": ["DH", "T"],
            "IH": ["IY", "EH"],
            "IY": ["IH"],
            "UH": ["UW", "AH"],
            "UW": ["UH"],
            "AA": ["AO", "AH"],
            "AO": ["AA", "OW"],
            "EH": ["IH", "AE"],
            "AE": ["EH", "AH"],
            "ER": ["AH", "AO"],
            "OW": ["AO", "UH"],
            "G": ["K"],
            "K": ["G"],
            "P": ["B"],
        }

    def _cache_key(self) -> str:
        payload = {
            "tokenizer_name": self.tokenizer_name,
            "max_char_delta": self.max_char_delta,
            "max_phoneme_distance": self.max_phoneme_distance,
            "top_k_neighbors": self.top_k_neighbors,
            "confusion_map": {k: list(v) for k, v in sorted(self.confusion_map.items())},
            "version": 1,
        }
        digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"phoneme_neighbors_{digest}.pkl"

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self._cache_key()

    def _load_or_build_neighbor_graph(self) -> Dict[str, List[str]]:
        cache_path = self._cache_path()
        if self.use_cache and cache_path is not None and cache_path.exists():
            print(f"Loading perturb graph from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print("Building perturb graph from scratch...")
        graph = self._build_neighbor_graph()

        if self.use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved perturb graph cache to: {cache_path}")

        return graph

    def _strip_stress(self, phones: str) -> Tuple[str, ...]:
        return tuple(re.sub(r"[012]", "", p) for p in phones.split())

    def _build_fast_lexicon(self) -> Dict[str, Dict]:
        lex = {}
        for raw_word, phones_list in pronouncing.lookup.items():
            word = raw_word.lower()
            if "(" in word or not re.fullmatch(r"[a-z']+", word):
                continue
            if not phones_list:
                continue
            arpabet = self._strip_stress(phones_list[0])
            token_len = len(self.text_to_ids(" " + word))
            lex[word] = {
                "arpabet": arpabet,
                "char_len": len(word),
                "token_len": token_len,
            }
        return lex

    def _group_by_token_len(self) -> Dict[int, List[str]]:
        grouped = defaultdict(list)
        for word, info in self.lexicon.items():
            grouped[info["token_len"]].append(word)
        return dict(grouped)

    def _build_neighbor_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}

        for token_len, words in self.by_token_len.items():
            for word in words:
                src_info = self.lexicon[word]
                candidates = []

                for cand in words:
                    if cand == word:
                        continue

                    cand_info = self.lexicon[cand]
                    if abs(cand_info["char_len"] - src_info["char_len"]) > self.max_char_delta:
                        continue

                    dist = self.phoneme_edit_distance(
                        src_info["arpabet"],
                        cand_info["arpabet"],
                        max_distance=self.max_phoneme_distance,
                    )
                    if dist > self.max_phoneme_distance:
                        continue

                    candidates.append((dist, abs(cand_info["char_len"] - src_info["char_len"]), cand))

                candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                graph[word] = [cand for _, _, cand in candidates[: self.top_k_neighbors]]

        return graph

    def ids_to_text(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def phoneme_edit_distance(a: Sequence[str], b: Sequence[str], max_distance: int) -> int:
        m, n = len(a), len(b)
        if abs(m - n) > max_distance:
            return max_distance + 1

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            row_min = max_distance + 1
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
                row_min = min(row_min, dp[i][j])
            if row_min > max_distance:
                return max_distance + 1
        return dp[m][n]

    @staticmethod
    def _match_case(src: str, dst: str) -> str:
        if src.isupper():
            return dst.upper()
        if src[:1].isupper():
            return dst.capitalize()
        return dst

    @staticmethod
    def _is_word(part: str) -> bool:
        return re.fullmatch(r"[A-Za-z']+", part) is not None

    def _parts_and_offsets(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        parts = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]+|\s+", text)
        offsets = []
        cursor = 0
        for part in parts:
            start = cursor
            end = cursor + len(part)
            offsets.append((start, end))
            cursor = end
        return parts, offsets

    def _char_span_visible(
        self,
        char_start: int,
        char_end: int,
        token_offsets: List[Tuple[int, int]],
        visible_mask: List[bool],
    ) -> bool:
        touched = False
        for (tok_start, tok_end), is_visible in zip(token_offsets, visible_mask):
            if tok_end <= char_start or tok_start >= char_end:
                continue
            touched = True
            if not is_visible:
                return False
        return touched

    def perturb_tokenized_sentence(
        self,
        token_ids: List[int],
        visible_token_mask: Optional[List[bool]] = None,
        max_perturbed_words: int = DEFAULT_MAX_PERTURBED_WORDS_PER_SAMPLE,
    ) -> Tuple[List[int], str, str, Dict[str, int]]:
        text = self.ids_to_text(token_ids)
        parts, offsets = self._parts_and_offsets(text)

        if visible_token_mask is None:
            visible_token_mask = [True] * len(token_ids)

        token_offsets = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )["offset_mapping"]

        eligible_indices = []
        for idx, part in enumerate(parts):
            if not self._is_word(part):
                continue

            lower = part.lower()
            if lower not in self.lexicon:
                continue
            if not self.neighbor_graph.get(lower):
                continue

            char_start, char_end = offsets[idx]
            if self._char_span_visible(char_start, char_end, token_offsets, visible_token_mask):
                eligible_indices.append(idx)

        self.rng.shuffle(eligible_indices)
        chosen_indices = set(eligible_indices[:max_perturbed_words])

        new_parts = []
        changed_words = 0
        attempted_words = len(chosen_indices)

        for idx, part in enumerate(parts):
            if idx not in chosen_indices:
                new_parts.append(part)
                continue

            lower = part.lower()
            neighbors = self.neighbor_graph.get(lower, [])
            if not neighbors:
                new_parts.append(part)
                continue

            replacement = self._match_case(part, self.rng.choice(neighbors))
            if replacement != part:
                changed_words += 1
            new_parts.append(replacement)

        perturbed_text = "".join(new_parts)
        perturbed_ids = self.text_to_ids(perturbed_text)

        if len(perturbed_ids) != len(token_ids):
            return token_ids, text, text, {
                "eligible_words": len(eligible_indices),
                "attempted_words": attempted_words,
                "changed_words": 0,
                "token_length_preserved": 0,
            }

        return perturbed_ids, text, perturbed_text, {
            "eligible_words": len(eligible_indices),
            "attempted_words": attempted_words,
            "changed_words": changed_words,
            "token_length_preserved": 1,
        }

    def perturb_tokenised_batch(
        self,
        token_ids_batch: List[List[int]],
        visible_token_masks: Optional[List[List[bool]]] = None,
        max_perturbed_words_per_sample: int = DEFAULT_MAX_PERTURBED_WORDS_PER_SAMPLE,
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        perturbed_batch = []
        total_eligible_words = 0
        total_attempted_words = 0
        total_changed_words = 0
        total_preserved = 0

        if visible_token_masks is None:
            visible_token_masks = [None] * len(token_ids_batch)

        for token_ids, visible_mask in zip(token_ids_batch, visible_token_masks):
            perturbed_ids, _, _, stats = self.perturb_tokenized_sentence(
                token_ids,
                visible_token_mask=visible_mask,
                max_perturbed_words=max_perturbed_words_per_sample,
            )
            perturbed_batch.append(perturbed_ids)
            total_eligible_words += stats["eligible_words"]
            total_attempted_words += stats["attempted_words"]
            total_changed_words += stats["changed_words"]
            total_preserved += stats["token_length_preserved"]

        batch_size = max(1, len(token_ids_batch))
        return perturbed_batch, {
            "eligible_words_total": total_eligible_words,
            "attempted_words_total": total_attempted_words,
            "changed_words_total": total_changed_words,
            "avg_eligible_words_per_sample": total_eligible_words / batch_size,
            "avg_attempted_words_per_sample": total_attempted_words / batch_size,
            "avg_changed_words_per_sample": total_changed_words / batch_size,
            "token_length_preserved_rate": total_preserved / batch_size,
        }


def run_examples(example_texts: List[str]) -> None:
    perturber = FastTokenizerAwarePhonemePerturber(
        tokenizer_name=TOKENIZER_NAME,
        seed=RANDOM_SEED,
        max_char_delta=DEFAULT_MAX_CHAR_DELTA,
        max_phoneme_distance=DEFAULT_MAX_PHONEME_DISTANCE,
        top_k_neighbors=DEFAULT_TOP_K_NEIGHBORS,
        cache_dir=".cache/phoneme_perturb",
        use_cache=True,
    )

    print("TOKENIZER:", TOKENIZER_NAME)
    print("N EXAMPLES:", len(example_texts))
    print("LEXICON SIZE:", len(perturber.lexicon))
    print("-" * 80)

    for i, text in enumerate(example_texts, start=1):
        original_ids = perturber.text_to_ids(text)
        perturbed_ids, original_text, perturbed_text, stats = perturber.perturb_tokenized_sentence(
            original_ids,
            visible_token_mask=[True] * len(original_ids),
            max_perturbed_words=DEFAULT_MAX_PERTURBED_WORDS_PER_SAMPLE,
        )
        print(f"EXAMPLE {i}")
        print("ORIGINAL TEXT :", original_text)
        print("ORIGINAL LEN  :", len(original_ids))
        print("PERTURBED TEXT:", perturbed_text)
        print("PERTURBED LEN :", len(perturbed_ids))
        print("ROUNDTRIP TEXT:", perturber.ids_to_text(perturbed_ids))
        print("STATS         :", stats)
        print("-" * 80)


if __name__ == "__main__":
    run_examples(EXAMPLE_TEXTS)