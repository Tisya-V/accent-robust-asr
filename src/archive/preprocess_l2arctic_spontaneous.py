#!/usr/bin/env python3
"""
Legacy preprocessing script for spontaneous/suitcase corpus.
No longer used in main pipeline — kept for reference.

To use: from src.archive.load_l2arctic_legacy import load_suitcase_corpus
"""

from pathlib import Path
import random
import torchaudio
import pandas as pd
from tqdm import tqdm

from textgrid import TextGrid  # praat-parsed TextGrid

from src.config import LOCAL_L2ARCTIC_DIR, SPONTANEOUS_SUBDIR
from src.archive.load_l2arctic_legacy import load_suitcase_corpus


TARGET_SR = 16000
MIN_SEC = 10
MAX_SEC = 25


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


def sec_to_frame(sec, sr=TARGET_SR):
    return int(sec * sr)


# ------------------------------------------------------------
# chunking via TextGrid
# ------------------------------------------------------------

def chunk_from_textgrid(wav_path, tg_path):
    audio = load_audio(wav_path)
    tg = TextGrid.fromFile(str(tg_path))

    # assume first tier is words (L2-Arctic standard)
    word_tier = None
    for tier in tg.tiers:
        if tier.name.lower().startswith("word"):
            word_tier = tier
            break

    if word_tier is None:
        raise ValueError(f"No word tier found in {tg_path}")

    chunks = []
    cur_words = []
    cur_start = None
    cur_end = None

    for interval in word_tier.intervals:
        w = interval.mark.strip()
        if w == "":
            continue

        start, end = interval.minTime, interval.maxTime

        if cur_start is None:
            cur_start = start

        cur_words.append(w)
        cur_end = end

        duration = cur_end - cur_start
        min_dur = random.randint(MIN_SEC, MAX_SEC)
        if duration >= min_dur:
                chunks.append({
                    "start": cur_start,
                    "end": cur_end,
                    "text": " ".join(cur_words)
                })

                cur_words = []
                cur_start = None
                cur_end = None

    return chunks, audio


# ------------------------------------------------------------
# main pipeline
# ------------------------------------------------------------

def build_spontaneous_chunks(out_dir):
    out_dir = Path(out_dir)
    wav_out = out_dir / "wav"
    txt_out = out_dir / "transcript"
    wav_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    utts = load_suitcase_corpus()

    manifest = []

    for u in tqdm(utts):
        wav_path = Path(u["wav_path"])
        tg_path = Path(u["textgrid"])

        if not tg_path.exists():
            continue

        try:
            chunks, audio = chunk_from_textgrid(wav_path, tg_path)
        except Exception as e:
            print(f"[WARN] skipping {wav_path}: {e}")
            continue

        for i, c in enumerate(chunks):
            start, end = c["start"], c["end"]

            chunk_audio = audio[
                sec_to_frame(start):sec_to_frame(end)
            ]

            chunk_id = f"{Path(wav_path).stem}_chunk{i}"

            out_wav = wav_out / f"{chunk_id}.wav"
            out_txt = txt_out / f"{chunk_id}.txt"

            torchaudio.save(
                out_wav,
                chunk_audio.unsqueeze(0),
                TARGET_SR
            )

            out_txt.write_text(c["text"])

            manifest.append({
                "utterance_id": chunk_id,
                "speaker": u["speaker"],
                "l1": u["l1"],
                "wav_path": str(out_wav),
                "text": c["text"],
                "domain": "spontaneous",
                "split": "ood"
            })

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved {len(manifest)} chunks → {out_dir}")


if __name__ == "__main__":
    build_spontaneous_chunks(f"{LOCAL_L2ARCTIC_DIR}/{SPONTANEOUS_SUBDIR}")
