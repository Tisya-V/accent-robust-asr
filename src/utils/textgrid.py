from dataclasses import dataclass
from pathlib import Path
from typing import List
from src.config import ENCODER_FRAME_RATE, SPEAKER_L1, SILENCE_LABELS
from src.utils.phonology import PHONE2ID
# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

@dataclass
class PhoneSegment:
    label:    str    # ARPAbet label (stress digits stripped)
    start:    float  # seconds
    end:      float  # seconds
    phone_id: int    # index into ARPABET_VOCAB; -1 if silence/unknown

    @property
    def start_frame(self) -> int:
        return int(self.start * ENCODER_FRAME_RATE)

    @property
    def end_frame(self) -> int:
        return max(self.start_frame + 1, int(self.end * ENCODER_FRAME_RATE))

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_lab_file(lab_path: str) -> List[PhoneSegment]:
    """
    Parse a CMU ARCTIC .lab file and return PhoneSegments.
    Silence tokens (pau) are excluded; stress digits are stripped.
    """
    path = Path(lab_path)
    if not path.exists():
        raise FileNotFoundError(f"LAB file not found: {lab_path}")

    raw = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                raw.append((float(parts[0]), parts[2]))
            except ValueError:
                pass

    segments: List[PhoneSegment] = []
    for i, (end_time, text) in enumerate(raw):
        label = text.rstrip("012").upper()
        if label in SILENCE_LABELS:
            continue
        start_time = raw[i - 1][0] if i > 0 else 0.0
        segments.append(PhoneSegment(
            label    = label,
            start    = start_time,
            end      = end_time,
            phone_id = PHONE2ID.get(label, -1),
        ))

    return segments


def parse_textgrid(
    textgrid_path: str,
    tier_name: str = "phones",
) -> List[PhoneSegment]:
    """
    Parse a Praat TextGrid and return PhoneSegments for the named tier.
    Handles both short-format and long-format TextGrids.
    Silence tokens are excluded; stress digits are stripped from labels.
    """
    path = Path(textgrid_path)
    if not path.exists():
        raise FileNotFoundError(f"TextGrid not found: {textgrid_path}")

    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]

    segments: List[PhoneSegment] = []
    in_target_tier = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "name =" in line:
            tier_label     = line.split("=", 1)[1].strip().strip('"')
            in_target_tier = (tier_label == tier_name)
        if in_target_tier and ("intervals [" in line or "intervals:" in line):
            try:
                xmin     = float(lines[i + 1].split("=")[1].strip())
                xmax     = float(lines[i + 2].split("=")[1].strip())
                text_val = lines[i + 3].split("=", 1)[1].strip().strip('"')
                label    = text_val.rstrip("012").upper()
                if label not in SILENCE_LABELS:
                    segments.append(PhoneSegment(
                        label    = label,
                        start    = xmin,
                        end      = xmax,
                        phone_id = PHONE2ID.get(label, -1),
                    ))
                i += 4
                continue
            except (IndexError, ValueError):
                pass
        i += 1

    return segments

