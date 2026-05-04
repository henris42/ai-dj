"""Supervised genre tagging with MTG's Discogs-Effnet (genre-discogs400 head).

Replaces the CLAP zero-shot tagger. CLAP scored audio against free-text prompts
("a rock song", "an ambient soundscape", …) which is brittle: the same track
gets a different label depending on prompt phrasing, and the labels often
disagree with what a human would call the music.

Discogs-Effnet is an EfficientNet trained on hundreds of thousands of audio
clips with their Discogs genre/style labels. The 400-class head used here
predicts sigmoid probabilities over the Discogs taxonomy ("Rock---Indie Rock",
"Electronic---Ambient", etc.); we aggregate those into the same 12 high-level
steering styles that the GUI buttons use, so the rest of the app doesn't need
to change.

Models live under `data/models/`:
  - `discogs-effnet-bs64-1.pb` — backbone (audio → 1280-d embeddings)
  - `genre_discogs400-discogs-effnet-1.pb` — head (1280-d → 400 sigmoid)
  - `genre_discogs400-discogs-effnet-1.json` — class labels + metadata

Audio is loaded as 30 s of 16 kHz mono from the middle of the track. The
backbone returns one embedding per 4 s window (hop 0.5 s); the per-window
sigmoid probabilities are averaged before mapping to the high-level styles."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
EFFNET_PB = MODELS_DIR / "discogs-effnet-bs64-1.pb"
HEAD_PB = MODELS_DIR / "genre_discogs400-discogs-effnet-1.pb"
LABELS_JSON = MODELS_DIR / "genre_discogs400-discogs-effnet-1.json"

SR = 16000
CLIP_SECONDS = 30.0
MIN_SECONDS = 4.0


def load_audio_16k(path: str, duration_s: float | None,
                   sample_seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Load `sample_seconds` of 16 kHz mono audio centred in the track."""
    target_len = int(SR * sample_seconds)
    if duration_s and duration_s > sample_seconds:
        offset = (duration_s - sample_seconds) / 2.0
    else:
        offset = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=sample_seconds)
    if len(y) < SR * MIN_SECONDS:
        raise ValueError(f"audio too short: {len(y) / SR:.1f}s")
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32, copy=False)


def _classify_discogs_label(label: str) -> str | None:
    """Map a Discogs `Genre---Style` label to one of the 12 high-level styles
    in `ai_dj.styles.STYLES`, or None to drop the label entirely (categories
    like Children's / Brass & Military / Non-Music aren't useful for steering)."""
    head, sub = label.split("---", 1)
    sub_l = sub.lower()
    if head == "Rock":
        metal_keys = ("metal", "doom", "thrash", "grindcore", "hardcore", "sludge", "crust")
        if any(k in sub_l for k in metal_keys):
            return "Metal"
        return "Rock"
    if head == "Pop":
        return "Pop"
    if head == "Electronic":
        if "ambient" in sub_l or "drone" in sub_l or "lo-fi" in sub_l or "downtempo" in sub_l:
            return "Ambient"
        if "disco" in sub_l or "house" in sub_l or "eurodance" in sub_l or "italo" in sub_l:
            return "Dance"
        return "Electronic"
    if head == "Classical":
        return "Classical"
    if head == "Jazz":
        return "Jazz"
    if head == "Hip Hop":
        return "Hip-Hop"
    if head == "Funk / Soul":
        if "disco" in sub_l:
            return "Dance"
        return "Soul"
    if head == "Blues":
        return "Soul"
    if head in ("Folk, World, & Country", "Latin", "Reggae"):
        return "Folk"
    if head == "Stage & Screen":
        return "Soundtrack"
    # Brass & Military, Children's, Non-Music — not useful for the steering UI.
    return None


class DiscogsTagger:
    """Tags audio with the Discogs-Effnet 400-class genre head, then aggregates
    to the high-level steering styles the GUI uses."""

    def __init__(self) -> None:
        for pb in (EFFNET_PB, HEAD_PB, LABELS_JSON):
            if not pb.exists():
                raise FileNotFoundError(
                    f"missing model file: {pb}\n"
                    "download from https://essentia.upf.edu/models/ — see "
                    "docstring at top of discogs_tag.py for URLs"
                )

        # Lazy import — pulling in essentia + TF takes ~3 s on first import.
        from essentia.standard import (  # type: ignore
            TensorflowPredictEffnetDiscogs, TensorflowPredict2D,
        )

        self._effnet = TensorflowPredictEffnetDiscogs(
            graphFilename=str(EFFNET_PB), output="PartitionedCall:1",
        )
        self._head = TensorflowPredict2D(
            graphFilename=str(HEAD_PB),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        with LABELS_JSON.open() as f:
            meta = json.load(f)
        self._labels: list[str] = meta["classes"]
        # Pre-compute label index -> high-level style (or None to skip).
        self._label_to_high: list[str | None] = [
            _classify_discogs_label(lab) for lab in self._labels
        ]
        kept = sum(1 for h in self._label_to_high if h is not None)
        logger.info("loaded Discogs-Effnet: %d Discogs labels → %d kept after high-level mapping",
                    len(self._labels), kept)

    def tag_batch(self, ys: Sequence[np.ndarray]) -> list[tuple[str, float, list[tuple[str, float]]]]:
        """Score each clip and return (top_style, top_score, top3) per clip,
        matching the interface of `ClapTagger.tag_batch` so `tag_library.py`
        works unchanged. Essentia processes one clip at a time internally; we
        loop here so the caller can still hand us batches for prefetch parity."""
        out: list[tuple[str, float, list[tuple[str, float]]]] = []
        for y in ys:
            embeddings = self._effnet(y)              # (T_emb, 1280)
            preds = self._head(embeddings)            # (T_emb, 400) sigmoid
            mean_preds = np.asarray(preds, dtype=np.float32).mean(axis=0)
            # Aggregate to high-level styles via max over the contributing
            # Discogs labels — one strong sub-style is enough to fire the
            # high-level bucket.
            per_style: dict[str, float] = {}
            for i, hi in enumerate(self._label_to_high):
                if hi is None:
                    continue
                p = float(mean_preds[i])
                if p > per_style.get(hi, 0.0):
                    per_style[hi] = p
            if not per_style:
                # Should be rare — would mean every Discogs prob mapped to a
                # dropped category (Brass & Military, Non-Music, …). Pick the
                # plain top label so we still record something.
                top_idx = int(np.argmax(mean_preds))
                fallback = self._labels[top_idx]
                out.append((fallback, float(mean_preds[top_idx]), [(fallback, float(mean_preds[top_idx]))]))
                continue
            ranked = sorted(per_style.items(), key=lambda x: -x[1])
            top_style, top_score = ranked[0]
            out.append((top_style, top_score, ranked[:3]))
        return out
