"""Zero-shot genre tagging with LAION-CLAP (music-tuned).

For each track we sample a 10s window at 48 kHz, encode with CLAP, and score it
against a set of text prompts grouped by style. The style with the best matching
prompt wins; the score is kept alongside so the GUI/planner can decide whether
the AI tag is confident enough to trust.

Text embeddings are precomputed once in the constructor — they're small and
constant — so tagging is a single forward pass per audio batch."""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor



logger = logging.getLogger(__name__)

MODEL_ID = "laion/clap-htsat-unfused"
SR = 48000
CLIP_SECONDS = 10.0
MIN_SECONDS = 2.0

# Style -> list of prompts. Styles match `ai_dj.styles.STYLES` keys so the GUI
# buttons and AI tags line up 1:1.
GENRE_PROMPTS: dict[str, list[str]] = {
    "Rock":       ["rock music", "a rock song", "alternative rock", "hard rock", "indie rock", "grunge music"],
    "Pop":        ["pop music", "a pop song", "mainstream pop", "synth pop", "electropop"],
    "Electronic": ["electronic music", "a techno track", "house music", "an IDM piece", "drum and bass",
                   "synthesizer music", "electronica"],
    "Dance":      ["dance music", "a disco song", "club music", "EDM"],
    "Ambient":    ["ambient music", "an ambient soundscape", "drone music", "lo-fi beats", "chillout music"],
    "Classical":  ["classical music", "an orchestral piece", "a baroque composition", "a string quartet",
                   "a piano sonata"],
    "Jazz":       ["jazz music", "a jazz combo", "bebop", "fusion jazz", "a jazz standard"],
    "Metal":      ["heavy metal", "death metal", "doom metal", "a metal song", "thrash metal"],
    "Hip-Hop":    ["hip hop", "rap music", "a hip hop track", "r&b", "trap music"],
    "Soul":       ["soul music", "funk music", "motown", "gospel"],
    "Folk":       ["folk music", "acoustic folk", "americana", "country music", "world music"],
    "Soundtrack": ["a film score", "a movie soundtrack", "cinematic music", "video game music"],
}


def load_audio_48k(path: str, duration_s: float | None, sample_seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Load a 48 kHz mono window centred in the track."""
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


class ClapTagger:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.proc = ClapProcessor.from_pretrained(MODEL_ID)
        self.model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()

        self._prompt_styles: list[str] = []
        self._prompts: list[str] = []
        for style, ps in GENRE_PROMPTS.items():
            for p in ps:
                self._prompt_styles.append(style)
                self._prompts.append(p)
        logger.info("loaded %s on %s, %d prompts over %d styles",
                    MODEL_ID, device, len(self._prompts), len(GENRE_PROMPTS))

    @torch.no_grad()
    def tag_batch(self, ys: Sequence[np.ndarray]) -> list[tuple[str, float, list[tuple[str, float]]]]:
        """Score a batch of 10s clips against the prompt set.

        Returns a list of (top_style, top_score, [(style, score), ...top-3]) per
        input clip. Uses the full ClapModel forward so both modalities go
        through their respective projections into the joint space."""
        inputs = self.proc(
            text=self._prompts,
            audio=list(ys),
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # logits_per_audio: (B_audio, N_text), already = logit_scale * (audio @ text.T)
        sims = out.logits_per_audio.float().cpu().numpy()

        results: list[tuple[str, float, list[tuple[str, float]]]] = []
        for row in sims:
            # Max-pool scores per style
            per_style: dict[str, float] = {}
            for i, style in enumerate(self._prompt_styles):
                s = float(row[i])
                if s > per_style.get(style, -1e9):
                    per_style[style] = s
            ranked = sorted(per_style.items(), key=lambda x: -x[1])
            top_style, top_score = ranked[0]
            results.append((top_style, top_score, ranked[:3]))
        return results
