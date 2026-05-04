"""MERT-based track embedding.

Audio loading is split from GPU inference: `load_audio` runs in worker threads and
returns fixed-length arrays so the main thread can stack them into uniform GPU
batches via `embed_batch`."""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import librosa
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)

MODEL_ID = "m-a-p/MERT-v1-95M"
SR = 24000
MIN_SECONDS = 2.0


def load_audio(path: str, duration_s: float | None, sample_seconds: float = 30.0) -> np.ndarray:
    """Load a mono 24 kHz window centred in the track, padded/truncated to exactly
    `sample_seconds`. Worker-thread safe (librosa+ffmpeg release the GIL)."""
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


class Embedder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
        )
        logger.info("loaded %s on %s", MODEL_ID, device)

    @torch.no_grad()
    def embed_batch(self, ys: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of same-length audio arrays. Returns (len(ys), 768) float32."""
        inputs = self.fe(list(ys), sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        vecs = out.last_hidden_state.mean(dim=1)  # (B, D)
        return vecs.cpu().float().numpy()

    def embed_file(self, path: str, duration_s: float | None = None, sample_seconds: float = 30.0) -> np.ndarray:
        y = load_audio(path, duration_s, sample_seconds)
        return self.embed_batch([y])[0]
