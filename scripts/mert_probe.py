"""MERT GPU probe — load MERT-v1-95M, embed 10s of audio, verify device."""
import sys
import time

import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MODEL_ID = "m-a-p/MERT-v1-95M"
SR = 24000
DURATION_S = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
if device == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")

t0 = time.time()
fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
print(f"model loaded in {time.time() - t0:.1f}s")
print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

wav = torch.sin(2 * torch.pi * 440 * torch.arange(SR * DURATION_S) / SR).unsqueeze(0)
print(f"input audio:  {wav.shape} @ {SR} Hz")

inputs = fe(wav.squeeze().numpy(), sampling_rate=SR, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    for run in range(4):
        t0 = time.time()
        out = model(**inputs, output_hidden_states=True)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  run {run}: {(time.time() - t0) * 1000:.1f} ms")
dt = time.time() - t0

hidden = out.last_hidden_state
pooled = hidden.mean(dim=1)
print(f"last_hidden:  {tuple(hidden.shape)}  on {hidden.device}")
print(f"n layers:     {len(out.hidden_states)}")
print(f"pooled:       {tuple(pooled.shape)}  norm={pooled.norm().item():.3f}")
print(f"inference:    {dt * 1000:.1f} ms")

if hidden.device.type != "cuda":
    print("FAIL: embedding did not run on GPU", file=sys.stderr)
    sys.exit(1)
print("OK")
