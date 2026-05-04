"""Feed real music files through CLAP and print the full prompt-score matrix."""
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from ai_dj import clap_tag

MODEL_ID = "laion/clap-htsat-unfused"
proc = ClapProcessor.from_pretrained(MODEL_ID)
model = ClapModel.from_pretrained(MODEL_ID).cuda().eval()

TEST_FILES = [
    ("ACDC - Back In Black (rock)",    "/mnt/e/Music/iTunes/iTunes Media/Music/AC_DC/Back In Black/06 Back In Black.mp3"),
    ("AFX - Analogue Bubblebath (electronic)", "/mnt/e/Music/iTunes/iTunes Media/Music/AFX/Analogue Bubblebath [EP]/01 Analogue Bubblebath.m4a"),
    ("Bach - Air from suite no. 3 (classical)", "/mnt/e/Music/iTunes/iTunes Media/Music/Compilations/Kauneinta taidemusiikkia osa 1/03 Air from suite no. 3.m4a"),
    ("Kraftwerk - Computer Love (electronic)", "/mnt/e/Music/iTunes/iTunes Media/Music/Kraftwerk/Computer World/05 Computer Love.m4a"),
    ("Queen - Hammer To Fall (rock)", "/mnt/e/Music/iTunes/iTunes Media/Music/Compilations/Greatest Hits II/14 Hammer To Fall.m4a"),
]

prompts = ["rock music", "heavy metal", "classical music", "an orchestral piece",
           "electronic music", "a techno track", "ambient music", "jazz music",
           "folk music", "hip hop", "pop music"]

for name, path in TEST_FILES:
    try:
        y = clap_tag.load_audio_48k(path, duration_s=None)
    except Exception as e:
        print(f"\n=== {name}\n   load failed: {e}")
        continue
    print(f"\n=== {name}")
    print(f"   audio rms={np.sqrt((y ** 2).mean()):.3f} len={len(y)/48000:.1f}s")
    inputs = proc(text=prompts, audio=[y], sampling_rate=48000, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    sims = out.logits_per_audio[0].cpu().tolist()
    ranked = sorted(zip(prompts, sims), key=lambda x: -x[1])
    for p, s in ranked[:5]:
        print(f"   {s:+.4f}  {p}")
