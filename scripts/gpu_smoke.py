"""Minimal GPU smoke test — verify PyTorch sees the RX 7800 XT via WSL ROCm."""
import sys

import torch

print(f"torch:          {torch.__version__}")
print(f"hip version:    {getattr(torch.version, 'hip', None)}")
print(f"cuda.is_available: {torch.cuda.is_available()}")
print(f"device_count:   {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("FAIL: no GPU visible to PyTorch", file=sys.stderr)
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = a @ b
torch.cuda.synchronize()
print(f"matmul result: shape={tuple(c.shape)} dtype={c.dtype} sum={c.sum().item():.2f}")
print("OK")
