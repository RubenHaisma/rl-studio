"""Modal CUDA smoke test — prove the rented GPU works before a real run.

Modal gives free monthly credits, so this check is effectively free (~$0.01,
~30-60s). Run it first whenever you touch the GPU path:

    uv sync --extra modal
    uv run modal run scripts/modal_cuda_smoke.py
    uv run modal run scripts/modal_cuda_smoke.py --gpu A10G

What it does: spin up a GPU container, print nvidia-smi, verify torch.cuda sees
the device, run one matmul to prove it computes, then tear down. Same proven
shape as Laava Studio's smoke test.
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("RL_MODAL_GPU", "T4")  # T4 | L4 | A10G | A100 | H100
TORCH_VERSION = os.environ.get("RL_MODAL_TORCH", "2.5.1")
PYTHON_VERSION = os.environ.get("RL_MODAL_PYTHON", "3.11")
TIMEOUT = int(os.environ.get("RL_MODAL_TIMEOUT_S", "300"))

app = modal.App("rl-studio-cuda-smoke")
image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
    f"torch=={TORCH_VERSION}"
)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT)
def smoke() -> str:
    """Run nvidia-smi + torch.cuda checks, return findings as a JSON string."""
    import json
    import subprocess

    out: dict = {"checks": {}}

    try:
        nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        out["checks"]["nvidia_smi"] = {
            "ok": nvidia_smi.returncode == 0,
            "first_line": nvidia_smi.stdout.splitlines()[0] if nvidia_smi.stdout else None,
        }
    except FileNotFoundError:
        out["checks"]["nvidia_smi"] = {"ok": False, "reason": "binary not in image"}

    import torch

    available = torch.cuda.is_available()
    out["checks"]["torch"] = {
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(available),
        "device_name": str(torch.cuda.get_device_name(0)) if available else None,
        "ok": bool(available),
    }

    if available:
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.randn(1024, 1024, device="cuda")
        out["checks"]["matmul"] = {"ok": True, "sum_result": float((x @ y).sum().item())}
    else:
        out["checks"]["matmul"] = {"ok": False, "reason": "no cuda device"}

    out["ok"] = all(c.get("ok", False) for c in out["checks"].values())
    payload = json.dumps(out, indent=2)
    print(payload)
    return payload


@app.local_entrypoint()
def main() -> None:
    import json

    payload = smoke.remote()
    result = json.loads(payload)
    print("\n=== summary ===")
    print(payload)
    print(f"\nimage: torch=={TORCH_VERSION} python={PYTHON_VERSION}  gpu={GPU}  timeout={TIMEOUT}s")
    if not result.get("ok"):
        raise SystemExit("smoke test failed")
