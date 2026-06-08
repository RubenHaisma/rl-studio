"""Run the `trl` backend on a rented Modal GPU — a thin wrapper over the shared
engine in ``rl_studio.lib.trl_runner``.

The pluggable model/dataset/reward logic lives in ``trl_runner`` so local and
Modal compute run *identical* code. This file only does the Modal-specific bits:
build a GPU container, mount the package + a ``rewards/`` dir, route the HF cache
to a volume, and hand the config to ``run_grpo``.

Auth is machine-level (``~/.modal.toml``); no key in ``.env``. Launch via the CLI:

    rl-studio train configs/grpo-qwen.yaml --json        # backend: trl, compute: modal

or directly:

    uv run modal run scripts/modal_grpo.py --config configs/grpo-qwen.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# GPU stack baked into the container; the local rl_studio source (the shared
# engine) and any rewards/ dir (custom reward_fn files) are mounted in.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.3",
    "transformers>=4.44",
    "trl>=0.10",
    "datasets>=2.20",
    "accelerate>=0.33",
    "pyyaml>=6.0",
)
image = image.add_local_python_source("rl_studio")
if Path("rewards").is_dir():  # so reward_fn: rewards/<file>.py works on Modal too
    image = image.add_local_dir("rewards", remote_path="/root/rewards")

app = modal.App("rl-studio-grpo", image=image)

GPU = os.environ.get("RL_MODAL_GPU", "A10G")  # T4 | L4 | A10G | A100 | H100
TIMEOUT_S = int(os.environ.get("RL_MODAL_TIMEOUT_S", str(60 * 60)))

volume = modal.Volume.from_name("rl-studio-grpo", create_if_missing=True)
MODEL_DIR = "/vol/output"

_secrets = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if os.environ.get("HF_TOKEN")
    else []
)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={"/vol": volume}, secrets=_secrets)
def train_grpo(cfg: dict) -> dict:
    """Route the HF cache to the volume, then run the shared engine."""
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/vol/hf/hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from rl_studio.lib.trl_runner import run_grpo

    result = run_grpo(cfg, output_dir=MODEL_DIR)
    volume.commit()
    return result


@app.local_entrypoint()
def main(config: str = "configs/grpo-qwen.yaml") -> None:
    import json

    import yaml

    cfg = yaml.safe_load(Path(config).read_text())
    print(f"launching GRPO on Modal: {cfg['model']} / {cfg['dataset']} (gpu={GPU})")
    result = train_grpo.remote(cfg)

    out = Path("results") / cfg.get("name", "grpo")
    out.mkdir(parents=True, exist_ok=True)
    (out / "modal_result.json").write_text(json.dumps(result, indent=2, default=str))
    history = result.get("history") or []
    curve = [
        {"step": e.get("step"), "reward": e.get("reward"), "loss": e.get("loss")}
        for e in history
        if "reward" in e or "loss" in e
    ]
    (out / "reward_curve.json").write_text(json.dumps(curve, indent=2))
    print(f"\nfinal reward {result.get('final_reward')} over {result.get('steps')} steps")
    print(f"reward curve ({len(curve)} points) -> {out / 'reward_curve.json'}")
    if not result.get("ok"):
        raise SystemExit("GRPO training failed")
