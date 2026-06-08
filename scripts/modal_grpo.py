"""LLM GRPO on a rented Modal GPU — the same algorithm as the verified numpy loop.

This applies the GRPO loop from ``src/rl_studio/lib/grpo.py`` to a real language
model via TRL's ``GRPOTrainer``: sample a group of completions per prompt, score
each with a verifiable reward, normalize advantages by the group mean/std (no
value network), and update with a KL penalty toward the reference model.

The task is GSM8K with a **verifiable correctness reward**: parse the model's
final numeric answer and check it against the gold answer — no neural reward
model, mirroring the toy task and real RLVR setups.

Setup follows the proven Modal pattern from Laava Studio:

- **Auth is machine-level.** ``modal`` reads ``~/.modal.toml`` (set once via
  ``modal token set``). There is no Modal key in ``.env``.
- **No HF token is required** for the default model (Qwen2.5-0.5B is open) or
  dataset (GSM8K is public). If you later target a gated model, put ``HF_TOKEN``
  in ``.env`` and it is forwarded into the container as a Modal Secret.
- **The HF cache lives on a Modal Volume**, so re-runs don't re-download weights.
- **Knobs are env-overridable**, so the same script does a cheap smoke or a full
  run without edits.

Run it::

    uv sync --extra modal                                   # light: just the launcher
    uv run modal run scripts/modal_cuda_smoke.py            # ~$0.01 GPU sanity check
    uv run modal run scripts/modal_grpo.py --config configs/grpo-qwen-smoke.yaml  # cheap
    uv run modal run scripts/modal_grpo.py --config configs/grpo-qwen.yaml        # full

The full stack (torch/trl/transformers/...) is built into the container image
below, not installed locally — launching only needs the ``modal`` extra.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import modal


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Lets HF_TOKEN / RL_MODAL_* knobs come
    from a gitignored .env. Runs at import time so the values are visible when the
    Modal Secret + GPU decorator are constructed below. No-op if the file is
    absent (e.g. inside the container)."""
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

# ---------------------------------------------------------------------------
# Image: the GPU stack baked into the container so the run is reproducible and
# nothing heavy is installed on the laptop that launches it.
# ---------------------------------------------------------------------------
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.3",
    "transformers>=4.44",
    "trl>=0.10",
    "datasets>=2.20",
    "accelerate>=0.33",
    "pyyaml>=6.0",
)

app = modal.App("rl-studio-grpo", image=image)

# Env-overridable knobs (Laava Studio pattern). The config file sets defaults;
# these let you swap GPU/timeout for a one-off run without editing anything.
GPU = os.environ.get("RL_MODAL_GPU", "A10G")  # T4 | L4 | A10G | A100 | H100
TIMEOUT_S = int(os.environ.get("RL_MODAL_TIMEOUT_S", str(60 * 60)))

# Persist the HF cache + trained adapters across runs.
volume = modal.Volume.from_name("rl-studio-grpo", create_if_missing=True)
MODEL_DIR = "/vol/output"

# Forward HF_TOKEN as a Modal Secret only when one is set locally (gated models).
# Not needed for the open default model — kept so a .env key "just works" later.
_secrets = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if os.environ.get("HF_TOKEN")
    else []
)


# ---------------------------------------------------------------------------
# Verifiable reward: parse the model's final answer, check it against gold.
# The LLM analogue of the toy task's deterministic reward fn.
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> str | None:
    matches = _NUM.findall(text)
    return matches[-1].replace(",", "") if matches else None


def _gold_answer(answer_field: str) -> str | None:
    # GSM8K gold answers end with "#### <number>".
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return _last_number(answer_field)


def correctness_reward(completions, answer, **_) -> list[float]:
    """TRL reward signature: one scalar per completion. 1.0 if the parsed final
    answer matches the gold answer, else 0.0 — verifiable, no reward model."""
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        pred = _last_number(text)
        target = _gold_answer(gold)
        rewards.append(1.0 if (pred is not None and pred == target) else 0.0)
    return rewards


SYSTEM_PROMPT = (
    "You are a careful math tutor. Reason step by step, then give the final "
    "answer as a single number on the last line."
)


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={"/vol": volume}, secrets=_secrets)
def train_grpo(cfg: dict) -> dict:
    """Run TRL GRPOTrainer on the rented GPU and return final metrics + curve."""
    # Route the HF cache to the persistent volume so re-runs skip the download.
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/vol/hf/hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    raw = load_dataset(cfg["dataset"], cfg.get("dataset_config", "main"), split=cfg["split"])

    def to_prompt(example: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    dataset = raw.map(to_prompt)

    grpo_config = GRPOConfig(
        output_dir=MODEL_DIR,
        num_generations=cfg["num_generations"],  # the group G
        learning_rate=float(cfg["learning_rate"]),
        beta=float(cfg["beta"]),  # KL coefficient toward the reference model
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),
        temperature=cfg["temperature"],
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        report_to=[],  # metrics returned to the launcher + written under results/
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=correctness_reward,
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(MODEL_DIR)
    volume.commit()

    log = trainer.state.log_history
    last = log[-1] if log else {}
    return {
        "ok": True,
        "model": model_name,
        "output_dir": MODEL_DIR,
        "final_loss": last.get("loss"),
        "final_reward": last.get("reward"),
        "steps": trainer.state.global_step,
        "history": log,  # full per-step log — the reward curve lives here
    }


@app.local_entrypoint()
def main(config: str = "configs/grpo-qwen.yaml") -> None:
    import json

    cfg = _load_config(config)
    print(
        f"launching GRPO on Modal: {cfg['model']} / {cfg['dataset']} "
        f"(gpu={GPU}, max_steps={cfg.get('max_steps')})"
    )
    result = train_grpo.remote(cfg)

    # Persist the committable artifact: the reward curve from a real LLM run.
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
