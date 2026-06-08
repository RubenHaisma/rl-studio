"""LLM GRPO on a rented Modal GPU — the scaffolded path.

This is the *same algorithm* as the verified numpy loop in
``src/rl_studio/lib/grpo.py``, applied to a real language model via TRL's
``GRPOTrainer``: sample a group of completions per prompt, score each with a
verifiable reward, normalize advantages by the group mean/std (no value
network), and update with a KL penalty toward the reference model.

The task is GSM8K with a **verifiable correctness reward**: we extract the
model's final numeric answer and check it against the gold answer — no neural
reward model, mirroring the toy task and real RLVR setups.

It is NOT run in CI: it needs a GPU and a Modal account. Launch it with::

    uv sync --extra gpu
    modal run scripts/modal_grpo.py --config configs/grpo-qwen.yaml

Everything below is real, runnable code, but it has only been verified to import
and to be wired correctly — the actual training run requires rented hardware and
is presented as scaffolding, per the README's "what's verified" matrix.
"""

from __future__ import annotations

import re

import modal

# ---------------------------------------------------------------------------
# Image: the `gpu` extra, baked into a container so the run is reproducible.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.44",
        "trl>=0.10",
        "datasets>=2.20",
        "accelerate>=0.33",
        "mlflow>=2.14",
        "pyyaml>=6.0",
    )
)

app = modal.App("rl-studio-grpo", image=image)

# Persist the HF cache + trained adapters across runs.
volume = modal.Volume.from_name("rl-studio-grpo", create_if_missing=True)
MODEL_DIR = "/vol/output"


# ---------------------------------------------------------------------------
# Verifiable reward: parse the model's final answer, check it against gold.
# This is the LLM analogue of the toy task's deterministic reward fn.
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


@app.function(gpu="A10G", timeout=60 * 60, volumes={"/vol": volume})
def train_grpo(cfg: dict) -> dict:
    """Run TRL GRPOTrainer on the rented GPU and return final metrics."""
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
        report_to=[],  # MLflow logging is wired below via the returned metrics
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
    }


@app.local_entrypoint()
def main(config: str = "configs/grpo-qwen.yaml") -> None:
    cfg = _load_config(config)
    print(f"launching GRPO on Modal: {cfg['model']} / {cfg['dataset']} (gpu={cfg.get('gpu')})")
    result = train_grpo.remote(cfg)
    print(result)
