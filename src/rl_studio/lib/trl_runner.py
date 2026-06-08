"""Pluggable TRL GRPO training — model, dataset, and reward all from config.

This is the engine the ``trl`` backend drives, shared by **both** compute targets
so they run identical code:

- ``compute: local`` — the CLI calls :func:`run_grpo` in-process on your GPU
  (needs the ``gpu`` extra).
- ``compute: modal`` — ``scripts/modal_grpo.py`` calls the same :func:`run_grpo`
  inside a rented Modal container.

The pluggable bits, all config-driven (point it at *any* task, no code edits for
the common cases):

- **model** — any HF causal-LM id (``model:``).
- **dataset** — any HF dataset (``dataset:`` / ``dataset_config:`` / ``split:``)
  with the question/answer columns named via ``prompt_column`` / ``answer_column``.
- **reward** — a built-in verifiable reward by name (``reward:``), or your own
  ``reward_fn: path/to/file.py:function``. No reward model — rewards are
  deterministic and checkable (RLVR-style).

Heavy deps (torch/trl/transformers/datasets) are imported *inside* :func:`run_grpo`
so importing this module — and its reward registry — stays light and CPU-only,
which keeps the rewards unit-testable without a GPU.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Reason step by step, then give the final "
    "answer as a single line at the end."
)

# ---------------------------------------------------------------------------
# Built-in verifiable rewards — pure python, no torch. TRL calls a reward with
# `completions=` plus the dataset columns as kwargs; we normalize the gold column
# to `answer` in run_grpo, so every built-in reads `completions` + `answer`.
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _text(completion: Any) -> str:
    """TRL gives chat completions as ``[{"role","content"}]``; plain ones as str."""
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return str(completion)


def _last_number(s: str) -> str | None:
    matches = _NUM.findall(s or "")
    return matches[-1].replace(",", "") if matches else None


def _gold_number(ans: str) -> str | None:
    if ans and "####" in ans:  # GSM8K-style "#### <number>"
        return ans.split("####")[-1].strip().replace(",", "")
    return _last_number(ans)


def numeric_match(completions: list, answer: list, **_: Any) -> list[float]:
    """1.0 when the completion's final number equals the gold final number."""

    def hit(completion: Any, gold: str) -> float:
        pred = _last_number(_text(completion))
        return 1.0 if (pred is not None and pred == _gold_number(gold)) else 0.0

    return [hit(c, g) for c, g in zip(completions, answer, strict=False)]


def exact_match(completions: list, answer: list, **_: Any) -> list[float]:
    """1.0 when the completion equals the gold answer (trimmed)."""
    return [
        1.0 if _text(c).strip() == str(g).strip() else 0.0
        for c, g in zip(completions, answer, strict=False)
    ]


def contains(completions: list, answer: list, **_: Any) -> list[float]:
    """1.0 when the gold answer appears anywhere in the completion."""
    return [
        1.0 if (str(g).strip() and str(g).strip() in _text(c)) else 0.0
        for c, g in zip(completions, answer, strict=False)
    ]


def regex_match(
    completions: list, answer: list | None = None, *, pattern: str | None = None, **_: Any
) -> list[float]:
    """1.0 when the completion matches ``reward_pattern`` (a regex)."""
    rx = re.compile(pattern) if pattern else None
    return [1.0 if (rx and rx.search(_text(c))) else 0.0 for c in completions]


REWARDS: dict[str, Callable] = {
    "numeric_match": numeric_match,
    "exact_match": exact_match,
    "contains": contains,
    "regex_match": regex_match,
}


def load_reward(cfg: dict) -> Callable:
    """Resolve the reward: a custom ``reward_fn: file.py:func``, else a built-in
    by ``reward:`` name (default ``numeric_match``)."""
    spec = cfg.get("reward_fn")
    if spec:
        path_str, _, func_name = spec.partition(":")
        func_name = func_name or "reward"
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"reward_fn file not found: {path}")
        mod_spec = importlib.util.spec_from_file_location("rl_custom_reward", path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, func_name):
            raise AttributeError(f"{path} has no function '{func_name}'")
        return getattr(module, func_name)

    name = cfg.get("reward", "numeric_match")
    if name not in REWARDS:
        raise KeyError(f"unknown reward '{name}'; built-ins: {sorted(REWARDS)} (or set reward_fn)")
    if name == "regex_match":
        pattern = cfg.get("reward_pattern")

        def _regex(completions: list, answer: Any = None, **kw: Any) -> list[float]:
            return regex_match(completions, answer, pattern=pattern, **kw)

        return _regex
    return REWARDS[name]


def _grpo_kwargs(cfg: dict, output_dir: str) -> dict:
    """Desired GRPOConfig kwargs, filtered to the installed TRL's fields so a
    version bump degrades gracefully instead of raising TypeError."""
    import dataclasses

    from trl import GRPOConfig

    desired = {
        "output_dir": output_dir,
        "num_generations": cfg.get("num_generations", 8),
        "learning_rate": float(cfg.get("learning_rate", 1e-6)),
        "beta": float(cfg.get("beta", 0.04)),
        "max_prompt_length": cfg.get("max_prompt_length"),
        "max_completion_length": cfg.get("max_completion_length", 256),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 8),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "num_train_epochs": cfg.get("num_train_epochs", 1),
        "max_steps": cfg.get("max_steps", -1),
        "temperature": cfg.get("temperature", 0.9),
        "bf16": cfg.get("bf16", True),
        "logging_steps": cfg.get("logging_steps", 1),
        "save_steps": cfg.get("save_steps", 100),
        "seed": cfg.get("seed", 0),
        "report_to": [],
    }
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    return {k: v for k, v in desired.items() if k in valid and v is not None}


def run_grpo(cfg: dict, *, output_dir: str = "artifacts/grpo-out") -> dict:
    """Train GRPO on (model, dataset, reward) from ``cfg`` and return metrics.

    Identical on local and Modal compute — only the *caller* differs.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    model_name = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    raw = load_dataset(cfg["dataset"], cfg.get("dataset_config"), split=cfg.get("split", "train"))
    prompt_col = cfg.get("prompt_column", "question")
    answer_col = cfg.get("answer_column", "answer")
    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    def to_prompt(example: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[prompt_col]},
            ],
            "answer": example[answer_col],  # normalize gold to `answer` for rewards
        }

    dataset = raw.map(to_prompt)
    reward = load_reward(cfg)

    from trl import GRPOConfig

    grpo_config = GRPOConfig(**_grpo_kwargs(cfg, output_dir))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)

    log = trainer.state.log_history
    last = log[-1] if log else {}
    return {
        "ok": True,
        "model": model_name,
        "output_dir": output_dir,
        "steps": trainer.state.global_step,
        "final_loss": last.get("loss"),
        "final_reward": last.get("reward"),
        "history": log,
    }
