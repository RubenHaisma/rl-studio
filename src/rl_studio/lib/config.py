"""Typed config loading. One YAML file in, one validated dataclass out.

Kept deliberately small: a dataclass per config shape, a loader that fails
loud with a :class:`CliError` instead of a stack trace. Mirrors the template's
``config.py`` shape, swapped for the GRPO toy-task hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_studio.output import CliError


@dataclass(slots=True)
class GRPOConfig:
    """Hyperparameters for the numpy GRPO toy loop, resolved from YAML + defaults.

    The task is verifiable-reward: emit a length ``seq_len`` sequence over a
    vocab of ``vocab`` digits (0..vocab-1) whose elements sum to ``target``.
    Reward is deterministic and checkable — no neural reward model.
    """

    name: str
    task: str = "digit_sum"  # the only verifiable task this repo ships
    seq_len: int = 4
    vocab: int = 10
    target: int = 18
    group_size: int = 16  # G completions sampled per step (the GRPO group)
    steps: int = 200
    lr: float = 0.5
    kl_coef: float = 0.02
    adv_eps: float = 1e-4  # numerical floor on the group-std normalizer
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfig:
        p = Path(path)
        if not p.is_file():
            raise CliError(f"config not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - exercised via CLI
            raise CliError(f"invalid yaml in {p}: {exc}") from exc
        if "name" not in raw:
            raise CliError(f"config {p} is missing required key: name")
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject configs the task can never satisfy, with a one-line reason."""
        if self.task != "digit_sum":
            raise CliError(
                f"this repo ships the verifiable 'digit_sum' task only, got: {self.task}"
            )
        if self.seq_len < 1 or self.vocab < 2:
            raise CliError("seq_len must be >=1 and vocab must be >=2")
        if self.group_size < 2:
            raise CliError("group_size must be >=2 (GRPO needs a group to normalize over)")
        max_sum = self.seq_len * (self.vocab - 1)
        if not 0 <= self.target <= max_sum:
            raise CliError(
                f"target {self.target} is unreachable: must be in [0, {max_sum}] "
                f"for seq_len={self.seq_len}, vocab={self.vocab}"
            )

    def as_params(self) -> dict[str, Any]:
        """Flat dict for MLflow ``log_params``."""
        return {
            "task": self.task,
            "seq_len": self.seq_len,
            "vocab": self.vocab,
            "target": self.target,
            "group_size": self.group_size,
            "steps": self.steps,
            "lr": self.lr,
            "kl_coef": self.kl_coef,
            "seed": self.seed,
        }
