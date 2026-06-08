"""Example custom reward — the escape hatch for tasks the built-ins don't cover.

Point a config at it with:

    reward_fn: rewards/example_reward.py:reward

A reward function receives TRL's ``completions`` plus the dataset columns as
keyword args (the gold column is normalized to ``answer``), and returns one float
per completion. It must be **verifiable**: deterministic and checkable, no neural
reward model. This one rewards a correct final number *and* a bit of visible
reasoning — the kind of shaping you'd write for a real task.
"""

from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return str(completion)


def _last_number(s: str) -> str | None:
    matches = _NUM.findall(s or "")
    return matches[-1].replace(",", "") if matches else None


def _gold(ans: str) -> str | None:
    return ans.split("####")[-1].strip().replace(",", "") if ans and "####" in ans else _last_number(ans)


def reward(completions: list, answer: list, **_: Any) -> list[float]:
    """1.0 for a correct final answer, +0.1 bonus for showing >20 chars of work."""
    out: list[float] = []
    for completion, gold in zip(completions, answer, strict=False):
        text = _text(completion)
        correct = _last_number(text) is not None and _last_number(text) == _gold(gold)
        score = 1.0 if correct else 0.0
        if correct and len(text.strip()) > 20:  # reward showing the reasoning
            score += 0.1
        out.append(score)
    return out
