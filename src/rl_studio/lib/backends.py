"""Backend dispatch — one agent-facing CLI, multiple RL engines behind it.

The point of rl-studio is **not** to reimplement RL — TRL, verl, Unsloth already
do that well. The point is to make a real engine *agent-operable*: a stable
``--json`` CLI, load-bearing exit codes, a preflight, and machine-parseable run
results, so Claude Code / Codex can run and iterate fine-tunes autonomously.

``rl-studio train <config>`` reads the config's ``backend`` and routes here,
returning the same result shape either way:

- ``builtin`` — the pure-numpy GRPO loop, in-process. CPU, CI-verified, offline,
  great for learning the mechanics.
- ``trl`` — real GRPO via TRL's ``GRPOTrainer`` on a rented Modal GPU
  (``scripts/modal_grpo.py``). The engine does the training; we dispatch to it
  and normalize the result. Ran-for-real; not CI-tested (needs a GPU).

Roadmap backends (same seam, not yet built): ``verl``, ``unsloth``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from rl_studio.output import CliError

BUILTIN = "builtin"
TRL = "trl"
ROADMAP = {"verl", "unsloth"}  # named so the seam is honest; not implemented

MODAL_SCRIPT = "scripts/modal_grpo.py"
# What the LAUNCHER needs locally. torch/trl/transformers build in the Modal
# container image, never on the user's machine — so launching only needs modal.
TRL_LAUNCH_DEPS = ["modal"]


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        raise CliError(f"config not found: {p}")
    try:
        return yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via CLI
        raise CliError(f"invalid yaml in {p}: {exc}") from exc


def read_backend(config_path: str | Path, override: str | None = None) -> str:
    """Resolve which engine drives this run: the ``--backend`` flag wins, else
    the config's ``backend`` key, else ``builtin``."""
    if override:
        return override
    return _read_yaml(config_path).get("backend", BUILTIN)


def run_trl(config_path: str, *, dry_run: bool) -> dict[str, Any]:
    """Dispatch a real GRPO run to TRL on a Modal GPU and normalize its result.

    With ``dry_run`` it returns the dispatch plan without spending anything — so
    an agent (or CI) can inspect exactly what would run. Without it, it launches
    the Modal job, then reads back ``results/<name>/modal_result.json``.
    """
    name = _read_yaml(config_path).get("name", "grpo")
    command = ["modal", "run", MODAL_SCRIPT, "--config", config_path]
    plan = {
        "ok": True,
        "backend": TRL,
        "name": name,
        "dispatch": "modal",
        "engine": "trl.GRPOTrainer",
        "would_run": " ".join(command),
        "note": "real GRPO on a rented Modal GPU — spends credits, takes minutes",
    }
    if dry_run:
        return plan

    missing = [m for m in TRL_LAUNCH_DEPS if importlib.util.find_spec(m) is None]
    if missing:
        raise CliError(
            f"trl backend needs the 'modal' launcher (missing: {', '.join(missing)}); "
            "install with: uv sync --extra modal  — or pass --dry-run to see the plan",
            code=2,
        )

    # Stream the engine's output to stderr so our stdout stays a clean JSON channel.
    proc = subprocess.run(command, stdout=sys.stderr, stderr=sys.stderr)  # noqa: S603
    if proc.returncode != 0:
        raise CliError(
            f"trl backend run failed (modal exit {proc.returncode})", code=proc.returncode
        )

    result_path = Path("results") / name / "modal_result.json"
    if not result_path.is_file():
        raise CliError(f"trl run finished but no result at {result_path}")
    raw = json.loads(result_path.read_text())
    history = raw.get("history") or []

    def _last(key: str) -> Any:
        # TRL's final log entry is a run summary without per-step reward/loss;
        # walk back to the last step that actually reported the metric.
        for entry in reversed(history):
            if entry.get(key) is not None:
                return entry[key]
        return None

    final_reward = raw.get("final_reward")
    final_loss = raw.get("final_loss")
    return {
        "ok": bool(raw.get("ok", True)),
        "backend": TRL,
        "name": name,
        "model": raw.get("model"),
        "metrics": {
            "final_reward": final_reward if final_reward is not None else _last("reward"),
            "final_loss": final_loss if final_loss is not None else _last("loss"),
            "steps": raw.get("steps"),
        },
        "results_path": str(result_path),
    }
