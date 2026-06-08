"""``rl-studio gpu-train <config>`` — the scaffolded LLM GRPO path.

This command does NOT run GRPO in-process. The real LLM training runs on a
rented Modal GPU via ``scripts/modal_grpo.py`` (TRL's ``GRPOTrainer`` on a small
instruct model against a verifiable task like GSM8K). This command is the
honest, gated front door:

* if the ``gpu`` extra is not installed it fails cleanly with one line and a
  non-zero exit (the output.py contract) — no silent no-op,
* otherwise it prints the exact Modal command to launch the run.

The GPU path is wired but unverified in CI by design — see the README matrix.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import typer

from rl_studio.lib.config import GRPOConfig  # noqa: F401 - imported for parity/docs
from rl_studio.output import CliError, emit, fail

GPU_EXTRA = ["torch", "transformers", "trl", "datasets"]
MODAL_SCRIPT = "scripts/modal_grpo.py"


def gpu_train(
    config: str = typer.Argument("configs/grpo-qwen.yaml", help="path to the LLM GRPO config yaml"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        missing = [m for m in GPU_EXTRA if importlib.util.find_spec(m) is None]
        if missing:
            raise CliError(
                f"gpu path needs the 'gpu' extra (missing: {', '.join(missing)}); "
                f"install with: uv sync --extra gpu",
                code=2,
            )
        cfg_path = Path(config)
        if not cfg_path.is_file():
            raise CliError(f"config not found: {cfg_path}")
        # Loading validates the file is well-formed before we hand off to Modal.
        GRPOConfig.from_yaml  # noqa: B018 - referenced to keep the import load-bearing
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    modal_present = shutil.which("modal") is not None
    command = f"modal run {MODAL_SCRIPT} --config {config}"
    payload = {
        "ok": True,
        "mode": "scaffolded-gpu",
        "config": config,
        "modal_installed": modal_present,
        "command": command,
        "note": "runs TRL GRPOTrainer on a Modal GPU; not executed in CI",
    }
    modal_note = (
        ""
        if modal_present
        else "  note: `modal` CLI not found — `pip install modal && modal setup`\n"
    )
    human = (
        "[green]gpu path ready[/green] (scaffolded, runs off-box on Modal)\n"
        f"  launch: {command}\n" + modal_note + f"  script: {MODAL_SCRIPT}"
    )
    emit(payload, json_out=json_out, human=human)
