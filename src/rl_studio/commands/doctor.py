"""``rl-studio doctor`` — the canonical "does this environment work?" check.

Drive it from CI, an agent loop, or a fresh checkout. Exits non-zero if any
hard requirement is missing so it composes in a shell ``&&`` chain. The GPU
extra is reported as informational only — the verified numpy path never needs it.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys

import typer

from rl_studio.lib.tracking import tracking_uri
from rl_studio.output import CliError, emit, fail

# Hard requirements for the verified (numpy) path.
REQUIRED = ["numpy", "mlflow", "yaml"]
# Optional heavy stack for the scaffolded LLM GRPO path (informational).
GPU_EXTRA = ["torch", "transformers", "trl", "datasets"]


def doctor(json_out: bool = typer.Option(False, "--json", help="machine-readable output")) -> None:
    checks: dict[str, object] = {}

    checks["python"] = sys.version.split()[0]
    checks["python_ok"] = sys.version_info >= (3, 11)

    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    checks["deps_missing"] = missing
    checks["deps_ok"] = not missing

    gpu_missing = [m for m in GPU_EXTRA if importlib.util.find_spec(m) is None]
    checks["gpu_extra_installed"] = not gpu_missing
    checks["gpu_extra_missing"] = gpu_missing

    checks["docker"] = shutil.which("docker") is not None
    checks["modal"] = shutil.which("modal") is not None
    checks["tracking_uri"] = tracking_uri()

    ok = bool(checks["python_ok"]) and bool(checks["deps_ok"])
    payload = {"ok": ok, "checks": checks}

    if not ok:
        why = "python<3.11" if not checks["python_ok"] else f"missing deps: {missing}"
        fail(CliError(f"environment not ready: {why}"), json_out=json_out)

    human = "[green]ok[/green] — environment ready (verified numpy GRPO path)\n" + "\n".join(
        f"  {k}: {v}" for k, v in checks.items()
    )
    emit(payload, json_out=json_out, human=human)
