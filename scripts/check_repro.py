"""Run a training command twice and assert it yields identical metrics — proof
the pipeline is deterministic (seeded), not accidentally reproducible.

A seed is a promise; this checks the promise holds. stdlib only.

    python scripts/check_repro.py -- uv run mlt train configs/iris.yaml
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def run_once(cmd: list[str]) -> dict[str, Any]:
    if "--json" not in cmd:
        cmd = [*cmd, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout)
        sys.exit(proc.returncode)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])  # the JSON payload is the last stdout line


def metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("metrics", payload)


def main() -> None:
    if "--" not in sys.argv:
        print("usage: check_repro.py -- <train command ...>")
        sys.exit(2)
    cmd = sys.argv[sys.argv.index("--") + 1 :]

    first, second = metrics(run_once(cmd)), metrics(run_once(cmd))
    if first == second:
        print("reproducible — identical metrics across two independent runs:")
        print(f"  {json.dumps(first)}")
        return
    print("NOT reproducible — metrics differ between runs:")
    print(f"  run 1: {json.dumps(first)}")
    print(f"  run 2: {json.dumps(second)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
