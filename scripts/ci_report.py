"""Render a metrics JSON (from `<cli> train --json`) as a GitHub-flavored
markdown summary: a metrics table plus a sparkline for any numeric series.

CI pipes this into `$GITHUB_STEP_SUMMARY` so every push posts the *actual
numbers* to the run summary — "CI proves the claim", not just lint + test.
stdlib only.

    uv run mlt train configs/iris.yaml --json | tee metrics.json
    python scripts/ci_report.py metrics.json >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import json
import sys
from typing import Any

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[Any]) -> str:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    span = (hi - lo) or 1.0
    cells = len(_SPARK) - 1
    return "".join(_SPARK[min(cells, int((v - lo) / span * cells))] for v in nums)


def rows(d: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(rows(value, f"{name}."))
        elif isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            out.append((name, f"`{sparkline(value)}`  ({len(value)} pts, last `{value[-1]:.4g}`)"))
        elif isinstance(value, (int, float, str, bool)):
            out.append((name, f"`{value}`"))
    return out


def main() -> None:
    raw = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    data = json.loads(raw)
    payload = data.get("metrics", data) if isinstance(data, dict) else {}
    name = data.get("name") or data.get("model") or "run"

    print(f"### 📊 Pipeline ran in CI: `{name}`\n")
    print("| metric | value |")
    print("| --- | --- |")
    for key, value in rows(payload):
        print(f"| `{key}` | {value} |")
    print("\n_Trained on this commit by the CI runner — numbers above are live, not pasted._")


if __name__ == "__main__":
    main()
