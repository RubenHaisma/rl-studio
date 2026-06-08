"""Run the README commands marked `<!-- ci-test -->`, asserting each exits 0.

Keeps the docs honest: a quickstart that drifts from the code breaks CI instead
of silently lying to readers. Opt-in per block (so service-dependent or
illustrative commands aren't run). stdlib only.

Mark a block by putting the comment on the line before its fence:

    <!-- ci-test -->
    ```bash
    uv run mlt doctor
    ```
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARK = "<!-- ci-test -->"
_BLOCK = re.compile(re.escape(MARK) + r"\s*\n```(?:bash|sh|console)?\n(.*?)```", re.S)


def main() -> None:
    readme = Path("README.md")
    if not readme.is_file():
        print("no README.md — nothing to verify")
        return

    blocks = _BLOCK.findall(readme.read_text(encoding="utf-8"))
    if not blocks:
        print(f"no {MARK} blocks found — nothing to verify")
        return

    ran = 0
    for block in blocks:
        for line in block.splitlines():
            cmd = line.strip()
            if not cmd or cmd.startswith("#"):
                continue
            print(f"$ {cmd}", flush=True)
            result = subprocess.run(cmd, shell=True)  # noqa: S602 - trusted README content
            if result.returncode != 0:
                print(f"\nREADME command failed (exit {result.returncode}): {cmd}")
                sys.exit(1)
            ran += 1
    print(f"\nok — {ran} README command(s) still run clean")


if __name__ == "__main__":
    main()
