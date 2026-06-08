"""Output contract shared by every command.

Two rules, enforced everywhere:

1. ``--json`` prints a single JSON object to stdout and nothing else.
2. Exit codes are load-bearing: ``0`` success, non-zero failure with one
   human-readable line on stderr.

Commands call :func:`emit` for success payloads and raise :class:`CliError`
for failures. They never ``print`` ad-hoc.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import typer
from rich.console import Console

_console = Console()
_err = Console(stderr=True)


class CliError(Exception):
    """A failure that maps to a non-zero exit code and one stderr line."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def emit(payload: dict[str, Any], *, json_out: bool, human: str | None = None) -> None:
    """Emit a success payload.

    With ``json_out`` the payload is the entire stdout. Otherwise a friendly
    ``human`` string (or a pretty dump of the payload) is printed.
    """
    if json_out:
        sys.stdout.write(json.dumps(payload, default=str) + "\n")
        return
    if human is not None:
        _console.print(human)
    else:
        _console.print_json(data=payload)


def fail(err: CliError, *, json_out: bool) -> None:
    """Render a failure and exit non-zero.

    In JSON mode the error is still a single JSON object on stdout so an agent
    parsing stdout never has to special-case the error path.
    """
    if json_out:
        sys.stdout.write(json.dumps({"ok": False, "error": err.message}) + "\n")
    else:
        _err.print(f"[red]error:[/red] {err.message}")
    raise typer.Exit(code=err.code)
