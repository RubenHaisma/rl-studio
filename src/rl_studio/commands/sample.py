"""``rl-studio sample <name> --n K`` — draw K completions with their rewards.

The qualitative counterpart to ``eval``: see what the learned policy actually
emits, each completion scored by the same deterministic reward fn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from rl_studio.lib import grpo
from rl_studio.output import CliError, emit, fail


def sample(
    name: str = typer.Argument(..., help="project name (artifacts/<name>/policy.npz)"),
    n: int = typer.Option(5, "--n", help="number of completions to sample"),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    seed: int = typer.Option(0, "--seed", help="sampling seed"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        if n < 1:
            raise CliError(f"--n must be >= 1, got {n}")
        policy_path = Path(out) / name / "policy.npz"
        policy = grpo.load_policy(policy_path)
        probs = grpo._softmax(policy.logits)
        rng = np.random.default_rng(seed)
        seqs = grpo.sample(probs, rng, n)
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    completions = []
    for seq in seqs:
        completions.append(
            {
                "sequence": [int(x) for x in seq],
                "sum": int(seq.sum()),
                "reward": round(grpo.reward(seq, policy.target), 4),
                "success": grpo.is_success(seq, policy.target),
            }
        )

    payload = {
        "ok": True,
        "name": name,
        "target": policy.target,
        "n": n,
        "completions": completions,
    }
    lines = [f"[green]sampled[/green] {n} from {name}  (target sum {policy.target})"]
    for c in completions:
        mark = "[green]✓[/green]" if c["success"] else "[yellow]·[/yellow]"
        lines.append(f"  {mark} {c['sequence']}  sum={c['sum']}  reward={c['reward']}")
    emit(payload, json_out=json_out, human="\n".join(lines))
