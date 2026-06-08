"""``rl-studio eval <name>`` — load a trained policy, report success vs baseline.

Reporting the honest random-policy baseline alongside the learned policy is
non-negotiable house style: a metric without a baseline is marketing.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rl_studio.lib import grpo
from rl_studio.output import CliError, emit, fail

EVAL_N = 2000


def eval(  # noqa: A001 - the subcommand name *is* "eval" by design
    name: str = typer.Argument(..., help="project name (artifacts/<name>/policy.npz)"),
    n: int = typer.Option(EVAL_N, "--n", help="number of completions to score"),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        policy_path = Path(out) / name / "policy.npz"
        policy = grpo.load_policy(policy_path)
        learned = grpo.evaluate(policy.logits, policy.target, n=n, seed=policy.seed + 1)
        baseline = grpo.random_baseline(
            policy.seq_len, policy.vocab, policy.target, n=n, seed=policy.seed + 1
        )
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    lift = learned["success_rate"] - baseline["success_rate"]
    payload = {
        "ok": True,
        "name": name,
        "task": {"seq_len": policy.seq_len, "vocab": policy.vocab, "target": policy.target},
        "n": n,
        "metrics": {
            "success_rate": round(learned["success_rate"], 4),
            "mean_reward": round(learned["mean_reward"], 4),
            "greedy_sum": learned["greedy_sum"],
            "greedy_success": learned["greedy_success"],
            "baseline_success_rate": round(baseline["success_rate"], 4),
            "lift_over_baseline": round(lift, 4),
        },
    }
    m = payload["metrics"]
    human = (
        f"[green]eval[/green] {name}  (target sum {policy.target}, "
        f"{policy.seq_len} digits in [0,{policy.vocab}))\n"
        f"  success_rate {m['success_rate']}  "
        f"(random baseline {m['baseline_success_rate']}, lift {m['lift_over_baseline']:+})\n"
        f"  greedy decode sums to {m['greedy_sum']} "
        f"({'hit' if m['greedy_success'] else 'miss'})"
    )
    emit(payload, json_out=json_out, human=human)
