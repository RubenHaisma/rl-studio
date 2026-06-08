"""``rl-studio train <config>`` — the one agent-facing entry to RL fine-tuning.

Reads the config's ``backend`` and routes to the engine, returning the same JSON
shape either way:

- ``builtin`` — the numpy GRPO loop in-process (CPU, CI-verified).
- ``trl`` — real GRPO via TRL on a rented Modal GPU (``scripts/modal_grpo.py``).

This is the combine: rl-studio doesn't reimplement RL, it makes real engines
agent-operable behind a stable ``--json`` CLI. ``--dry-run`` prints the dispatch
plan without spending anything.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rl_studio.lib import backends, grpo, tracking
from rl_studio.lib.config import GRPOConfig
from rl_studio.output import CliError, emit, fail

# Sample size for the post-training eval / baseline (deterministic, seeded).
EVAL_N = 2000


def train(
    config: str = typer.Argument(..., help="path to a GRPO config yaml"),
    out: str = typer.Option("artifacts", "--out", help="where to write the policy (builtin)"),
    backend: str = typer.Option(None, "--backend", help="override config backend: builtin|trl"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the dispatch plan, don't run"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        which = backends.read_backend(config, override=backend)
        if which == backends.TRL:
            _emit_trl(backends.run_trl(config, dry_run=dry_run), json_out=json_out, dry_run=dry_run)
            return
        if which in backends.ROADMAP:
            raise CliError(f"backend '{which}' is on the roadmap, not implemented yet", code=2)
        if which != backends.BUILTIN:
            raise CliError(f"unknown backend '{which}' (known: builtin, trl)")
        _train_builtin(config, out, dry_run=dry_run, json_out=json_out)
    except CliError as exc:
        fail(exc, json_out=json_out)


def _emit_trl(result: dict, *, json_out: bool, dry_run: bool) -> None:
    if dry_run:
        human = (
            f"[green]plan[/green] trl · compute={result['compute']} → {result['dispatch']}\n"
            f"  engine {result['engine']} · model {result.get('model')}\n"
            f"  dataset {result.get('dataset')} · reward {result.get('reward')}\n"
            f"  would run: {result['would_run']}\n"
            f"  {result['note']}"
        )
    else:
        m = result["metrics"]
        human = (
            f"[green]trained[/green] {result['name']} via TRL "
            f"({result.get('compute')}, {result.get('model')})\n"
            f"  final reward {m['final_reward']} over {m['steps']} steps\n"
            f"  results -> {result['results_path']}"
        )
    emit(result, json_out=json_out, human=human)


def _train_builtin(config: str, out: str, *, dry_run: bool, json_out: bool) -> None:
    cfg = GRPOConfig.from_yaml(config)
    if dry_run:
        plan = {
            "ok": True,
            "backend": "builtin",
            "name": cfg.name,
            "dispatch": "in-process",
            "would_run": f"numpy GRPO — {cfg.steps} steps, group {cfg.group_size}",
        }
        emit(
            plan,
            json_out=json_out,
            human=f"[green]plan[/green] builtin numpy GRPO: {plan['would_run']}",
        )
        return

    out_dir = Path(out) / cfg.name
    result = grpo.train(cfg)
    policy_path = out_dir / "policy.npz"
    grpo.save_policy(policy_path, result.logits, cfg)
    baseline = grpo.random_baseline(cfg.seq_len, cfg.vocab, cfg.target, n=EVAL_N, seed=cfg.seed)

    with tracking.run(experiment=cfg.name, run_name="grpo"):
        tracking.log_params(cfg.as_params())
        # Per-step curves (the proof it learns) go in as metric series.
        for step, (r, kl, sr) in enumerate(
            zip(
                result.history["mean_reward"],
                result.history["kl"],
                result.history["success_rate"],
                strict=True,
            )
        ):
            tracking.log_metric_step("mean_reward", r, step)
            tracking.log_metric_step("kl", kl, step)
            tracking.log_metric_step("success_rate", sr, step)
        tracking.log_metrics(
            {
                "final_mean_reward": result.final_mean_reward,
                "final_success_rate": result.final_success_rate,
                "baseline_success_rate": baseline["success_rate"],
                "lift_over_baseline": result.final_success_rate - baseline["success_rate"],
            }
        )

    first_reward = result.history["mean_reward"][0]
    payload = {
        "ok": True,
        "backend": "builtin",
        "name": cfg.name,
        "policy_path": str(policy_path),
        "steps": cfg.steps,
        "group_size": cfg.group_size,
        "metrics": {
            "first_mean_reward": round(first_reward, 4),
            "final_mean_reward": round(result.final_mean_reward, 4),
            "final_success_rate": round(result.final_success_rate, 4),
            "final_kl": round(result.history["kl"][-1], 4),
            "baseline_success_rate": round(baseline["success_rate"], 4),
            "lift_over_baseline": round(result.final_success_rate - baseline["success_rate"], 4),
        },
    }
    m = payload["metrics"]
    human = (
        f"[green]trained[/green] {cfg.name} (GRPO, {cfg.steps} steps, group={cfg.group_size})\n"
        f"  mean_reward {m['first_mean_reward']} -> {m['final_mean_reward']}  "
        f"(KL {m['final_kl']})\n"
        f"  success_rate {m['final_success_rate']}  "
        f"(random baseline {m['baseline_success_rate']}, lift {m['lift_over_baseline']:+})\n"
        f"  policy -> {policy_path}"
    )
    emit(payload, json_out=json_out, human=human)
