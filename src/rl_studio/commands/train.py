"""``rl-studio train <config>`` — run the numpy GRPO toy loop.

Logs the per-step reward/KL curve to MLflow and writes the learned policy to
``artifacts/<name>/policy.npz``. Reports the final mean reward and strict
success rate against an honest random-policy baseline.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rl_studio.lib import grpo, tracking
from rl_studio.lib.config import GRPOConfig
from rl_studio.output import CliError, emit, fail

# Sample size for the post-training eval / baseline (deterministic, seeded).
EVAL_N = 2000


def train(
    config: str = typer.Argument(..., help="path to a GRPO config yaml"),
    out: str = typer.Option("artifacts", "--out", help="where to write the policy"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        cfg = GRPOConfig.from_yaml(config)
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
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    first_reward = result.history["mean_reward"][0]
    payload = {
        "ok": True,
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
