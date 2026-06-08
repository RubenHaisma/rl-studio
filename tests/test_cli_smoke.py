"""Smoke tests — the contract CI enforces on every push.

Two layers:

1. **Shell tests** — does the CLI run, is ``--json`` valid JSON, are exit codes
   load-bearing, does the full train -> eval -> sample loop work on CPU.
2. **Algorithm tests** — the GRPO loop must *demonstrably learn*: mean reward
   rises over training and the learned policy beats the random baseline by a
   wide margin. If this fails, fix the algorithm — never weaken the assertion.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from rl_studio.cli import app
from rl_studio.lib import grpo
from rl_studio.lib.config import GRPOConfig

runner = CliRunner()


# ---------------------------------------------------------------------------
# shell contract
# ---------------------------------------------------------------------------
def test_version_json():
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["version"]


def test_doctor_json_is_valid_json():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "ok" in payload and "checks" in payload


def test_bad_config_exits_nonzero():
    result = runner.invoke(app, ["train", "does-not-exist.yaml", "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_unreachable_target_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: bad\nseq_len: 2\nvocab: 10\ntarget: 999\n")
    result = runner.invoke(app, ["train", str(cfg), "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_gpu_train_fails_cleanly_without_extra():
    # The `gpu` extra (torch/trl/...) is not installed in the dev/CI env, so
    # this must fail with a non-zero exit and a one-line JSON error — never a
    # stack trace, never a silent success.
    result = runner.invoke(app, ["gpu-train", "configs/grpo-qwen.yaml", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "gpu" in payload["error"]


# ---------------------------------------------------------------------------
# the GRPO algorithm actually learns (the credible bit)
# ---------------------------------------------------------------------------
def test_grpo_learns_beats_baseline():
    cfg = GRPOConfig.from_yaml("configs/toy-grpo.yaml")
    result = grpo.train(cfg)

    first = result.history["mean_reward"][0]
    final = result.final_mean_reward
    # Reward must rise substantially over training.
    assert final > first + 0.3, f"reward did not rise: {first:.3f} -> {final:.3f}"
    assert final > 0.8, f"policy did not converge: final mean reward {final:.3f}"

    baseline = grpo.random_baseline(cfg.seq_len, cfg.vocab, cfg.target, n=2000, seed=cfg.seed)
    # The strict success rate must clear the random baseline by a wide margin.
    assert result.final_success_rate > baseline["success_rate"] + 0.3, (
        f"trained {result.final_success_rate:.3f} vs baseline {baseline['success_rate']:.3f}"
    )


def test_grpo_kl_is_tracked_and_responsive():
    # A higher KL penalty must measurably hold the policy closer to the
    # reference — proves the KL term is real, not decorative.
    import dataclasses

    base = GRPOConfig.from_yaml("configs/toy-grpo.yaml")
    loose = grpo.train(dataclasses.replace(base, kl_coef=0.0))
    tight = grpo.train(dataclasses.replace(base, kl_coef=0.2))
    assert tight.history["kl"][-1] < loose.history["kl"][-1]


# ---------------------------------------------------------------------------
# full CLI loop: train -> eval -> sample
# ---------------------------------------------------------------------------
def test_train_eval_sample_loop(tmp_path):
    out = str(tmp_path / "artifacts")

    trained = runner.invoke(app, ["train", "configs/toy-grpo.yaml", "--out", out, "--json"])
    assert trained.exit_code == 0, trained.stdout
    tpayload = json.loads(trained.stdout)
    assert tpayload["ok"] is True
    # the learned policy must beat the random baseline
    assert tpayload["metrics"]["lift_over_baseline"] > 0
    assert tpayload["metrics"]["final_mean_reward"] > tpayload["metrics"]["first_mean_reward"]

    evaled = runner.invoke(app, ["eval", "digit-sum", "--out", out, "--n", "1000", "--json"])
    assert evaled.exit_code == 0, evaled.stdout
    epayload = json.loads(evaled.stdout)
    assert epayload["ok"] is True
    assert epayload["metrics"]["lift_over_baseline"] > 0

    sampled = runner.invoke(app, ["sample", "digit-sum", "--out", out, "--n", "5", "--json"])
    assert sampled.exit_code == 0, sampled.stdout
    spayload = json.loads(sampled.stdout)
    assert spayload["ok"] is True
    assert len(spayload["completions"]) == 5
    for c in spayload["completions"]:
        assert c["sum"] == sum(c["sequence"])


def test_eval_missing_policy_exits_nonzero(tmp_path):
    out = str(tmp_path / "artifacts")
    result = runner.invoke(app, ["eval", "nope", "--out", out, "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False
