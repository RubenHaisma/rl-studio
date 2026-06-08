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

import pytest
from typer.testing import CliRunner

from rl_studio.cli import app
from rl_studio.lib import backends, grpo, trl_runner
from rl_studio.lib.config import GRPOConfig
from rl_studio.output import CliError

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


# ---------------------------------------------------------------------------
# backend seam: one `train` entry routes to builtin (numpy) or trl (real engine)
# ---------------------------------------------------------------------------
def test_backend_detection():
    assert backends.read_backend("configs/toy-grpo.yaml") == "builtin"
    assert backends.read_backend("configs/grpo-qwen.yaml") == "trl"
    # the --backend flag overrides the config
    assert backends.read_backend("configs/toy-grpo.yaml", override="trl") == "trl"


def test_train_trl_dry_run_shows_plan_without_gpu():
    # --dry-run must work with no GPU and no modal launcher: it just prints the
    # dispatch plan an agent would execute.
    result = runner.invoke(app, ["train", "configs/grpo-qwen.yaml", "--dry-run", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["backend"] == "trl"
    assert payload["dispatch"] == "modal"
    assert "modal run" in payload["would_run"]


def test_train_builtin_dry_run():
    result = runner.invoke(app, ["train", "configs/toy-grpo.yaml", "--dry-run", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["backend"] == "builtin"
    assert payload["dispatch"] == "in-process"


def test_trl_backend_requires_launcher(monkeypatch):
    # Without the `modal` launcher a real (non-dry-run) trl dispatch must fail
    # cleanly, never attempt to run. (Simulate modal absent.)
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(CliError):
        backends.run_trl("configs/grpo-qwen.yaml", dry_run=False)


# ---------------------------------------------------------------------------
# pluggable trl backend: reward registry, custom reward, compute targets
# ---------------------------------------------------------------------------
def test_reward_registry_numeric_match():
    comps = [
        [{"role": "assistant", "content": "let me think ... the answer is 42"}],
        [{"role": "assistant", "content": "7"}],
    ]
    assert trl_runner.numeric_match(comps, ["#### 42", "#### 9"]) == [1.0, 0.0]


def test_reward_registry_exact_and_contains():
    assert trl_runner.exact_match(["hello"], ["hello"]) == [1.0]
    assert trl_runner.contains(["the cat sat"], ["cat"]) == [1.0]
    assert trl_runner.contains(["dog"], ["cat"]) == [0.0]


def test_load_reward_builtin_and_unknown():
    assert trl_runner.load_reward({"reward": "exact_match"}) is trl_runner.exact_match
    with pytest.raises(KeyError):
        trl_runner.load_reward({"reward": "nope"})


def test_load_reward_custom_file():
    fn = trl_runner.load_reward({"reward_fn": "rewards/example_reward.py:reward"})
    scores = fn([[{"role": "assistant", "content": "work it out: the answer is 42"}]], ["#### 42"])
    assert scores[0] >= 1.0  # correct + reasoning bonus


def test_train_trl_local_dry_run():
    result = runner.invoke(app, ["train", "configs/grpo-local.yaml", "--dry-run", "--json"])
    assert result.exit_code == 0, result.stdout
    p = json.loads(result.stdout)
    assert p["backend"] == "trl" and p["compute"] == "local" and p["dispatch"] == "in-process"


def test_train_trl_modal_dry_run_has_pluggable_fields():
    result = runner.invoke(app, ["train", "configs/grpo-qwen.yaml", "--dry-run", "--json"])
    p = json.loads(result.stdout)
    assert p["compute"] == "modal"
    assert p["model"] and p["dataset"] and p["reward"]  # the pluggable knobs surface in the plan


def test_compute_local_requires_gpu_extra(monkeypatch):
    # compute=local without torch/trl must fail cleanly, never half-run.
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(CliError):
        backends.run_trl("configs/grpo-local.yaml", dry_run=False)


def test_unknown_compute_rejected(tmp_path):
    cfg = tmp_path / "x.yaml"
    cfg.write_text("name: x\nbackend: trl\ncompute: martian\nmodel: m\ndataset: d\n")
    with pytest.raises(CliError):
        backends.run_trl(str(cfg), dry_run=True)
