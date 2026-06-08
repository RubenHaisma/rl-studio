# AGENTS.md

Agent instructions for **rl-studio**, in the cross-tool [AGENTS.md](https://agents.md) format — read natively by Codex, Cursor, GitHub Copilot, Windsurf, Amp, Devin, and others. Claude Code reads `CLAUDE.md`, which is a **symlink to this file**, so there is a single source of truth for every tool. Humans: see [README.md](README.md).

## What this repo is

An **agent-operable harness for GRPO fine-tuning**. It does not reimplement RL — it makes a real engine drivable by a coding agent through one `--json` CLI. `rl-studio train <config>` routes by the config's `backend`:

1. **`builtin`** (numpy, CPU, CI-verified): `src/rl_studio/lib/grpo.py` — a real, readable GRPO loop on a verifiable-reward toy task that demonstrably learns.
2. **`trl`** (real LLM, GPU): dispatches `scripts/modal_grpo.py` — TRL `GRPOTrainer` on Qwen2.5-0.5B / GSM8K on a rented Modal GPU (ran for real, reward 0.26→0.48). Returns the **same JSON shape** as builtin.

Roadmap backends (same seam): `verl`, `unsloth`. The dispatch layer is `src/rl_studio/lib/backends.py`.

## Driving the CLI as an agent

Every command is **non-interactive**, takes **`--json`**, and uses **load-bearing exit codes** — drive the whole loop and parse results with no TTY, no UI, no service.

```bash
rl-studio doctor --json
# -> {"ok": true, "checks": {... "modal": true, "gpu_extra_installed": false}}

rl-studio train configs/toy-grpo.yaml --json            # backend: builtin (CPU, in-process)
# -> {"ok": true, "backend": "builtin", "name": "digit-sum", "metrics": {"final_success_rate": 0.875, "lift_over_baseline": 0.809, ...}}

rl-studio train configs/grpo-qwen.yaml --dry-run --json # backend: trl — inspect the plan, NO spend
# -> {"ok": true, "backend": "trl", "dispatch": "modal", "would_run": "modal run scripts/modal_grpo.py --config ...", ...}

rl-studio train configs/grpo-qwen.yaml --json           # backend: trl — launches TRL on a Modal GPU
# -> {"ok": true, "backend": "trl", "metrics": {"final_reward": ..., "steps": ...}, "results_path": "results/..."}

rl-studio eval digit-sum --json      # success rate vs random baseline (builtin policies)
rl-studio sample digit-sum --n 5 --json
```

**Backend routing:** `train` reads `backend:` from the config (`builtin` | `trl`), or `--backend` overrides it. Always `--dry-run` a `trl` run first to surface the plan/cost before spending. The dispatch returns the same shape regardless of engine, so don't special-case it.

**Contract:** with `--json`, stdout is exactly one JSON object (success *or* `{"ok": false, "error": "..."}`); exit `0` = success, non-zero = failure with one stderr line. Parse stdout, branch on the exit code. Discover the surface with `rl-studio --help`.

## Set up your own RL pipeline (fast)

This repo is a working template — point it at *your* task, keep the loop.

**A new verifiable-reward task (CPU, minutes):**
1. Edit `reward(seq, target)` and `is_success(...)` in `src/rl_studio/lib/grpo.py` — return a higher scalar for better outputs (deterministic, checkable; no reward model).
2. Adjust the action space (`seq_len`, `vocab`) and hyperparameters in `configs/toy-grpo.yaml`.
3. `rl-studio train configs/<your>.yaml --json` — the group-relative advantage, KL penalty, and MLflow logging are already wired.

**A real LLM (GPU via Modal, no infra to manage):**
1. `uv sync --extra modal` (just the launcher — torch builds in the container, not on your laptop). Auth is machine-level: `modal token set` once.
2. Edit `correctness_reward(...)` in `scripts/modal_grpo.py` for your verifiable reward, and the model/dataset/hparams in `configs/grpo-qwen.yaml`.
3. Smoke cheap first, then run: `modal run scripts/modal_grpo.py --config configs/grpo-qwen-smoke.yaml` then `--config configs/grpo-qwen.yaml`. The reward curve lands in `results/<name>/`.

`GRPOConfig` kwargs are filtered to the installed TRL's fields, so a TRL version bump degrades gracefully instead of crashing.

## Setup (for the agent's environment)

```bash
uv sync --extra dev        # verified path: numpy only, no torch
uv run rl-studio doctor --json
```

## Hard rules (when editing this repo)

1. **CLI-first.** Every capability is an `rl-studio` subcommand. No notebook-only or script-only flows.
2. **`--json` on every command**, with load-bearing exit codes (see `src/rl_studio/output.py`). `gpu-train` without the `gpu` extra fails cleanly, non-zero — never a silent no-op.
3. **MLflow is the single source of truth**; per-step reward/KL curves go in as metric series. Falls back to `sqlite:///mlflow.db`.
4. **Report a baseline with every metric.** The policy is always reported vs the random-policy baseline.
5. **The GRPO loop must demonstrably learn.** If `test_grpo_learns_beats_baseline` fails, fix the algorithm — never weaken the assertion.
6. **Marimo `.py`, never `.ipynb`.**

## GRPO invariants (don't regress these)

- Advantage is **group-relative**: `(r − group_mean) / (group_std + ε)`. No value network — this is the defining trick.
- KL penalty is toward the **frozen reference (initial) policy**, tracked every step, and stays responsive (a test guards this).
- Rewards stay **verifiable** (deterministic, checkable) — no learned reward model on either path.

## Build / test / verify

```bash
uv run ruff check src tests scripts
uv run pytest                     # smoke + the "GRPO learns" assertion
make repro                        # determinism: train twice, identical metrics
make readme                       # run the README's <!-- ci-test --> commands
```

## Layout

```
src/rl_studio/
  cli.py        # Typer app, one command per capability
  output.py     # emit()/fail() — output + exit-code contract
  commands/     # doctor, train, eval, sample, gpu-train, version
  lib/grpo.py   # the real numpy GRPO loop
configs/        # toy-grpo.yaml (verified) · grpo-qwen[-smoke].yaml (LLM/GPU)
scripts/        # modal_grpo.py, modal_cuda_smoke.py + stdlib CI helpers
results/        # committed reward curves from real GPU runs
```

## What NOT to add

- `.ipynb` notebooks. A neural reward model (defeats the verifiable-reward point). `torch`/`trl`/`transformers` in main deps — they live in the `gpu` extra. Commands without `--json`.
