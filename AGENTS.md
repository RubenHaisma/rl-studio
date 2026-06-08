# AGENTS.md

Agent instructions for **rl-studio**, in the cross-tool [AGENTS.md](https://agents.md) format — read natively by Codex, Cursor, GitHub Copilot, Windsurf, Amp, Devin, and others. Claude Code reads `CLAUDE.md`, which is a **symlink to this file**, so there is a single source of truth for every tool. Humans: see [README.md](README.md).

## What this repo is

An **agent-operable harness for GRPO fine-tuning**. It does not reimplement RL — it makes a real engine drivable by a coding agent through one `--json` CLI. `rl-studio train <config>` routes by the config's `backend`:

1. **`builtin`** (numpy, CPU, CI-verified): `src/rl_studio/lib/grpo.py` — a real, readable GRPO loop on a verifiable-reward toy task that demonstrably learns.
2. **`trl`** (real LLM): TRL `GRPOTrainer` via the shared engine `src/rl_studio/lib/trl_runner.py`. **Pluggable** — `model` / `dataset` / `reward` are config-driven (point it at anything). Two compute targets, same engine: `compute: modal` (rented GPU, ran for real, reward 0.26→0.48) and `compute: local` (your own GPU, in-process). Same JSON shape as builtin.

Roadmap backends (same seam): `verl`, `unsloth`. Dispatch layer: `src/rl_studio/lib/backends.py`.

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

Mostly **config, not code**. `rl-studio train <config>` is the one entry; the config picks engine, compute, model, dataset, and reward.

**A real LLM on your task (no code for common cases):**
1. Copy `configs/grpo-qwen.yaml`. Set `model`, `dataset` (+ `dataset_config` / `split`), `prompt_column`, `answer_column`, and `reward` (built-in: `numeric_match` | `exact_match` | `contains` | `regex_match`; `regex_match` also reads `reward_pattern`).
2. Pick compute: `compute: modal` (rented GPU — `uv sync --extra modal`, `modal token set` once) or `compute: local` (your own GPU — `uv sync --extra gpu`).
3. `rl-studio train configs/<yours>.yaml --dry-run --json` to inspect the plan + cost, then drop `--dry-run` to run. Results land in `results/<name>/`.

**A custom reward (escape hatch):** write a verifiable function in `rewards/<yours>.py` — signature `(completions, answer, **kw) -> list[float]`, deterministic/checkable, no reward model — and set `reward_fn: rewards/<yours>.py:reward`. See `rewards/example_reward.py`.

**Learn / CI / offline:** the `builtin` backend (`configs/toy-grpo.yaml`) is a real numpy GRPO loop; edit `reward(...)` in `src/rl_studio/lib/grpo.py` for a CPU toy task.

The shared engine is `src/rl_studio/lib/trl_runner.py` (local + Modal run identical code). `GRPOConfig` kwargs are filtered to the installed TRL's fields, so a version bump degrades gracefully.

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
