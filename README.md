# rl-studio

[![ci](https://github.com/RubenHaisma/rl-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenHaisma/rl-studio/actions/workflows/ci.yml)

**Agent-drivable GRPO fine-tuning — run real RL from Claude Code or Codex via one `--json` CLI.** rl-studio doesn't reimplement RL (TRL/verl/Unsloth already do that well). It makes a real engine **agent-operable**: a stable command surface, load-bearing exit codes, a preflight, and machine-parseable run results — so a coding agent can run and iterate fine-tunes autonomously instead of writing throwaway scripts and scraping stdout.

> **One command, pluggable backends.** `rl-studio train <config>` reads the config's `backend` and routes:
> - **`builtin`** — a real, readable **GRPO** loop in pure numpy (CPU, CI-verified, offline). Learn the mechanic; run it anywhere in under a second.
> - **`trl`** — real GRPO via TRL's `GRPOTrainer` on a rented Modal GPU (Qwen2.5-0.5B on GSM8K). Same `--json` result shape; the engine does the training, we make it drivable.
>
> Roadmap backends (same seam): `verl`, `unsloth`. GRPO is the method behind DeepSeek-R1-style reasoning training — drop the value network, use the **group mean** as the baseline.

## What's actually implemented (not a framework call)

The numpy loop in [`src/rl_studio/lib/grpo.py`](src/rl_studio/lib/grpo.py) does the real thing:

- **A categorical policy** — a `(seq_len, vocab)` numpy logits table; each position is an independent softmax categorical, so `log π(seq) = Σ_t log softmax(logits[t])[seq[t]]`.
- **A verifiable task (RLVR-style)** — emit a length-`N` digit sequence whose sum equals a target. Reward is deterministic and checkable (`exp(-|sum − target| / scale)`, 1.0 on target). No neural reward model.
- **The GRPO step** — sample a **group** of `G` completions, score each, compute the **group-relative advantage** `A_i = (r_i − mean) / (std + ε)` (the baseline is the group mean — *no value network*), then a REINFORCE policy-gradient step on the logits using `A_i · ∇log π(seq_i)`.
- **A KL penalty** toward the frozen reference (initial) policy, tracked every step.
- **It learns.** Mean reward rises and converges; the strict `sum == target` success rate beats the random baseline by a wide margin. A test asserts this — see below.

## Backends

One agent-facing command, several engines behind it. The config picks the engine **and compute target**; the result shape is identical, so an agent never special-cases it.

| backend | compute | engine | runs on | status |
| --- | --- | --- | --- | --- |
| `builtin` | — | pure-numpy GRPO loop | CPU, in-process | ✅ CI-verified, learns |
| `trl` | `modal` | TRL `GRPOTrainer` | rented Modal GPU | ✅ ran for real (reward 0.26→0.48) |
| `trl` | `local` | TRL `GRPOTrainer` | your own GPU, in-process | ✅ same engine; routing CI-tested |
| `verl`, `unsloth` | — | — | — | 🧭 roadmap (same seam) |

```bash
rl-studio train configs/toy-grpo.yaml             # builtin → numpy, here, now
rl-studio train configs/grpo-qwen.yaml --dry-run  # trl/modal → plan only, no spend
rl-studio train configs/grpo-qwen.yaml            # trl/modal → real GRPO on a rented GPU
rl-studio train configs/grpo-local.yaml           # trl/local → real GRPO on your own GPU
```

### Pluggable — point it at any task (no code edits)

The `trl` backend is fully config-driven:

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct   # any HF causal LM
dataset: openai/gsm8k               # any HF dataset
prompt_column: question             # which column is the prompt
answer_column: answer               # which column is the gold answer
reward: numeric_match               # built-in: numeric_match | exact_match | contains | regex_match
# reward_fn: rewards/example_reward.py:reward   # ...or your own verifiable reward
compute: modal                      # modal (rented) | local (your GPU)
```

Built-in rewards cover the verifiable-reward 80% (math / answer-matching / format); `reward_fn` is the escape hatch for anything else — a deterministic, checkable function, no reward model. `--dry-run` shows the resolved model/dataset/reward/compute before you spend; `--backend` overrides the config.

## Quickstart

<!-- ci-test -->
```bash
uv sync --extra dev                         # install (light: numpy, no torch)
uv run rl-studio doctor                      # environment readiness (--json for CI)
uv run rl-studio train configs/toy-grpo.yaml # run the numpy GRPO loop
uv run rl-studio eval digit-sum              # success rate vs random baseline
uv run rl-studio sample digit-sum --n 5      # see what the policy emits
```

> The block above is marked `<!-- ci-test -->` — **CI runs these exact commands on every push**, so this quickstart can never silently drift from the code.

Output of `train` (human mode):

```
trained digit-sum (GRPO, 300 steps, group=24)
  mean_reward 0.38 -> 0.95  (KL 0.31)
  success_rate 0.46  (random baseline 0.07, lift +0.39)
  policy -> artifacts/digit-sum/policy.npz
```

Everything is also available via `make`: `make demo` runs the full train → eval → sample loop.

### Spin up your own RL pipeline in minutes

A working template, not a demo to read — and mostly **config, not code**:

- **Your task, no code:** copy `configs/grpo-qwen.yaml`, point `model` / `dataset` / `prompt_column` / `answer_column` / `reward` at your task, choose `compute: modal` (rented GPU) or `local` (your own), then `rl-studio train configs/<yours>.yaml`. Dataset mapping, GRPO config, reward registry, and MLflow logging are already wired.
- **A custom reward:** drop a function in `rewards/<yours>.py` and set `reward_fn: rewards/<yours>.py:reward` — see [`rewards/example_reward.py`](rewards/example_reward.py). Verifiable only (deterministic, checkable; no reward model).
- **Learn the mechanic (CPU):** the `builtin` backend (`configs/toy-grpo.yaml`) is a real GRPO loop in numpy you can read end to end.

Full adapt-it guide for humans and agents: [`AGENTS.md`](AGENTS.md).

## Drive it with Claude Code

This is the point of the harness: hand the loop to your coding agent. Because every command is `--json` with load-bearing exit codes, an agent can run experiments and react to the numbers — no human in the loop.

A real agent session looks like:

```
You:    "Get the GRPO policy above 0.9 success rate. Use rl-studio."
Agent:  $ rl-studio doctor --json                     # {"ok": true, ...}
        $ rl-studio train configs/toy-grpo.yaml --json # reads {"final_success_rate": 0.71, ...}
        # 0.71 < 0.9 — bump the group size and steps, rerun:
        $ rl-studio train configs/toy-grpo.yaml --backend builtin --json   # {"final_success_rate": 0.94}
        "Hit 0.94. Raised group_size 24→48 and steps 300→500; reward curve in MLflow."
```

For a real LLM run, the agent dry-runs first to show the plan and cost, then dispatches:

```
$ rl-studio train configs/grpo-qwen.yaml --dry-run --json   # {"would_run": "modal run ...", "note": "spends credits"}
$ rl-studio train configs/grpo-qwen.yaml --json             # launches TRL on a Modal GPU, returns final metrics
```

The `AGENTS.md` in this repo (symlinked to `CLAUDE.md`) tells the agent the contract, so Claude Code / Codex pick it up automatically.

## CLI surface

```
rl-studio doctor [--json]                                  # is this environment ready?
rl-studio train <config> [--backend builtin|trl]           # THE entry: routes to the engine
          [--dry-run] [--out] [--json]                     #   --dry-run prints the plan, no spend
rl-studio eval <name> [--n] [--json]                       # success rate vs random-policy baseline
rl-studio sample <name> --n K [--seed] [--json]            # K completions with their rewards
rl-studio gpu-train <config> [--json]                      # explainer for the trl/Modal path (gated)
rl-studio version [--json]
```

Every command emits a single JSON object with `--json` and returns a load-bearing exit code (`0` ok, non-zero failure with one stderr line). See [`src/rl_studio/output.py`](src/rl_studio/output.py).

Tracking UI (optional): `make up` starts MLflow on `localhost:5050`, then `export MLFLOW_TRACKING_URI=http://localhost:5050`.

## Running on real GPUs (Modal or local)

The `builtin` backend runs on CPU; the `trl` backend runs real GRPO on a GPU — rented (`compute: modal`) or your own (`compute: local`). `rl-studio train` is the entry, but the lower-level paths are available too:

```bash
uv sync --extra gpu                       # torch / transformers / trl / datasets / modal
uv run rl-studio gpu-train configs/grpo-qwen.yaml   # prints the Modal launch command
modal run scripts/modal_grpo.py --config configs/grpo-qwen.yaml
```

[`scripts/modal_grpo.py`](scripts/modal_grpo.py) runs TRL's `GRPOTrainer` on `Qwen/Qwen2.5-0.5B-Instruct` against GSM8K with a verifiable correctness reward (parse the final answer, check it). [`configs/grpo-qwen.yaml`](configs/grpo-qwen.yaml) holds the hyperparameters. Without the `gpu` extra, `rl-studio gpu-train` fails cleanly with one line and a non-zero exit — no silent no-op.

### This actually ran on a GPU

The LLM path isn't just wired — it was run on a Modal A10G (200 GRPO steps, ~25 min). The verifiable GSM8K answer-correctness reward rose from **0.256 → 0.475** as the policy learned to get more answers right:

```
reward over 200 steps:  ▂▄▃▃▁▃▄▂▆▂▃▂▁▃▅▂▁▄▃▃▄▄▅▄▃▂▅▃▅▃▄▄▃▁▃▂▁▃▄█   0.26 → 0.48
```

The full curve and run metadata are committed under [`results/grpo-qwen/`](results/grpo-qwen/) — noisy, as expected for a 0.5B model with small batches, but clearly trending up. (It's marked "not run in CI" below because *CI* never rents a GPU, not because it hasn't run.)

## Notebooks (marimo)

```bash
uv run marimo edit notebooks/01_reward_curve.py  # reward + success-rate curve from MLflow
uv run marimo edit notebooks/02_kl.py            # KL-to-reference drift over training
```

## What's verified

| Path                                                   | Status                              |
| ------------------------------------------------------ | ----------------------------------- |
| numpy GRPO loop **learns** (reward rises, beats baseline) | ✅ verified (asserted in tests, CPU) |
| `train` → `eval` → `sample` loop on CPU                | ✅ verified                          |
| `--json` contract + load-bearing exit codes            | ✅ verified                          |
| `pytest` smoke suite + ruff in CI                      | ✅ verified                          |
| MLflow local sqlite store (reward/KL curves logged)    | ✅ verified                          |
| MLflow server via docker-compose                       | 🟡 compose provided, runs locally    |
| LLM GRPO (TRL `GRPOTrainer`, Qwen2.5-0.5B, GSM8K)      | ✅ **ran on a Modal A10G** — reward 0.26→0.48, curve in `results/grpo-qwen/`; not run *in CI* (no GPU) |
| unified `train` → `trl` dispatch (one command → GPU engine) | ✅ ran end-to-end (smoke + full); routing + clean-fail are CI-tested |
| pluggable model / dataset / reward (registry + `reward_fn`) | ✅ reward registry + custom loader unit-tested; carried through dispatch |
| `compute: local` (same engine, your own GPU)           | 🟡 wired — same `run_grpo` Modal runs; routing + clean-fail CI-tested; not run here (no local CUDA) |
| Modal GPU launch (`scripts/modal_grpo.py`)             | ✅ ran on Modal; not in CI            |

The honest split: the **GRPO algorithm** is verified end-to-end on a CPU toy task; the **LLM application** of the same algorithm is wired against TRL + Modal but requires a rented GPU and an account, so it is presented as scaffolding, not a passing test.

## Agent-friendly by design

Every command is non-interactive, emits a single JSON object with `--json`, and returns a load-bearing exit code — so AI coding agents (**Codex, Claude Code, Cursor, Copilot, Windsurf, …**) and plain scripts can run and parse the GRPO loop with no TTY, no UI, no screen-scraping.

```bash
rl-studio train configs/toy-grpo.yaml --json   # -> {"ok": true, "metrics": {"final_success_rate": ...}}   exit 0
```

Agent instructions live in [`AGENTS.md`](AGENTS.md) — the [cross-tool standard](https://agents.md) read natively by Codex, Cursor, Copilot, and more. `CLAUDE.md` is a symlink to it, so every tool reads one source of truth.

## CI does more than lint

Most repos' CI checks that the code *parses*. This one checks that the GRPO loop *works* — three things beyond lint + tests, all stdlib, no extra deps:

1. **It runs the loop and publishes the numbers.** Every push trains the numpy GRPO loop and posts a live metrics table (reward, success rate, lift over baseline) to the GitHub Actions [run summary](https://github.com/RubenHaisma/rl-studio/actions) (`scripts/ci_report.py`).
2. **It keeps the docs honest.** The Quickstart block is marked `<!-- ci-test -->` and `scripts/test_readme.py` runs those exact commands in CI — drift fails the build.
3. **It proves determinism.** `scripts/check_repro.py` trains twice and asserts identical metrics — a seed is a promise, and CI verifies it holds.

Run them locally: `make summary`, `make readme`, `make repro`.

## How it relates to the template

This is a domain repo over [`ml-pipeline-template`](https://github.com/rubenhaisma/ml-pipeline-template): same `output.py` / `tracking.py` shell, same CLI/`--json`/exit-code/baseline discipline, with `lib/pipeline.py` (a sklearn classifier) swapped for `lib/grpo.py` (a real GRPO loop).

## License

Apache-2.0.
