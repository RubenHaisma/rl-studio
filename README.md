# rl-studio

**CLI-first GRPO reinforcement-learning fine-tuning.** A real, minimal **GRPO** (Group-Relative Policy Optimization) loop you can read, run, and verify — train → eval-against-baseline → sample, tracked in MLflow, driven by one machine-readable binary. Built in the house style of [`ml-pipeline-template`](https://github.com/rubenhaisma/ml-pipeline-template): `--json` everywhere, load-bearing exit codes, honest baselines.

> GRPO is the RL method behind DeepSeek-R1-style reasoning training: drop the value network and use the **group mean** of sampled completions as the baseline. This repo implements that mechanic for real — twice. A **verified** pure-numpy loop on a verifiable-reward toy task that runs on CPU in CI in under a second and *demonstrably learns*, and a **scaffolded** LLM path (TRL `GRPOTrainer` on `Qwen2.5-0.5B` against GSM8K, rented on a Modal GPU) that is wired but honestly marked unverified.

## What's actually implemented (not a framework call)

The numpy loop in [`src/rl_studio/lib/grpo.py`](src/rl_studio/lib/grpo.py) does the real thing:

- **A categorical policy** — a `(seq_len, vocab)` numpy logits table; each position is an independent softmax categorical, so `log π(seq) = Σ_t log softmax(logits[t])[seq[t]]`.
- **A verifiable task (RLVR-style)** — emit a length-`N` digit sequence whose sum equals a target. Reward is deterministic and checkable (`exp(-|sum − target| / scale)`, 1.0 on target). No neural reward model.
- **The GRPO step** — sample a **group** of `G` completions, score each, compute the **group-relative advantage** `A_i = (r_i − mean) / (std + ε)` (the baseline is the group mean — *no value network*), then a REINFORCE policy-gradient step on the logits using `A_i · ∇log π(seq_i)`.
- **A KL penalty** toward the frozen reference (initial) policy, tracked every step.
- **It learns.** Mean reward rises and converges; the strict `sum == target` success rate beats the random baseline by a wide margin. A test asserts this — see below.

## Quickstart

```bash
uv sync --extra dev                         # install (light: numpy, no torch)
uv run rl-studio doctor                      # environment readiness (--json for CI)
uv run rl-studio train configs/toy-grpo.yaml # run the numpy GRPO loop
uv run rl-studio eval digit-sum              # success rate vs random baseline
uv run rl-studio sample digit-sum --n 5      # see what the policy emits
```

Output of `train` (human mode):

```
trained digit-sum (GRPO, 300 steps, group=24)
  mean_reward 0.38 -> 0.95  (KL 0.31)
  success_rate 0.46  (random baseline 0.07, lift +0.39)
  policy -> artifacts/digit-sum/policy.npz
```

Everything is also available via `make`: `make demo` runs the full train → eval → sample loop.

## CLI surface

```
rl-studio doctor [--json]                       # is this environment ready?
rl-studio train <config> [--out] [--json]       # numpy GRPO loop, logs reward/KL curve to MLflow
rl-studio eval <name> [--n] [--json]            # success rate vs random-policy baseline
rl-studio sample <name> --n K [--seed] [--json] # K completions with their rewards
rl-studio gpu-train <config> [--json]           # scaffolded LLM GRPO via TRL/Modal (gated)
rl-studio version [--json]
```

Every command emits a single JSON object with `--json` and returns a load-bearing exit code (`0` ok, non-zero failure with one stderr line). See [`src/rl_studio/output.py`](src/rl_studio/output.py).

Tracking UI (optional): `make up` starts MLflow on `localhost:5050`, then `export MLFLOW_TRACKING_URI=http://localhost:5050`.

## The GPU path (scaffolded, off-box)

The verified path runs on CPU. The real LLM training is rented on a GPU and is *not* run in CI:

```bash
uv sync --extra gpu                       # torch / transformers / trl / datasets / modal
uv run rl-studio gpu-train configs/grpo-qwen.yaml   # prints the Modal launch command
modal run scripts/modal_grpo.py --config configs/grpo-qwen.yaml
```

[`scripts/modal_grpo.py`](scripts/modal_grpo.py) runs TRL's `GRPOTrainer` on `Qwen/Qwen2.5-0.5B-Instruct` against GSM8K with a verifiable correctness reward (parse the boxed answer, check it). [`configs/grpo-qwen.yaml`](configs/grpo-qwen.yaml) holds the hyperparameters. Without the `gpu` extra, `rl-studio gpu-train` fails cleanly with one line and a non-zero exit — no silent no-op.

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
| LLM GRPO (TRL `GRPOTrainer`, Qwen2.5-0.5B, GSM8K)      | 🟠 scaffolded, **not run in CI** (needs a GPU) |
| Modal GPU launch (`scripts/modal_grpo.py`)             | 🟠 scaffolded, **not run in CI**     |

The honest split: the **GRPO algorithm** is verified end-to-end on a CPU toy task; the **LLM application** of the same algorithm is wired against TRL + Modal but requires a rented GPU and an account, so it is presented as scaffolding, not a passing test.

## How it relates to the template

This is a domain repo over [`ml-pipeline-template`](https://github.com/rubenhaisma/ml-pipeline-template): same `output.py` / `tracking.py` shell, same CLI/`--json`/exit-code/baseline discipline, with `lib/pipeline.py` (a sklearn classifier) swapped for `lib/grpo.py` (a real GRPO loop).

## License

Apache-2.0.
