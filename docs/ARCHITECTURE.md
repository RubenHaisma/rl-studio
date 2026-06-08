# Architecture

```
┌──────────────────────────────────────────────────────────┐
│  rl-studio CLI (Typer)                                     │
│  doctor · train · eval · sample · gpu-train · version      │
└───────┬───────────────────────────────────┬──────────────┘
        │ verified (numpy, CPU)              │ scaffolded (GPU, off-box)
        ▼                                    ▼
  ┌──────────────┐                     ┌──────────────────────┐
  │ lib/grpo.py  │  ── trains ──►      │ scripts/modal_grpo.py │
  │ numpy GRPO   │                     │ TRL GRPOTrainer       │
  │ loop         │  ── logs ───┐       │ Qwen2.5-0.5B / GSM8K  │
  └──────┬───────┘             │       │ on a rented Modal GPU │
         │                     ▼       └──────────────────────┘
         │               ┌──────────┐
         │               │  MLflow  │  reward / KL / success-rate curves
         │               │  runs +  │  (sqlite:///mlflow.db fallback)
         │               │  metrics │
         ▼               └──────────┘
  artifacts/<name>/policy.npz  ──►  rl-studio eval / sample
```

## The GRPO loop (`lib/grpo.py`) — the verified core

The algorithm, not a framework call:

1. **Policy.** A `(seq_len, vocab)` numpy logits table. Each position is an
   independent softmax categorical, so a completion is one digit sampled per
   position and `log π(seq) = Σ_t log softmax(logits[t])[seq[t]]`.
2. **Verifiable reward.** Emit a sequence summing to `target`; reward is
   `exp(-|sum − target| / scale)` (1.0 on target), with the strict `sum == target`
   used only for reporting `success_rate`. No neural reward model (RLVR-style).
3. **Group-relative advantage (the defining GRPO trick).** Each step samples a
   group of `G` completions and computes `A_i = (r_i − mean) / (std + ε)`. The
   baseline is the group mean — there is **no value network**, which is what
   distinguishes GRPO from PPO.
4. **Policy gradient.** REINFORCE on the logits: `A_i · ∇log π(seq_i)`, where for
   a per-position softmax `∂log π/∂logits[t,v] = 1{seq[t]=v} − probs[t,v]`.
5. **KL penalty.** Subtract `kl_coef · ∇KL(π ‖ π_ref)` toward the frozen initial
   policy. KL is tracked every step; a higher `kl_coef` measurably holds the
   policy closer to the reference (a test guards this).

It converges: mean reward rises from ~0.29 to ~0.93 and the strict success rate
goes from a ~0.07 random baseline to ~0.88 in a few hundred CPU steps.

## Output contract (`output.py`)
Verbatim from `ml-pipeline-template`. Every command funnels through `emit()`
(success) or `fail()` (error):

- `--json` → exactly one JSON object on stdout, success or failure.
- human mode → rich-formatted line(s) on stdout, `error: ...` on stderr.
- failure → non-zero exit, always. `gpu-train` without the `gpu` extra exits
  non-zero with one line — no silent no-op.

## Tracking (`lib/tracking.py`)
MLflow with a local `sqlite:///mlflow.db` fallback so a fresh checkout works with
no services. `train` logs hyperparameters as params, the **per-step reward, KL,
and success-rate as metric series** (the proof it learns), and final summary
metrics including `lift_over_baseline`. Export
`MLFLOW_TRACKING_URI=http://localhost:5050` (after `make up`) for the shared
backend in `docker-compose.yml` and the curve UI.

## The scaffolded GPU path (`scripts/modal_grpo.py`)
The same GRPO algorithm on a real LLM: TRL's `GRPOTrainer` on
`Qwen/Qwen2.5-0.5B-Instruct` against GSM8K, with a verifiable correctness reward
(parse the model's final number, check it against gold). Runs on a rented Modal
GPU; the heavy stack (`torch`/`trl`/`transformers`/`datasets`/`accelerate`/`modal`)
lives in the `gpu` extra. It is **not run in CI** — see the README's verified
matrix. `rl-studio gpu-train` is the gated front door that validates the config
and prints the `modal run` command, failing cleanly if the extra is absent.

## Why this shape
The reusable value is the shell: a CLI an agent can drive end-to-end,
machine-readable everywhere, with tracking and an honest baseline baked in. This
repo keeps that shell and swaps `lib/pipeline.py` (a sklearn classifier in the
template) for `lib/grpo.py` (a real GRPO loop). The verified path proves the
*algorithm*; the scaffolded path shows the *application* to an LLM without
pretending a GPU run happened in CI.
