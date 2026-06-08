# rl-studio — agent rules

For Claude Code and similar agents. Humans read `README.md`.

## What this repo is
A domain repo over `ml-pipeline-template`: same CLI/`--json`/exit-code/tracking
shell, with the trivial sklearn model swapped for a **real GRPO loop**. It
demonstrates Group-Relative Policy Optimization two ways:

1. **Verified (numpy, CPU, in CI):** `src/rl_studio/lib/grpo.py` is a real GRPO
   loop on a verifiable-reward toy task (emit digits summing to a target). It
   actually learns — a test asserts reward rises and beats the random baseline.
2. **Scaffolded (LLM, GPU, off-box):** `scripts/modal_grpo.py` runs TRL's
   `GRPOTrainer` on Qwen2.5-0.5B against GSM8K, rented on Modal. Wired, honestly
   marked unverified, **never run in CI**.

## Hard rules
1. **CLI-first.** Every capability ships as an `rl-studio` subcommand. No
   notebook-only or script-only flows.
2. **`--json` on every command.** Machine-readable output is a contract.
3. **Exit codes mean something.** `0` success; non-zero failure with one
   human-readable line on stderr (see `output.py`). `gpu-train` without the
   `gpu` extra fails cleanly, non-zero — never a silent no-op.
4. **MLflow is the single source of truth** for runs/params/metrics. The
   per-step reward/KL curves go in as metric series. Falls back to
   `sqlite:///mlflow.db` with no server running.
5. **Report a baseline with every metric.** The learned policy is always
   reported against the random-policy baseline. A policy that doesn't beat it
   is a finding to surface, not a number to hide.
6. **The GRPO loop must demonstrably learn.** If `test_grpo_learns_beats_baseline`
   fails, fix the algorithm (lr/steps/reward shaping) — **never weaken the
   assertion to make it pass**.
7. **Marimo `.py`, never `.ipynb`.**

## GRPO invariants (don't regress these)
- The advantage is **group-relative**: `(r - group_mean) / (group_std + eps)`.
  No value network. This is the defining trick — keep it.
- The KL penalty is toward the **frozen reference (initial) policy**, tracked
  every step. It must remain responsive (higher `kl_coef` → lower final KL); a
  test guards this.
- The reward must stay **verifiable** (deterministic, checkable) — no learned
  reward model on either path.

## Layout
```
src/rl_studio/
  cli.py            # Typer app, one command per capability
  output.py         # emit()/fail() — the output + exit-code contract (verbatim from template)
  commands/         # doctor, train, eval, sample, gpu-train, version
  lib/              # config, tracking (verbatim), grpo (the numpy loop)
configs/            # toy-grpo.yaml (verified) + grpo-qwen.yaml (scaffolded)
scripts/            # modal_grpo.py — TRL GRPOTrainer on a rented GPU
notebooks/          # marimo .py — reward curve, KL drift
tests/              # pytest — shell contract + "GRPO learns" assertion
```

## What NOT to add
- `.ipynb` notebooks.
- A neural reward model (defeats the verifiable-reward point).
- `torch`/`trl`/`transformers` in the main deps — they live in the `gpu` extra
  so the verified path and CI stay light.
- Cloud-specific training SDKs beyond the Modal scaffold.
- Commands without `--json`.
