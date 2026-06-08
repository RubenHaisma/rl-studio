"""A real, minimal GRPO loop in pure numpy on a verifiable toy task.

This is the *verified* path: no torch, no GPU, no neural reward model. It runs
on CPU in CI in well under a second and demonstrably learns. The point is to
implement the GRPO mechanics honestly so the algorithm — not a framework call —
is on display.

The policy
==========
A small categorical policy over a tiny action space, parameterized by a numpy
logits table of shape ``(seq_len, vocab)``. Each position is an independent
softmax categorical, so a completion is one digit sampled per position and

    log pi(seq) = sum_t log softmax(logits[t])[seq[t]].

The task (verifiable reward, à la RLVR)
=======================================
Emit a length-``seq_len`` sequence of digits in ``[0, vocab)`` whose sum equals
``target``. The reward is deterministic and checkable with no learned model:

    reward(seq) = exp(-|sum(seq) - target| / scale)

which is 1.0 exactly on target and decays smoothly otherwise. Smooth shaping
(rather than a 0/1 hit) keeps the group-relative advantage informative even
before the policy lands its first exact hit. ``success`` is the strict
``sum == target`` check, reported separately and never used in the gradient.

The GRPO step (the defining trick)
==================================
For each step we sample a GROUP of ``G`` completions from the current policy,
score each with the reward fn, and compute the **group-relative advantage**

    A_i = (r_i - mean(r)) / (std(r) + eps)

i.e. the baseline is the group mean — there is no value network, which is the
whole point of GRPO vs PPO. We then take a policy-gradient ascent step on the
logits using ``A_i * grad log pi(seq_i)``, plus a KL penalty pulling the policy
back toward the frozen reference (initial) policy. We track mean reward, the KL
to the reference, and the strict success rate every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_studio.lib.config import GRPOConfig
from rl_studio.output import CliError

# Width of the reward kernel: a sum that is `REWARD_SCALE` away from target
# earns reward 1/e. Small enough that the target is clearly preferred.
REWARD_SCALE = 2.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, numerically stable."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def reward(seq: np.ndarray, target: int) -> float:
    """Deterministic, checkable reward: peaks at 1.0 when sum == target."""
    return float(np.exp(-abs(int(seq.sum()) - target) / REWARD_SCALE))


def is_success(seq: np.ndarray, target: int) -> bool:
    """The strict, unshaped objective used for reporting (never for gradients)."""
    return int(seq.sum()) == target


def sample(probs: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample ``n`` completions. ``probs`` is ``(seq_len, vocab)``.

    Returns an int array of shape ``(n, seq_len)``.
    """
    seq_len, vocab = probs.shape
    out = np.empty((n, seq_len), dtype=np.int64)
    for t in range(seq_len):
        out[:, t] = rng.choice(vocab, size=n, p=probs[t])
    return out


def policy_kl(probs: np.ndarray, ref_probs: np.ndarray) -> float:
    """Mean over positions of KL(pi || pi_ref) for the per-position categoricals."""
    eps = 1e-12
    kl = np.sum(probs * (np.log(probs + eps) - np.log(ref_probs + eps)), axis=-1)
    return float(kl.mean())


@dataclass(slots=True)
class TrainResult:
    """Everything ``train`` produces: the learned policy plus the curves."""

    logits: np.ndarray
    ref_logits: np.ndarray
    history: dict[str, list[float]]  # step-indexed: mean_reward, kl, success_rate
    final_mean_reward: float
    final_success_rate: float


def train(cfg: GRPOConfig) -> TrainResult:
    """Run the numpy GRPO loop and return the learned logits + training curves.

    Pure function of ``cfg`` (seeded): no I/O, no MLflow — the command layer
    handles persistence and tracking. This keeps the algorithm unit-testable.
    """
    rng = np.random.default_rng(cfg.seed)

    # Start from a uniform-ish policy with tiny noise to break ties.
    logits = rng.normal(0.0, 0.01, size=(cfg.seq_len, cfg.vocab))
    ref_logits = logits.copy()  # frozen reference for the KL penalty
    ref_probs = _softmax(ref_logits)

    history: dict[str, list[float]] = {"mean_reward": [], "kl": [], "success_rate": []}

    for _ in range(cfg.steps):
        probs = _softmax(logits)

        # 1. Sample a GROUP of G completions from the current policy.
        group = sample(probs, rng, cfg.group_size)  # (G, seq_len)

        # 2. Score each with the verifiable reward fn.
        rewards = np.array([reward(seq, cfg.target) for seq in group])
        successes = np.array([is_success(seq, cfg.target) for seq in group], dtype=float)

        # 3. Group-relative advantage — the GRPO trick. Baseline = group mean,
        #    no value network.
        advantages = (rewards - rewards.mean()) / (rewards.std() + cfg.adv_eps)

        # 4. Policy-gradient ascent on the logits.
        #    For a per-position softmax, d log pi(seq)/d logits[t, v]
        #      = (1{seq[t]==v} - probs[t, v]).
        #    The REINFORCE estimator averages A_i * that over the group.
        grad = np.zeros_like(logits)
        for i in range(cfg.group_size):
            seq = group[i]
            onehot = np.zeros_like(logits)
            onehot[np.arange(cfg.seq_len), seq] = 1.0
            grad += advantages[i] * (onehot - probs)
        grad /= cfg.group_size

        # 5. KL penalty pulling pi back toward the reference. The gradient of
        #    KL(pi || pi_ref) wrt logits[t] is probs[t] * (log pi - log pi_ref
        #    - KL_t), i.e. descent here shrinks drift from the reference.
        log_ratio = np.log(probs + 1e-12) - np.log(ref_probs + 1e-12)
        kl_per_pos = np.sum(probs * log_ratio, axis=-1, keepdims=True)
        kl_grad = probs * (log_ratio - kl_per_pos)

        logits += cfg.lr * grad - cfg.lr * cfg.kl_coef * kl_grad

        history["mean_reward"].append(float(rewards.mean()))
        history["kl"].append(policy_kl(_softmax(logits), ref_probs))
        history["success_rate"].append(float(successes.mean()))

    return TrainResult(
        logits=logits,
        ref_logits=ref_logits,
        history=history,
        final_mean_reward=history["mean_reward"][-1] if history["mean_reward"] else 0.0,
        final_success_rate=history["success_rate"][-1] if history["success_rate"] else 0.0,
    )


def evaluate(logits: np.ndarray, target: int, n: int, seed: int) -> dict[str, float]:
    """Greedy + sampled success rate for a policy over ``n`` completions."""
    rng = np.random.default_rng(seed)
    probs = _softmax(logits)
    sampled = sample(probs, rng, n)
    success = np.mean([is_success(seq, target) for seq in sampled])
    mean_reward = np.mean([reward(seq, target) for seq in sampled])

    greedy = probs.argmax(axis=-1)
    return {
        "success_rate": float(success),
        "mean_reward": float(mean_reward),
        "greedy_sum": int(greedy.sum()),
        "greedy_success": bool(is_success(greedy, target)),
        "n": int(n),
    }


def random_baseline(seq_len: int, vocab: int, target: int, n: int, seed: int) -> dict[str, float]:
    """Honest baseline: success rate of a uniform random policy. House style —
    a metric without a baseline is marketing, not evaluation."""
    rng = np.random.default_rng(seed)
    uniform = np.full((seq_len, vocab), 1.0 / vocab)
    sampled = sample(uniform, rng, n)
    success = np.mean([is_success(seq, target) for seq in sampled])
    mean_reward = np.mean([reward(seq, target) for seq in sampled])
    return {"success_rate": float(success), "mean_reward": float(mean_reward), "n": int(n)}


def save_policy(path: Path, logits: np.ndarray, cfg: GRPOConfig) -> None:
    """Persist the learned logits + the task spec needed to reload and score it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        logits=logits,
        seq_len=cfg.seq_len,
        vocab=cfg.vocab,
        target=cfg.target,
        seed=cfg.seed,
    )


@dataclass(slots=True)
class LoadedPolicy:
    """A trained policy reloaded from disk, with the task spec it was trained on."""

    logits: np.ndarray
    seq_len: int
    vocab: int
    target: int
    seed: int


def load_policy(path: Path) -> LoadedPolicy:
    if not path.is_file():
        raise CliError(f"policy not found: {path} (train one first)")
    data = np.load(path)
    return LoadedPolicy(
        logits=data["logits"],
        seq_len=int(data["seq_len"]),
        vocab=int(data["vocab"]),
        target=int(data["target"]),
        seed=int(data["seed"]),
    )
