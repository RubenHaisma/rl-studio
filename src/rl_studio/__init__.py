"""rl_studio — CLI-first GRPO reinforcement-learning fine-tuning studio.

Built in the house style of ``ml-pipeline-template``: CLI-first, ``--json`` on
every command, load-bearing exit codes, MLflow as the single source of truth,
marimo (not Jupyter) for exploration.

Two paths share one shell:

* a **verified** pure-numpy GRPO loop on a verifiable toy task (runs on CPU, in
  CI, in seconds — and demonstrably learns), and
* a **scaffolded** LLM GRPO path (TRL ``GRPOTrainer`` on a small model, rented
  on a Modal GPU) that is wired but honestly marked unverified.
"""

__version__ = "0.1.0"
