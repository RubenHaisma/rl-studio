"""MLflow is the single source of truth for runs, params, metrics, models.

Falls back to a local ``sqlite:///mlflow.db`` store when no tracking server is
up, so the happy path works on a fresh checkout with zero services running.
(MLflow 3 deprecated the bare ``./mlruns`` file store; sqlite is the supported
local backend.) Point at the docker-compose MLflow by exporting
``MLFLOW_TRACKING_URI``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


@contextmanager
def run(experiment: str, run_name: str) -> Iterator[Any]:
    """Open an MLflow run against the configured (or local) tracking store."""
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        yield active


def log_params(params: dict[str, Any]) -> None:
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_metric_step(key: str, value: float, step: int) -> None:
    """Log one point of a metric curve (per-step reward/KL series)."""
    mlflow.log_metric(key, float(value), step=step)
