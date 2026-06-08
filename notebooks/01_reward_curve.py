"""Reward + success-rate curve — does the GRPO policy actually learn?

Reads the per-step metric series logged by `rl-studio train` and plots mean
reward and strict success rate over training. Marimo, not Jupyter: plain Python
that diffs, greps, and edits like source.

Run with: marimo edit notebooks/01_reward_curve.py
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # GRPO Reward Curve
        Mean reward and strict success rate per step, pulled from the MLflow run
        logged by `rl-studio train`. A rising reward curve is the proof the loop
        learns; the success rate is the unshaped objective (sum == target).
        """
    )
    return


@app.cell
def __():
    import os

    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(search_all_experiments=True, order_by=["start_time DESC"])
    return client, mlflow, runs


@app.cell
def __(client, mo, runs):
    if runs.empty:
        out = mo.md("_No runs yet — run `rl-studio train configs/toy-grpo.yaml` first._")
        latest_run_id = None
    else:
        latest_run_id = runs.iloc[0]["run_id"]
        out = mo.md(f"Latest run: `{latest_run_id}`")
    out
    return (latest_run_id,)


@app.cell
def __(client, latest_run_id, mo):
    if latest_run_id is None:
        chart = mo.md("")
    else:
        reward = client.get_metric_history(latest_run_id, "mean_reward")
        success = client.get_metric_history(latest_run_id, "success_rate")
        steps = [m.step for m in reward]
        rvals = [m.value for m in reward]
        svals = [m.value for m in success]
        rows = "\n".join(
            f"| {s} | {r:.3f} | {sr:.3f} |"
            for s, r, sr in list(zip(steps, rvals, svals))[:: max(1, len(steps) // 20)]
        )
        chart = mo.md(
            "| step | mean_reward | success_rate |\n|---|---|---|\n"
            + rows
            + f"\n\n**first reward** {rvals[0]:.3f} → **final** {rvals[-1]:.3f}"
        )
    chart
    return


if __name__ == "__main__":
    app.run()
