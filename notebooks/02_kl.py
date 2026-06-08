"""KL-to-reference drift — is the policy staying anchored to its reference?

Reads the per-step `kl` series logged by `rl-studio train`. The KL penalty is
the second half of GRPO (after the group-relative advantage): it pulls the
policy back toward the frozen reference so it improves reward without collapsing.
A rising-then-plateauing KL means the policy moved as far as the task demanded;
a runaway KL is a finding to surface, not hide.

Run with: marimo edit notebooks/02_kl.py
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
        # KL to Reference
        How far the GRPO policy drifts from its initial (reference) policy over
        training, from the MLflow run logged by `rl-studio train`. Compare against
        `final_kl` and `final_mean_reward` — higher reward usually costs more KL.
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
        view = mo.md("")
    else:
        kl = client.get_metric_history(latest_run_id, "kl")
        steps = [m.step for m in kl]
        vals = [m.value for m in kl]
        rows = "\n".join(
            f"| {s} | {v:.4f} |"
            for s, v in list(zip(steps, vals))[:: max(1, len(steps) // 20)]
        )
        view = mo.md(
            "| step | kl_to_reference |\n|---|---|\n"
            + rows
            + f"\n\n**final KL** {vals[-1]:.4f}  (max {max(vals):.4f})"
        )
    view
    return


if __name__ == "__main__":
    app.run()
