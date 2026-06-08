"""rl-studio — the binary. One Typer app, one subcommand per capability.

House rules (enforced in CI):
- every command takes ``--json``
- exit codes are load-bearing (0 ok, non-zero failure)
- no command writes state outside MLflow + ./artifacts
"""

from __future__ import annotations

import typer

from rl_studio import __version__
from rl_studio.commands.doctor import doctor
from rl_studio.commands.eval import eval
from rl_studio.commands.gpu_train import gpu_train
from rl_studio.commands.sample import sample
from rl_studio.commands.train import train

app = typer.Typer(
    name="rl-studio",
    help="CLI-first GRPO RL fine-tuning studio — train, eval, sample, tracked in MLflow.",
    no_args_is_help=True,
    add_completion=False,
)

app.command()(doctor)
app.command()(train)
app.command()(eval)
app.command()(sample)
app.command(name="gpu-train")(gpu_train)


@app.command()
def version(json_out: bool = typer.Option(False, "--json")) -> None:
    """Print the rl-studio version."""
    if json_out:
        typer.echo(f'{{"version": "{__version__}"}}')
    else:
        typer.echo(__version__)


if __name__ == "__main__":
    app()
