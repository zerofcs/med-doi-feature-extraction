from __future__ import annotations

from rich.console import Console
from dotenv import load_dotenv
import typer


console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Medical literature extraction & screening CLI",
)


def main():
    load_dotenv()
    from .commands import register
    register(app, console)
    app()


if __name__ == "__main__":
    main()
