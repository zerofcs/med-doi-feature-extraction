from __future__ import annotations

from rich.console import Console
from dotenv import load_dotenv
import typer


console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False, help="Config-driven medical literature extraction CLI")


def _register_all() -> None:
    from .commands_data import register as register_data
    from .commands_run import register as register_run
    from .commands_post import register as register_post
    from .menus import register as register_menus

    register_data(app, console)
    register_run(app, console)
    register_post(app, console)
    register_menus(app, console)


def main():
    load_dotenv()
    _register_all()
    app()


if __name__ == "__main__":
    main()
