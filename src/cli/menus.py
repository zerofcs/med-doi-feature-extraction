from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt
from rich.table import Table
from rich import box

from .helpers import discover_pipelines_meta, select_config_interactive
from .registry import get as _get
import inspect
from .commands_data import register as register_data  # for type hints only


def register(app: typer.Typer, console: Console) -> None:
    @app.callback(invoke_without_command=True)
    def entry(ctx: typer.Context):
        if ctx.invoked_subcommand is not None:
            return
        load_dotenv()

        # Show welcome header
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Medical Literature Extraction CLI[/bold cyan]\n"
            "[dim]Automated data extraction from medical literature[/dim]",
            style="bold green",
            border_style="green"
        ))
        console.print()

        # Handle config selection
        cfg_path: Optional[Path] = None
        use_env = os.getenv("MED_CONFIG_PATH")
        if use_env and Path(use_env).exists():
            cfg_path = Path(use_env)
            console.print(f"[dim]Using MED_CONFIG_PATH:[/dim] [cyan]{cfg_path}[/cyan]")
            console.print()
        else:
            # Automatically show config selection
            try:
                cfg_path = select_config_interactive(console)
                console.print()
            except Exception as e:
                console.print(f"[yellow]{e}[/yellow]")
                console.print()

        # Build menu table
        metas = discover_pipelines_meta()

        # Create actions menu table
        menu_table = Table(
            title="Available Actions",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="blue"
        )
        menu_table.add_column("#", justify="right", style="dim", width=4)
        menu_table.add_column("Command", style="bold", width=20)
        menu_table.add_column("Description", style="dim")

        # Reduced action set
        actions = [
            ("test", "Test Extraction", "Test extraction on a single record (with preview)"),
            ("run", "Run Extraction", "Execute batch extraction on dataset"),
            ("validate", "Validate Results", "Check quality of extraction results"),
            ("retry", "Retry Failed", "Retry failed extractions"),
            (None, "Exit", "Exit the application"),
        ]

        # Add standard actions
        for i, (cmd_name, label, desc) in enumerate(actions, 1):
            menu_table.add_row(str(i), label, desc)

        # No dynamic pipeline commands in trimmed menu

        console.print(menu_table)
        console.print()
        console.print("[dim]Tip: You can also run commands directly, e.g., 'cli run --help'[/dim]")
        console.print()

        choice = IntPrompt.ask("[bold cyan]→[/bold cyan] Select action", default=1)

        # Handle standard actions
        if 1 <= choice <= len(actions):
            cmd_name, label, _ = actions[choice - 1]
            if cmd_name is None:
                console.print("[green]Goodbye![/green]")
                raise typer.Exit(0)
            fn = _get(cmd_name)
            sig = inspect.signature(fn)

            # Build kwargs with explicit None values for all optional parameters
            # to avoid Typer passing OptionInfo objects as defaults
            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "config":
                    kwargs["config"] = cfg_path
                elif param.default != inspect.Parameter.empty:
                    # For optional parameters, explicitly pass None
                    kwargs[param_name] = None

            ctx.invoke(fn, **kwargs)
            return

        console.print("[yellow]Invalid selection[/yellow]")
        raise typer.Exit(0)
