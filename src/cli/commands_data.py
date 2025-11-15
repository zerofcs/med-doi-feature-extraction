from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..config.schema import ConfigSchema
from ..engine import ExtractionEngine
from ..models import Record
from ..services.audit_service import AuditService
from .helpers import (
    load_config,
    load_tabular,
    data_quality_table,
    show_spinner_status,
    select_provider_interactive,
    select_model_interactive,
    display_record_preview,
    styled_confirm,
    prompt_file_path,
)
from .registry import add as _add


def register(app: typer.Typer, console: Console) -> None:
    @app.command()
    def test(
        config: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Pipeline overlay YAML"),
        input: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Input CSV/Excel"),
        id_column: str = typer.Option("DOI", help="Primary key column in input"),
        skip: int = typer.Option(0, help="Skip first N records"),
        provider: Optional[str] = typer.Option(None, help="Override provider (openai/ollama)"),
        model: Optional[str] = typer.Option(None, help="Override model name"),
        strategy: Optional[str] = typer.Option(None, help="Model selection strategy override"),
    ):
        """Test extraction on a single record with a brief data preview."""
        load_dotenv()
        cfg = load_config(config)

        # Show pipeline context
        pipeline_name = cfg.pipeline or "extraction"
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Testing [{pipeline_name.upper()}] Pipeline[/bold cyan]",
            style="bold green",
            border_style="green"
        ))
        console.print()

        # Interactive LLM configuration if not provided via flags
        if not provider or not isinstance(provider, str):
            console.print("[yellow]→ LLM Provider[/yellow]")
            provider = select_provider_interactive(console, cfg)
            console.print()

        # Load data first to show file info
        if not input:
            input_path = prompt_file_path("Input file path", default="data/final_test.csv", console=console)
        else:
            input_path = input

        # Model selection (no estimates for single test record)
        if not model or not isinstance(model, str):
            console.print("[yellow]→ Model Selection[/yellow]")
            model = select_model_interactive(console, cfg, provider, total_records=None, batch_size=None)
            console.print()

        # Always use "manual" strategy to use the selected model
        strategy = "manual"

        # Apply model overrides
        raw = cfg.model_dump(mode="python")
        raw.setdefault("llm", {}).update({"default_provider": provider})
        if model:
            raw.setdefault("llm", {}).setdefault("openai", {})["model"] = model
            raw.setdefault("llm", {}).setdefault("default_openai_model", model)
        raw.setdefault("llm", {}).update({"model_selection_strategy": strategy})
        cfg = ConfigSchema.model_validate(raw)
        with show_spinner_status(console, "Loading data"):
            df = load_tabular(input_path)

        # Brief preview: file summary and data quality
        console.print()
        summary = Table(title="File Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        summary.add_column("Property", style="bold")
        summary.add_column("Value", style="cyan")
        summary.add_row("Rows", str(len(df)))
        summary.add_row("Columns", str(len(df.columns)))
        summary.add_row("File", str(input_path))
        console.print(summary)

        console.print()
        important_cols = [id_column, "DOI", "Abstract", "Abstract Note", "Title"]
        console.print(data_quality_table(df, important_cols))

        # Handle None skip value
        skip_value = skip if isinstance(skip, int) else 0

        if skip_value >= len(df):
            console.print(f"[red]Error: Skip ({skip_value}) exceeds number of rows ({len(df)})[/red]")
            raise typer.Exit(1)

        # Get test record
        row = df.iloc[skip_value]
        data = {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
        key = str(data.get(id_column) or data.get("DOI") or skip_value)
        record = Record(key=key, data=data)

        # Show record preview
        console.print()
        display_record_preview(console, data, id_column)

        # Confirmation before extraction
        console.print()
        if not styled_confirm(f"Test extraction on record '{key}'?", default=True):
            console.print("[yellow]Test cancelled[/yellow]")
            raise typer.Exit(0)

        # Initialize engine
        audit_dir = Path(cfg.output.directory) / "_audit"
        audit = AuditService(audit_dir)
        engine = ExtractionEngine(cfg, audit)

        # Test extraction with spinner
        console.print()

        with console.status("[bold green]Extracting data...", spinner="dots"):
            result = asyncio.run(engine.process_record_async(record, force=True, strategy=strategy))

        if not result:
            console.print("[red]✗ No result returned (extraction failed)[/red]")
            raise typer.Exit(1)

        # Display results
        console.print()
        table = Table(title="Extraction Result", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Field", style="bold")
        table.add_column("Value", style="cyan")

        for k, v in (result.normalized or result.extracted).items():
            table.add_row(str(k), str(v))

        # Add metadata rows
        table.add_row("", "", end_section=True)
        confidence_color = "green" if result.confidence.overall >= 0.8 else "yellow" if result.confidence.overall >= 0.5 else "red"
        table.add_row("[bold]Confidence[/bold]", f"[{confidence_color}]{result.confidence.overall:.2f}[/{confidence_color}]")
        table.add_row("[bold]Valid[/bold]", f"[green]✓[/green]" if result.valid else "[red]✗[/red]")

        if result.errors:
            table.add_row("[bold]Errors[/bold]", f"[red]{', '.join(result.errors)}[/red]")

        if result.transparency:
            table.add_row("", "", end_section=True)
            table.add_row("[bold]Provider[/bold]", str(result.transparency.provider))
            table.add_row("[bold]Model[/bold]", str(result.transparency.model))
            table.add_row("[bold]Cost[/bold]", f"${result.transparency.cost:.4f}" if result.transparency.cost else "-")

        console.print(table)
        console.print()
    _add("test", test)
