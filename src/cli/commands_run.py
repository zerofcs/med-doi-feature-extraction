from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.prompt import Prompt, Confirm

from ..config.schema import ConfigSchema
from ..engine import ExtractionEngine
from ..services.audit_service import AuditService
from .helpers import (
    load_config,
    load_tabular,
    select_provider_interactive,
    select_model_interactive,
    styled_confirm,
    prompt_file_path,
)
from .registry import add as _add


def register(app: typer.Typer, console: Console) -> None:
    @app.command()
    def run(
        config: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Pipeline overlay YAML"),
        input: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Input CSV/Excel"),
        id_column: str = typer.Option("DOI", help="Primary key column in input"),
        skip: int = typer.Option(0, help="Skip first N records"),
        limit: Optional[int] = typer.Option(None, help="Limit number of records"),
        batch_size: Optional[int] = typer.Option(None, help="Number of records to process concurrently (default: 1)"),
        model: Optional[str] = typer.Option(None, help="Override OpenAI model name (skips interactive selection)"),
        force: bool = typer.Option(False, help="Force reprocess even if output exists"),
        delay: Optional[float] = typer.Option(None, help="Delay between batches in seconds (default: 0)"),
    ):
        """Execute batch extraction on dataset with parallel processing support."""
        load_dotenv()

        # Load configuration
        cfg = load_config(config)

        # Show pipeline context
        from rich.panel import Panel
        pipeline_name = cfg.pipeline or "extraction"
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Running [{pipeline_name.upper()}] Pipeline[/bold cyan]",
            style="bold green",
            border_style="green"
        ))
        console.print()

        # Interactive prompts for processing parameters
        console.print("[yellow]→ Processing Configuration[/yellow]")

        # Get input path with autocomplete
        if not input:
            input_path = prompt_file_path("Input file path", default="data/final_test.csv", console=console)
        else:
            input_path = input

        # Handle None values for parameters with interactive prompts
        id_column_value = id_column if isinstance(id_column, str) else "DOI"

        if not skip or not isinstance(skip, int):
            skip_value = Prompt.ask("[cyan]  Skip first N records[/cyan]", default="0")
            skip_value = int(skip_value) if skip_value.isdigit() else 0
        else:
            skip_value = skip

        if not limit or not isinstance(limit, int):
            limit_input = Prompt.ask("[cyan]  Limit number of records (leave empty for all)[/cyan]", default="")
            limit_value = int(limit_input) if limit_input.isdigit() else None
        else:
            limit_value = limit

        if not batch_size or not isinstance(batch_size, int):
            batch_size_value = Prompt.ask(
                "[cyan]  Batch size (concurrent records)[/cyan]",
                default=str(cfg.processing.batch_size)
            )
            batch_size_value = int(batch_size_value) if batch_size_value.isdigit() else cfg.processing.batch_size
        else:
            batch_size_value = batch_size

        if not force or not isinstance(force, bool):
            force_value = Confirm.ask("[cyan]  Force reprocess existing outputs?[/cyan]", default=False)
        else:
            force_value = force

        console.print()

        # Load file to count records for cost/time estimates
        console.print("[dim]Loading file to calculate estimates...[/dim]")
        df = load_tabular(input_path)
        total_rows = len(df)

        # Calculate actual records to process
        start_idx = skip_value
        end_idx = min(total_rows, start_idx + limit_value) if limit_value else total_rows
        total_records = max(0, end_idx - start_idx)

        console.print(f"[dim]File contains {total_rows} total rows, will process {total_records} records[/dim]")
        console.print()

        # Interactive LLM configuration with cost/time estimates
        console.print("[yellow]→ LLM Configuration[/yellow]")

        provider = select_provider_interactive(console, cfg)
        console.print()

        # Model selection with estimates based on actual record count (allow CLI override)
        if model and isinstance(model, str):
            selected_model = model
        else:
            selected_model = select_model_interactive(
                console, cfg, provider, total_records=total_records, batch_size=batch_size_value
            )
        console.print()

        # Always use "manual" strategy to use the selected model
        strategy = "manual"

        # Apply configuration overrides
        raw = cfg.model_dump(mode="python")
        raw.setdefault("llm", {}).update({"default_provider": provider})
        if selected_model:
            raw.setdefault("llm", {}).setdefault("openai", {})["model"] = selected_model
            raw.setdefault("llm", {}).setdefault("default_openai_model", selected_model)
        raw.setdefault("llm", {}).update({"model_selection_strategy": strategy})

        if batch_size_value is not None:
            raw.setdefault("processing", {})["batch_size"] = int(batch_size_value)

        # Set delay to 0
        raw.setdefault("processing", {})["delay_between_requests"] = 0

        cfg = ConfigSchema.model_validate(raw)

        # Show processing parameters summary
        from rich.table import Table
        console.print()
        summary = Table(title="Extraction Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        summary.add_column("Parameter", style="bold")
        summary.add_column("Value", style="cyan")

        summary.add_row("Pipeline", pipeline_name)
        summary.add_row("Config", str(config or 'interactive selection'))
        summary.add_row("Input File", str(input_path))
        summary.add_row("ID Column", id_column_value)
        summary.add_row("Skip Records", str(skip_value))
        summary.add_row("Limit Records", str(limit_value or 'all'))
        summary.add_row("Batch Size", str(batch_size_value))
        summary.add_row("Provider", provider)
        summary.add_row("Model", selected_model or "default")
        summary.add_row("Force Reprocess", "Yes" if force_value else "No")

        console.print(summary)

        # Initialize audit service to get session ID
        audit_dir = Path(cfg.output.directory) / "_audit"
        audit = AuditService(audit_dir)
        session_id = audit.session_id

        # Show output location
        console.print()
        output_path = Path(cfg.output.directory) / "sessions" / pipeline_name / session_id
        console.print(f"[dim]Session ID:[/dim] [cyan]{session_id}[/cyan]")
        console.print(f"[dim]Output Directory:[/dim] [cyan]{output_path}[/cyan]")

        # Confirmation before processing
        console.print()
        if not styled_confirm("Ready to process extraction?", default=True):
            console.print("[yellow]Extraction cancelled[/yellow]")
            raise typer.Exit(0)

        console.print()

        # Snapshot configuration (audit was already initialized above for session ID)
        audit.snapshot_config(cfg.model_dump(mode="json"))
        engine = ExtractionEngine(cfg, audit)

        # Run extraction
        asyncio.run(
            engine.run(
                input_path=input_path,
                id_column=id_column_value,
                skip=skip_value,
                limit=limit_value,
                force=force_value,
                strategy=strategy
            )
        )
    _add("run", run)
