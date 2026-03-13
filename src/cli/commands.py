from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ..config.schema import ConfigSchema
from ..engine import ExtractionEngine
from ..models import Record
from ..services.audit_service import AuditService
from .helpers import (
    load_config,
    load_tabular,
    select_provider_interactive,
    select_model_interactive,
    create_progress,
    prompt_file_path,
    styled_confirm,
)


def register(app: typer.Typer, console: Console) -> None:

    @app.command()
    def run(
        config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True, help="Pipeline config YAML"),
        input: Optional[Path] = typer.Option(None, "--input", "-i", exists=True, readable=True, help="Input CSV/Excel"),
        id_column: Optional[str] = typer.Option(None, "--id-column", help="Primary key column (default from config or DOI)"),
        skip: int = typer.Option(0, help="Skip first N records"),
        limit: Optional[int] = typer.Option(None, help="Limit number of records"),
        batch_size: Optional[int] = typer.Option(None, help="Concurrent records per batch"),
        provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai/ollama)"),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model name (skips interactive selection)"),
        force: bool = typer.Option(False, help="Force reprocess existing outputs"),
    ):
        """Run a pipeline (extraction or screening) on a dataset."""
        cfg = load_config(config)
        task_type = cfg.task or "extract"
        pipeline_name = cfg.pipeline or "pipeline"

        # Resolve input file
        if not input:
            default_file = "data/data-abstracts.csv" if task_type == "screen" else "data/final_test.csv"
            input_path = prompt_file_path("Input file", default=default_file, console=console)
        else:
            input_path = input

        # Resolve id column: flag > config > default
        id_col = id_column or (cfg.input.id_column if cfg.input else "DOI")

        # Resolve batch size: flag > config
        bs = batch_size or cfg.processing.batch_size

        # Load data for record count
        console.print()
        df = load_tabular(input_path)
        total_rows = len(df)
        actual = max(0, min(total_rows - skip, limit or total_rows - skip))
        console.print(f"[dim]{total_rows} rows in file, will process {actual}[/dim]")
        console.print()

        # Provider and model selection -- skip interactive if flags given
        if provider:
            selected_provider = provider
        else:
            console.print("[yellow]LLM Configuration[/yellow]")
            selected_provider = select_provider_interactive(console, cfg)
            console.print()

        if model:
            selected_model = model
        else:
            selected_model = select_model_interactive(console, cfg, selected_provider, total_records=actual, batch_size=bs)
        console.print()

        # Apply overrides
        raw = cfg.model_dump(mode="python")
        raw.setdefault("llm", {}).update({
            "default_provider": selected_provider,
            "model_selection_strategy": "manual",
        })
        if selected_model:
            raw.setdefault("llm", {}).setdefault("openai", {})["model"] = selected_model
            raw["llm"]["default_openai_model"] = selected_model
        raw.setdefault("processing", {})["batch_size"] = bs
        raw.setdefault("processing", {})["delay_between_requests"] = 0
        cfg = ConfigSchema.model_validate(raw)

        # Summary
        summary = Table(title=f"{task_type.upper()} Pipeline: {pipeline_name}", box=box.ROUNDED, show_header=False)
        summary.add_column("Param", style="bold")
        summary.add_column("Value", style="cyan")
        summary.add_row("Input", str(input_path))
        summary.add_row("ID Column", id_col)
        summary.add_row("Records", f"{actual} (skip={skip}, limit={limit or 'all'})")
        summary.add_row("Batch Size", str(bs))
        summary.add_row("Provider/Model", f"{selected_provider} / {selected_model or 'default'}")
        summary.add_row("Force Reprocess", "yes" if force else "no")
        if task_type == "screen" and cfg.screen:
            summary.add_row("Topic", cfg.screen.topic[:80])
        console.print(summary)
        console.print()

        if not styled_confirm("Start processing?", default=True):
            raise typer.Exit(0)

        # Run
        audit = AuditService(Path(cfg.output.directory) / "_audit")
        console.print(f"\n[dim]Session: {audit.session_id}[/dim]")
        audit.snapshot_config(cfg.model_dump(mode="json"))
        engine = ExtractionEngine(cfg, audit)

        asyncio.run(engine.run(
            input_path=input_path,
            id_column=id_col,
            skip=skip,
            limit=limit,
            force=force,
            strategy="manual",
        ))

        # Final summary
        s = audit.summary
        console.print()
        result_table = Table(title="Results", box=box.ROUNDED, show_header=False)
        result_table.add_column("Metric", style="bold")
        result_table.add_column("Value", style="cyan")
        result_table.add_row("Processed", str(s.succeeded))
        result_table.add_row("Skipped", str(s.skipped))
        result_table.add_row("Failed", str(s.failed))
        result_table.add_row("Total Cost", f"${s.cost_total:.4f}")
        result_table.add_row("Output", str(engine.output.session_dir))
        console.print(result_table)
        console.print()

    @app.command()
    def test(
        config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True, help="Pipeline config YAML"),
        input: Optional[Path] = typer.Option(None, "--input", "-i", exists=True, readable=True, help="Input CSV/Excel"),
        id_column: Optional[str] = typer.Option(None, "--id-column", help="Primary key column"),
        skip: int = typer.Option(0, help="Record index to test (0-based)"),
        provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai/ollama)"),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model name"),
    ):
        """Test a pipeline on a single record."""
        cfg = load_config(config)
        task_type = cfg.task or "extract"
        pipeline_name = cfg.pipeline or "pipeline"

        # Resolve input file
        if not input:
            default_file = "data/data-abstracts.csv" if task_type == "screen" else "data/final_test.csv"
            input_path = prompt_file_path("Input file", default=default_file, console=console)
        else:
            input_path = input

        id_col = id_column or (cfg.input.id_column if cfg.input else "DOI")

        # Load data
        df = load_tabular(input_path)
        if skip >= len(df):
            console.print(f"[red]Skip ({skip}) exceeds row count ({len(df)})[/red]")
            raise typer.Exit(1)

        row = df.iloc[skip]
        data = {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
        key = str(data.get(id_col) or data.get("DOI") or skip)
        record = Record(key=key, data=data)

        # Show record preview
        console.print()
        console.print(f"[bold]{task_type.upper()}[/bold] pipeline: [cyan]{pipeline_name}[/cyan]")
        console.print(f"Record: [cyan]{key}[/cyan]")
        if "Title" in data:
            title = str(data["Title"])[:120]
            console.print(f"Title: {title}")
        if "Abstract" in data:
            abstract = str(data["Abstract"])[:200]
            console.print(f"Abstract: {abstract}...")
        console.print()

        # Provider and model selection -- skip interactive if flags given
        if provider:
            selected_provider = provider
        else:
            selected_provider = select_provider_interactive(console, cfg)
            console.print()

        if model:
            selected_model = model
        else:
            selected_model = select_model_interactive(console, cfg, selected_provider)
        console.print()

        # Apply overrides
        raw = cfg.model_dump(mode="python")
        raw.setdefault("llm", {}).update({
            "default_provider": selected_provider,
            "model_selection_strategy": "manual",
        })
        if selected_model:
            raw.setdefault("llm", {}).setdefault("openai", {})["model"] = selected_model
            raw["llm"]["default_openai_model"] = selected_model
        cfg = ConfigSchema.model_validate(raw)

        # Run
        audit = AuditService(Path(cfg.output.directory) / "_audit")
        engine = ExtractionEngine(cfg, audit)

        with console.status("[bold green]Processing...", spinner="dots"):
            result = asyncio.run(engine.process_record_async(record, force=True, strategy="manual"))

        if not result:
            console.print("[red]Extraction failed -- no result returned[/red]")
            raise typer.Exit(1)

        # Display result
        console.print()
        table = Table(title="Result", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Field", style="bold")
        table.add_column("Value", style="cyan")

        for k, v in (result.normalized or result.extracted).items():
            table.add_row(str(k), str(v))

        table.add_row("", "", end_section=True)
        table.add_row("Valid", "[green]yes[/green]" if result.valid else "[red]no[/red]")
        if result.errors:
            table.add_row("Errors", ", ".join(result.errors))
        table.add_row("Provider", result.transparency.provider)
        table.add_row("Model", result.transparency.model)
        if result.transparency.cost:
            table.add_row("Cost", f"${result.transparency.cost:.4f}")

        console.print(table)
        console.print()
