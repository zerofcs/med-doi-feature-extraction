from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Callable

import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.prompt import IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.status import Status
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import Validator, ValidationError

from ..config.loader import ConfigLoader
from ..config.schema import ConfigSchema
from ..models import InputRecord
from ..utils import slugify


def detect_pipeline_configs() -> List[Path]:
    paths: List[Path] = []
    preferred = Path("config/pipelines")
    if preferred.exists():
        paths.extend(sorted(preferred.glob("*.yaml")))
    else:
        legacy = Path("pipelines")
        if legacy.exists():
            paths.extend(sorted(legacy.glob("*.yaml")))
    return paths


def discover_pipelines_meta() -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for path in detect_pipeline_configs():
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        pipeline = data.get("pipeline")
        if isinstance(pipeline, str):
            label = pipeline
        elif isinstance(pipeline, dict):
            label = pipeline.get("label") or path.stem.replace("_", " ").title()
        else:
            label = path.stem.replace("_", " ").title()
        cmd = slugify(label)
        if not cmd.startswith("extract-"):
            cmd = f"extract-{cmd}"
        metas.append({"path": path, "label": label, "slug": cmd})
    return metas


def select_config_interactive(console: Console) -> Path:
    env_path = os.getenv("MED_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Get pipeline metadata with labels
    metas = discover_pipelines_meta()
    if not metas:
        raise ValueError("No pipeline overlays found under 'config/pipelines/'.")

    # Create styled table
    table = Table(
        title="Available Configurations",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="blue"
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Configuration", style="cyan")
    table.add_column("Path", style="dim")

    for i, meta in enumerate(metas, 1):
        table.add_row(str(i), meta["label"], str(meta["path"]))

    console.print(table)
    idx = IntPrompt.ask("[bold cyan]→[/bold cyan] Select configuration", default=1)
    idx = max(1, min(idx, len(metas)))
    return metas[idx - 1]["path"]


def load_config(path: Optional[Path]) -> ConfigSchema:
    cfg_path = path or select_config_interactive(Console())
    loader = ConfigLoader(cfg_path)
    cfg = loader.load()
    return cfg


def load_tabular(input_path: Path | str) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def data_quality_table(df: pd.DataFrame, important_cols: Sequence[str]) -> Table:
    table = Table(
        title="Data Quality Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Column", style="bold")
    table.add_column("Non-empty %", justify="right")
    table.add_column("Non-null", justify="right")
    table.add_column("Total", justify="right")
    total = len(df)
    for col in important_cols:
        if col in df.columns:
            non_null = df[col].notna() & (df[col].astype(str).str.len() > 0)
            count = int(non_null.sum())
            pct = 100.0 * count / total if total else 0.0
            # Color code the percentage
            color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            table.add_row(col, f"[{color}]{pct:.1f}%[/{color}]", str(count), str(total))
    return table


def load_excel_data(path: Path | str, *, require_doi: bool = True) -> List[InputRecord]:
    df = load_tabular(path)
    records: List[InputRecord] = []
    for _, row in df.iterrows():
        data = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        try:
            rec = InputRecord.model_validate(data)
            if require_doi and not rec.DOI:
                continue
            records.append(rec)
        except Exception:
            continue
    return records


def choose_from_list(console: Console, title: str, options: List[str], default_index: int = 0) -> int:
    table = Table(title=title, show_header=False, box=box.SIMPLE)
    for i, opt in enumerate(options, 1):
        table.add_row(f"{i}", opt)
    console.print(table)
    idx = IntPrompt.ask("Select option #", default=default_index + 1)
    return max(1, min(idx, len(options))) - 1


def create_selection_table(
    title: str,
    items: List[Dict[str, Any]],
    columns: List[tuple[str, str]] = None,
    show_index: bool = True
) -> Table:
    """
    Create a styled selection table for user choices.

    Args:
        title: Table title
        items: List of items (dicts) to display
        columns: List of (column_name, dict_key) tuples. Defaults to [("Item", "label")]
        show_index: Whether to show index column

    Returns:
        Rich Table instance
    """
    if columns is None:
        columns = [("Item", "label")]

    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

    if show_index:
        table.add_column("#", justify="right", style="dim")

    for col_name, _ in columns:
        table.add_column(col_name)

    for i, item in enumerate(items, 1):
        row_data = []
        if show_index:
            row_data.append(str(i))
        for _, dict_key in columns:
            value = item.get(dict_key, "")
            row_data.append(str(value))
        table.add_row(*row_data)

    return table


def create_progress() -> Progress:
    """
    Create a Rich Progress instance with custom columns for extraction tasks.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def create_results_table(title: str = "Extraction Results") -> Table:
    """
    Create a live-updating results table.

    Args:
        title: Table title

    Returns:
        Rich Table instance
    """
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Record ID", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Fields", justify="right")
    table.add_column("Details", style="dim")
    return table


def add_result_row(table: Table, record_id: str, success: bool, fields_count: int = 0, details: str = ""):
    """
    Add a result row to the results table.

    Args:
        table: Table to add row to
        record_id: Record identifier
        success: Whether extraction succeeded
        fields_count: Number of fields extracted
        details: Additional details or error message
    """
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    fields = f"[green]{fields_count}[/green]" if success else "[dim]-[/dim]"
    details_text = details[:50] + "..." if len(details) > 50 else details
    table.add_row(record_id, status, fields, details_text)


def styled_confirm(message: str, default: bool = False) -> bool:
    """
    Show a styled confirmation prompt.

    Args:
        message: Confirmation message
        default: Default choice

    Returns:
        User's boolean choice
    """
    return Confirm.ask(f"[yellow]?[/yellow] {message}", default=default)


def show_spinner_status(console: Console, message: str):
    """
    Create a context manager for spinner status.

    Args:
        console: Console instance
        message: Status message

    Returns:
        Status context manager
    """
    return console.status(f"[bold green]{message}...", spinner="dots")


def select_provider_interactive(console: Console, cfg: ConfigSchema) -> str:
    """
    Interactive menu to select LLM provider.

    Args:
        console: Console instance
        cfg: Configuration schema

    Returns:
        Selected provider name ('openai' or 'ollama')
    """
    default_provider = cfg.llm.default_provider

    table = Table(title="Select LLM Provider", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="dim")

    providers = [
        ("openai", "OpenAI (GPT models)"),
        ("ollama", "Ollama (Local models)")
    ]

    for i, (provider_id, label) in enumerate(providers, 1):
        status = "[green]default[/green]" if provider_id == default_provider else ""
        table.add_row(str(i), label, status)

    console.print(table)

    default_choice = 1 if default_provider == "openai" else 2
    choice = IntPrompt.ask("[bold cyan]→[/bold cyan] Select provider", default=default_choice)

    return providers[choice - 1][0] if 1 <= choice <= len(providers) else default_provider


def select_model_interactive(
    console: Console,
    cfg: ConfigSchema,
    provider: str,
    total_records: Optional[int] = None,
    batch_size: Optional[int] = None
) -> str:
    """
    Interactive menu to select model with pricing and estimated cost/time.

    Args:
        console: Console instance
        cfg: Configuration schema
        provider: Selected provider ('openai' or 'ollama')
        total_records: Total number of records to process (for estimates)
        batch_size: Batch size for processing (for time estimates)

    Returns:
        Selected model name
    """
    if provider == "ollama":
        return cfg.llm.ollama.model  # Ollama has single model config

    # OpenAI models
    default_model = cfg.llm.default_openai_model
    models = cfg.llm.openai.models

    if not models:
        return default_model

    # Estimate average tokens per record
    avg_input_tokens = 1000  # Conservative estimate for medical abstracts
    avg_output_tokens = 200  # Typical extraction output

    # Create table with estimate columns if we have record count
    show_estimates = total_records is not None and total_records > 0

    table = Table(title="Select Model", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Input Cost", justify="right")
    table.add_column("Output Cost", justify="right")
    table.add_column("Speed", style="dim")

    if show_estimates:
        table.add_column("Est. Cost", justify="right", style="yellow")
        table.add_column("Est. Time", justify="right", style="yellow")

    model_list = list(models.items())
    for i, (model_name, model_cfg) in enumerate(model_list, 1):
        is_default = model_name == default_model
        name_display = f"[green]{model_name}[/green]" if is_default else model_name

        input_cost = f"${model_cfg.pricing.input_per_1m:.2f}/1M"
        output_cost = f"${model_cfg.pricing.output_per_1m:.2f}/1M"

        # Infer speed from threshold (lower threshold = faster/cheaper model)
        speed = "fast" if model_cfg.complexity_threshold < 0.5 else "medium" if model_cfg.complexity_threshold < 0.8 else "slow"

        row = [str(i), name_display, input_cost, output_cost, speed]

        if show_estimates and total_records:
            # Calculate estimated cost
            input_cost_total = (avg_input_tokens * total_records * model_cfg.pricing.input_per_1m) / 1_000_000
            output_cost_total = (avg_output_tokens * total_records * model_cfg.pricing.output_per_1m) / 1_000_000
            total_cost = input_cost_total + output_cost_total

            # Calculate estimated time (3-8 seconds per record depending on model)
            seconds_per_record = 3 if speed == "fast" else 5 if speed == "medium" else 8
            total_seconds = total_records * seconds_per_record
            if batch_size and batch_size > 1:
                # Adjust for concurrent processing
                total_seconds = total_seconds / batch_size

            # Format time nicely
            if total_seconds < 60:
                time_str = f"~{int(total_seconds)}s"
            elif total_seconds < 3600:
                time_str = f"~{int(total_seconds/60)}m"
            else:
                hours = int(total_seconds / 3600)
                mins = int((total_seconds % 3600) / 60)
                time_str = f"~{hours}h {mins}m"

            row.append(f"${total_cost:.2f}")
            row.append(time_str)

        table.add_row(*row)

    console.print(table)
    if show_estimates:
        console.print(f"[dim]Estimates based on {total_records} records, {avg_input_tokens} input tokens, {avg_output_tokens} output tokens per record[/dim]")

    default_choice = next((i for i, (name, _) in enumerate(model_list, 1) if name == default_model), 1)
    choice = IntPrompt.ask("[bold cyan]→[/bold cyan] Select model", default=default_choice)

    return model_list[choice - 1][0] if 1 <= choice <= len(model_list) else default_model


# Strategy selection removed - always use "manual" to use the selected model


def display_record_preview(console: Console, record: Dict[str, Any], id_column: str) -> None:
    """
    Display preview of a specific record before testing.

    Args:
        console: Console instance
        record: Record data dictionary
        id_column: Column name for record ID
    """
    table = Table(title="Record Preview", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value", style="cyan")

    # Show key fields
    key = record.get(id_column) or record.get("DOI") or "Unknown"
    table.add_row("ID", str(key))

    if "Title" in record:
        title = str(record["Title"])[:100] + "..." if len(str(record["Title"])) > 100 else str(record["Title"])
        table.add_row("Title", title)

    if "Abstract" in record or "Abstract Note" in record:
        abstract = record.get("Abstract") or record.get("Abstract Note") or ""
        abstract_preview = str(abstract)[:200] + "..." if len(str(abstract)) > 200 else str(abstract)
        table.add_row("Abstract", abstract_preview)

    console.print(table)


class DataFileValidator(Validator):
    """Validator for data file paths (CSV/Excel)."""

    def validate(self, document):
        text = document.text
        if not text:
            return  # Allow empty for default

        path = Path(text)
        if not path.exists():
            raise ValidationError(message="File does not exist")

        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise ValidationError(message="File must be .csv, .xlsx, or .xls")


def prompt_file_path(
    message: str = "Input file path",
    default: str = "data/final_test.csv",
    console: Optional[Console] = None
) -> Path:
    """
    Prompt for file path with autocomplete support.

    Args:
        message: Prompt message
        default: Default file path
        console: Optional Console instance for rich formatting

    Returns:
        Selected file path
    """
    if console:
        console.print(f"[cyan]{message}[/cyan] (Tab for autocomplete)")

    # Create path completer with filtering for data files
    completer = PathCompleter(
        only_directories=False,
        file_filter=lambda filename: filename.endswith(('.csv', '.xlsx', '.xls'))
    )

    # Prompt with autocomplete
    result = prompt(
        f"{message}: ",
        default=default,
        completer=completer,
        complete_while_typing=True,
        validator=DataFileValidator(),
        validate_while_typing=False
    )

    return Path(result.strip())

