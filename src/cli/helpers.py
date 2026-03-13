from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.prompt import IntPrompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import Validator, ValidationError

from ..config.loader import ConfigLoader
from ..config.schema import ConfigSchema


def detect_pipeline_configs() -> List[Path]:
    paths: List[Path] = []
    preferred = Path("config/pipelines")
    if preferred.exists():
        paths.extend(sorted(preferred.glob("*.yaml")))
    return paths


def select_config_interactive(console: Console) -> Path:
    env_path = os.getenv("MED_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    configs = detect_pipeline_configs()
    if not configs:
        raise ValueError("No pipeline configs found under 'config/pipelines/'.")

    table = Table(title="Available Pipelines", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Pipeline", style="cyan")
    table.add_column("Path", style="dim")

    for i, path in enumerate(configs, 1):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        label = data.get("pipeline") or path.stem
        table.add_row(str(i), str(label), str(path))

    console.print(table)
    idx = IntPrompt.ask("[bold cyan]->[/bold cyan] Select pipeline", default=1)
    idx = max(1, min(idx, len(configs)))
    return configs[idx - 1]


def load_config(path: Optional[Path]) -> ConfigSchema:
    cfg_path = path or select_config_interactive(Console())
    return ConfigLoader(cfg_path).load()


def load_tabular(input_path: Path | str) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )


def styled_confirm(message: str, default: bool = False) -> bool:
    return Confirm.ask(f"[yellow]?[/yellow] {message}", default=default)


def select_provider_interactive(console: Console, cfg: ConfigSchema) -> str:
    default_provider = cfg.llm.default_provider

    table = Table(title="LLM Provider", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Provider", style="cyan")
    table.add_column("", style="dim")

    providers = [("openai", "OpenAI"), ("ollama", "Ollama (local)")]
    for i, (pid, label) in enumerate(providers, 1):
        marker = "[green]default[/green]" if pid == default_provider else ""
        table.add_row(str(i), label, marker)

    console.print(table)
    default_choice = 1 if default_provider == "openai" else 2
    choice = IntPrompt.ask("[bold cyan]->[/bold cyan] Select provider", default=default_choice)
    return providers[choice - 1][0] if 1 <= choice <= len(providers) else default_provider


def select_model_interactive(
    console: Console,
    cfg: ConfigSchema,
    provider: str,
    total_records: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> str:
    if provider == "ollama":
        return cfg.llm.ollama.model

    default_model = cfg.llm.default_openai_model
    models = cfg.llm.openai.models
    if not models:
        return default_model

    show_estimates = total_records is not None and total_records > 0

    table = Table(title="Model", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    if show_estimates:
        table.add_column("Est. Cost", justify="right", style="yellow")

    model_list = list(models.items())
    for i, (name, mcfg) in enumerate(model_list, 1):
        display = f"[green]{name}[/green]" if name == default_model else name
        row = [str(i), display, f"${mcfg.pricing.input_per_1m:.2f}", f"${mcfg.pricing.output_per_1m:.2f}"]
        if show_estimates and total_records:
            cost = (1000 * total_records * mcfg.pricing.input_per_1m + 200 * total_records * mcfg.pricing.output_per_1m) / 1_000_000
            row.append(f"${cost:.2f}")
        table.add_row(*row)

    console.print(table)
    default_choice = next((i for i, (n, _) in enumerate(model_list, 1) if n == default_model), 1)
    choice = IntPrompt.ask("[bold cyan]->[/bold cyan] Select model", default=default_choice)
    return model_list[choice - 1][0] if 1 <= choice <= len(model_list) else default_model


class _DataFileValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text:
            return
        p = Path(text)
        if not p.exists():
            raise ValidationError(message="File does not exist")
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise ValidationError(message="Must be .csv, .xlsx, or .xls")


def prompt_file_path(
    message: str = "Input file path",
    default: str = "data/final_test.csv",
    console: Optional[Console] = None,
) -> Path:
    if console:
        console.print(f"[cyan]{message}[/cyan] (Tab for autocomplete)")

    completer = PathCompleter(
        only_directories=False,
        file_filter=lambda f: f.endswith((".csv", ".xlsx", ".xls")),
    )
    result = prompt(
        f"{message}: ",
        default=default,
        completer=completer,
        complete_while_typing=True,
        validator=_DataFileValidator(),
        validate_while_typing=False,
    )
    return Path(result.strip())
