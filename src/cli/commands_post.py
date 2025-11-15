from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
import yaml
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

from ..config.schema import ConfigSchema
from ..engine import ExtractionEngine
from ..models import Record
from ..services.audit_service import AuditService
from .helpers import load_config, load_tabular, create_progress, show_spinner_status
from ..utils import deep_merge
from .registry import add as _add


# Removed _flatten_result_payload (was used only by the export command)


def register(app: typer.Typer, console: Console) -> None:
    # Removed benchmark command to narrow CLI

    @app.command()
    def retry(
        config: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Pipeline overlay YAML"),
        session: Optional[str] = typer.Option(None, help="Session ID to retry failures from"),
    ):
        """Retry failed extractions from a previous session."""
        load_dotenv()
        cfg = load_config(config)
        audit_dir = Path(cfg.output.directory) / "_audit"

        # Locate failures file for session
        with show_spinner_status(console, "Locating failures"):
            if not session:
                files = sorted(audit_dir.glob("failures_*.jsonl"), reverse=True)
                if not files:
                    console.print("[yellow]No failures found[/yellow]")
                    raise typer.Exit(0)
                failures_path = files[0]
                session = failures_path.stem.replace("failures_", "")
            else:
                failures_path = audit_dir / f"failures_{session}.jsonl"

        if not failures_path.exists():
            console.print(f"[yellow]No failures file found for session {session}[/yellow]")
            raise typer.Exit(0)

        # Discover input metadata from events log
        events_path = audit_dir / f"events_{session}.jsonl"
        input_path = None
        id_column = "DOI"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "session_start":
                        input_path = obj.get("input_path")
                        id_column = obj.get("id_column") or id_column
                        break
                except Exception:
                    pass
        if not input_path:
            # Fallback prompt
            input_path = typer.prompt("Input CSV/Excel path for this session", default="data/final_test.csv")

        # Load input file
        df = load_tabular(Path(input_path))
        # Collect failures
        failures: List[Dict[str, Any]] = []
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                failures.append(obj)
            except Exception:
                continue

        # Retry plan from config extras
        extras = getattr(cfg, "model_extra", {}) or {}
        retry_plan: Dict[str, Any] = extras.get("retry_plan", {}) if isinstance(extras, dict) else {}

        console.print()
        console.print(f"[bold]Session:[/bold] [cyan]{session}[/cyan]")
        console.print(f"[bold]Failures:[/bold] [cyan]{len(failures)}[/cyan]")
        console.print()

        retried = 0
        recovered = 0
        remaining: List[Dict[str, Any]] = []

        # Use a separate audit session for retries
        retry_audit = AuditService(Path(cfg.output.directory) / f"_audit_retry_{session}")
        engine = ExtractionEngine(cfg, retry_audit)

        # Process retries with progress bar
        progress = create_progress()
        with progress:
            retry_task = progress.add_task("Retrying failures", total=len(failures))

            for entry in failures:
                key = entry.get("key")
                category = entry.get("failure_category") or "unknown"
                retry_count = int(entry.get("retry_count") or 0)
                plan = retry_plan.get(category, {}) if isinstance(retry_plan, dict) else {}
                max_retries = int(plan.get("max_retries", 1)) if isinstance(plan, dict) else 1

                if retry_count >= max_retries:
                    entry["retry_count"] = retry_count
                    remaining.append(entry)
                    progress.update(retry_task, advance=1)
                    continue

                # Build per-record config overrides
                overrides = plan.get("overrides", {}) if isinstance(plan, dict) else {}
                raw_cfg = cfg.model_dump(mode="python")
                if overrides:
                    raw_cfg = deep_merge(raw_cfg, overrides)
                cfg_one = ConfigSchema.model_validate(raw_cfg)

                # Recreate record from input
                row = None
                if id_column in df.columns:
                    try:
                        row = df[df[id_column].astype(str) == str(key)].iloc[0]
                    except Exception:
                        row = None
                if row is None:
                    entry["failure_category"] = "missing_input"
                    entry["retry_count"] = retry_count + 1
                    remaining.append(entry)
                    progress.update(retry_task, advance=1)
                    continue

                data = {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}
                record = Record(key=str(key), data=data)

                # Process using per-record engine; force to overwrite outputs
                engine_one = ExtractionEngine(cfg_one, retry_audit)
                retried += 1
                result = asyncio.run(engine_one.process_record_async(record, force=True))
                if result is not None:
                    recovered += 1
                else:
                    # Remains failed; increment count and keep
                    entry["retry_count"] = retry_count + 1
                    remaining.append(entry)

                progress.update(retry_task, advance=1)

        # Rewrite failures file with remaining
        with failures_path.open("w", encoding="utf-8") as f:
            for e in remaining:
                f.write(json.dumps(e) + "\n")

        # Summary table
        console.print()
        table = Table(title="Retry Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right", style="cyan")
        table.add_row("Retried", str(retried))
        table.add_row("Recovered", f"[green]{recovered}[/green]")
        table.add_row("Remaining", f"[yellow]{len(remaining)}[/yellow]" if remaining else "[green]0[/green]")
        recovery_rate = (recovered / retried * 100) if retried else 0
        table.add_row("Recovery Rate", f"{recovery_rate:.1f}%")
        console.print(table)
        console.print()
    _add("retry", retry)

    # Removed export command to narrow CLI

    @app.command(name="validate")
    def validate_output(
        config: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Pipeline overlay YAML"),
        source_dir: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Directory of extracted YAML files"),
    ):
        """Check quality of extraction results and field coverage."""
        load_dotenv()
        cfg = load_config(config)
        src = source_dir or Path(cfg.output.directory)

        # Find files
        with show_spinner_status(console, "Scanning result files"):
            files = sorted(Path(src).glob("*.yaml"))

        if not files:
            console.print(f"[yellow]No YAML files found in {src}[/yellow]")
            raise typer.Exit(0)

        console.print()
        console.print(f"[bold]Validating results from:[/bold] [cyan]{src}[/cyan]")
        console.print()

        # Analyze files
        confs: List[float] = []
        field_counts: Dict[str, int] = {}
        total = 0

        progress = create_progress()
        with progress:
            scan_task = progress.add_task("Analyzing results", total=len(files))

            for f in files:
                try:
                    data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                    total += 1
                    c = ((data.get("confidence") or {}).get("overall") if isinstance(data.get("confidence"), dict) else None)
                    if isinstance(c, (int, float)):
                        confs.append(float(c))
                    norm = data.get("normalized") or {}
                    if isinstance(norm, dict):
                        for k, v in norm.items():
                            if v not in (None, ""):
                                field_counts[k] = field_counts.get(k, 0) + 1
                except Exception:
                    pass
                progress.update(scan_task, advance=1)

        # Quality summary
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        high_conf = sum(1 for x in confs if x >= 0.7)
        med_conf = sum(1 for x in confs if 0.5 <= x < 0.7)
        low_conf = sum(1 for x in confs if x < 0.5)

        console.print()
        qtable = Table(title="Quality Validation Report", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        qtable.add_column("Metric", style="bold")
        qtable.add_column("Value", justify="right", style="cyan")
        qtable.add_row("Files scanned", str(total))
        qtable.add_row("Avg confidence", f"{avg_conf:.2f}")
        qtable.add_row("", "", end_section=True)
        qtable.add_row("[green]High (≥0.7)[/green]", f"[green]{high_conf}[/green] ({high_conf/total*100:.1f}%)" if total else "0")
        qtable.add_row("[yellow]Medium (0.5-0.7)[/yellow]", f"[yellow]{med_conf}[/yellow] ({med_conf/total*100:.1f}%)" if total else "0")
        qtable.add_row("[red]Low (<0.5)[/red]", f"[red]{low_conf}[/red] ({low_conf/total*100:.1f}%)" if total else "0")
        console.print(qtable)

        # Field coverage
        console.print()
        ctable = Table(title="Field Coverage", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        ctable.add_column("Field", style="bold")
        ctable.add_column("Count", justify="right")
        ctable.add_column("Coverage %", justify="right")
        for k, cnt in sorted(field_counts.items()):
            pct = 100.0 * cnt / total if total else 0.0
            color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            ctable.add_row(k, str(cnt), f"[{color}]{pct:.1f}%[/{color}]")
        console.print(ctable)
        console.print()
    _add("validate", validate_output)
