#!/usr/bin/env python3
"""
Main CLI application for medical literature DOI scraping and analysis.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd
import yaml
import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
import uuid

from .core.models import InputRecord, BenchmarkResult
from .extractors.doi_extractor import DOIExtractor
from .core.audit import AuditLogger
from .core.quality import QualityValidator

# Initialize Typer app with custom context settings
app = typer.Typer(
    name="med-doi-scraper",
    help="Research-grade medical literature DOI scraping and analysis pipeline",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "-help", "--help"]}
)

console = Console()


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update dict 'base' with 'override' keys, returning base."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _get_config_title(path: Path) -> str:
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        title = data.get('title')
        if isinstance(title, str) and title.strip():
            return title.strip()
    except Exception:
        pass
    # Fallback to filename-based title
    name = path.stem.replace('settings.', '').replace('_', ' ').title()
    return name


def load_config() -> dict:
    """Load configuration from file."""
    # Allow override via env var so the interactive menu can switch configs
    override_path = os.getenv("MED_CONFIG_PATH")
    overlay_path = Path(override_path) if override_path else None

    # Determine overlay default
    if overlay_path is None:
        for cand in [Path("config/settings.doi.yaml"), Path("config/settings.country.yaml")]:
            if cand.exists():
                overlay_path = cand
                break

    base_path = Path("config/settings.base.yaml")
    if not overlay_path and not base_path.exists():
        console.print("[red]Error: no config file found. Expected one of:[/red]")
        console.print("  - config/settings.base.yaml")
        console.print("  - config/settings.doi.yaml")
        console.print("  - config/settings.country.yaml")
        console.print("Set MED_CONFIG_PATH to a valid config path or run the CLI interactively to choose one.")
        raise typer.Exit(1)

    # Load base, then overlay
    config = {}
    if base_path.exists():
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        # Remove title from loaded config to avoid leaking into runtime dict
        config.pop('title', None)

    if overlay_path and overlay_path.exists():
        with open(overlay_path, 'r') as f:
            overlay = yaml.safe_load(f) or {}
        overlay.pop('title', None)
        config = _deep_update(config, overlay)

    return config


def _choose_from_list(prompt_text: str, options: list, default: Optional[str] = None) -> str:
    """Interactive numeric selector; falls back to typed value if matched.

    Shows a numbered list and accepts either the number or the exact option value.
    """
    if not options:
        raise ValueError("No options to choose from")
    # Determine default index
    default_index = 0
    if default and default in options:
        default_index = options.index(default)

    # Display menu
    console.print(f"\n[bold]{prompt_text}[/bold]")
    for i, opt in enumerate(options, start=1):
        if i - 1 == default_index:
            console.print(f"  {i}) {opt} [dim](default)[/dim]")
        else:
            console.print(f"  {i}) {opt}")

    valid_choices = [str(i) for i in range(1, len(options) + 1)] + options
    choice = Prompt.ask(
        "Select",
        choices=valid_choices,
        default=str(default_index + 1),
    )
    if choice.isdigit():
        idx = int(choice) - 1
        return options[idx]
    return choice


def _openai_model_options(config: dict) -> List[str]:
    models = list((config.get('llm', {}).get('openai', {}).get('models', {}) or {}).keys())
    if models:
        return models
    # Sensible fallback
    return ["gpt-5-nano", "gpt-5-mini", "gpt-5"]


def _default_openai_model(config: dict) -> str:
    return config.get('llm', {}).get('default_openai_model', 'gpt-5-nano')


def _list_available_configs() -> list:
    """List available config settings files to choose from."""
    cfg_dir = Path("config")
    if not cfg_dir.exists():
        return []
    # Only list pipeline-specific settings files; base is not selectable
    configs = sorted([p for p in cfg_dir.glob("settings.*.y*ml") if p.name != "settings.base.yaml"]) 
    return configs


def _select_config() -> bool:
    """Config selection submenu. Returns True if a config was selected and set."""
    while True:
        configs = _list_available_configs()
        if not configs:
            console.print("[yellow]No config files found in ./config[/yellow]")
            # Offer to enter custom
            no_cfg_choice = Prompt.ask("Enter a custom path (p) or back (b)", choices=["p", "b"], default="b")
            if no_cfg_choice == "b":
                return False
            custom = Prompt.ask("Enter path to YAML config", default="config/settings.yaml")
            path = Path(custom)
            if path.exists():
                os.environ["MED_CONFIG_PATH"] = str(path)
                console.print(f"\n[green]✓ Using config:[/green] {path}")
                return True
            console.print(f"[red]Config file not found: {path}[/red]")
            continue

        console.print("\n[bold]Available configs:[/bold]")
        for i, c in enumerate(configs, start=1):
            title = _get_config_title(c)
            console.print(f"  {i}) {title} [dim]({c})[/dim]")
        console.print("  p) Enter a custom path")
        console.print("  b) Back")

        valid_numeric = [str(i) for i in range(1, len(configs) + 1)]
        selection = Prompt.ask(
            "Pick a config",
            choices=valid_numeric + ["p", "b"],
            default="1",
        )

        if selection == "b":
            return False
        if selection == "p":
            custom = Prompt.ask("Enter path to YAML config", default="config/settings.yaml")
            path = Path(custom)
        else:
            path = configs[int(selection) - 1]

        if not path.exists():
            console.print(f"[red]Config file not found: {path}[/red]")
            continue

        os.environ["MED_CONFIG_PATH"] = str(path)
        console.print(f"\n[green]✓ Using config:[/green] {path}")
        return True


@app.callback(invoke_without_command=True)
def entry(ctx: typer.Context):
    """Interactive entrypoint when no subcommand is provided."""
    if ctx.invoked_subcommand:
        return

    while True:
        console.print("\n[bold cyan]LIM - Data Feature Extraction[/bold cyan]")
        console.print("[bold]Main menu:[/bold]")
        console.print("  c) Choose config")
        console.print("  g) Generate new config (agent/wizard)")
        console.print("  h) Help")
        console.print("  q) Quit")

        selection = Prompt.ask(
            "Select option",
            choices=["c", "g", "h", "q"],
            default="c",
        )

        if selection == "q":
            raise typer.Exit(0)
        if selection == "h":
            typer.echo(typer.main.get_command(app).get_help(ctx))
            continue
        if selection == "g":
            try:
                generate_extractor(fallback_wizard=True)  # type: ignore[name-defined]
            except Exception as e:
                console.print(f"[red]Error starting generator: {e}[/red]")
            continue
        if selection == "c":
            if _select_config():
                _interactive_actions()
            # After actions loop or back, show main menu again
            continue


def load_excel_data(file_path: str, skip: int = 0, limit: Optional[int] = None) -> Tuple[List[InputRecord], int]:
    """Load data from Excel file with skip and limit support.
    
    Returns:
        - List of InputRecord objects
        - Total number of records with DOIs in the file (before skip/limit)
    """
    try:
        df = pd.read_excel(file_path)
        
        # First, filter for records with DOIs and create records
        all_records = []
        for idx, row in df.iterrows():
            try:
                # Helper to handle NaN values
                def get_value(key, default=None):
                    val = row.get(key)
                    if pd.isna(val):
                        return default
                    return val
                
                record = InputRecord(
                    key=get_value('Key', ''),
                    item_type=get_value('Item Type'),
                    publication_year=get_value('Publication Year'),
                    author=get_value('Author'),
                    number_of_authors=get_value('Number of Authors'),
                    title=get_value('Title'),
                    publication_title=get_value('Publication Title'),
                    isbn=get_value('ISBN'),
                    issn=get_value('ISSN'),
                    doi=get_value('DOI'),
                    url=get_value('Url'),
                    abstract_note=get_value('Abstract Note'),
                    date=str(get_value('Date')) if get_value('Date') else None,
                    issue=str(get_value('Issue')) if get_value('Issue') else None,
                    volume=str(get_value('Volume')) if get_value('Volume') else None,
                    journal_abbreviation=get_value('Journal Abbreviation'),
                    author_affiliation_new=get_value('Author Affiliation New'),
                    trimmed_author_list=get_value('Trimmed Author List')
                )
                if record.doi:  # Only include records with DOI
                    all_records.append(record)
            except Exception as e:
                console.print(f"[yellow]Warning: Skipping record {idx}: {e}[/yellow]")
        
        total_with_doi = len(all_records)
        
        # Apply skip and limit
        if skip > 0:
            all_records = all_records[skip:]
        
        if limit:
            all_records = all_records[:limit]
        
        return all_records, total_with_doi
    
    except Exception as e:
        console.print(f"[red]Error loading Excel file: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def preview(
    file: str = typer.Option("data-source.xlsx", "--file", "-f", help="Path to Excel file containing DOI records"),
    skip: int = typer.Option(0, "--skip", "-s", help="Number of records to skip before showing preview"),
    rows: int = typer.Option(10, "--rows", "-r", help="Number of rows to display in preview")
):
    """
    Preview data from the Excel spreadsheet.
    
    Examples:
        python cli.py preview
        python cli.py preview --skip 100 --rows 5
        python cli.py preview -f mydata.xlsx -s 50 -r 20
    """
    console.print(f"\n[bold]Previewing {file}...[/bold]\n")
    
    try:
        df = pd.read_excel(file)
        
        # Show summary
        console.print(f"[green]Total rows:[/green] {len(df)}")
        console.print(f"[green]Total columns:[/green] {len(df.columns)}")
        
        # Data quality summary
        console.print("\n[bold]Data Quality Summary:[/bold]")
        quality_table = Table(show_header=True, header_style="bold magenta")
        quality_table.add_column("Field", style="cyan")
        quality_table.add_column("Available", justify="right")
        quality_table.add_column("Percentage", justify="right")
        
        fields = ['DOI', 'Abstract Note', 'Author', 'Title', 'Publication Year']
        for field in fields:
            if field in df.columns:
                count = df[field].notna().sum()
                percentage = (count / len(df)) * 100
                quality_table.add_row(
                    field,
                    str(count),
                    f"{percentage:.1f}%"
                )
        
        console.print(quality_table)
        
        # Show sample records
        doi_records = df[df['DOI'].notna()]
        total_doi_records = len(doi_records)
        
        # Apply skip
        if skip > 0:
            doi_records = doi_records.iloc[skip:]
            console.print(f"[yellow]Skipping first {skip} records[/yellow]")
        
        # Apply limit
        doi_records = doi_records.head(rows)
        
        if skip > 0:
            console.print(f"\n[bold]Records {skip+1}-{skip+len(doi_records)} of {total_doi_records} total with DOIs:[/bold]")
        else:
            console.print(f"\n[bold]First {rows} records with DOIs:[/bold]")
        
        for idx, row in doi_records.iterrows():
            console.print(f"\n[yellow]Record {idx + 1}:[/yellow]")
            console.print(f"  DOI: {row.get('DOI', 'N/A')}")
            console.print(f"  Title: {str(row.get('Title', 'N/A'))[:100]}...")
            console.print(f"  Authors: {str(row.get('Author', 'N/A'))[:100]}...")
            if pd.notna(row.get('Abstract Note')):
                console.print(f"  Abstract: {str(row['Abstract Note'])[:150]}...")
            else:
                console.print(f"  Abstract: [red]Missing[/red]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test(
    file: str = typer.Option("data-source.xlsx", "--file", "-f", help="Path to Excel file containing DOI records"),
    skip: int = typer.Option(0, "--skip", "-s", help="Number of records to skip before selecting test record"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use (ollama/openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific OpenAI model (gpt-5-nano/gpt-5-mini/gpt-5)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Model selection strategy (cost-optimized/balanced/accuracy-first)")
):
    """
    Test extraction with a single record.
    
    Useful for debugging and verifying the pipeline works correctly.
    
    Examples:
        python cli.py test
        python cli.py test --skip 100 --provider openai
        python cli.py test -s 50 -p ollama
    """
    console.print("\n[bold]Running test extraction...[/bold]\n")
    
    config = load_config()
    config['processing']['test_mode'] = True
    config['processing']['test_limit'] = 1
    
    # Load one record
    records, total = load_excel_data(file, skip=skip, limit=1)
    if not records:
        console.print("[red]No records with DOI found[/red]")
        raise typer.Exit(1)
    
    record = records[0]
    if skip > 0:
        console.print(f"[yellow]Skipped {skip} records[/yellow]")
    console.print(f"[green]Testing record {skip + 1} of {total} total records with DOIs[/green]")
    console.print(f"[green]DOI:[/green] {record.doi}")
    console.print(f"[green]Title:[/green] {record.title[:100] if record.title else 'N/A'}...")
    
    # Interactive provider/model selection if not provided
    if provider is None:
        provider = _choose_from_list("Provider", ["ollama", "openai"], default=config.get('llm', {}).get('default_provider', 'openai'))
    if provider == 'openai' and model is None and strategy is None:
        # Let user choose model directly; strategy remains optional
        model = _choose_from_list("OpenAI model", _openai_model_options(config), default=_default_openai_model(config))

    # Apply model configuration
    if provider:
        config['llm']['default_provider'] = provider
    if model:
        config['llm']['default_openai_model'] = model
    if strategy:
        config['llm']['model_selection_strategy'] = strategy
    
    # Initialize components
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_logger = AuditLogger(session_id, config=config)
    engine = DOIExtractor(config, audit_logger, session_id)
    
    # Run extraction
    async def run_test():
        result, error = await engine.extract_from_record(
            record, 
            force_provider=provider, 
            force_model=model
        )
        return result, error
    
    with console.status("[bold green]Extracting..."):
        result, error = asyncio.run(run_test())
    
    if error:
        console.print(f"\n[red]Extraction failed:[/red] {error}")
    else:
        console.print("\n[green]Extraction successful![/green]")
        
        if result:
            # Display results
            console.print("\n[bold]Extracted Data (Three Fields):[/bold]")
            console.print(f"  Year: {result.year}")
            
            console.print(f"\n  [cyan]FIELD 1 - Study Design:[/cyan]")
            console.print(f"    {result.study_design if result.study_design else 'N/A'}")
            if result.study_design_other:
                console.print(f"    Other specification: {result.study_design_other}")
            
            console.print(f"\n  [cyan]FIELD 2 - Subspecialty Focus:[/cyan]")
            console.print(f"    {result.subspecialty_focus if result.subspecialty_focus else 'N/A'}")
            if result.subspecialty_focus_other:
                console.print(f"    Other specification: {result.subspecialty_focus_other}")
            
            console.print(f"\n  [cyan]FIELD 3 - Priority Topic:[/cyan]")
            if getattr(result, 'priority_topic', None):
                console.print(f"    {result.priority_topic}")
            else:
                console.print("    N/A")
            
            console.print(f"\n  [green]Overall Confidence:[/green] {result.confidence_scores.overall:.2f}")
            console.print(f"  [yellow]Human Review Required:[/yellow] {result.transparency_metadata.human_review_required}")
            
            if result.transparency_metadata.warning_logs:
                console.print(f"  [yellow]Warnings:[/yellow]")
                for warning in result.transparency_metadata.warning_logs:
                    console.print(f"    • {warning}")
            
            if result.transparency_metadata.validation_flags:
                console.print(f"  [red]Validation Flags:[/red] {', '.join(result.transparency_metadata.validation_flags)}")
    
    audit_logger.finalize_session()


@app.command(name="extract-doi")
def extract_doi(
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to Excel file containing DOI records"),
    skip: Optional[int] = typer.Option(None, "--skip", "-s", help="Number of records to skip before processing"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of records to process"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use (ollama/openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific OpenAI model (gpt-5-nano/gpt-5-mini/gpt-5)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Model selection strategy (cost-optimized/balanced/accuracy-first)"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Maximum cost per extraction ($)"),
    force: Optional[bool] = typer.Option(None, "--force", help="Force reprocess existing DOIs (ignore cache)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Number of records to process concurrently")
):
    """
    Run DOI extraction pipeline on medical literature dataset.

    Processes medical literature records to extract three classification fields:
    1. Study Design
    2. Subspecialty Focus
    3. Priority Topic (single selection)

    Interactive mode: Run without flags to be prompted for all options.

    Examples:
        python cli.py extract-doi                          # Interactive mode
        python cli.py extract-doi --skip 100 --limit 50    # Process records 101-150
        python cli.py extract-doi -s 500 -l 100 -p openai  # Skip 500, process 100 with OpenAI
        python cli.py extract-doi --force --batch-size 10  # Reprocess all with batch size 10
    """
    console.print("\n[bold]Medical Literature Extraction Pipeline[/bold]\n")
    
    # Interactive prompts for missing parameters
    if file is None:
        file = Prompt.ask("[cyan]Excel file to process[/cyan]", default="data-source.xlsx")
    
    # First load to show stats only if we need to prompt for skip/limit
    if skip is None or limit is None:
        console.print(f"Loading data from {file}...")
        preview_records, total_with_doi = load_excel_data(file, skip=0, limit=None)
        console.print(f"[green]Total records with DOIs: {total_with_doi}[/green]")
        
        if skip is None:
            skip = IntPrompt.ask("[cyan]Number of records to skip[/cyan]", default=0)
        
        if limit is None:
            remaining = total_with_doi - skip
            console.print(f"[dim]Remaining after skip: {remaining} records[/dim]")
            limit_input = Prompt.ask(
                f"[cyan]Number of records to process (Enter for all {remaining})[/cyan]",
                default=""
            )
            limit = int(limit_input) if limit_input else None
    else:
        # Set defaults if provided
        if skip is None:
            skip = 0
        # Default file if not set
        if file is None:
            file = "data-source.xlsx"
    
    if provider is None:
        cfg_for_defaults = load_config()
        provider = _choose_from_list(
            "LLM provider",
            ["ollama", "openai"],
            default=cfg_for_defaults.get('llm', {}).get('default_provider', 'openai')
        )
    
    # Handle OpenAI model selection
    if provider == "openai":
        if model is None and strategy is None:
            model_choice = _choose_from_list("OpenAI model selection", ["strategy", "specific-model"], default="strategy")
            if model_choice == "strategy":
                strategy = _choose_from_list(
                    "Model selection strategy",
                    ["cost-optimized", "balanced", "accuracy-first", "manual"],
                    default=cfg_for_defaults.get('llm', {}).get('model_selection_strategy', 'cost-optimized')
                )
            else:
                model = _choose_from_list(
                    "OpenAI model",
                    _openai_model_options(cfg_for_defaults),
                    default=_default_openai_model(cfg_for_defaults)
                )
    
    if force is None:
        force = Confirm.ask("[cyan]Force reprocess existing DOIs?[/cyan]", default=False)
    
    if batch_size is None:
        cfg_processing_default = cfg_for_defaults.get('processing', {}).get('batch_size', 1)
        batch_size = IntPrompt.ask("[cyan]Batch size for concurrent processing[/cyan]", default=int(cfg_processing_default))
    
    console.print("\n[bold]Medical Literature Extraction Pipeline[/bold]\n")
    
    # Load configuration
    config = load_config()
    if force:
        config['processing']['force_reprocess'] = True
    if batch_size:
        config['processing']['batch_size'] = batch_size
    
    # Apply model selection configuration
    if provider:
        config['llm']['default_provider'] = provider
    if model:
        config['llm']['default_openai_model'] = model
    if strategy:
        config['llm']['model_selection_strategy'] = strategy
    if max_cost:
        config['llm']['openai']['cost_limits']['max_cost_per_extraction'] = max_cost
    
    # Load data with skip and limit
    console.print(f"\nLoading records from {file}...")
    records, total = load_excel_data(file, skip=skip, limit=limit)
    
    # Display processing range
    if skip > 0 or limit:
        start_record = skip + 1
        end_record = skip + len(records)
        console.print(f"[yellow]Processing records {start_record}-{end_record} of {total} total[/yellow]")
    else:
        console.print(f"[green]Processing all {len(records)} records with DOIs[/green]")
    
    if not records:
        console.print("[red]No records to process[/red]")
        raise typer.Exit(1)
    
    # Confirm processing
    if len(records) > 10:
        if not Confirm.ask(f"Process {len(records)} records?"):
            raise typer.Exit(0)
    
    # Initialize components
    session_id = f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    audit_logger = AuditLogger(session_id, config=config)
    engine = DOIExtractor(config, audit_logger, session_id)
    
    console.print(f"\n[bold]Session ID:[/bold] {session_id}")
    console.print(f"[bold]Output directory:[/bold] {config['output']['directory']}")
    console.print(f"[bold]LLM Provider:[/bold] {config['llm']['default_provider']}")
    console.print("")
    
    # Process records
    async def process_all():
        results = []
        
        # Counters for different categories
        reduced_confidence_count = 0
        failed_count = 0
        processed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Three separate progress bars
            overall_task = progress.add_task(
                "[cyan]Overall progress...",
                total=len(records)
            )
            reduced_confidence_task = progress.add_task(
                "[yellow]Reduced confidence (missing abstract)...",
                total=len(records)
            )
            failed_task = progress.add_task(
                "[red]Failed extractions...",
                total=len(records)
            )
            
            batch_size = config['processing']['batch_size']
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_results = await engine.process_batch(batch, batch_size=batch_size, force_provider=provider, force_model=model)
                results.extend(batch_results)
                
                # Process batch results and update counters
                for result, error in batch_results:
                    processed_count += 1
                    
                    # Check if this record has reduced confidence due to missing abstract
                    if result:
                        # Reduced confidence due to known warnings (e.g., missing abstract)
                        if any("Missing abstract" in w for w in result.transparency_metadata.warning_logs):
                            reduced_confidence_count += 1
                    
                    # Check if failed
                    if error and "Already processed" not in error:
                        failed_count += 1
                
                # Update progress bars
                progress.update(overall_task, completed=processed_count)
                progress.update(reduced_confidence_task, completed=reduced_confidence_count)
                progress.update(failed_task, completed=failed_count)
        
        return results
    
    # Run extraction
    results = asyncio.run(process_all())
    
    # Finalize session
    summary = audit_logger.finalize_session()
    
    # Display summary
    console.print("\n" + "="*60)
    console.print("[bold green]Extraction Complete![/bold green]")
    console.print("="*60)
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Total Records", str(summary.total_records))
    summary_table.add_row("Successful", str(summary.successful_extractions))
    summary_table.add_row("Failed", str(summary.failed_extractions))
    summary_table.add_row("Skipped (Already Processed)", str(summary.skipped_already_processed))
    summary_table.add_row("Average Processing Time", f"{summary.average_processing_time:.2f}s")
    summary_table.add_row("Human Review Required", str(summary.human_review_required_count))
    
    console.print(summary_table)
    
    if summary.failure_categories:
        console.print("\n[bold]Failure Categories:[/bold]")
        for category, count in summary.failure_categories.items():
            console.print(f"  {category}: {count}")
    
    console.print(f"\n[bold]Session logs:[/bold] output/logs/session_{session_id}.log")
    console.print(f"[bold]Failures:[/bold] output/failures/failures_{session_id}.yaml")
    console.print(f"[bold]Extracted data:[/bold] {config['output']['directory']}/")


@app.command()
def extract(
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to Excel file containing DOI records"),
    skip: Optional[int] = typer.Option(None, "--skip", "-s", help="Number of records to skip before processing"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of records to process"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use (ollama/openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific OpenAI model (gpt-5-nano/gpt-5-mini/gpt-5)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Model selection strategy (cost-optimized/balanced/accuracy-first)"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Maximum cost per extraction ($)"),
    force: Optional[bool] = typer.Option(None, "--force", help="Force reprocess existing DOIs (ignore cache)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Number of records to process concurrently")
):
    """
    [DEPRECATED] Use 'extract-doi' instead. This command is kept for backward compatibility.

    Run DOI extraction pipeline on medical literature dataset.
    """
    console.print("[yellow]Note: 'extract' command is deprecated. Please use 'extract-doi' instead.[/yellow]\n")
    # Call the new command with same parameters
    extract_doi(file=file, skip=skip, limit=limit, provider=provider, model=model,
                strategy=strategy, max_cost=max_cost, force=force, batch_size=batch_size)


@app.command(name="extract-country")
def extract_country(
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to country.xlsx file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV file path"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use (ollama/openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific OpenAI model (gpt-5-nano/gpt-5-mini/gpt-5)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Number of records to process concurrently"),
    skip: Optional[int] = typer.Option(None, "--skip", "-s", help="Number of rows to skip before processing"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of rows to process")
):
    """
    Extract first author country information from affiliation text.

    Processes author affiliation data from Excel file to extract:
    1. First Author Name
    2. First Author Full Location/Affiliation
    3. First Author Country

    Reads from columns:
    - Column 2 (preferred): First author affiliation
    - Column 1 (fallback): Full author list with affiliations

    Output: CSV file with original text, extracted fields, confidence scores, and metadata.

    Examples:
        python cli.py extract-country                              # Interactive mode
        python cli.py extract-country --file country.xlsx          # Process country.xlsx
        python cli.py extract-country -f country.xlsx -o results.csv  # Custom output
    """
    from .extractors.country_extractor import CountryExtractionEngine

    console.print("\n[bold]Author Country Extraction Pipeline[/bold]\n")

    # Load configuration first to use defaults in prompts
    config = load_config()

    # Interactive prompts for missing parameters
    if file is None:
        file = Prompt.ask("[cyan]Excel file to process[/cyan]", default="country.xlsx")

    if output is None:
        output = Prompt.ask("[cyan]Output CSV file path[/cyan]", default="output/country_extracted.csv")

    if provider is None:
        provider_default = config.get('llm', {}).get('default_provider', 'openai')
        provider = _choose_from_list("LLM provider", ["ollama", "openai"], default=provider_default)

    # Handle OpenAI model selection
    if provider == "openai" and model is None:
        model_default = _default_openai_model(config)
        model_options = _openai_model_options(config)
        model = _choose_from_list("OpenAI model", model_options, default=model_default)

    if batch_size is None:
        batch_default = int(config.get('processing', {}).get('batch_size', 5))
        batch_size = IntPrompt.ask("[cyan]Batch size for concurrent processing[/cyan]", default=batch_default)

    # Optional limit/skip prompts
    if skip is None:
        skip = IntPrompt.ask("[cyan]Skip rows before processing[/cyan]", default=0)
    if limit is None:
        limit = IntPrompt.ask("[cyan]Limit rows to process (0 = all)[/cyan]", default=0)
        if limit == 0:
            limit = None

    # Override config with command-line options
    if provider:
        config.setdefault('llm', {})['default_provider'] = provider

    # Create session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    # Pass full config so logging.console can be honored (avoids progress glitches)
    audit_logger = AuditLogger(session_id, output_dir="output", config=config)

    # Initialize country extraction engine
    engine = CountryExtractionEngine(config, audit_logger, session_id)

    console.print(f"\n[bold]Loading data from {file}...[/bold]")

    try:
        # Load records from Excel
        records = engine.load_country_xlsx(Path(file))
        total = len(records)
        # Apply skip/limit
        if skip:
            records = records[skip:]
        if limit:
            records = records[:limit]
        console.print(f"[green]Loaded {len(records)} records[/green] [dim](from {total})[/dim]\n")

        # Process records
        console.print("[bold]Processing records...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Extracting country data...", total=len(records))

            async def run_extraction():
                results = await engine.process_batch(
                    records,
                    batch_size=batch_size,
                    force_provider=provider,
                    force_model=model
                )
                return results

            # Run async extraction
            results = asyncio.run(run_extraction())

            # Collect successful extractions
            successful = []
            failed = []
            for result, error in results:
                progress.advance(task)
                if result:
                    successful.append(result)
                else:
                    failed.append(error)

        # Save results to CSV
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        engine.save_results_to_csv(successful, output_path)

        # Display summary
        console.print(f"\n[bold green]Extraction Complete![/bold green]")
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total records: {len(records)}")
        console.print(f"  Successful: {len(successful)}")
        console.print(f"  Failed: {len(failed)}")

        if successful:
            avg_conf = sum(r.confidence_scores.overall for r in successful) / len(successful)
            console.print(f"  Average confidence: {avg_conf:.2%}")

        console.print(f"\n[bold]Output file:[/bold] {output_path}")
        console.print(f"[bold]Session logs:[/bold] output/logs/session_{session_id}.log")

    except FileNotFoundError:
        console.print(f"[red]Error: File '{file}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
    finally:
        audit_logger.finalize_session()


@app.command()
def benchmark(
    file: str = typer.Option("data-source.xlsx", "--file", "-f", help="Path to Excel file containing DOI records"),
    sample_size: Optional[int] = typer.Option(None, "--sample-size", "-n", help="Number of records to benchmark"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated list of models to test (default: all)"),
    skip: int = typer.Option(0, "--skip", "-s", help="Number of records to skip before sampling")
):
    """
    Benchmark different OpenAI models for accuracy and cost.
    
    Tests each model on the same dataset and compares:
    - Extraction success rate
    - Average confidence scores  
    - Processing time
    - Total cost and cost per extraction
    - Error categories
    
    Examples:
        python cli.py benchmark                                    # Test all models on default sample
        python cli.py benchmark --sample-size 100                 # Custom sample size
        python cli.py benchmark --models gpt-5-nano,gpt-5-mini    # Test specific models
        python cli.py benchmark -f mydata.xlsx --skip 50          # Custom file and starting point
    """
    console.print("\n[bold]Model Benchmarking[/bold]\n")
    
    config = load_config()
    
    # Determine sample size
    if sample_size is None:
        sample_size = config.get('processing', {}).get('benchmark_sample_size', 50)
    
    # Determine models to test from config
    available_models = _openai_model_options(config)
    if models:
        test_models = [m.strip() for m in models.split(',')]
        test_models = [m for m in test_models if m in available_models]
    else:
        test_models = available_models
    
    console.print(f"[green]Testing models: {', '.join(test_models)}[/green]")
    console.print(f"[green]Sample size: {sample_size}[/green]")
    console.print(f"[green]Starting from record: {skip + 1}[/green]\n")
    
    # Load sample data
    console.print(f"Loading sample data from {file}...")
    records, total = load_excel_data(file, skip=skip, limit=sample_size)
    
    if not records:
        console.print("[red]No records to benchmark[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Loaded {len(records)} records for benchmarking[/green]\n")
    
    if not Confirm.ask(f"Proceed with benchmarking {len(test_models)} models on {len(records)} records?"):
        raise typer.Exit(0)
    
    # Run benchmarks
    results = {}
    
    for model_name in test_models:
        console.print(f"\n[bold cyan]Testing {model_name}...[/bold cyan]")
        
        # Configure for this model
        test_config = config.copy()
        test_config['llm']['default_provider'] = 'openai'
        test_config['llm']['default_openai_model'] = model_name
        test_config['llm']['model_selection_strategy'] = 'manual'
        
        # Initialize components
        session_id = f"benchmark_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audit_logger = AuditLogger(session_id, config=test_config)
        from .extractors.doi_extractor import DOIExtractor
        engine = DOIExtractor(test_config, audit_logger, session_id)
        
        # Run extractions
        model_results = {
            'successful': 0,
            'failed': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'confidence_scores': [],
            'processing_times': [],  # Track individual times for statistics
            'error_categories': {}
        }
        
        async def run_model_benchmark():
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]Testing {model_name}..."),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {model_name}", total=len(records))
                
                for record in records:
                    try:
                        result, error = await engine.extract_from_record(
                            record, 
                            force_provider='openai',
                            force_model=model_name
                        )
                        
                        if result:
                            model_results['successful'] += 1
                            model_results['confidence_scores'].append(result.confidence_scores.overall)

                            # Extract processing time from transparency metadata
                            if hasattr(result.transparency_metadata, 'processing_time_seconds'):
                                time_seconds = result.transparency_metadata.processing_time_seconds
                                model_results['total_time'] += time_seconds
                                model_results['processing_times'].append(time_seconds)

                            # Extract cost from transparency metadata if available
                            if hasattr(result.transparency_metadata, 'processing_cost'):
                                model_results['total_cost'] += result.transparency_metadata.processing_cost
                        else:
                            model_results['failed'] += 1
                            
                            # Categorize error
                            category = engine._categorize_failure(Exception(error))
                            model_results['error_categories'][category] = model_results['error_categories'].get(category, 0) + 1
                        
                    except Exception as e:
                        model_results['failed'] += 1
                        category = engine._categorize_failure(e)
                        model_results['error_categories'][category] = model_results['error_categories'].get(category, 0) + 1
                    
                    progress.update(task, advance=1)
        
        # Run benchmark for this model
        asyncio.run(run_model_benchmark())
        
        # Calculate averages
        total_records = model_results['successful'] + model_results['failed']
        avg_confidence = sum(model_results['confidence_scores']) / len(model_results['confidence_scores']) if model_results['confidence_scores'] else 0.0
        avg_cost = model_results['total_cost'] / total_records if total_records > 0 else 0.0

        # Calculate timing statistics
        processing_times = model_results['processing_times']
        if processing_times:
            min_time = min(processing_times)
            max_time = max(processing_times)
            sorted_times = sorted(processing_times)
            median_time = sorted_times[len(sorted_times) // 2]
        else:
            min_time = max_time = median_time = 0.0

        # Store results
        results[model_name] = BenchmarkResult(
            model_name=model_name,
            total_records=total_records,
            successful_extractions=model_results['successful'],
            failed_extractions=model_results['failed'],
            average_confidence=avg_confidence,
            average_processing_time=model_results['total_time'] / total_records if total_records > 0 else 0.0,
            min_processing_time=min_time,
            max_processing_time=max_time,
            median_processing_time=median_time,
            total_cost=model_results['total_cost'],
            average_cost_per_extraction=avg_cost,
            error_categories=model_results['error_categories']
        )
        
        audit_logger.finalize_session()
    
    # Display comparison results
    console.print("\n" + "="*80)
    console.print("[bold green]Benchmark Results[/bold green]")
    console.print("="*80)
    
    # Create comparison table
    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Success Rate", justify="right")
    comparison_table.add_column("Avg Confidence", justify="right")
    comparison_table.add_column("Avg Time (s)", justify="right")
    comparison_table.add_column("Total Cost", justify="right")
    comparison_table.add_column("Cost/Record", justify="right")
    comparison_table.add_column("Failures", justify="right")

    for model_name, result in results.items():
        success_rate = (result.successful_extractions / result.total_records) * 100 if result.total_records > 0 else 0

        comparison_table.add_row(
            model_name,
            f"{success_rate:.1f}%",
            f"{result.average_confidence:.3f}",
            f"{result.average_processing_time:.2f}",
            f"${result.total_cost:.4f}",
            f"${result.average_cost_per_extraction:.4f}",
            str(result.failed_extractions)
        )
    
    console.print(comparison_table)
    
    # Cost efficiency analysis
    console.print(f"\n[bold]Cost Efficiency Analysis:[/bold]")
    sorted_by_cost = sorted(results.items(), key=lambda x: x[1].average_cost_per_extraction)
    cheapest_model = sorted_by_cost[0]
    most_expensive = sorted_by_cost[-1]
    
    cost_ratio = most_expensive[1].average_cost_per_extraction / cheapest_model[1].average_cost_per_extraction
    console.print(f"  Most cost-effective: {cheapest_model[0]} (${cheapest_model[1].average_cost_per_extraction:.4f}/record)")
    console.print(f"  Most expensive: {most_expensive[0]} (${most_expensive[1].average_cost_per_extraction:.4f}/record)")
    console.print(f"  Cost difference: {cost_ratio:.1f}x")
    
    # Accuracy analysis
    console.print(f"\n[bold]Accuracy Analysis:[/bold]")
    sorted_by_confidence = sorted(results.items(), key=lambda x: x[1].average_confidence, reverse=True)
    highest_confidence = sorted_by_confidence[0]
    lowest_confidence = sorted_by_confidence[-1]

    console.print(f"  Highest confidence: {highest_confidence[0]} ({highest_confidence[1].average_confidence:.3f})")
    console.print(f"  Lowest confidence: {lowest_confidence[0]} ({lowest_confidence[1].average_confidence:.3f})")

    # Performance analysis
    console.print(f"\n[bold]Performance Analysis:[/bold]")
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1].average_processing_time)
    fastest_model = sorted_by_speed[0]
    slowest_model = sorted_by_speed[-1]

    console.print(f"  Fastest: {fastest_model[0]} ({fastest_model[1].average_processing_time:.2f}s avg)")
    console.print(f"  Slowest: {slowest_model[0]} ({slowest_model[1].average_processing_time:.2f}s avg)")

    # Show timing breakdown for each model
    for model_name, result in results.items():
        if result.min_processing_time and result.max_processing_time and result.median_processing_time:
            console.print(f"  {model_name}: min={result.min_processing_time:.2f}s, "
                         f"median={result.median_processing_time:.2f}s, "
                         f"max={result.max_processing_time:.2f}s")

    # Recommendations
    console.print(f"\n[bold]Recommendations:[/bold]")
    console.print("  For cost optimization: Use gpt-5-nano for simple extractions")
    console.print("  For balanced performance: Use gpt-5-mini for most cases")
    console.print("  For highest accuracy: Use gpt-5 for critical extractions")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    benchmark_file = f"output/benchmarks/benchmark_results_{timestamp}.yaml"
    Path(benchmark_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(benchmark_file, 'w') as f:
        benchmark_data = {
            'timestamp': timestamp,
            'sample_size': sample_size,
            'models_tested': test_models,
            'results': {name: result.model_dump() for name, result in results.items()}
        }
        yaml.dump(benchmark_data, f, default_flow_style=False)
    
    console.print(f"\n[green]Detailed results saved to: {benchmark_file}[/green]")


@app.command()
def retry(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-sid", help="Specific session ID to retry failures from"),
    max_retries: int = typer.Option(2, "--max-retries", "-m", help="Maximum retry attempts per failed record")
):
    """
    Retry failed extractions from previous sessions.
    
    Automatically categorizes failures and applies appropriate retry strategies:
    - missing_abstract: Skipped (cannot proceed without abstract)
    - llm_error: Retry with fallback provider
    - timeout: Retry with longer timeout
    - parsing_error: Retry with simpler format request
    
    Examples:
        python cli.py retry                           # Retry all failures
        python cli.py retry --max-retries 3           # Try up to 3 times
        python cli.py retry --session-id extract_...  # Retry specific session
    """
    console.print("\n[bold]Retrying Failed Extractions[/bold]\n")
    
    config = load_config()
    
    # Create audit logger to load failures
    temp_logger = AuditLogger("retry_temp", config=config)
    failures = temp_logger.load_failures_for_retry()
    
    if not failures:
        console.print("[green]No failures found to retry[/green]")
        return
    
    # Filter by session if specified
    if session_id:
        failures = [f for f in failures if getattr(f, 'processing_session_id', None) == session_id]
    
    console.print(f"[yellow]Found {len(failures)} failures to retry[/yellow]")
    
    # Group by failure category
    categories = {}
    for failure in failures:
        if failure.failure_category not in categories:
            categories[failure.failure_category] = []
        categories[failure.failure_category].append(failure)
    
    console.print("\n[bold]Failure categories:[/bold]")
    for category, items in categories.items():
        console.print(f"  {category}: {len(items)}")
    
    if not Confirm.ask("\nProceed with retry?"):
        raise typer.Exit(0)
    
    # Initialize new session for retries
    retry_session_id = f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_logger = AuditLogger(retry_session_id, config=config)
    engine = DOIExtractor(config, audit_logger, retry_session_id)
    validator = QualityValidator(config.get('quality', {}))
    
    console.print(f"\n[bold]Retry Session ID:[/bold] {retry_session_id}\n")
    
    # Process retries
    async def retry_all():
        success_count = 0
        still_failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Retrying failures...",
                total=len(failures)
            )
            
            for failure in failures:
                if failure.retry_count >= max_retries:
                    console.print(f"[red]Skipping {failure.doi}: Max retries exceeded[/red]")
                    still_failed += 1
                    progress.update(task, advance=1)
                    continue
                
                # Get retry strategy
                strategy = validator.suggest_retry_strategy(
                    failure.failure_category,
                    failure.retry_count
                )
                
                console.print(f"[yellow]Retrying {failure.doi}: {strategy['message']}[/yellow]")
                
                # Recreate input record from saved data
                try:
                    record = InputRecord(**failure.input_data)
                    
                    # Apply strategy modifications
                    if strategy['action'] == 'retry_with_partial':
                        # Continue even with missing data
                        pass
                    elif strategy['action'] == 'retry_with_fallback':
                        # Force different provider
                        provider = 'openai' if config['llm']['default_provider'] == 'ollama' else 'ollama'
                        result, error = await engine.extract_from_record(record, force_provider=provider)
                    else:
                        result, error = await engine.extract_from_record(record)
                    
                    if result:
                        success_count += 1
                        console.print(f"[green]✓ Successfully processed {failure.doi}[/green]")
                    else:
                        still_failed += 1
                        console.print(f"[red]✗ Still failed: {error}[/red]")
                    
                except Exception as e:
                    still_failed += 1
                    console.print(f"[red]✗ Retry error: {e}[/red]")
                
                progress.update(task, advance=1)
        
        return success_count, still_failed
    
    success, failed = asyncio.run(retry_all())
    
    # Finalize
    audit_logger.finalize_session()
    
    console.print("\n" + "="*60)
    console.print("[bold]Retry Complete![/bold]")
    console.print(f"  Successfully recovered: {success}")
    console.print(f"  Still failed: {failed}")
    console.print("="*60)


@app.command()
def export(
    output_dir: str = typer.Option("output/extracted", "--output-dir", "-d", help="Directory containing extracted YAML files"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv/excel)"),
    output_file: Optional[str] = typer.Option(None, "--output-file", "-o", help="Custom output file name")
):
    """
    Export extracted data to CSV or Excel format.
    
    Aggregates all extracted YAML files into a single spreadsheet for analysis.
    Includes all three classification fields, confidence scores, and metadata.
    
    Examples:
        python cli.py export                              # Export to CSV
        python cli.py export --format excel               # Export to Excel
        python cli.py export -f excel -o results.xlsx     # Custom filename
        python cli.py export -d output/custom             # Custom source directory
    """
    console.print(f"\n[bold]Exporting data from {output_dir}...[/bold]\n")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        console.print(f"[red]Directory {output_dir} not found[/red]")
        raise typer.Exit(1)
    
    # Load all YAML files
    yaml_files = list(output_path.glob("*.yaml"))
    if not yaml_files:
        console.print("[red]No YAML files found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(yaml_files)} extracted records")
    
    # Load data
    data = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading files...", total=len(yaml_files))
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                record = yaml.safe_load(f)
                
                # Flatten for export with enhanced confidence and model info
                confidence_scores = record.get('confidence_scores', {})
                transparency_metadata = record.get('transparency_metadata', {})
                
                flat_record = {
                    'doi': record['doi'],
                    'title': record.get('title'),
                    'authors': record.get('authors'),
                    'journal': record.get('journal'),
                    'year': record.get('year'),
                    'study_design': record.get('study_design'),
                    'study_design_other': record.get('study_design_other'),
                    'subspecialty_focus': record.get('subspecialty_focus'),
                    'subspecialty_focus_other': record.get('subspecialty_focus_other'),
                    'priority_topic': record.get('priority_topic'),
                    
                    # Enhanced confidence scores
                    'overall_confidence': confidence_scores.get('overall'),
                    'study_design_confidence': confidence_scores.get('study_design'),
                    'subspecialty_focus_confidence': confidence_scores.get('subspecialty_focus'), 
                    'priority_topics_confidence': confidence_scores.get('priority_topics'),
                    
                    # Model and processing info
                    'llm_provider': transparency_metadata.get('llm_provider_used'),
                    'llm_model_version': transparency_metadata.get('llm_model_version'),
                    'processing_time_seconds': transparency_metadata.get('processing_time_seconds'),
                    'processing_cost': transparency_metadata.get('processing_cost'),
                    'input_tokens': transparency_metadata.get('input_tokens'),
                    'output_tokens': transparency_metadata.get('output_tokens'),
                    'retry_count': transparency_metadata.get('retry_count', 0),
                    
                    # Quality indicators
                    'human_review_required': transparency_metadata.get('human_review_required'),
                    'has_warnings': len(transparency_metadata.get('warning_logs', [])) > 0,
                    'warning_count': len(transparency_metadata.get('warning_logs', [])),
                    'validation_flags': '; '.join(transparency_metadata.get('validation_flags', [])),
                    
                    # Processing metadata
                    'processing_session': transparency_metadata.get('processing_session_id'),
                    'processing_date': transparency_metadata.get('processing_timestamp'),
                    'prompt_version_hash': transparency_metadata.get('prompt_version_hash')
                }
                data.append(flat_record)
            
            progress.update(task, advance=1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = 'xlsx' if format == 'excel' else 'csv'
        output_file = f"output/exports/extracted_data_{timestamp}.{extension}"
    
    # Ensure export directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    if format == 'excel':
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    console.print(f"\n[green]✓ Exported {len(data)} records to {output_file}[/green]")
    
    # Show summary statistics
    console.print("\n[bold]Export Summary:[/bold]")
    console.print(f"  High confidence (≥0.8): {len(df[df['overall_confidence'] >= 0.8])}")
    console.print(f"  Medium confidence (0.6-0.8): {len(df[(df['overall_confidence'] >= 0.6) & (df['overall_confidence'] < 0.8)])}")
    console.print(f"  Low confidence (<0.6): {len(df[df['overall_confidence'] < 0.6])}")
    console.print(f"  Needs human review: {df['human_review_required'].sum()}")


@app.command()
def validate(
    output_dir: str = typer.Option("output/extracted", "--output-dir", "-d", help="Directory containing extracted YAML files")
):
    """
    Validate extraction quality and generate quality report.
    
    Analyzes all extracted data to provide:
    - Confidence score distribution
    - Field coverage statistics
    - Human review requirements
    - Data quality metrics
    
    Examples:
        python cli.py validate                    # Validate default directory
        python cli.py validate -d output/custom   # Validate custom directory
    """
    console.print(f"\n[bold]Validating extractions from {output_dir}...[/bold]\n")
    
    output_path = Path(output_dir)
    yaml_files = list(output_path.glob("*.yaml"))
    
    if not yaml_files:
        console.print("[red]No extracted files found[/red]")
        raise typer.Exit(1)
    
    # Load extracted data as raw dicts
    extracted_data = []
    
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            # Note: This is simplified - proper deserialization would be needed
            extracted_data.append(data)
    
    validator = QualityValidator()
    
    # Generate report
    console.print(f"Analyzing {len(extracted_data)} extractions...")
    
    # Calculate statistics
    high_confidence = sum(1 for d in extracted_data if d.get('confidence_scores', {}).get('overall', 0) >= 0.8)
    medium_confidence = sum(1 for d in extracted_data if 0.6 <= d.get('confidence_scores', {}).get('overall', 0) < 0.8)
    low_confidence = sum(1 for d in extracted_data if d.get('confidence_scores', {}).get('overall', 0) < 0.6)
    needs_review = sum(1 for d in extracted_data if d.get('transparency_metadata', {}).get('human_review_required'))
    
    # Display report
    console.print("\n[bold]Quality Validation Report[/bold]")
    console.print("="*60)
    
    report_table = Table(show_header=True, header_style="bold magenta")
    report_table.add_column("Metric", style="cyan")
    report_table.add_column("Count", justify="right")
    report_table.add_column("Percentage", justify="right")
    
    total = len(extracted_data)
    report_table.add_row("Total Extractions", str(total), "100%")
    report_table.add_row("High Confidence (≥0.8)", str(high_confidence), f"{high_confidence/total*100:.2f}%")
    report_table.add_row("Medium Confidence (0.6-0.8)", str(medium_confidence), f"{medium_confidence/total*100:.2f}%")
    report_table.add_row("Low Confidence (<0.6)", str(low_confidence), f"{low_confidence/total*100:.2f}%")
    report_table.add_row("Needs Human Review", str(needs_review), f"{needs_review/total*100:.2f}%")
    
    console.print(report_table)
    
    # Field coverage
    console.print("\n[bold]Field Coverage:[/bold]")
    fields = ['year', 'study_design', 'subspecialty_focus', 'priority_topic']
    
    for field in fields:
        coverage = sum(1 for d in extracted_data if d.get(field))
        console.print(f"  {field}: {coverage}/{total} ({coverage/total*100:.2f}%)")


@app.command(name="generate-extractor")
def generate_extractor(
    modify: Optional[str] = typer.Option(None, "--modify", help="Modify existing extractor (e.g., 'doi', 'country')"),
    fork: Optional[str] = typer.Option(None, "--fork", help="Fork existing extractor as template"),
    name: Optional[str] = typer.Option(None, "--name", help="Name for new extractor"),
    fallback_wizard: Optional[bool] = typer.Option(False, "--fallback-wizard", help="Use local interactive wizard instead of the AI agent")
):
    """
    AI-powered extraction pipeline generator.

    Interactive conversational interface to design and generate extraction configurations.
    The AI agent will:
    1. Ask clarifying questions about what you want to extract
    2. Help you define fields and their types
    3. Generate field definitions (YAML)
    4. Generate extraction prompts (YAML)
    5. Provide code templates for implementation

    Modes:
    - Create new: Run without options for guided creation
    - Modify: --modify <name> to update existing extractor
    - Fork: --fork <name> to clone and customize existing extractor

    Examples:
        python cli.py generate-extractor
        python cli.py generate-extractor --modify doi
        python cli.py generate-extractor --fork country --name institution
    """
    from .pipeline_generators.extractor_generator import generator_agent
    from .pipeline_generators.config_generator import ConfigGenerator
    from agents import Runner
    import os

    console.print("\n[bold cyan]AI Extraction Pipeline Generator[/bold cyan]")
    console.print("[dim]Powered by OpenAI Agents[/dim]\n")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        if fallback_wizard or Confirm.ask(
            "OPENAI_API_KEY not set. Use the local config wizard instead?", default=True
        ):
            _run_config_wizard()
            return
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("Please set your OpenAI API key to use the AI generator:")
        console.print("  export OPENAI_API_KEY='your-api-key-here'")
        raise typer.Exit(1)

    # Build initial message based on mode
    if modify:
        initial_message = f"I want to modify the '{modify}' extractor"
        mode = "modify"
    elif fork:
        if name:
            initial_message = f"I want to create a new extractor called '{name}' based on the '{fork}' extractor"
        else:
            initial_message = f"I want to fork the '{fork}' extractor"
        mode = "fork"
    else:
        if name:
            initial_message = f"I want to create a new extraction pipeline called '{name}'"
        else:
            initial_message = "I want to create a new extraction pipeline"
        mode = "new"

    if fallback_wizard:
        _run_config_wizard()
        return

    console.print("[bold]Starting AI conversation...[/bold]\n")
    console.print(f"[dim]Mode: {mode}[/dim]")
    console.print(f"[dim]Initial message: {initial_message}[/dim]\n")

    try:
        # Run the agent conversation
        async def run_conversation():
            result = await Runner.run(generator_agent, input=initial_message)
            return result

        result = asyncio.run(run_conversation())

        console.print(f"\n[green]✓ Agent session completed[/green]")
        console.print(f"\n{result.final_output}")

        console.print("\n[bold cyan]Next Steps:[/bold cyan]")
        console.print("1. Review the generated configuration files")
        console.print("2. Create the extractor class (template provided above)")
        console.print("3. Add the CLI command (code snippet provided above)")
        console.print("4. Test your new extraction pipeline")

    except KeyboardInterrupt:
        console.print("\n[yellow]Agent conversation cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        # Offer fallback wizard on schema errors or any agent init failure
        msg = str(e)
        console.print(f"\n[red]Error during agent conversation: {msg}[/red]")
        if "additionalProperties" in msg or "pydantic" in msg.lower():
            console.print("\n[yellow]The Agents runtime seems incompatible with your Pydantic version.[/yellow]")
            if Confirm.ask("Use the local interactive wizard instead?", default=True):
                _run_config_wizard()
                return
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


def _interactive_actions():
    """Offer common actions after selecting a config, in a loop."""
    def _detect_pipeline() -> str:
        cfg_path = os.getenv("MED_CONFIG_PATH", "")
        if "country" in cfg_path.lower():
            return "country"
        if "doi" in cfg_path.lower():
            return "doi"
        # Fallback: infer from output directory
        try:
            cfg = load_config()
            out_dir = (cfg.get('output', {}) or {}).get('directory', '')
            if 'country' in str(out_dir).lower():
                return "country"
        except Exception:
            pass
        return "doi"

    while True:
        try:
            cfg = load_config()
        except Exception:
            cfg = {}

        pipeline = _detect_pipeline()

        console.print("\n[bold]What would you like to do next?[/bold]")
        if pipeline == "country":
            console.print("  1) Preview data")
            console.print("  2) Test single record")
            console.print("  3) Extract Country (affiliations)")
            console.print("  q) Quit")
            valid_choices = ["1", "2", "3", "q"]
            default_choice = "3"
        else:
            console.print("  1) Preview data")
            console.print("  2) Test single record")
            console.print("  3) Extract DOIs")
            console.print("  5) Export results")
            console.print("  6) Validate quality")
            console.print("  7) Retry failures")
            console.print("  8) Benchmark models")
            console.print("  q) Quit")
            valid_choices = ["1","2","3","5","6","7","8","q"]
            default_choice = "3"

        choice = Prompt.ask(
            "Select action",
            choices=valid_choices,
            default=default_choice,
        )

        if choice == "q":
            return

        # Common defaults
        default_file = "data-source.xlsx"
        default_provider = cfg.get('llm', {}).get('default_provider', 'openai')
        default_batch = cfg.get('processing', {}).get('batch_size', 1)

        if pipeline == "country" and choice == "1":
            # Preview country.xlsx
            from pandas import read_excel
            file = Prompt.ask("Excel file", default="country.xlsx")
            try:
                df = read_excel(file)
            except Exception as e:
                console.print(f"[red]Error loading file: {e}[/red]")
            else:
                console.print(f"[green]Total rows:[/green] {len(df)}  [green]Total columns:[/green] {len(df.columns)}")
                console.print("\n[bold]Sample (first 5 rows):[/bold]")
                sample = df.head(5)
                for idx, row in sample.iterrows():
                    col1 = str(row.iloc[0]) if len(row) > 0 else ''
                    col2 = str(row.iloc[1]) if len(row) > 1 else ''
                    console.print(f"  [dim]Row {idx+1}[/dim]\n    Col1: {col1[:120]}\n    Col2: {col2[:120]}")
        elif pipeline == "country" and choice == "2":
            # Test a single row
            from .extractors.country_extractor import CountryExtractionEngine
            cfg = load_config()
            session_id = f"country_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            audit_logger = AuditLogger(session_id, config=cfg)
            engine = CountryExtractionEngine(cfg, audit_logger, session_id)
            file = Prompt.ask("Excel file", default="country.xlsx")
            skip_rows = IntPrompt.ask("Skip rows before test", default=0)
            provider = _choose_from_list("LLM provider", ["ollama", "openai"], default=cfg.get('llm',{}).get('default_provider','openai'))
            model = None
            if provider == "openai":
                model = _choose_from_list("OpenAI model", ["gpt-5-nano","gpt-5-mini","gpt-5"], default=cfg.get('llm',{}).get('default_openai_model','gpt-5-nano'))
            try:
                records = engine.load_country_xlsx(Path(file))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            else:
                if skip_rows >= len(records):
                    console.print("[red]Skip exceeds rows[/red]")
                else:
                    record = records[skip_rows]
                    async def run_one():
                        return await engine.extract_from_record(record, force_provider=provider, force_model=model)
                    result, error = asyncio.run(run_one())
                    if error:
                        console.print(f"[red]Extraction failed:[/red] {error}")
                    else:
                        console.print("[green]Extraction successful![/green]")
                        console.print(f"  First author: {result.first_author}")
                        console.print(f"  Location: {result.full_location}")
                        console.print(f"  Country: {result.country}")
                        console.print(f"  Confidence: {result.confidence_scores.overall:.2f}")
            audit_logger.finalize_session()
        elif pipeline == "country" and choice == "3":
            # Delegate to existing interactive prompts in the command
            extract_country(file=None, output=None, provider=None, model=None, batch_size=None)
        elif pipeline == "doi" and choice == "1":
            file = Prompt.ask("Excel file", default=default_file)
            skip = IntPrompt.ask("Skip rows", default=0)
            rows = IntPrompt.ask("Rows to preview", default=10)
            preview(file=file, skip=skip, rows=rows)
        elif pipeline == "doi" and choice == "2":
            file = Prompt.ask("Excel file", default=default_file)
            skip = IntPrompt.ask("Skip rows before test", default=0)
            provider = _choose_from_list("Provider", ["ollama","openai"], default=default_provider)
            model = None
            if provider == "openai":
                model = _choose_from_list("OpenAI model", _openai_model_options(cfg), default=_default_openai_model(cfg))
            test(file=file, skip=skip, provider=provider, model=model, strategy=None)
        elif pipeline == "doi" and choice == "3":
            file = Prompt.ask("Excel file", default=default_file)
            skip = IntPrompt.ask("Skip", default=0)
            limit = IntPrompt.ask("Limit (0 for all)", default=0)
            limit = None if limit == 0 else limit
            provider = _choose_from_list("Provider", ["ollama","openai"], default=default_provider)
            model = None
            strategy = None
            if provider == "openai":
                model = _choose_from_list("OpenAI model", _openai_model_options(cfg), default=_default_openai_model(cfg))
                strategy = _choose_from_list("Model selection strategy", ["cost-optimized","balanced","accuracy-first","manual"], default=cfg.get('llm',{}).get('model_selection_strategy','cost-optimized'))
            max_cost = Prompt.ask("Max cost per extraction ($, blank to skip)", default="")
            max_cost_val = float(max_cost) if max_cost.strip() else None
            force = Confirm.ask("Force reprocess existing?", default=False)
            batch_size = IntPrompt.ask("Batch size", default=int(default_batch))
            extract_doi(
                file=file,
                skip=skip,
                limit=limit,
                provider=provider,
                model=model,
                strategy=strategy,
                max_cost=max_cost_val,
                force=force,
                batch_size=batch_size,
            )
        elif pipeline == "doi" and choice == "5":
            out_dir = Prompt.ask("Output directory", default=cfg.get('output',{}).get('directory','output/extracted'))
            fmt = Prompt.ask("Export format", choices=["csv","excel"], default="csv")
            out_file = Prompt.ask("Output file (blank = auto)", default="")
            export(output_dir=out_dir, format=fmt, output_file=(out_file or None))
        elif pipeline == "doi" and choice == "6":
            out_dir = Prompt.ask("Output directory", default=cfg.get('output',{}).get('directory','output/extracted'))
            validate(output_dir=out_dir)
        elif pipeline == "doi" and choice == "7":
            session_id = Prompt.ask("Retry specific session id (blank = all)", default="")
            max_retries = IntPrompt.ask("Max retries per record", default=2)
            retry(session_id=(session_id or None), max_retries=max_retries)
        elif pipeline == "doi" and choice == "8":
            file = Prompt.ask("Excel file", default=default_file)
            sample_size = IntPrompt.ask("Sample size (0 = default)", default=0)
            sample_size = None if sample_size == 0 else sample_size
            models = Prompt.ask("Models (comma, blank=all)", default="")
            skip = IntPrompt.ask("Skip rows", default=0)
            benchmark(file=file, sample_size=sample_size, models=(models or None), skip=skip)

        # Ask to continue
        cont = Confirm.ask("\nRun another action?", default=False)
        if not cont:
            return


def _run_config_wizard():
    """Local interactive wizard to generate field and prompt configs without the agent runtime."""
    from .pipeline_generators.config_generator import ConfigGenerator
    console.print("\n[bold cyan]Config Wizard (local)[/bold cyan]")

    mode = Prompt.ask("Mode", choices=["new", "modify", "fork"], default="new")

    # List existing extractors by scanning config/fields
    existing = []
    fields_dir = Path("config/fields")
    if fields_dir.exists():
        for f in fields_dir.glob("*_fields.yaml"):
            existing.append(f.stem.replace("_fields", ""))
    existing = sorted(set(existing))

    base_name = None
    if mode in ("modify", "fork"):
        if not existing:
            console.print("[yellow]No existing extractors found; switching to 'new' mode[/yellow]")
            mode = "new"
        else:
            console.print("Existing extractors: " + ", ".join(existing))
            base_name = Prompt.ask("Select extractor to modify/fork", choices=existing)

    if mode == "fork" or mode == "new":
        name = Prompt.ask("New extractor name (snake_case)")
    else:
        # modify in place
        name = base_name

    # Load base configs if forking/modifying
    base_fields = {}
    base_prompts = {"system": "", "extraction": ""}
    if mode in ("modify", "fork") and base_name:
        fields_path = Path(f"config/fields/{base_name}_fields.yaml")
        prompts_path = Path(f"config/prompts/{base_name}_prompts.yaml")
        if fields_path.exists():
            base_fields = yaml.safe_load(fields_path.read_text()) or {}
        if prompts_path.exists():
            base_prompts = yaml.safe_load(prompts_path.read_text()) or base_prompts

    # Build fields interactively
    console.print("\n[bold]Define fields[/bold]")
    fields = base_fields if mode in ("modify", "fork") else {}
    while True:
        add = Confirm.ask("Add or edit a field?", default=(len(fields) == 0))
        if not add:
            break
        fname = Prompt.ask("Field name (snake_case)")
        ftype = Prompt.ask("Field type", choices=["text", "enum", "numeric", "boolean"], default="text")
        if ftype == "enum":
            values_str = Prompt.ask("Allowed values (comma-separated)", default=",")
            allowed = [v.strip() for v in values_str.split(',') if v.strip()]
            fields[fname] = allowed
        else:
            fields[fname] = []

    # Prompts
    console.print("\n[bold]Prompts[/bold]")
    default_system = base_prompts.get("system", "You are an assistant extracting structured fields.")
    system_prompt = Prompt.ask("System prompt", default=default_system)
    default_extraction = base_prompts.get("extraction", "Fill JSON with required fields.")
    extraction_prompt = Prompt.ask("Extraction prompt template", default=default_extraction)

    # Generate files
    fields_path, fields_content = ConfigGenerator.create_field_config(name, fields)
    prompts_path, prompts_content = ConfigGenerator.create_prompt_config(name, system_prompt, extraction_prompt)

    console.print("\n[bold]Preview field config:[/bold]")
    console.print(fields_content)
    console.print("\n[bold]Preview prompt config:[/bold]")
    console.print(prompts_content)

    if Confirm.ask("Save these files?", default=True):
        ok1 = ConfigGenerator.save_file(fields_path, fields_content)
        ok2 = ConfigGenerator.save_file(prompts_path, prompts_content)
        if ok1 and ok2:
            console.print(f"[green]✓ Saved:[/green] {fields_path}")
            console.print(f"[green]✓ Saved:[/green] {prompts_path}")
        else:
            console.print("[red]Failed to save one or more files[/red]")

    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("- Implement your extractor using BaseExtractor (see src/extractors/doi_extractor.py for reference)")
    console.print(f"- Update CLI to add a command for '{name}' if needed")
    console.print("- Test with: python cli.py preview/test/extract-doi depending on your pipeline")

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
