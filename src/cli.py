#!/usr/bin/env python3
"""
Main CLI application for medical literature DOI scraping and analysis.
"""

import asyncio
import sys
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

from .models import InputRecord, BenchmarkResult
from .extractor import ExtractionEngine
from .audit import AuditLogger
from .quality import QualityValidator

# Initialize Typer app with custom context settings
app = typer.Typer(
    name="med-doi-scraper",
    help="Research-grade medical literature DOI scraping and analysis pipeline",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "-help", "--help"]}
)

console = Console()


def load_config() -> dict:
    """Load configuration from file."""
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        console.print("[red]Error: config/settings.yaml not found[/red]")
        raise typer.Exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    engine = ExtractionEngine(config, audit_logger, session_id)
    
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
            console.print(f"    {result.study_design.value if result.study_design else 'N/A'}")
            if result.study_design_other:
                console.print(f"    Other specification: {result.study_design_other}")
            
            console.print(f"\n  [cyan]FIELD 2 - Subspecialty Focus:[/cyan]")
            console.print(f"    {result.subspecialty_focus.value if result.subspecialty_focus else 'N/A'}")
            if result.subspecialty_focus_other:
                console.print(f"    Other specification: {result.subspecialty_focus_other}")
            
            console.print(f"\n  [cyan]FIELD 3 - Priority Topics:[/cyan]")
            if result.priority_topics:
                for topic in result.priority_topics:
                    console.print(f"    • {topic.value}")
                if result.priority_topics_details:
                    console.print("    [dim]Details:[/dim]")
                    for detail in result.priority_topics_details:
                        console.print(f"      - {detail}")
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
    Run full extraction pipeline on the dataset.
    
    Processes medical literature records to extract three classification fields:
    1. Study Design
    2. Subspecialty Focus 
    3. Priority Topics alignment
    
    Interactive mode: Run without flags to be prompted for all options.
    
    Examples:
        python cli.py extract                          # Interactive mode
        python cli.py extract --skip 100 --limit 50    # Process records 101-150
        python cli.py extract -s 500 -l 100 -p openai  # Skip 500, process 100 with OpenAI
        python cli.py extract --force --batch-size 10  # Reprocess all with batch size 10
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
        provider = Prompt.ask(
            "[cyan]LLM provider[/cyan]",
            choices=["ollama", "openai"],
            default="ollama"
        )
    
    # Handle OpenAI model selection
    if provider == "openai":
        if model is None and strategy is None:
            model_choice = Prompt.ask(
                "[cyan]OpenAI model selection[/cyan]",
                choices=["strategy", "specific-model"],
                default="strategy"
            )
            
            if model_choice == "strategy":
                strategy = Prompt.ask(
                    "[cyan]Model selection strategy[/cyan]",
                    choices=["cost-optimized", "balanced", "accuracy-first"],
                    default="cost-optimized"
                )
            else:
                model = Prompt.ask(
                    "[cyan]Specific OpenAI model[/cyan]",
                    choices=["gpt-5-nano", "gpt-5-mini", "gpt-5"],
                    default="gpt-5-nano"
                )
    
    if force is None:
        force = Confirm.ask("[cyan]Force reprocess existing DOIs?[/cyan]", default=False)
    
    if batch_size is None:
        batch_size = IntPrompt.ask("[cyan]Batch size for concurrent processing[/cyan]", default=1)
    
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
    engine = ExtractionEngine(config, audit_logger, session_id)
    
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
                batch_results = await engine.process_batch(batch, batch_size=1, force_provider=provider, force_model=model)
                results.extend(batch_results)
                
                # Process batch results and update counters
                for result, error in batch_results:
                    processed_count += 1
                    
                    # Check if this record has reduced confidence due to missing abstract
                    if result and result.confidence_scores.overall <= 0.6:
                        # Check if it's due to missing abstract
                        if "Missing abstract" in str(result.transparency_metadata.warning_logs):
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
    
    # Determine models to test
    available_models = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5']
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
        engine = ExtractionEngine(test_config, audit_logger, session_id)
        
        # Run extractions
        model_results = {
            'successful': 0,
            'failed': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'confidence_scores': [],
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
        
        # Store results
        results[model_name] = BenchmarkResult(
            model_name=model_name,
            total_records=total_records,
            successful_extractions=model_results['successful'],
            failed_extractions=model_results['failed'],
            average_confidence=avg_confidence,
            average_processing_time=model_results['total_time'] / total_records if total_records > 0 else 0.0,
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
    comparison_table.add_column("Total Cost", justify="right")
    comparison_table.add_column("Cost/Record", justify="right")
    comparison_table.add_column("Failures", justify="right")
    
    for model_name, result in results.items():
        success_rate = (result.successful_extractions / result.total_records) * 100 if result.total_records > 0 else 0
        
        comparison_table.add_row(
            model_name,
            f"{success_rate:.1f}%",
            f"{result.average_confidence:.3f}",
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
        failures = [f for f in failures if session_id in str(f.doi)]
    
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
    engine = ExtractionEngine(config, audit_logger, retry_session_id)
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
                
                # Flatten for export
                flat_record = {
                    'doi': record['doi'],
                    'title': record.get('title'),
                    'year': record.get('year'),
                    'study_design': record.get('study_design'),
                    'study_design_other': record.get('study_design_other'),
                    'subspecialty_focus': record.get('subspecialty_focus'),
                    'subspecialty_focus_other': record.get('subspecialty_focus_other'),
                    'priority_topics': '; '.join(record.get('priority_topics', [])) if isinstance(record.get('priority_topics'), list) else record.get('priority_topics'),
                    'priority_topics_details': '; '.join(record.get('priority_topics_details', [])) if record.get('priority_topics_details') else '',
                    'overall_confidence': record.get('confidence_scores', {}).get('overall'),
                    'human_review_required': record.get('transparency_metadata', {}).get('human_review_required'),
                    'has_warnings': len(record.get('transparency_metadata', {}).get('warning_logs', [])) > 0,
                    'processing_session': record.get('transparency_metadata', {}).get('processing_session_id'),
                    'processing_date': record.get('transparency_metadata', {}).get('processing_timestamp')
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
    
    # Load extracted data
    from .models import ExtractedData
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
    fields = ['year', 'study_design', 'subspecialty_focus', 'priority_topics']
    
    for field in fields:
        coverage = sum(1 for d in extracted_data if d.get(field))
        console.print(f"  {field}: {coverage}/{total} ({coverage/total*100:.2f}%)")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()