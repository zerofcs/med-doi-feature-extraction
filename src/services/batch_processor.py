"""Batch processing service for parallel extraction execution."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import pandas as pd
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn


@dataclass
class BatchResult:
    """Result from processing a single record."""
    record_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchProcessor:
    """Handles parallel batch processing with progress tracking."""

    def __init__(self, batch_size: int = 1, max_workers: Optional[int] = None):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of records to process concurrently
            max_workers: Maximum worker threads (defaults to batch_size)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or batch_size

    def process_dataframe(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.Series, int], Dict[str, Any]],
        id_column: str = "DOI",
        skip: int = 0,
        limit: Optional[int] = None,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None
    ) -> List[BatchResult]:
        """
        Process DataFrame records in parallel batches.

        Args:
            df: Input DataFrame
            process_func: Function that processes a single record (row, index) -> result dict
            id_column: Column name to use as record identifier
            skip: Number of records to skip from start
            limit: Maximum number of records to process
            progress: Optional Rich Progress instance for tracking
            task_id: Optional Progress task ID to update

        Returns:
            List of BatchResult objects
        """
        # Apply skip and limit
        total_records = len(df)
        start_idx = skip
        end_idx = min(start_idx + limit, total_records) if limit else total_records

        df_subset = df.iloc[start_idx:end_idx]
        records_to_process = len(df_subset)

        results: List[BatchResult] = []

        # Sequential processing (batch_size = 1)
        if self.batch_size == 1:
            for idx, row in df_subset.iterrows():
                record_id = str(row[id_column])
                try:
                    result_data = process_func(row, idx)
                    results.append(BatchResult(
                        record_id=record_id,
                        success=True,
                        data=result_data
                    ))
                except Exception as e:
                    results.append(BatchResult(
                        record_id=record_id,
                        success=False,
                        error=str(e)
                    ))

                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

        # Parallel batch processing
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_record = {}
                for idx, row in df_subset.iterrows():
                    record_id = str(row[id_column])
                    future = executor.submit(self._safe_process, process_func, row, idx, record_id)
                    future_to_record[future] = record_id

                # Collect results as they complete
                for future in as_completed(future_to_record):
                    result = future.result()
                    results.append(result)

                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)

        return results

    def _safe_process(
        self,
        process_func: Callable,
        row: pd.Series,
        idx: int,
        record_id: str
    ) -> BatchResult:
        """
        Safely execute process function with error handling.

        Args:
            process_func: Function to execute
            row: DataFrame row to process
            idx: Row index
            record_id: Record identifier

        Returns:
            BatchResult with success/error status
        """
        try:
            result_data = process_func(row, idx)
            return BatchResult(
                record_id=record_id,
                success=True,
                data=result_data
            )
        except Exception as e:
            return BatchResult(
                record_id=record_id,
                success=False,
                error=str(e)
            )

    @staticmethod
    def create_progress() -> Progress:
        """
        Create a Rich Progress instance with custom columns.

        Returns:
            Configured Progress instance
        """
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
