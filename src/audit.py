"""
Audit trail and transparency logging system for research reproducibility.
"""

import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from rich.console import Console
from rich.logging import RichHandler

from .models import AuditLogEntry, SessionSummary, ProcessingFailure


class AuditLogger:
    """Manages audit trails and transparency logging."""
    
    def __init__(self, session_id: str, output_dir: str = "output", config: dict = None):
        """Initialize audit logger."""
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.failures_dir = self.output_dir / "failures"
        self.config = config or {}
        
        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        
        # Session log file
        self.session_log_file = self.logs_dir / f"session_{session_id}.jsonl"
        self.failure_log_file = self.failures_dir / f"failures_{session_id}.yaml"
        
        # Initialize session summary
        self.session_summary = SessionSummary(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # Setup logging
        self._setup_logging()
        
        # Audit entries buffer
        self.audit_buffer: List[AuditLogEntry] = []
        self.failures: List[ProcessingFailure] = []
    
    def _setup_logging(self):
        """Setup rich logging handler."""
        # Configure logger
        self.logger = logging.getLogger(f"audit_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        
        # Only add console handler if enabled in config
        if self.config.get('logging', {}).get('console', True):
            console = Console(stderr=True)
            handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True
            )
            self.logger.addHandler(handler)
        
        # Also log to file
        file_handler = logging.FileHandler(
            self.logs_dir / f"session_{self.session_id}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def log_event(
        self,
        doi: str,
        event_type: str,
        event_data: Dict[str, Any],
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Log an audit event."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            session_id=self.session_id,
            doi=doi,
            event_type=event_type,
            event_data=event_data,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            processing_time_ms=processing_time_ms
        )
        
        # Add to buffer
        self.audit_buffer.append(entry)
        
        # Write to file immediately for safety
        with open(self.session_log_file, 'a') as f:
            f.write(entry.model_dump_json() + '\n')
        
        # Log to console
        self.logger.info(f"[{event_type}] DOI: {doi} - {event_data.get('message', '')}")
    
    def log_failure(
        self,
        doi: str,
        key: str,
        failure_category: str,
        failure_reason: str,
        input_data: Dict[str, Any],
        traceback: Optional[str] = None
    ):
        """Log a processing failure."""
        failure = ProcessingFailure(
            doi=doi,
            key=key,
            failure_category=failure_category,
            failure_reason=failure_reason,
            failure_timestamp=datetime.now(),
            input_data=input_data,
            traceback=traceback
        )
        
        self.failures.append(failure)
        
        # Update session summary
        if failure_category not in self.session_summary.failure_categories:
            self.session_summary.failure_categories[failure_category] = 0
        self.session_summary.failure_categories[failure_category] += 1
        self.session_summary.failed_extractions += 1
        
        # Save failures immediately
        self._save_failures()
        
        # Log to console
        self.logger.error(f"[FAILURE] DOI: {doi} - {failure_category}: {failure_reason}")
    
    def _save_failures(self):
        """Save failures to YAML file."""
        failures_data = [f.model_dump() for f in self.failures]
        with open(self.failure_log_file, 'w') as f:
            yaml.dump(failures_data, f, default_flow_style=False)
    
    def update_session_stats(
        self,
        successful: bool = False,
        skipped: bool = False,
        processing_time: Optional[float] = None,
        llm_provider: Optional[str] = None,
        confidence_score: Optional[float] = None,
        human_review_required: bool = False,
        has_warnings: bool = False
    ):
        """Update session statistics."""
        if successful:
            self.session_summary.successful_extractions += 1
        if skipped:
            self.session_summary.skipped_already_processed += 1
        
        if processing_time:
            # Update average processing time
            total = self.session_summary.successful_extractions + self.session_summary.failed_extractions
            if total > 0:
                current_avg = self.session_summary.average_processing_time
                self.session_summary.average_processing_time = (
                    (current_avg * (total - 1) + processing_time) / total
                )
        
        if llm_provider:
            if llm_provider not in self.session_summary.llm_provider_stats:
                self.session_summary.llm_provider_stats[llm_provider] = 0
            self.session_summary.llm_provider_stats[llm_provider] += 1
        
        if confidence_score is not None:
            # Bucket confidence scores
            bucket = f"{int(confidence_score * 10) / 10:.2f}"
            if bucket not in self.session_summary.confidence_distribution:
                self.session_summary.confidence_distribution[bucket] = 0
            self.session_summary.confidence_distribution[bucket] += 1
        
        if human_review_required:
            self.session_summary.human_review_required_count += 1
        
        if has_warnings:
            self.session_summary.records_with_warnings += 1
    
    def save_raw_llm_interaction(
        self,
        doi: str,
        prompt: str,
        response: str,
        provider: str,
        model: str
    ) -> str:
        """Save raw LLM interaction for transparency."""
        # Create hash for this interaction
        interaction_hash = hashlib.md5(
            f"{doi}{prompt}{response}".encode()
        ).hexdigest()[:8]
        
        # Save to separate file (sanitize DOI for filename)
        safe_doi = doi.replace('/', '_').replace(':', '_')
        interaction_file = self.logs_dir / "llm_interactions" / f"{safe_doi}_{interaction_hash}.json"
        interaction_file.parent.mkdir(exist_ok=True)
        
        interaction_data = {
            "doi": doi,
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "response": response,
            "hash": interaction_hash
        }
        
        with open(interaction_file, 'w') as f:
            json.dump(interaction_data, f, indent=2)
        
        return interaction_hash
    
    def get_prompt_version_hash(self, prompt_template: str) -> str:
        """Generate hash for prompt version tracking."""
        return hashlib.md5(prompt_template.encode()).hexdigest()[:8]
    
    def finalize_session(self):
        """Finalize session and save summary."""
        self.session_summary.end_time = datetime.now()
        self.session_summary.total_records = (
            self.session_summary.successful_extractions +
            self.session_summary.failed_extractions +
            self.session_summary.skipped_already_processed
        )
        
        # Save session summary
        summary_file = self.logs_dir / f"summary_{self.session_id}.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(self.session_summary.model_dump(), f, default_flow_style=False)
        
        self.logger.info(f"Session {self.session_id} finalized")
        self.logger.info(f"Total: {self.session_summary.total_records}")
        self.logger.info(f"Successful: {self.session_summary.successful_extractions}")
        self.logger.info(f"Failed: {self.session_summary.failed_extractions}")
        self.logger.info(f"Skipped: {self.session_summary.skipped_already_processed}")
        
        return self.session_summary
    
    def load_failures_for_retry(self) -> List[ProcessingFailure]:
        """Load failures from previous sessions for retry."""
        all_failures = []
        
        # Load all failure files
        for failure_file in self.failures_dir.glob("failures_*.yaml"):
            with open(failure_file, 'r') as f:
                failures_data = yaml.safe_load(f)
                if failures_data:
                    for failure_dict in failures_data:
                        # Convert dict back to ProcessingFailure
                        failure_dict['failure_timestamp'] = datetime.fromisoformat(
                            failure_dict['failure_timestamp']
                        )
                        if failure_dict.get('last_retry_timestamp'):
                            failure_dict['last_retry_timestamp'] = datetime.fromisoformat(
                                failure_dict['last_retry_timestamp']
                            )
                        all_failures.append(ProcessingFailure(**failure_dict))
        
        return all_failures