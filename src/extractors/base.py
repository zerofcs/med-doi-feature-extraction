"""
Abstract base class for extraction pipelines.

Defines the interface that all extractors must implement.
"""

import asyncio
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from ..core.llm_engine import LLMEngine
from ..core.audit import AuditLogger
from ..core.quality import QualityValidator


class BaseExtractor(ABC):
    """
    Abstract base class for extraction pipelines.

    Each extraction type (DOI, Country, etc.) should inherit from this
    and implement the abstract methods.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        audit_logger: AuditLogger,
        session_id: str
    ):
        """
        Initialize base extractor.

        Args:
            config: Full configuration dictionary
            audit_logger: Audit logging instance
            session_id: Unique session identifier
        """
        self.config = config
        self.audit_logger = audit_logger
        self.session_id = session_id

        # Initialize LLM engine (shared infrastructure)
        self.llm_engine = LLMEngine(config)

        # Initialize quality validator (extraction-agnostic)
        self.quality_validator = QualityValidator(config.get('quality', {}))

        # Output directory
        self.output_dir = Path(config.get('output', {}).get('directory', 'output/extracted'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load extraction-specific configuration
        self.prompts = self.load_prompts()
        self.field_options = self.load_field_options()

    @abstractmethod
    def load_prompts(self) -> Dict[str, str]:
        """
        Load extraction-specific prompt templates.

        Returns:
            Dictionary of prompt templates (e.g., {'system': ..., 'extraction': ...})
        """
        pass

    @abstractmethod
    def load_field_options(self) -> Dict[str, List[str]]:
        """
        Load extraction-specific field options.

        Returns:
            Dictionary mapping field names to allowed values
        """
        pass

    @abstractmethod
    def build_prompt(self, record: Any) -> str:
        """
        Build extraction prompt for a single record.

        Args:
            record: Input record (type varies by extractor)

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def assess_complexity(self, record: Any) -> float:
        """
        Assess extraction complexity for a record.

        Args:
            record: Input record

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        pass

    @abstractmethod
    def parse_llm_response(self, response_content: str, provider: Any) -> Dict[str, Any]:
        """
        Parse LLM response into structured data.

        Args:
            response_content: Raw LLM response text
            provider: LLM provider instance

        Returns:
            Parsed structured data dictionary
        """
        pass

    @abstractmethod
    def create_extracted_data(
        self,
        record: Any,
        extracted: Dict[str, Any],
        confidence_scores: Any,
        transparency_metadata: Any,
        processing_time: float
    ) -> Any:
        """
        Create extraction data model from parsed response.

        Args:
            record: Original input record
            extracted: Parsed LLM response
            confidence_scores: Calculated confidence scores
            transparency_metadata: Audit metadata
            processing_time: Processing time in seconds

        Returns:
            ExtractedData instance (type varies by extractor)
        """
        pass

    @abstractmethod
    def save_extraction(self, data: Any, output_path: Path):
        """
        Save extracted data to file.

        Args:
            data: Extracted data instance
            output_path: Output file path
        """
        pass

    @abstractmethod
    def get_record_identifier(self, record: Any) -> str:
        """
        Get unique identifier for a record (for logging).

        Args:
            record: Input record

        Returns:
            Unique identifier string
        """
        pass

    @abstractmethod
    def get_output_path(self, record: Any) -> Path:
        """
        Get output file path for a record.

        Args:
            record: Input record

        Returns:
            Output file path
        """
        pass

    def normalize_choice(self, value: Optional[str], allowed: List[str]) -> Optional[str]:
        """
        Normalize a free-text choice to an allowed option (case-insensitive).

        Falls back to 'Other' when available if no close match can be found.

        Args:
            value: User-provided value
            allowed: List of allowed values

        Returns:
            Normalized value or None
        """
        if value is None:
            return None
        v = str(value).strip()

        # Exact case-insensitive match
        for opt in allowed:
            if v.lower() == opt.lower():
                return opt

        # Contains match
        for opt in allowed:
            if opt.lower() in v.lower() or v.lower() in opt.lower():
                return opt

        # Fallback to Other
        for opt in allowed:
            if opt.lower() == 'other':
                return opt

        return v

    def categorize_failure(self, error: Exception) -> str:
        """
        Categorize failure for retry strategy.

        Args:
            error: Exception that occurred

        Returns:
            Failure category string
        """
        error_str = str(error).lower()

        if "abstract" in error_str:
            return "missing_abstract"
        elif "timeout" in error_str:
            return "timeout"
        elif "json" in error_str or "parse" in error_str:
            return "parsing_error"
        elif "validation" in error_str:
            return "validation_error"
        else:
            return "llm_error"

    async def extract_from_record(
        self,
        record: Any,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Extract structured data from a single record.

        Template method that orchestrates the extraction process using
        abstract methods implemented by subclasses.

        Args:
            record: Input record
            force_provider: Optional provider override
            force_model: Optional model override

        Returns:
            Tuple of (ExtractedData or None, error_message or None)
        """
        identifier = self.get_record_identifier(record)
        output_file = self.get_output_path(record)

        # Check if already processed
        if output_file.exists() and not self.config.get('force_reprocess', False):
            self.audit_logger.log_event(
                doi=identifier,
                event_type="skipped",
                event_data={"message": "Already processed"}
            )
            self.audit_logger.update_session_stats(skipped=True)
            return None, "Already processed"

        try:
            # Start processing
            start_time = datetime.now()
            self.audit_logger.log_event(
                doi=identifier,
                event_type="start",
                event_data={"identifier": identifier}
            )

            # Build prompt
            prompt = self.build_prompt(record)
            prompt_hash = self.audit_logger.get_prompt_version_hash(self.prompts.get('extraction', ''))

            # Assess complexity and select provider
            complexity = self.assess_complexity(record)
            provider_name, provider = self.llm_engine.select_provider(
                complexity_score=complexity,
                force_provider=force_provider,
                force_model=force_model
            )

            # Log LLM request
            self.audit_logger.log_event(
                doi=identifier,
                event_type="llm_request",
                event_data={"provider": provider_name, "model": provider.config.get('model')},
                llm_prompt=prompt
            )

            # Define fallback callback
            def on_fallback(from_provider, to_provider, error):
                self.audit_logger.log_event(
                    doi=identifier,
                    event_type="fallback",
                    event_data={"from": from_provider, "to": to_provider, "reason": error}
                )

            # Generate extraction via LLM engine
            response, final_model_version = await self.llm_engine.generate(
                prompt=prompt,
                system_prompt=self.prompts.get('system', ''),
                provider_name=provider_name,
                provider=provider,
                enable_fallback=True,
                fallback_callback=on_fallback
            )

            # Log LLM response
            llm_hash = self.audit_logger.save_raw_llm_interaction(
                doi=identifier,
                prompt=prompt,
                response=response.content,
                provider=provider_name,
                model=final_model_version
            )

            self.audit_logger.log_event(
                doi=identifier,
                event_type="llm_response",
                event_data={"provider": provider_name, "hash": llm_hash},
                llm_response=response.content[:500],
                processing_time_ms=response.processing_time * 1000
            )

            # Parse response (extractor-specific)
            extracted = self.parse_llm_response(response.content, provider)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create extracted data (extractor-specific)
            # Note: Subclasses handle confidence calculation and metadata creation
            extracted_data = self.create_extracted_data(
                record=record,
                extracted=extracted,
                confidence_scores=None,  # Subclass will calculate
                transparency_metadata=None,  # Subclass will create
                processing_time=processing_time
            )

            # Save extraction (extractor-specific)
            self.save_extraction(extracted_data, output_file)

            # Log success
            self.audit_logger.log_event(
                doi=identifier,
                event_type="completed",
                event_data={"success": True}
            )

            # Update stats
            self.audit_logger.update_session_stats(
                successful=True,
                processing_time=processing_time,
                llm_provider=provider_name
            )

            return extracted_data, None

        except Exception as e:
            # Log failure
            import traceback
            error_msg = str(e)
            tb = traceback.format_exc()

            self.audit_logger.log_failure(
                doi=identifier,
                key=identifier,
                failure_category=self.categorize_failure(e),
                failure_reason=error_msg,
                input_data={"identifier": identifier},
                traceback=tb
            )

            return None, error_msg

    async def process_batch(
        self,
        records: List[Any],
        batch_size: int = 10,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> List[Tuple[Optional[Any], Optional[str]]]:
        """
        Process a batch of records concurrently.

        Args:
            records: List of input records
            batch_size: Number of concurrent extractions
            force_provider: Optional provider override
            force_model: Optional model override

        Returns:
            List of (ExtractedData or None, error or None) tuples
        """
        results = []

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            tasks = [
                self.extract_from_record(record, force_provider=force_provider, force_model=force_model)
                for record in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Add delay between batches
            if i + batch_size < len(records):
                delay = self.config.get('processing', {}).get('delay_between_requests', 1.0)
                await asyncio.sleep(delay)

        return results
