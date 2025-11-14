"""
Country extraction engine for author affiliation analysis.

Extracts first author name, location, and country from affiliation text.
"""

import asyncio
import json
import yaml
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import traceback
import pandas as pd

from ..core.models import (
    CountryInputRecord,
    CountryExtractedData,
    CountryConfidenceScores,
    TransparencyMetadata,
)
from ..providers import LLMProvider, OllamaProvider, OpenAIProvider
from ..core.audit import AuditLogger
from ..core.quality import QualityValidator


class CountryExtractionEngine:
    """Extraction engine for processing author affiliation country data."""

    def __init__(
        self, config: Dict[str, Any], audit_logger: AuditLogger, session_id: str
    ):
        """Initialize country extraction engine."""
        self.config = config
        self.audit_logger = audit_logger
        self.session_id = session_id
        self.quality_validator = QualityValidator(config.get("quality", {}))

        # Output directory
        self.output_dir = Path(
            config.get("output", {}).get("directory", "output/country_extracted")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM providers
        self.providers = self._initialize_providers()
        self.default_provider = config.get("llm", {}).get("default_provider", "ollama")

        # Model selection configuration
        self.model_selection_strategy = config.get("llm", {}).get(
            "model_selection_strategy", "manual"
        )
        self.default_openai_model = config.get("llm", {}).get(
            "default_openai_model", "gpt-5-nano"
        )

        # Load prompts
        self.prompts = self._load_prompts()
        # Load country options
        self.field_options = self._load_field_options()

    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Initialize available LLM providers."""
        providers = {}

        # Initialize Ollama if configured
        if "ollama" in self.config.get("llm", {}):
            try:
                ollama_provider = OllamaProvider(self.config["llm"]["ollama"])
                is_valid, message = ollama_provider.validate_connection()
                if is_valid:
                    providers["ollama"] = ollama_provider
            except Exception:
                pass

        # Initialize OpenAI models if configured
        if "openai" in self.config.get("llm", {}):
            openai_config = self.config["llm"]["openai"]

            if "models" in openai_config:
                for model_name in openai_config["models"]:
                    try:
                        provider_key = f"openai-{model_name}"
                        openai_provider = OpenAIProvider(openai_config, model_name)
                        is_valid, message = openai_provider.validate_connection()
                        if is_valid:
                            providers[provider_key] = openai_provider
                    except Exception:
                        pass
            else:
                try:
                    openai_provider = OpenAIProvider(openai_config)
                    is_valid, message = openai_provider.validate_connection()
                    if is_valid:
                        providers["openai"] = openai_provider
                except Exception:
                    pass

        if not providers:
            raise ValueError("No LLM providers available")

        return providers

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates."""
        prompts_file = Path("config/prompts/country_prompts.yaml")
        if prompts_file.exists():
            with open(prompts_file, "r") as f:
                return yaml.safe_load(f)

        # Default prompts
        return {
            "system": """You are a research assistant specializing in extracting author affiliation information.""",
            "extraction": """Extract first author name, location, and country from the affiliation text.""",
        }

    def _load_field_options(self) -> Dict[str, List[str]]:
        """Load country options."""
        fields_file = Path("config/fields/country_fields.yaml")
        if fields_file.exists():
            with open(fields_file, "r") as f:
                data = yaml.safe_load(f) or {}
                return {"country": [str(v) for v in (data.get("country") or [])]}

        return {"country": ["Other"]}

    async def extract_from_record(
        self,
        record: CountryInputRecord,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> Tuple[Optional[CountryExtractedData], Optional[str]]:
        """
        Extract country data from a single affiliation record.

        Returns:
            - CountryExtractedData if successful
            - Error message if failed
        """
        identifier = f"row_{record.row_number}"

        try:
            # Start processing
            start_time = datetime.now()
            self.audit_logger.log_event(
                doi=identifier,
                event_type="start",
                event_data={"row": record.row_number},
            )

            # Get both author text (Column 1) and affiliation text (Column 2)
            author_text = record.full_author_text or ""
            affiliation_text = (
                record.first_author_affiliation or record.full_author_text or ""
            )

            if not author_text.strip() and not affiliation_text.strip():
                return None, "No author or affiliation text provided"

            # Build prompt with both texts
            prompt = self._build_prompt(author_text, affiliation_text)
            prompt_hash = self.audit_logger.get_prompt_version_hash(
                self.prompts["extraction"]
            )

            # Select provider
            provider_name, provider = self._select_provider(
                force_provider=force_provider, force_model=force_model
            )

            # Log LLM request
            self.audit_logger.log_event(
                doi=identifier,
                event_type="llm_request",
                event_data={
                    "provider": provider_name,
                    "model": provider.config.get("model"),
                },
                llm_prompt=prompt,
            )

            # Generate extraction
            try:
                # print(f"[DEBUG] Calling provider.generate() - Provider: {provider_name}")
                response = await provider.generate(
                    prompt=prompt, system_prompt=self.prompts["system"]
                )
                # print(f"[DEBUG] Provider returned response - Content length: {len(response.content) if response else 0}")
                final_model_version = response.model
            except Exception as e:
                # Try fallback provider if available
                if len(self.providers) > 1:
                    fallback_name = [
                        k for k in self.providers.keys() if k != provider_name
                    ][0]
                    self.audit_logger.log_event(
                        doi=identifier,
                        event_type="fallback",
                        event_data={
                            "from": provider_name,
                            "to": fallback_name,
                            "reason": str(e),
                        },
                    )
                    provider = self.providers[fallback_name]
                    response = await provider.generate(
                        prompt=prompt, system_prompt=self.prompts["system"]
                    )
                    final_model_version = response.model
                else:
                    raise

            # Log LLM response
            llm_hash = self.audit_logger.save_raw_llm_interaction(
                doi=identifier,
                prompt=prompt,
                response=response.content,
                provider=provider_name,
                model=final_model_version,
            )

            self.audit_logger.log_event(
                doi=identifier,
                event_type="llm_response",
                event_data={"provider": provider_name, "hash": llm_hash},
                llm_response=response.content[:500],
                processing_time_ms=response.processing_time * 1000,
            )

            # Parse response
            if isinstance(provider, (OllamaProvider, OpenAIProvider)):
                extracted = provider.parse_structured_output(response.content)
            else:
                extracted = json.loads(response.content)

            # Normalize country using data-driven options
            country = self._normalize_choice(
                extracted.get("country"), self.field_options["country"]
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(
                extracted=extracted,
                affiliation_text=affiliation_text,
                llm_confidence=response.confidence,
            )

            # Create transparency metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            transparency_metadata = TransparencyMetadata(
                processing_session_id=self.session_id,
                processing_timestamp=start_time,
                llm_provider_used=provider_name,
                llm_model_version=final_model_version,
                prompt_version_hash=prompt_hash,
                processing_time_seconds=processing_time,
                processing_cost=getattr(response, "cost", None),
                input_tokens=getattr(response, "input_tokens", None),
                output_tokens=getattr(response, "output_tokens", None),
                warning_logs=[],
                has_raw_llm_response=True,
            )

            # Create extracted data object
            extracted_data = CountryExtractedData(
                row_number=record.row_number,
                original_text=affiliation_text,
                first_author=extracted.get("first_author"),
                full_location=extracted.get("full_location"),
                country=country,
                confidence_scores=confidence_scores,
                transparency_metadata=transparency_metadata,
                processing_metadata={
                    "processed_at": datetime.now().isoformat(),
                    "extraction_success": True,
                    "session_id": self.session_id,
                    "processing_time_seconds": processing_time,
                },
            )

            # Log success
            self.audit_logger.log_event(
                doi=identifier,
                event_type="completed",
                event_data={"confidence": confidence_scores.overall},
            )

            # Update stats
            self.audit_logger.update_session_stats(
                successful=True,
                processing_time=processing_time,
                llm_provider=provider_name,
                processing_cost=getattr(response, "cost", None),
                model_name=final_model_version,
                input_tokens=getattr(response, "input_tokens", None),
                output_tokens=getattr(response, "output_tokens", None),
            )

            return extracted_data, None

        except Exception as e:
            # Log failure
            error_msg = str(e)
            tb = traceback.format_exc()

            self.audit_logger.log_failure(
                doi=identifier,
                key=identifier,
                failure_category=self._categorize_failure(e),
                failure_reason=error_msg,
                input_data={"row": record.row_number, "text": affiliation_text[:200]},
                traceback=tb,
            )

            return None, error_msg

    def _build_prompt(self, author_text: str, affiliation_text: str) -> str:
        """Build extraction prompt with both author and affiliation text."""
        country_options = "\n  - ".join(
            [f'"{opt}"' for opt in self.field_options["country"]]
        )

        return self.prompts["extraction"].format(
            author_text=author_text,
            affiliation_text=affiliation_text,
            country_options=country_options,
        )

    def _select_provider(
        self, force_provider: Optional[str] = None, force_model: Optional[str] = None
    ) -> Tuple[str, LLMProvider]:
        """Select provider/model."""
        if force_provider:
            if force_model and f"openai-{force_model}" in self.providers:
                provider_name = f"openai-{force_model}"
                return provider_name, self.providers[provider_name]
            elif force_provider in self.providers:
                return force_provider, self.providers[force_provider]

        # Use default
        if self.default_provider != "openai":
            provider_name = self.default_provider
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]

        # OpenAI selection
        if self.model_selection_strategy == "manual":
            provider_name = f"openai-{self.default_openai_model}"
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]

        # Fallback to first available
        provider_name = list(self.providers.keys())[0]
        return provider_name, self.providers[provider_name]

    def _calculate_confidence(
        self, extracted: Dict[str, Any], affiliation_text: str, llm_confidence: float
    ) -> CountryConfidenceScores:
        """Calculate confidence scores."""
        # Base confidence from LLM
        base = llm_confidence

        # Confidence for first_author
        author_conf = base
        if not extracted.get("first_author"):
            author_conf = 0.0
        elif len(extracted.get("first_author", "")) < 5:  # Very short name
            author_conf *= 0.7

        # Confidence for full_location
        location_conf = base
        if not extracted.get("full_location"):
            location_conf = 0.0
        elif len(extracted.get("full_location", "")) < 10:  # Very short location
            location_conf *= 0.7

        # Confidence for country
        country_conf = base
        if extracted.get("country") == "Other":
            country_conf *= 0.7

        overall = sum([author_conf, location_conf, country_conf]) / 3.0

        return CountryConfidenceScores(
            first_author=author_conf,
            full_location=location_conf,
            country=country_conf,
            overall=overall,
        )

    def _normalize_choice(
        self, value: Optional[str], allowed: List[str]
    ) -> Optional[str]:
        """Normalize country choice."""
        if value is None:
            return None
        v = str(value).strip()

        # Exact match
        for opt in allowed:
            if v.lower() == opt.lower():
                return opt

        # Contains match
        for opt in allowed:
            if opt.lower() in v.lower() or v.lower() in opt.lower():
                return opt

        # Fallback to Other
        for opt in allowed:
            if opt.lower() == "other":
                return opt

        return v

    def _categorize_failure(self, error: Exception) -> str:
        """Categorize failure."""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return "timeout"
        elif "json" in error_str or "parse" in error_str:
            return "parsing_error"
        else:
            return "llm_error"

    async def process_batch(
        self,
        records: List[CountryInputRecord],
        batch_size: int = 10,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> List[Tuple[Optional[CountryExtractedData], Optional[str]]]:
        """Process a batch of records concurrently."""
        results = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            tasks = [
                self.extract_from_record(
                    record, force_provider=force_provider, force_model=force_model
                )
                for record in batch
            ]
            # Add timeout to prevent indefinite hangs (30 seconds per batch)
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks), timeout=30.0
                )
                results.extend(batch_results)
            except asyncio.TimeoutError:
                print(
                    f"[ERROR] Batch timeout after 30 seconds. Batch records: {[r.get('index', '?') for r in batch]}"
                )
                self.audit_logger.log_event(
                    doi="batch_timeout",
                    event_type="error",
                    event_data={
                        "message": "Batch timeout",
                        "batch_indices": [r.get("index", "?") for r in batch],
                    },
                )
                # Add None results for failed batch
                results.extend([None] * len(batch))

            # Add delay between batches
            if i + batch_size < len(records):
                delay = self.config.get("processing", {}).get(
                    "delay_between_requests", 1.0
                )
                await asyncio.sleep(delay)

        return results

    def save_results_to_csv(
        self, results: List[CountryExtractedData], output_path: Path
    ):
        """Save extraction results to CSV file."""
        if not results:
            return

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "row_number",
                "original_text",
                "first_author",
                "full_location",
                "country",
                "confidence_overall",
                "confidence_author",
                "confidence_location",
                "confidence_country",
                "llm_provider",
                "llm_model",
                "processing_time_seconds",
                "session_id",
                "processing_timestamp",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in results:
                writer.writerow(
                    {
                        "row_number": data.row_number,
                        "original_text": data.original_text,
                        "first_author": data.first_author or "",
                        "full_location": data.full_location or "",
                        "country": data.country or "",
                        "confidence_overall": f"{data.confidence_scores.overall:.4f}",
                        "confidence_author": f"{data.confidence_scores.first_author:.4f}",
                        "confidence_location": f"{data.confidence_scores.full_location:.4f}",
                        "confidence_country": f"{data.confidence_scores.country:.4f}",
                        "llm_provider": data.transparency_metadata.llm_provider_used,
                        "llm_model": data.transparency_metadata.llm_model_version,
                        "processing_time_seconds": f"{data.transparency_metadata.processing_time_seconds:.2f}",
                        "session_id": data.transparency_metadata.processing_session_id,
                        "processing_timestamp": data.transparency_metadata.processing_timestamp.isoformat(),
                    }
                )

    def load_country_xlsx(self, file_path: Path) -> List[CountryInputRecord]:
        """Load records from country.xlsx file."""
        df = pd.read_excel(file_path)

        records = []
        for idx, row in df.iterrows():
            # Get columns by index (0-based)
            col1 = row.iloc[0] if len(row) > 0 else None  # "Cut off First"
            col2 = row.iloc[1] if len(row) > 1 else None  # "Shorten To country line"

            # Convert to string, handle NaN
            col1_str = str(col1) if pd.notna(col1) else None
            col2_str = str(col2) if pd.notna(col2) else None

            record = CountryInputRecord(
                row_number=idx + 1,  # 1-indexed for users
                full_author_text=col1_str,
                first_author_affiliation=col2_str,
            )
            records.append(record)

        return records
