"""
Core extraction engine for medical literature analysis.
"""

import asyncio
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import traceback

from .models import (
    InputRecord, ExtractedData, TransparencyMetadata,
    StudyDesign, SubspecialtyFocus, PriorityTopics
)
from .providers import LLMProvider, OllamaProvider, OpenAIProvider
from .audit import AuditLogger
from .quality import QualityValidator


class ExtractionEngine:
    """Main extraction engine for processing DOIs."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        audit_logger: AuditLogger,
        session_id: str
    ):
        """Initialize extraction engine."""
        self.config = config
        self.audit_logger = audit_logger
        self.session_id = session_id
        self.quality_validator = QualityValidator(config.get('quality', {}))
        
        # Output directory
        self.output_dir = Path(config.get('output', {}).get('directory', 'output/extracted'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM providers
        self.providers = self._initialize_providers()
        self.default_provider = config.get('llm', {}).get('default_provider', 'ollama')
        
        # Model selection configuration
        self.model_selection_strategy = config.get('llm', {}).get('model_selection_strategy', 'manual')
        self.default_openai_model = config.get('llm', {}).get('default_openai_model', 'gpt-5-nano')
        self.auto_fallback = config.get('processing', {}).get('auto_fallback', True)
        self.fallback_threshold = config.get('processing', {}).get('fallback_confidence_threshold', 0.6)
        
        # Load prompts
        self.prompts = self._load_prompts()
    
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Initialize available LLM providers."""
        providers = {}
        
        # Initialize Ollama if configured
        if 'ollama' in self.config.get('llm', {}):
            try:
                ollama_provider = OllamaProvider(self.config['llm']['ollama'])
                is_valid, message = ollama_provider.validate_connection()
                if is_valid:
                    providers['ollama'] = ollama_provider
                    pass  # Ollama connected successfully
                else:
                    pass  # Ollama connection failed
            except Exception as e:
                pass  # Ollama initialization failed
        
        # Initialize OpenAI models if configured
        if 'openai' in self.config.get('llm', {}):
            openai_config = self.config['llm']['openai']
            
            # Check if we have model-specific configurations
            if 'models' in openai_config:
                # Initialize each configured model as a separate provider
                for model_name in openai_config['models']:
                    try:
                        provider_key = f"openai-{model_name}"
                        openai_provider = OpenAIProvider(openai_config, model_name)
                        is_valid, message = openai_provider.validate_connection()
                        if is_valid:
                            providers[provider_key] = openai_provider
                    except Exception as e:
                        pass  # Model initialization failed
            else:
                # Legacy single model configuration
                try:
                    openai_provider = OpenAIProvider(openai_config)
                    is_valid, message = openai_provider.validate_connection()
                    if is_valid:
                        providers['openai'] = openai_provider
                except Exception as e:
                    pass  # OpenAI initialization failed
        
        if not providers:
            raise ValueError("No LLM providers available")
        
        return providers
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates."""
        prompts_file = Path('config/prompts.yaml')
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default prompts if file doesn't exist
        return {
            'system': """You are a medical research assistant specializing in plastic surgery literature analysis.
Extract THREE specific classification fields from medical abstracts.""",
            
            'extraction': """Analyze this medical literature and extract three classification fields..."""
        }
    
    async def extract_from_record(
        self,
        record: InputRecord,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> Tuple[Optional[ExtractedData], Optional[str]]:
        """
        Extract structured data from a single record.
        
        Returns:
            - ExtractedData if successful
            - Error message if failed
        """
        doi = record.doi
        if not doi:
            return None, "No DOI provided"
        
        # Check if already processed
        output_file = self.output_dir / f"{doi.replace('/', '_')}.yaml"
        if output_file.exists() and not self.config.get('force_reprocess', False):
            self.audit_logger.log_event(
                doi=doi,
                event_type="skipped",
                event_data={"message": "Already processed"}
            )
            self.audit_logger.update_session_stats(skipped=True)
            return None, "Already processed"
        
        try:
            # Start processing
            start_time = datetime.now()
            self.audit_logger.log_event(
                doi=doi,
                event_type="start",
                event_data={"key": record.key, "title": record.title}
            )
            
            # Log warnings for missing data
            warning_logs = []
            
            # Handle missing abstract with reduced confidence (don't fail)
            if not record.abstract_note:
                warning_logs.append("Missing abstract - confidence will be limited to 60%")
                self.audit_logger.log_event(
                    doi=doi,
                    event_type="warning",
                    event_data={"message": "Missing abstract - confidence will be limited to 60%"}
                )
            if not record.author:
                warning_logs.append("Missing author information")
            if not record.author_affiliation_new:
                warning_logs.append("Missing author affiliations")
            if not record.publication_year:
                warning_logs.append("Missing publication year")
            
            if warning_logs:
                for warning in warning_logs:
                    self.audit_logger.log_event(
                        doi=doi,
                        event_type="warning",
                        event_data={"message": warning}
                    )
            
            # Build prompt
            prompt = self._build_prompt(record)
            prompt_hash = self.audit_logger.get_prompt_version_hash(self.prompts['extraction'])
            
            # Select provider and model based on strategy
            provider_name, provider = self._select_optimal_provider(
                record=record,
                force_provider=force_provider,
                force_model=force_model
            )
            
            # Log LLM request
            self.audit_logger.log_event(
                doi=doi,
                event_type="llm_request",
                event_data={"provider": provider_name, "model": provider.config.get('model')},
                llm_prompt=prompt
            )
            
            # Generate extraction
            try:
                response = await provider.generate(
                    prompt=prompt,
                    system_prompt=self.prompts['system']
                )
            except Exception as e:
                # Try fallback provider if available
                if len(self.providers) > 1:
                    fallback_name = [k for k in self.providers.keys() if k != provider_name][0]
                    self.audit_logger.log_event(
                        doi=doi,
                        event_type="fallback",
                        event_data={"from": provider_name, "to": fallback_name, "reason": str(e)}
                    )
                    provider = self.providers[fallback_name]
                    provider_name = fallback_name
                    response = await provider.generate(
                        prompt=prompt,
                        system_prompt=self.prompts['system']
                    )
                else:
                    raise
            
            # Log LLM response
            llm_hash = self.audit_logger.save_raw_llm_interaction(
                doi=doi,
                prompt=prompt,
                response=response.content,
                provider=provider_name,
                model=response.model
            )
            
            self.audit_logger.log_event(
                doi=doi,
                event_type="llm_response",
                event_data={"provider": provider_name, "hash": llm_hash},
                llm_response=response.content[:500],  # Log first 500 chars
                processing_time_ms=response.processing_time * 1000
            )
            
            # Parse response
            if isinstance(provider, (OllamaProvider, OpenAIProvider)):
                extracted = provider.parse_structured_output(response.content)
            else:
                extracted = json.loads(response.content)
            
            # Extract year from publication_year if available
            year = None
            if record.publication_year:
                year = str(int(record.publication_year))
            
            # Calculate confidence scores
            confidence_scores = self.quality_validator.calculate_confidence_scores(
                extracted_data=extracted,
                input_record=record,
                llm_confidence=response.confidence,
                warning_logs=warning_logs
            )
            
            # Create transparency metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            transparency_metadata = TransparencyMetadata(
                processing_session_id=self.session_id,
                processing_timestamp=start_time,
                llm_provider_used=provider_name,
                llm_model_version=response.model,
                prompt_version_hash=prompt_hash,
                processing_time_seconds=processing_time,
                warning_logs=warning_logs,
                has_raw_llm_response=True
            )
            
            # Store "Other" specifications if present
            if extracted.get('subspecialty_focus_other'):
                transparency_metadata.other_specifications['subspecialty_focus'] = extracted['subspecialty_focus_other']
            if extracted.get('study_design_other'):
                transparency_metadata.other_specifications['study_design'] = extracted['study_design_other']
            
            # Create extracted data object
            extracted_data = ExtractedData(
                doi=doi,
                title=record.title,
                abstract=record.abstract_note,
                authors=record.author,
                journal=record.publication_title,
                year=year,
                # Field 1: Study Design - Single selection
                study_design=self._parse_enum(extracted.get('study_design'), StudyDesign),
                study_design_other=extracted.get('study_design_other'),
                # Field 2: Subspecialty Focus - Single selection
                subspecialty_focus=self._parse_enum(extracted.get('subspecialty_focus'), SubspecialtyFocus),
                subspecialty_focus_other=extracted.get('subspecialty_focus_other'),
                # Field 3: Priority Topics - Multiple selections with details
                priority_topics=self._parse_enum_list(extracted.get('priority_topics', []), PriorityTopics),
                priority_topics_details=extracted.get('priority_topics_details', []),
                confidence_scores=confidence_scores,
                transparency_metadata=transparency_metadata,
                processing_metadata={
                    "processed_at": datetime.now().isoformat(),
                    "extraction_success": True,
                    "session_id": self.session_id,
                    "processing_time_seconds": processing_time
                }
            )
            
            # Validate extraction
            is_valid, validation_flags, needs_review = self.quality_validator.validate_extraction(extracted_data)
            
            extracted_data.transparency_metadata.validation_flags = validation_flags
            extracted_data.transparency_metadata.human_review_required = needs_review
            
            if not is_valid and self.config.get('strict_validation', False):
                raise ValueError(f"Validation failed: {', '.join(validation_flags)}")
            
            # Save extraction
            self._save_extraction(extracted_data, output_file)
            
            # Log success
            self.audit_logger.log_event(
                doi=doi,
                event_type="completed",
                event_data={
                    "confidence": confidence_scores.overall,
                    "needs_review": needs_review,
                    "validation_flags": validation_flags,
                    "has_warnings": len(warning_logs) > 0
                }
            )
            
            # Update stats
            self.audit_logger.update_session_stats(
                successful=True,
                processing_time=processing_time,
                llm_provider=provider_name,
                confidence_score=confidence_scores.overall,
                human_review_required=needs_review,
                has_warnings=len(warning_logs) > 0
            )
            
            return extracted_data, None
            
        except Exception as e:
            # Log failure
            error_msg = str(e)
            tb = traceback.format_exc()
            
            self.audit_logger.log_failure(
                doi=doi,
                key=record.key,
                failure_category=self._categorize_failure(e),
                failure_reason=error_msg,
                input_data=record.model_dump(),
                traceback=tb
            )
            
            return None, error_msg
    
    def _assess_complexity(self, record: InputRecord) -> float:
        """Assess extraction complexity (0.0=simple, 1.0=complex)."""
        complexity_score = 0.0
        
        # Abstract length and quality
        if record.abstract_note:
            abstract_len = len(record.abstract_note)
            if abstract_len > 2000:
                complexity_score += 0.3  # Long abstract
            elif abstract_len > 1000:
                complexity_score += 0.2
            elif abstract_len < 200:
                complexity_score += 0.1  # Very short might be missing info
        else:
            complexity_score += 0.4  # Missing abstract is complex
        
        # Technical/medical terminology density
        if record.abstract_note:
            technical_terms = ['randomized', 'placebo', 'systematic', 'meta-analysis', 
                             'retrospective', 'prospective', 'cohort', 'case-control',
                             'statistical', 'multivariate', 'regression', 'p-value',
                             'confidence interval', 'odds ratio', 'hazard ratio']
            
            abstract_lower = record.abstract_note.lower()
            term_count = sum(1 for term in technical_terms if term in abstract_lower)
            complexity_score += min(term_count / 10, 0.3)  # Cap at 0.3
        
        # Missing critical fields
        missing_fields = 0
        if not record.title:
            missing_fields += 1
        if not record.author:
            missing_fields += 1
        if not record.publication_year:
            missing_fields += 1
        
        complexity_score += missing_fields * 0.1
        
        # Cap at 1.0
        return min(complexity_score, 1.0)
    
    def _select_optimal_provider(
        self, 
        record: InputRecord, 
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> Tuple[str, LLMProvider]:
        """Select the optimal provider/model for the given record."""
        
        # If forced provider/model, use those
        if force_provider:
            if force_model and f"openai-{force_model}" in self.providers:
                provider_name = f"openai-{force_model}"
                return provider_name, self.providers[provider_name]
            elif force_provider in self.providers:
                return force_provider, self.providers[force_provider]
        
        # For non-OpenAI providers, use simple selection
        if self.default_provider != 'openai':
            provider_name = self.default_provider
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]
            else:
                # Fallback to first available
                provider_name = list(self.providers.keys())[0]
                return provider_name, self.providers[provider_name]
        
        # Smart OpenAI model selection
        if self.model_selection_strategy == 'manual':
            # Use specified default model
            provider_name = f"openai-{self.default_openai_model}"
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]
        
        elif self.model_selection_strategy == 'cost-optimized':
            # Start with cheapest model that can handle complexity
            complexity = self._assess_complexity(record)
            
            # Try models in order of cost (cheapest first)
            model_order = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5']
            for model in model_order:
                provider_name = f"openai-{model}"
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    if provider.can_handle_complexity(complexity):
                        return provider_name, provider
        
        elif self.model_selection_strategy == 'balanced':
            # Use gpt-5-mini for most cases, nano for simple, gpt-5 for complex
            complexity = self._assess_complexity(record)
            if complexity < 0.3:
                model = 'gpt-5-nano'
            elif complexity > 0.8:
                model = 'gpt-5'
            else:
                model = 'gpt-5-mini'
            
            provider_name = f"openai-{model}"
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]
        
        elif self.model_selection_strategy == 'accuracy-first':
            # Always use most accurate model
            provider_name = f"openai-gpt-5"
            if provider_name in self.providers:
                return provider_name, self.providers[provider_name]
        
        # Fallback to any available OpenAI model or any provider
        for provider_name in self.providers:
            if provider_name.startswith('openai-'):
                return provider_name, self.providers[provider_name]
        
        # Last resort - any provider
        provider_name = list(self.providers.keys())[0]
        return provider_name, self.providers[provider_name]
    
    def _build_prompt(self, record: InputRecord) -> str:
        """Build extraction prompt from record."""
        return self.prompts['extraction'].format(
            title=record.title or "Not provided",
            authors=record.author or "Not provided",
            journal=record.publication_title or "Not provided",
            year=int(record.publication_year) if record.publication_year else "Not provided",
            abstract=record.abstract_note or "Not provided",
            affiliations=record.author_affiliation_new or "Not provided"
        )
    
    def _parse_enum(self, value: Optional[str], enum_class) -> Optional[Any]:
        """Parse string to enum value."""
        if not value:
            return None
        
        # Try to match enum value
        for enum_val in enum_class:
            if enum_val.value.lower() == value.lower():
                return enum_val
        
        # Default to Other if available
        if hasattr(enum_class, 'OTHER'):
            return enum_class.OTHER
        
        return None
    
    def _parse_enum_list(self, values: List[str], enum_class) -> List[Any]:
        """Parse list of strings to enum values."""
        result = []
        for value in values:
            parsed = self._parse_enum(value, enum_class)
            if parsed:
                result.append(parsed)
        return result
    
    def _categorize_failure(self, error: Exception) -> str:
        """Categorize failure for retry strategy."""
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
    
    def _save_extraction(self, data: ExtractedData, output_file: Path):
        """Save extraction to YAML file."""
        # Convert to dict with readable enum values
        data_dict = data.model_dump()
        
        # Convert enums to strings
        if data.subspecialty_focus:
            data_dict['subspecialty_focus'] = data.subspecialty_focus.value
        if data.study_design:
            data_dict['study_design'] = data.study_design.value
        if data.priority_topics:
            data_dict['priority_topics'] = [t.value for t in data.priority_topics]
        
        # Convert datetime to string
        data_dict['transparency_metadata']['processing_timestamp'] = (
            data.transparency_metadata.processing_timestamp.isoformat()
        )
        
        # Save to YAML
        with open(output_file, 'w') as f:
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
    
    async def process_batch(
        self,
        records: List[InputRecord],
        batch_size: int = 10,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> List[Tuple[Optional[ExtractedData], Optional[str]]]:
        """Process a batch of records concurrently."""
        results = []
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            tasks = [self.extract_from_record(record, force_provider=force_provider, force_model=force_model) for record in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Add delay between batches
            if i + batch_size < len(records):
                delay = self.config.get('processing', {}).get('delay_between_requests', 1.0)
                await asyncio.sleep(delay)
        
        return results