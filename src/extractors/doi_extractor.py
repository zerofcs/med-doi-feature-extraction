"""
DOI extraction pipeline for medical literature classification.

Extracts study_design, subspecialty_focus, and priority_topic from medical abstracts.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseExtractor
from ..core.models import InputRecord, ExtractedData, ConfidenceScores, TransparencyMetadata
from ..providers import OllamaProvider, OpenAIProvider
from ..utils import sanitize_filename


class DOIExtractor(BaseExtractor):
    """Extractor for medical DOI classification fields."""

    def load_prompts(self) -> Dict[str, str]:
        """Load DOI-specific prompt templates."""
        prompts_file = Path('config/prompts/doi_prompts.yaml')
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                return yaml.safe_load(f)

        # Fallback prompts
        return {
            'system': """You are a medical research assistant specializing in plastic surgery literature analysis.
Extract THREE specific classification fields from medical abstracts.""",
            'extraction': """Analyze this medical literature and extract three classification fields..."""
        }

    def load_field_options(self) -> Dict[str, List[str]]:
        """Load DOI-specific field options."""
        fields_file = Path('config/fields/doi_fields.yaml')
        if fields_file.exists():
            with open(fields_file, 'r') as f:
                data = yaml.safe_load(f) or {}

                def to_list(x):
                    return [str(v) for v in (x or [])]

                return {
                    'study_design': to_list(data.get('study_design')),
                    'subspecialty_focus': to_list(data.get('subspecialty_focus')),
                    'priority_topic': to_list(data.get('priority_topic')),
                }

        # Fallback minimal defaults
        return {
            'study_design': [
                "Randomized controlled trial", "Prospective cohort", "Retrospective cohort",
                "Cross-sectional study", "Case-control study", "Systematic review",
                "Case series", "Case report", "Preclinical Experimental", "Other"
            ],
            'subspecialty_focus': [
                "Aesthetic / Cosmetic (non-breast)", "Breast", "Craniofacial",
                "Hand/Upper extremity & Peripheral Nerve", "Burn", "Generalized Cutaneous Disorders",
                "Head & Neck Reconstruction", "Trunk / Pelvic / Lower-Limb Reconstruction",
                "Gender-affirming Surgery", "Education & Technology", "Other"
            ],
            'priority_topic': ["Other"]
        }

    def build_prompt(self, record: InputRecord) -> str:
        """Build extraction prompt from DOI record."""
        # Create formatted field lists for injection into prompt
        study_design_options = "\n  - ".join([f'"{opt}"' for opt in self.field_options['study_design']])
        subspecialty_focus_options = "\n  - ".join([f'"{opt}"' for opt in self.field_options['subspecialty_focus']])
        priority_topic_options = "\n    • ".join(self.field_options['priority_topic'])

        # Format the prompt with both record data and field options
        return self.prompts['extraction'].format(
            title=record.title or "Not provided",
            authors=record.author or "Not provided",
            journal=record.publication_title or "Not provided",
            year=int(record.publication_year) if record.publication_year else "Not provided",
            abstract=record.abstract_note or "Not provided",
            affiliations=record.author_affiliation_new or "Not provided",
            study_design_options=study_design_options,
            subspecialty_focus_options=subspecialty_focus_options,
            priority_topic_options=priority_topic_options
        )

    def assess_complexity(self, record: InputRecord) -> float:
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

    def parse_llm_response(self, response_content: str, provider: Any) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        if isinstance(provider, (OllamaProvider, OpenAIProvider)):
            return provider.parse_structured_output(response_content)
        else:
            return json.loads(response_content)

    def create_extracted_data(
        self,
        record: InputRecord,
        extracted: Dict[str, Any],
        confidence_scores: Any,
        transparency_metadata: Any,
        processing_time: float,
        llm_confidence: float = 1.0
    ) -> ExtractedData:
        """Create ExtractedData model from parsed response."""
        # Log warnings for missing data
        warning_logs = []

        if not record.abstract_note:
            warning_logs.append("Missing abstract - confidence will be limited to 60%")
        if not record.author:
            warning_logs.append("Missing author information")
        if not record.author_affiliation_new:
            warning_logs.append("Missing author affiliations")
        if not record.publication_year:
            warning_logs.append("Missing publication year")

        # Calculate confidence scores using actual LLM confidence from provider
        confidence_scores = self.quality_validator.calculate_confidence_scores(
            extracted_data=extracted,
            input_record=record,
            llm_confidence=llm_confidence,
            warning_logs=warning_logs
        )

        # Extract year
        year = None
        if record.publication_year:
            year = str(int(record.publication_year))

        # Create transparency metadata
        prompt_hash = self.audit_logger.get_prompt_version_hash(self.prompts.get('extraction', ''))

        transparency_metadata = TransparencyMetadata(
            processing_session_id=self.session_id,
            processing_timestamp=datetime.now(),
            llm_provider_used="unknown",  # Will be set by parent
            llm_model_version="unknown",  # Will be set by parent
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

        # Normalize choices using data-driven options
        study_design = self.normalize_choice(extracted.get('study_design'), self.field_options['study_design'], record.doi)
        subspecialty_focus = self.normalize_choice(extracted.get('subspecialty_focus'), self.field_options['subspecialty_focus'], record.doi)
        pt_value = extracted.get('priority_topic')
        if not pt_value and isinstance(extracted.get('priority_topics'), list) and extracted.get('priority_topics'):
            pt_value = extracted['priority_topics'][0]
        priority_topic = self.normalize_choice(pt_value, self.field_options['priority_topic'], record.doi)

        # Create extracted data object
        extracted_data = ExtractedData(
            doi=record.doi,
            title=record.title,
            abstract=record.abstract_note,
            authors=record.author,
            journal=record.publication_title,
            year=year,
            study_design=study_design,
            study_design_other=extracted.get('study_design_other'),
            subspecialty_focus=subspecialty_focus,
            subspecialty_focus_other=extracted.get('subspecialty_focus_other'),
            priority_topic=priority_topic,
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

        if not is_valid and self.config.get('processing', {}).get('strict_validation', False):
            raise ValueError(f"Validation failed: {', '.join(validation_flags)}")

        return extracted_data

    def save_extraction(self, data: ExtractedData, output_path: Path):
        """Save extraction to YAML file."""
        # Convert to dict
        data_dict = data.model_dump()
        # Ensure single priority_topic key (backward compatible cleanup)
        if 'priority_topics' in data_dict:
            data_dict.pop('priority_topics', None)

        # Convert datetime to string
        data_dict['transparency_metadata']['processing_timestamp'] = (
            data.transparency_metadata.processing_timestamp.isoformat()
        )

        # Save to YAML
        with open(output_path, 'w') as f:
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

    def get_record_identifier(self, record: InputRecord) -> str:
        """Get unique identifier for a record."""
        return record.doi if record.doi else "unknown"

    def get_output_path(self, record: InputRecord) -> Path:
        """Get output file path for a record."""
        doi = record.doi if record.doi else "unknown"
        safe_name = sanitize_filename(doi.replace('/', '_'))
        return self.output_dir / f"{safe_name}.yaml"
