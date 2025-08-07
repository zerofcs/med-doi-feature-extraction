"""
Quality validation and confidence scoring system.
"""

from typing import Dict, Any, List, Tuple, Optional
from .models import ExtractedData, ConfidenceScores, InputRecord


class QualityValidator:
    """Validates extraction quality and calculates confidence scores."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality validator."""
        self.config = config or {}
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.review_threshold = self.config.get('review_threshold', 0.7)
    
    def calculate_confidence_scores(
        self,
        extracted_data: Dict[str, Any],
        input_record: InputRecord,
        llm_confidence: float = 1.0,
        warning_logs: List[str] = None
    ) -> ConfidenceScores:
        """
        Calculate weighted confidence scores based on available input fields.
        
        Field availability affects maximum achievable confidence:
        - Title only: Max 35% confidence
        - Title + Abstract: Max 60% confidence  
        - Title + Abstract + Authors/Affiliations: Max 85% confidence
        - All fields (incl. extra/manual tags): Max 100% confidence
        """
        scores = ConfidenceScores()
        warning_logs = warning_logs or []
        
        # Determine maximum confidence based on available input fields
        max_confidence = self._calculate_max_confidence_from_inputs(input_record)
        
        # Base confidence starts with LLM confidence, capped by input field availability
        base_confidence = min(llm_confidence, max_confidence)
        
        # Apply warning penalties
        if warning_logs:
            # Reduce confidence by 0.05 for each warning
            base_confidence *= (1.0 - (0.05 * len(warning_logs)))
            base_confidence = max(base_confidence, 0.3)  # Floor at 0.3
        
        # Field 1: Subspecialty Focus (single selection)
        if extracted_data.get('subspecialty_focus'):
            if extracted_data['subspecialty_focus'].lower() == 'other':
                # Lower confidence for "Other"
                scores.subspecialty_focus = base_confidence * 0.7
                if not extracted_data.get('subspecialty_focus_other'):
                    # Even lower if no specification provided
                    scores.subspecialty_focus *= 0.8
            else:
                scores.subspecialty_focus = base_confidence * 0.9
        else:
            scores.subspecialty_focus = 0.0
        
        # Field 2: Suggested Edits (multiple selections)
        if extracted_data.get('suggested_edits'):
            edits = extracted_data['suggested_edits']
            if isinstance(edits, list) and len(edits) > 0:
                # Check if "Other" is in the list
                has_other = any('other' in str(e).lower() for e in edits)
                if has_other:
                    scores.suggested_edits = base_confidence * 0.75
                    if not extracted_data.get('suggested_edits_other'):
                        scores.suggested_edits *= 0.85
                else:
                    scores.suggested_edits = base_confidence * 0.9
                
                # Reduce confidence if too many categories selected (might indicate uncertainty)
                if len(edits) > 5:
                    scores.suggested_edits *= 0.85
            else:
                scores.suggested_edits = 0.3
        else:
            scores.suggested_edits = 0.0
        
        # Field 3: Priority Topics (multiple selections with details)
        if extracted_data.get('priority_topics'):
            topics = extracted_data['priority_topics']
            details = extracted_data.get('priority_topics_details', [])
            
            if isinstance(topics, list) and len(topics) > 0:
                scores.priority_topics = base_confidence * 0.85
                
                # Bonus confidence if specific details are provided
                if details and len(details) > 0:
                    scores.priority_topics = min(scores.priority_topics * 1.1, 1.0)
                
                # Reduce confidence if too many topics (might indicate uncertainty)
                if len(topics) > 6:
                    scores.priority_topics *= 0.8
            else:
                scores.priority_topics = 0.3
        else:
            scores.priority_topics = 0.0
        
        # Calculate overall confidence
        field_scores = [
            scores.subspecialty_focus,
            scores.suggested_edits,
            scores.priority_topics
        ]
        
        # Weight the three fields equally
        valid_scores = [s for s in field_scores if s > 0]
        if valid_scores:
            scores.overall = sum(valid_scores) / len(valid_scores)
        else:
            scores.overall = 0.0
        
        # Apply penalties for critical issues
        if "Missing author affiliations" in warning_logs:
            scores.overall *= 0.95  # Small penalty
        if "Missing publication year" in warning_logs:
            scores.overall *= 0.98  # Very small penalty
        
        return scores
    
    def _calculate_max_confidence_from_inputs(self, input_record: InputRecord) -> float:
        """
        Calculate maximum achievable confidence based on available input fields.
        
        Field importance hierarchy:
        1. Abstract (most critical for medical classification)
        2. Title (essential context)
        3. Authors/Affiliations (institutional context) 
        4. Publication details (year, journal, etc.)
        5. Additional metadata (extra/manual tags)
        """
        confidence = 0.0
        
        # Title is baseline requirement (35% max if only title)
        if input_record.title and input_record.title.strip():
            confidence = 0.35
        else:
            return 0.1  # Minimal confidence without title
        
        # Abstract is most important for medical classification (jumps to 60%)
        if input_record.abstract_note and input_record.abstract_note.strip():
            confidence = 0.60
            
            # Author information adds institutional context (up to 75%)
            if (input_record.author and input_record.author.strip()) or \
               (input_record.author_affiliation_new and input_record.author_affiliation_new.strip()):
                confidence = 0.75
                
                # Publication details add credibility context (up to 85%)
                pub_details = [
                    input_record.publication_year,
                    input_record.publication_title,
                    input_record.journal_abbreviation,
                    input_record.doi
                ]
                if any(detail for detail in pub_details if detail):
                    confidence = 0.85
                    
                    # Complete metadata allows full confidence (up to 100%)
                    additional_fields = [
                        input_record.volume,
                        input_record.issue,
                        input_record.issn,
                        input_record.url,
                        input_record.trimmed_author_list
                    ]
                    if sum(1 for field in additional_fields if field) >= 2:
                        confidence = 1.0
        
        return confidence
    
    def validate_extraction(
        self,
        extracted_data: ExtractedData
    ) -> Tuple[bool, List[str], bool]:
        """
        Validate extracted data quality.
        
        Returns:
            - is_valid: Whether extraction meets minimum quality
            - validation_flags: List of quality issues
            - needs_human_review: Whether human review is recommended
        """
        validation_flags = []
        needs_human_review = False
        
        # Check overall confidence
        if extracted_data.confidence_scores.overall < self.min_confidence_threshold:
            validation_flags.append(f"Low overall confidence: {extracted_data.confidence_scores.overall:.2f}")
            needs_human_review = True
        
        # Check for missing critical fields
        if not extracted_data.subspecialty_focus:
            validation_flags.append("Missing Field 1: Subspecialty Focus")
        
        if not extracted_data.suggested_edits:
            validation_flags.append("Missing Field 2: Suggested Edits")
        
        if not extracted_data.priority_topics:
            validation_flags.append("Missing Field 3: Priority Topics")
        
        # Check for "Other" selections that need review
        if extracted_data.subspecialty_focus and extracted_data.subspecialty_focus.value == "Other":
            if not extracted_data.subspecialty_focus_other:
                validation_flags.append("Field 1 'Other' selected without specification")
                needs_human_review = True
        
        if extracted_data.suggested_edits:
            has_other = any(e.value == "Other" for e in extracted_data.suggested_edits)
            if has_other and not extracted_data.suggested_edits_other:
                validation_flags.append("Field 2 'Other' selected without specification")
                needs_human_review = True
        
        # Check if review is needed based on confidence threshold
        if extracted_data.confidence_scores.overall < self.review_threshold:
            needs_human_review = True
        
        # Check for extraction anomalies
        if extracted_data.suggested_edits and len(extracted_data.suggested_edits) > 7:
            validation_flags.append("Unusually high number of suggested edits")
            needs_human_review = True
        
        if extracted_data.priority_topics and len(extracted_data.priority_topics) > 6:
            validation_flags.append("Unusually high number of priority topics")
            needs_human_review = True
        
        # Check for warnings that might need review
        if extracted_data.transparency_metadata.warning_logs:
            if len(extracted_data.transparency_metadata.warning_logs) > 2:
                needs_human_review = True
        
        # Determine if valid
        is_valid = (
            extracted_data.subspecialty_focus is not None and
            extracted_data.suggested_edits is not None and
            extracted_data.priority_topics is not None and
            extracted_data.confidence_scores.overall >= self.min_confidence_threshold
        )
        
        return is_valid, validation_flags, needs_human_review
    
    def suggest_retry_strategy(
        self,
        failure_category: str,
        retry_count: int
    ) -> Optional[Dict[str, Any]]:
        """Suggest retry strategy based on failure type."""
        strategies = {
            "missing_abstract": {
                "action": "skip",
                "message": "Abstract is required - cannot proceed without it"
            },
            "missing_data": {
                "action": "skip" if retry_count > 0 else "retry_with_partial",
                "message": "Missing non-critical data, attempting with available fields"
            },
            "llm_error": {
                "action": "retry_with_fallback" if retry_count < 2 else "manual_review",
                "message": "LLM processing error, switching providers"
            },
            "timeout": {
                "action": "retry_with_longer_timeout" if retry_count < 2 else "skip",
                "message": "Processing timeout, increasing timeout duration"
            },
            "validation_error": {
                "action": "retry_with_adjusted_prompt" if retry_count < 1 else "manual_review",
                "message": "Validation failed, adjusting extraction prompt"
            },
            "parsing_error": {
                "action": "retry_with_simpler_format" if retry_count < 2 else "manual_review",
                "message": "Output parsing failed, requesting simpler format"
            }
        }
        
        return strategies.get(failure_category, {
            "action": "manual_review",
            "message": "Unknown failure type, requires manual review"
        })
    
    def generate_quality_report(
        self,
        extracted_data_list: List[ExtractedData]
    ) -> Dict[str, Any]:
        """Generate quality report for a batch of extractions."""
        report = {
            "total_extractions": len(extracted_data_list),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "needs_review": 0,
            "has_other_selections": 0,
            "missing_fields": {
                "subspecialty_focus": 0,
                "suggested_edits": 0,
                "priority_topics": 0
            },
            "validation_issues": {},
            "confidence_distribution": {},
            "field_coverage": {}
        }
        
        for data in extracted_data_list:
            # Confidence buckets
            if data.confidence_scores.overall >= 0.8:
                report["high_confidence"] += 1
            elif data.confidence_scores.overall >= 0.6:
                report["medium_confidence"] += 1
            else:
                report["low_confidence"] += 1
            
            # Review needed
            if data.transparency_metadata.human_review_required:
                report["needs_review"] += 1
            
            # Check for "Other" selections
            if (data.subspecialty_focus and data.subspecialty_focus.value == "Other") or \
               (data.suggested_edits and any(e.value == "Other" for e in data.suggested_edits)):
                report["has_other_selections"] += 1
            
            # Missing fields
            if not data.subspecialty_focus:
                report["missing_fields"]["subspecialty_focus"] += 1
            if not data.suggested_edits:
                report["missing_fields"]["suggested_edits"] += 1
            if not data.priority_topics:
                report["missing_fields"]["priority_topics"] += 1
            
            # Validation issues
            for flag in data.transparency_metadata.validation_flags:
                if flag not in report["validation_issues"]:
                    report["validation_issues"][flag] = 0
                report["validation_issues"][flag] += 1
            
            # Field coverage
            fields = ["subspecialty_focus", "suggested_edits", "priority_topics"]
            for field in fields:
                if field not in report["field_coverage"]:
                    report["field_coverage"][field] = 0
                if getattr(data, field, None):
                    report["field_coverage"][field] += 1
        
        # Calculate percentages
        if report["total_extractions"] > 0:
            for field in report["field_coverage"]:
                report["field_coverage"][field] = (
                    report["field_coverage"][field] / report["total_extractions"] * 100
                )
        
        return report