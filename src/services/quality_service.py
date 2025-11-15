from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..models import Confidence, Record


class QualityService:
    def __init__(
        self,
        *,
        min_confidence_threshold: float = 0.5,
        review_threshold: float = 0.7,
        input_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.review_threshold = review_threshold
        self.input_signals = input_signals or []

    def calculate_complexity_score(self, record: Record) -> float:
        """Very simple heuristic: longer abstracts => higher complexity."""
        text = str(record.data.get("abstract") or record.data.get("Abstract") or "")
        length = len(text)
        if length == 0:
            return 0.2
        if length < 500:
            return 0.4
        if length < 1500:
            return 0.7
        return 0.9

    def calculate_confidence(self, llm_response_conf: float, record: Record) -> Confidence:
        # Calculate input quality based on configured signals
        if self.input_signals:
            input_quality = 0.0
            total_weight = 0.0
            for signal in self.input_signals:
                field = signal.get("field")
                weight = signal.get("weight", 1.0)
                if field in record.data and record.data[field]:
                    input_quality += weight
                total_weight += weight
            input_quality = input_quality / total_weight if total_weight > 0 else 0.5
        else:
            # Fallback to abstract-based for backward compatibility
            input_quality = 1.0 if (record.data.get("abstract") or record.data.get("Abstract")) else 0.7

        # Blend input quality with LLM response confidence
        overall = max(0.0, min(1.0, 0.6 * llm_response_conf + 0.4 * input_quality))
        return Confidence(overall=overall, fields={})

    def validate_extraction(self, extracted: Dict[str, Any], confidence: Confidence) -> Tuple[bool, list[str]]:
        errs: list[str] = []
        if confidence.overall < self.min_confidence_threshold:
            errs.append(f"low_confidence:{confidence.overall:.2f}")
        # Placeholder for required fields / rule DSL
        valid = len(errs) == 0
        return valid, errs

