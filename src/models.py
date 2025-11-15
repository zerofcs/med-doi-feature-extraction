from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .providers.base import LLMResponse as ProviderLLMResponse


class Record(BaseModel):
    """Normalized input record used for prompting."""
    key: str
    data: Dict[str, Any]


class InputRecord(BaseModel):
    """Simple schema for tabular inputs with DOI and common metadata.

    Used by CLI helpers for preview/test flows; the engine remains domain-agnostic.
    """
    DOI: Optional[str] = None
    Title: Optional[str] = None
    Authors: Optional[str] = None
    Journal: Optional[str] = None
    Year: Optional[int] = None
    Abstract: Optional[str] = None
    Affiliations: Optional[str] = None


class Confidence(BaseModel):
    overall: float = 0.0
    fields: Dict[str, float] = Field(default_factory=dict)


class TransparencyMetadata(BaseModel):
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    processing_time: Optional[float] = None
    system_prompt_hash: Optional[str] = None
    user_prompt_hash: Optional[str] = None


class ExtractionResult(BaseModel):
    key: str
    input: Dict[str, Any]
    extracted: Dict[str, Any]
    normalized: Dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence = Field(default_factory=Confidence)
    valid: bool = True
    errors: List[str] = Field(default_factory=list)
    transparency: TransparencyMetadata
    raw_llm: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionSummary(BaseModel):
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    cost_total: float = 0.0
    notes: Dict[str, Any] = Field(default_factory=dict)
