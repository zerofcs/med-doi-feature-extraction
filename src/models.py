"""
Pydantic data models for the medical literature extraction pipeline.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class StudyDesign(str, Enum):
    """FIRST FIELD: Study design classification."""
    RANDOMIZED_CONTROLLED_TRIAL = "Randomized controlled trial"
    PROSPECTIVE_COHORT = "Prospective cohort"
    RETROSPECTIVE_COHORT = "Retrospective cohort"
    CROSS_SECTIONAL = "Cross-sectional study"
    CASE_CONTROL = "Case-control study"
    SYSTEMATIC_REVIEW = "Systematic review"
    CASE_SERIES = "Case series"
    CASE_REPORT = "Case report"
    PRECLINICAL_EXPERIMENTAL = "Preclinical Experimental"
    OTHER = "Other"


class SubspecialtyFocus(str, Enum):
    """SECOND FIELD: Subspecialty focus categories."""
    AESTHETIC_COSMETIC = "Aesthetic / Cosmetic (non-breast)"
    BREAST = "Breast"
    CRANIOFACIAL = "Craniofacial"
    HAND_UPPER_PERIPHERAL = "Hand/Upper extremity & Peripheral Nerve"
    BURN = "Burn"
    CUTANEOUS_DISORDERS = "Generalized Cutaneous Disorders"
    HEAD_NECK_RECONSTRUCTION = "Head & Neck Reconstruction"
    TRUNK_PELVIC_LOWER = "Trunk / Pelvic / Lower-Limb Reconstruction"
    GENDER_AFFIRMING = "Gender-affirming Surgery"
    EDUCATION_TECHNOLOGY = "Education & Technology"
    OTHER = "Other"




class PriorityTopics(str, Enum):
    """THIRD FIELD: Alignment with Priority Topics in the Plastic Surgery Community."""
    PATIENT_SAFETY = "Patient Safety & Clinical Standards"
    IMPLANTS_DEVICES = "Implants and Device-Related Issues"
    AESTHETIC_POLICY = "Aesthetic Surgery Policy and Regulation"
    TECHNOLOGY_INNOVATION = "Technology and Innovation"
    HEALTH_SYSTEM_PRACTICE = "Health System and Practice Regulation"
    LEGISLATION_ADVOCACY = "Legislation, Advocacy & Public Health"
    WELLNESS_WORKFORCE = "Wellness & Workforce"
    SPECIAL_PROCEDURES = "Special Procedures and Emerging Topics"


class InputRecord(BaseModel):
    """Input record from Excel spreadsheet."""
    key: str = Field(..., description="Unique record key")
    item_type: Optional[str] = Field(None, description="Item type")
    publication_year: Optional[float] = Field(None, description="Year of publication")
    author: Optional[str] = Field(None, description="Author list")
    number_of_authors: Optional[int] = Field(None, description="Number of authors")
    title: Optional[str] = Field(None, description="Article title")
    publication_title: Optional[str] = Field(None, description="Journal/publication name")
    isbn: Optional[str] = Field(None, description="ISBN")
    issn: Optional[str] = Field(None, description="ISSN")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="URL")
    abstract_note: Optional[str] = Field(None, description="Abstract content")
    date: Optional[str] = Field(None, description="Publication date")
    issue: Optional[str] = Field(None, description="Issue number")
    volume: Optional[str] = Field(None, description="Volume number")
    journal_abbreviation: Optional[str] = Field(None, description="Journal abbreviation")
    author_affiliation_new: Optional[str] = Field(None, description="Author affiliations")
    trimmed_author_list: Optional[str] = Field(None, description="Trimmed author list")
    
    @validator('doi')
    def clean_doi(cls, v):
        """Clean and standardize DOI format."""
        if v:
            return v.strip().replace(' ', '')
        return v


class ConfidenceScores(BaseModel):
    """Confidence scores for each extracted field."""
    study_design: float = Field(default=0.0, ge=0.0, le=1.0)
    subspecialty_focus: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_topics: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


class TransparencyMetadata(BaseModel):
    """Transparency and audit metadata for research reproducibility."""
    processing_session_id: str
    processing_timestamp: datetime
    llm_provider_used: str
    llm_model_version: str
    prompt_version_hash: str
    processing_time_seconds: float
    retry_count: int = 0
    validation_flags: List[str] = Field(default_factory=list)
    warning_logs: List[str] = Field(default_factory=list)  # For missing non-critical data
    human_review_required: bool = False
    alternative_extractions: Dict[str, Any] = Field(default_factory=dict)
    has_raw_llm_response: bool = True
    confidence_explanation: Optional[str] = None
    other_specifications: Dict[str, str] = Field(default_factory=dict)  # When "Other" is selected


class ExtractedData(BaseModel):
    """Extracted data from a single DOI."""
    # Original data
    doi: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    
    # THREE REQUIRED EXTRACTED FIELDS
    # Field 1: Study Design - Single selection
    study_design: Optional[StudyDesign] = None
    study_design_other: Optional[str] = None  # Specification if "Other" selected
    
    # Field 2: Subspecialty Focus - Single selection
    subspecialty_focus: Optional[SubspecialtyFocus] = None
    subspecialty_focus_other: Optional[str] = None  # Specification if "Other" selected
    
    # Field 3: Priority Topics - Multiple selections allowed
    priority_topics: Optional[List[PriorityTopics]] = Field(default_factory=list)
    priority_topics_details: Optional[List[str]] = Field(default_factory=list)  # Specific sub-items
    
    # Quality metrics
    confidence_scores: ConfidenceScores = Field(default_factory=ConfidenceScores)
    
    # Transparency metadata
    transparency_metadata: TransparencyMetadata
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingFailure(BaseModel):
    """Record of a processing failure for retry management."""
    doi: str
    key: str
    failure_category: str  # "missing_abstract", "llm_error", "timeout", "validation_error"
    failure_reason: str
    failure_timestamp: datetime
    retry_count: int = 0
    last_retry_timestamp: Optional[datetime] = None
    input_data: Dict[str, Any]
    traceback: Optional[str] = None


class CostSummary(BaseModel):
    """Cost tracking for a session."""
    total_cost: float = 0.0
    cost_by_model: Dict[str, float] = Field(default_factory=dict)
    token_usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)  # model -> {input, output, total}
    average_cost_per_extraction: float = 0.0
    cost_warnings: List[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """Results from model benchmarking."""
    model_name: str
    total_records: int
    successful_extractions: int
    failed_extractions: int
    average_confidence: float
    average_processing_time: float
    total_cost: float
    average_cost_per_extraction: float
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict)
    error_categories: Dict[str, int] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    """Summary of a processing session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    failed_no_abstract: int = 0  # Track abstract failures separately
    skipped_already_processed: int = 0
    records_with_warnings: int = 0  # Track records with missing non-critical data
    average_processing_time: float = 0.0
    failure_categories: Dict[str, int] = Field(default_factory=dict)
    llm_provider_stats: Dict[str, int] = Field(default_factory=dict)
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    human_review_required_count: int = 0
    cost_summary: CostSummary = Field(default_factory=CostSummary)
    model_usage_stats: Dict[str, int] = Field(default_factory=dict)  # Track which models were used


class AuditLogEntry(BaseModel):
    """Single audit log entry for transparency."""
    timestamp: datetime
    session_id: str
    doi: str
    event_type: str  # "start", "llm_request", "llm_response", "validation", "save", "error", "warning"
    event_data: Dict[str, Any]
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    processing_time_ms: Optional[float] = None