from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class ModelPricing(BaseModel):
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


class OpenAIModelConfig(BaseModel):
    max_tokens: int = 2000
    temperature: float = 0.1
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    confidence_multiplier: float = 1.0
    complexity_threshold: float = 0.5


class OpenAIConfig(BaseModel):
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60
    models: Dict[str, OpenAIModelConfig] = Field(default_factory=dict)
    cost_limits: Dict[str, float] = Field(default_factory=dict)


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout: int = 120
    temperature: float = 0.1


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_openai_model: str = "gpt-5-nano"
    model_selection_strategy: str = "cost-optimized"
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class ProcessingConfig(BaseModel):
    test_mode: bool = False
    test_limit: int = 1
    batch_size: int = 1
    delay_between_requests: float = 1.0
    force_reprocess: bool = False
    strict_validation: bool = False
    benchmark_sample_size: int = 50
    auto_fallback: bool = True
    fallback_confidence_threshold: float = 0.6


class InputConfig(BaseModel):
    """Input data configuration including column mapping."""
    column_map: Optional[Dict[str, str]] = None


class QualitySignal(BaseModel):
    """Configuration for a single input quality signal."""
    field: str
    weight: float = 1.0
    required: bool = False


class QualityConfig(BaseModel):
    min_confidence_threshold: float = 0.5
    review_threshold: float = 0.7
    input_signals: List[QualitySignal] = Field(default_factory=list)


class OutputConfig(BaseModel):
    directory: str = "output/extracted"
    format: str = "yaml"  # or csv/json
    csv_aggregate: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = False
    file: bool = True


class PromptTemplates(BaseModel):
    # Simple container for system and user templates (strings)
    system: Optional[str] = None
    extraction: Optional[str] = None


class ConfigSchema(BaseModel):
    """Master configuration model reflecting merged YAML.

    Extra fields are allowed to avoid blocking incremental adoption.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    title: Optional[str] = None
    pipeline: Optional[str] = None  # Pipeline name (e.g., "doi", "country")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    input: Optional[InputConfig] = None
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompts: Optional[PromptTemplates] = None

