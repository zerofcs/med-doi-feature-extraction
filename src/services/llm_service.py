from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from ..config.schema import ConfigSchema
from ..models import Record
from ..providers.base import LLMResponse
from ..providers.openai_provider import OpenAIProvider
from ..providers.ollama_provider import OllamaProvider


class LLMService:
    def __init__(self, cfg: ConfigSchema) -> None:
        self.cfg = cfg
        self.default_provider = cfg.llm.default_provider

        # Instantiate providers lazily
        self._providers: Dict[str, Any] = {}

    def _get_provider(self, provider_name: str, *, model_name: Optional[str] = None):
        if provider_name == "openai":
            # Pass model_name so provider can load model-specific config
            return OpenAIProvider(cfg_to_dict(self.cfg.llm.openai), model_name=model_name)
        elif provider_name == "ollama":
            return OllamaProvider(cfg_to_dict(self.cfg.llm.ollama))
        raise ValueError(f"Unknown provider: {provider_name}")

    async def execute_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        record: Record,
        strategy: Optional[str] = None,
    ) -> LLMResponse:
        provider_name = self.default_provider
        llm_cfg = self.cfg.llm
        model_strategy = (strategy or llm_cfg.model_selection_strategy).lower()

        # Select initial model for OpenAI when using strategy
        model_candidates: List[str] = []
        if provider_name == "openai":
            # For manual strategy, use the explicitly configured model
            if model_strategy == "manual":
                selected_model = llm_cfg.openai.model
                model_candidates = [selected_model]
                start_ix = 0
            else:
                # Basic 3-tier strategy for automatic selection
                nano = llm_cfg.default_openai_model or "gpt-5-nano"
                mini = "gpt-5-mini"
                full = "gpt-5"
                model_candidates = [nano, mini, full]

                # Choose starting point by complexity
                from .quality_service import QualityService

                q = QualityService(
                    min_confidence_threshold=self.cfg.quality.min_confidence_threshold,
                    review_threshold=self.cfg.quality.review_threshold,
                )
                complexity = q.calculate_complexity_score(record)
                if model_strategy in {"cost-optimized", "balanced"}:
                    if complexity < 0.5:
                        start_ix = 0
                    elif complexity < 0.8:
                        start_ix = 1
                    else:
                        start_ix = 2
                elif model_strategy in {"accuracy-first"}:
                    start_ix = 2
                else:  # unknown strategy -> default to nano
                    start_ix = 0
        else:
            model_candidates = [None]  # Ollama provider ignores this list
            start_ix = 0

        # Attempt with fallback when enabled
        last_exc: Optional[Exception] = None
        for ix in range(start_ix, len(model_candidates)):
            model = model_candidates[ix]
            provider = self._get_provider(provider_name, model_name=model)
            try:
                resp = await provider.generate(user_prompt, system_prompt=system_prompt)
                # Early stopping: if auto_fallback disabled, return immediately
                if not self.cfg.processing.auto_fallback:
                    return resp
                # If confidence meets threshold, return; else, fallback to next
                if resp.confidence >= self.cfg.processing.fallback_confidence_threshold:
                    return resp
                last_exc = None
            except Exception as e:
                last_exc = e
                continue

        if last_exc:
            raise last_exc
        # If we got here, last attempt returned low confidence but with a response
        return resp  # type: ignore[name-defined]


def cfg_to_dict(model: Any) -> Dict[str, Any]:
    return model.model_dump(mode="python") if hasattr(model, "model_dump") else dict(model)

