"""
OpenAI LLM provider implementation.
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider for API-based LLM inference with multi-model support."""

    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        """Initialize OpenAI provider."""
        super().__init__(config)

        # Model selection - prioritize parameter, then config, then fallback
        self.model_name = model_name or config.get("model", "gpt-4o-mini")

        # Load model-specific configuration if available
        self.models_config = config.get("models", {})
        if self.model_name in self.models_config:
            model_config = self.models_config[self.model_name]
            self.max_tokens = model_config.get("max_tokens", 2000)
            self.temperature = model_config.get("temperature", 0.1)
            self.pricing = model_config.get("pricing", {})
            self.confidence_multiplier = model_config.get("confidence_multiplier", 1.0)
            self.complexity_threshold = model_config.get("complexity_threshold", 0.5)
        else:
            # Fallback to legacy config
            self.max_tokens = config.get("max_tokens", 2000)
            self.temperature = config.get("temperature", 0.1)
            self.pricing = {
                "input_per_1m": 0.0,
                "output_per_1m": 0.0,
            }  # Unknown pricing
            self.confidence_multiplier = 1.0
            self.complexity_threshold = 0.5

        # Cost management
        self.cost_limits = config.get("cost_limits", {})
        self.daily_cost = 0.0  # Track daily spending

        # Get API key from environment or config
        api_key = os.getenv("OPENAI_API_KEY") or config.get("api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
                "or add 'api_key' to the OpenAI configuration section. "
                "Example: export OPENAI_API_KEY='sk-...'"
            )

        # Set timeout for API calls
        timeout = config.get("timeout", 60)  # Default 60 seconds
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response from OpenAI model."""
        try:
            start_time = time.time()

            # Debug logging
            # print(f"[DEBUG] OpenAI API call starting - Model: {self.model_name}")
            # print(f"[DEBUG] API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
            # print(f"[DEBUG] Prompt length: {len(prompt)} chars")

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Make API call
            # GPT-5 models use max_completion_tokens instead of max_tokens
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "seed": 42,  # For reproducibility (when supported)
                "response_format": {"type": "json_object"}
                if "json" in prompt.lower()
                else {"type": "text"},
            }

            # Only request logprobs for models that support it (not gpt-5)
            if not self.model_name.startswith("gpt-5"):
                api_params["logprobs"] = True
                api_params["top_logprobs"] = 1

            # Use correct token parameter based on model
            if self.model_name.startswith("gpt-5"):
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens

            # print(f"[DEBUG] Calling OpenAI API with params: model={api_params['model']}, temp={api_params['temperature']}")
            response = await self.client.chat.completions.create(**api_params)

            processing_time = time.time() - start_time
            # print(f"[DEBUG] OpenAI API call completed in {processing_time:.2f}s")

            # Calculate cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Calculate dynamic confidence from response quality signals
            confidence = self._calculate_confidence(response)

            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response.model_dump(),
                provider="openai",
                model=self.model_name,
                processing_time=processing_time,
                token_count=response.usage.total_tokens if response.usage else None,
                confidence=confidence,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing."""
        if not self.pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * self.pricing.get("input_per_1m", 0.0)
        output_cost = (output_tokens / 1_000_000) * self.pricing.get(
            "output_per_1m", 0.0
        )
        total_cost = input_cost + output_cost

        # Update daily cost tracking
        self.daily_cost += total_cost

        return total_cost

    def _calculate_confidence(self, response: Any) -> float:
        """
        Calculate dynamic confidence based on response quality signals.

        Uses multiple signals:
        1. Logprobs: Average token probability (lower perplexity = higher confidence)
        2. Finish reason: Penalize incomplete responses
        3. Refusal: Zero confidence if model refused
        4. Model-specific multiplier: Account for known model capabilities

        Args:
            response: OpenAI API response object

        Returns:
            Confidence score between 0.0 and 1.0
        """
        import math

        choice = response.choices[0]

        # Check for refusal (GPT-5 models)
        if hasattr(choice.message, "refusal") and choice.message.refusal:
            return 0.0

        # Check finish reason
        finish_reason = choice.finish_reason
        if finish_reason == "length":
            # Response truncated - lower confidence
            finish_penalty = 0.7
        elif finish_reason == "content_filter":
            # Content filtered - very low confidence
            finish_penalty = 0.3
        elif finish_reason == "stop":
            # Normal completion - no penalty
            finish_penalty = 1.0
        else:
            # Unknown finish reason - slight penalty
            finish_penalty = 0.9

        # Calculate logprob-based confidence
        logprob_confidence = 1.0  # Default if logprobs not available

        if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
            # Extract log probabilities for each token
            logprobs = [
                token.logprob
                for token in choice.logprobs.content
                if token.logprob is not None
            ]

            if logprobs:
                # Calculate average logprob
                avg_logprob = sum(logprobs) / len(logprobs)

                # Convert to probability using exp(logprob)
                # Average probability across all tokens
                avg_prob = math.exp(avg_logprob)

                # Scale to confidence (sigmoid-like curve)
                # High avg_prob (>0.8) -> high confidence
                # Low avg_prob (<0.3) -> low confidence
                if avg_prob >= 0.8:
                    logprob_confidence = 0.85 + (avg_prob - 0.8) * 0.75  # 0.85-1.0
                elif avg_prob >= 0.5:
                    logprob_confidence = 0.65 + (avg_prob - 0.5) * 0.67  # 0.65-0.85
                elif avg_prob >= 0.3:
                    logprob_confidence = 0.45 + (avg_prob - 0.3) * 1.0  # 0.45-0.65
                else:
                    logprob_confidence = (
                        avg_prob * 1.5
                    )  # 0.0-0.45 (linear for very low)

                logprob_confidence = min(max(logprob_confidence, 0.0), 1.0)

        # Combine signals
        base_confidence = logprob_confidence * finish_penalty

        # Apply model-specific multiplier
        adjusted_confidence = base_confidence * self.confidence_multiplier

        # Clamp to [0, 1]
        return min(max(adjusted_confidence, 0.0), 1.0)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            "model_name": self.model_name,
            "pricing": self.pricing,
            "confidence_multiplier": self.confidence_multiplier,
            "complexity_threshold": self.complexity_threshold,
            "daily_cost": self.daily_cost,
            "cost_limits": self.cost_limits,
        }

    def can_handle_complexity(self, complexity_score: float) -> bool:
        """Check if this model can handle the given complexity level."""
        return complexity_score <= self.complexity_threshold

    def is_cost_within_limits(self, estimated_cost: float) -> bool:
        """Check if the estimated cost is within configured limits."""
        max_cost = self.cost_limits.get("max_cost_per_extraction", float("inf"))
        max_daily = self.cost_limits.get("max_daily_cost", float("inf"))

        if estimated_cost > max_cost:
            return False
        if self.daily_cost + estimated_cost > max_daily:
            return False

        return True

    def validate_connection(self) -> Tuple[bool, str]:
        """Validate connection to OpenAI API."""
        try:
            # Try a simple synchronous test
            import openai

            api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")

            if not api_key:
                return False, (
                    "Missing API key. Set OPENAI_API_KEY environment variable "
                    "or add 'api_key' to OpenAI config"
                )

            client = openai.OpenAI(api_key=api_key)

            # List models to test connection
            models = client.models.list()
            return True, f"Connected to OpenAI API with model {self.model_name}"

        except openai.AuthenticationError as e:
            return False, (
                f"OpenAI authentication failed: {str(e)}. "
                "Check your API key is valid and active"
            )
        except openai.APIConnectionError as e:
            return False, (
                f"OpenAI network connection failed: {str(e)}. "
                "Check your internet connection or try again later"
            )
        except Exception as e:
            return False, f"OpenAI API validation error: {str(e)}"

    def parse_structured_output(self, content: str) -> Dict[str, Any]:
        """
        Parse structured output from model response.

        OpenAI supports forced JSON format via response_format: json_object,
        so we try direct parsing first.
        """
        from ..utils import parse_json_from_text

        return parse_json_from_text(content, try_direct_parse=True)
