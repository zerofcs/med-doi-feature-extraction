"""
Ollama LLM provider implementation.
"""

import asyncio
from typing import Dict, Any, Optional, Tuple
import ollama
from .base import LLMProvider, LLMResponse
import time


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama3.1")
        self.timeout = config.get("timeout", 120)
        self.temperature = config.get("temperature", 0.1)
        self.client = ollama.Client(host=self.base_url)

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response from Ollama model."""
        try:
            start_time = time.time()

            # Combine system and user prompts
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Make synchronous call (ollama library doesn't have async yet)
            # Wrap with timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "seed": 42,  # For reproducibility
                    },
                ),
                timeout=self.timeout,
            )

            processing_time = time.time() - start_time

            return LLMResponse(
                content=response["message"]["content"],
                raw_response=response,
                provider="ollama",
                model=self.model,
                processing_time=processing_time,
                token_count=response.get("eval_count"),
                confidence=0.9,  # Default confidence for local models
            )

        except asyncio.TimeoutError:
            raise Exception(
                f"Ollama generation timed out after {self.timeout} seconds. Check if Ollama is running and responsive."
            )
        except Exception as e:
            raise Exception(f"Ollama generation failed: {str(e)}")

    def validate_connection(self) -> Tuple[bool, str]:
        """Validate connection to Ollama."""
        try:
            # Check if Ollama is running
            response = self.client.list()

            # Handle ollama ListResponse object
            model_names = []
            if hasattr(response, "models"):
                # It's a ListResponse object
                for model in response.models:
                    if hasattr(model, "model"):
                        model_names.append(model.model)
                    elif hasattr(model, "name"):
                        model_names.append(model.name)
            elif isinstance(response, dict) and "models" in response:
                # It's a dict response
                for m in response["models"]:
                    if isinstance(m, dict) and "name" in m:
                        model_names.append(m["name"])

            # Check if required model is available
            if any(self.model in name for name in model_names):
                return True, f"Connected to Ollama with model {self.model}"
            elif model_names:
                return (
                    False,
                    f"Model {self.model} not found. Available: {', '.join(model_names[:3])}",
                )
            else:
                return False, "No models found in Ollama"

        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}"

    def parse_structured_output(self, content: str) -> Dict[str, Any]:
        """
        Parse structured output from model response.

        Ollama does not support forced JSON format, so we use extraction strategies only.
        """
        from ..utils import parse_json_from_text

        return parse_json_from_text(content, try_direct_parse=False)
