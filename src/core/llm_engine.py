"""
Generic LLM orchestration engine.

This module provides provider-independent LLM functionality including:
- Provider initialization and management
- Model selection strategies
- Complexity assessment
- Prompt execution with fallback handling
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime

from ..providers import LLMProvider, OllamaProvider, OpenAIProvider, LLMResponse


class LLMEngine:
    """
    Generic LLM orchestration engine.

    Handles provider management, model selection, and prompt execution
    without any extraction-specific logic.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM engine.

        Args:
            config: Full configuration dict with 'llm' and 'processing' sections
        """
        self.config = config

        # Initialize LLM providers
        self.providers = self._initialize_providers()
        self.default_provider = config.get('llm', {}).get('default_provider', 'ollama')

        # Model selection configuration
        self.model_selection_strategy = config.get('llm', {}).get('model_selection_strategy', 'manual')
        self.default_openai_model = config.get('llm', {}).get('default_openai_model', 'gpt-5-nano')
        self.auto_fallback = config.get('processing', {}).get('auto_fallback', True)
        self.fallback_threshold = config.get('processing', {}).get('fallback_confidence_threshold', 0.6)

    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Initialize available LLM providers."""
        providers = {}
        provider_errors = []  # Track which providers failed and why

        # Initialize Ollama if configured
        if 'ollama' in self.config.get('llm', {}):
            try:
                ollama_provider = OllamaProvider(self.config['llm']['ollama'])
                is_valid, message = ollama_provider.validate_connection()
                if is_valid:
                    providers['ollama'] = ollama_provider
                    logger.info(f"Ollama provider initialized: {message}")
                else:
                    provider_errors.append(f"  ✗ ollama: {message}")
            except Exception as e:
                provider_errors.append(f"  ✗ ollama: Initialization failed - {str(e)}")

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
                            logger.info(f"OpenAI provider initialized: {message}")
                        else:
                            provider_errors.append(f"  ✗ {provider_key}: {message}")
                    except Exception as e:
                        provider_errors.append(f"  ✗ openai-{model_name}: {str(e)}")
            else:
                # Legacy single model configuration
                try:
                    openai_provider = OpenAIProvider(openai_config)
                    is_valid, message = openai_provider.validate_connection()
                    if is_valid:
                        providers['openai'] = openai_provider
                        logger.info(f"OpenAI provider initialized: {message}")
                    else:
                        provider_errors.append(f"  ✗ openai: {message}")
                except Exception as e:
                    provider_errors.append(f"  ✗ openai: {str(e)}")

        if not providers:
            error_msg = "No LLM providers available. Attempted initialization:\n"
            error_msg += "\n".join(provider_errors) if provider_errors else "  (no providers configured)"
            error_msg += "\n\nSuggestions:"

            # Check if OpenAI was attempted
            openai_attempted = any('openai' in err for err in provider_errors)
            ollama_attempted = any('ollama' in err for err in provider_errors)

            if openai_attempted:
                error_msg += "\n  • For OpenAI: Set OPENAI_API_KEY environment variable (export OPENAI_API_KEY='sk-...')"

            if not ollama_attempted or (ollama_attempted and not openai_attempted):
                error_msg += "\n  • For local LLM: Install Ollama from https://ollama.ai/download"
                error_msg += "\n    Then configure it in your config file under 'llm.ollama'"

            raise ValueError(error_msg)

        return providers

    def select_provider(
        self,
        complexity_score: float = 0.5,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None
    ) -> Tuple[str, LLMProvider]:
        """
        Select the optimal provider/model based on complexity and strategy.

        Args:
            complexity_score: Complexity score between 0.0 (simple) and 1.0 (complex)
            force_provider: Force specific provider (e.g., 'openai', 'ollama')
            force_model: Force specific model (e.g., 'gpt-5-nano')

        Returns:
            Tuple of (provider_name, provider_instance)
        """
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
            model_order = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5']
            for model in model_order:
                provider_name = f"openai-{model}"
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    if provider.can_handle_complexity(complexity_score):
                        return provider_name, provider

        elif self.model_selection_strategy == 'balanced':
            # Use gpt-5-mini for most cases, nano for simple, gpt-5 for complex
            if complexity_score < 0.3:
                model = 'gpt-5-nano'
            elif complexity_score > 0.8:
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

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        provider_name: str,
        provider: LLMProvider,
        enable_fallback: bool = True,
        fallback_callback: Optional[Callable] = None
    ) -> Tuple[LLMResponse, str]:
        """
        Generate LLM response with optional fallback handling.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            provider_name: Name of primary provider
            provider: Primary provider instance
            enable_fallback: Enable automatic fallback on failure and low confidence
            fallback_callback: Optional callback when fallback occurs (provider_from, provider_to, error)

        Returns:
            Tuple of (response, final_model_version)
        """
        try:
            response = await provider.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            final_model_version = response.model

            # Check confidence-based fallback (only if auto_fallback is enabled)
            if (enable_fallback and
                self.auto_fallback and
                response.confidence < self.fallback_threshold and
                len(self.providers) > 1):

                # Try to find a better provider
                fallback_name = self._select_fallback_provider(provider_name)
                if fallback_name:
                    # Notify callback if provided
                    if fallback_callback:
                        fallback_callback(
                            provider_name,
                            fallback_name,
                            f"Low confidence ({response.confidence:.2f} < {self.fallback_threshold})"
                        )

                    fallback_provider = self.providers[fallback_name]
                    fallback_response = await fallback_provider.generate(
                        prompt=prompt,
                        system_prompt=system_prompt
                    )

                    # Use fallback response if it has higher confidence
                    if fallback_response.confidence > response.confidence:
                        return fallback_response, fallback_response.model

            return response, final_model_version

        except Exception as e:
            # Try fallback provider if available (error-based fallback)
            if enable_fallback and len(self.providers) > 1:
                fallback_name = self._select_fallback_provider(provider_name)
                if fallback_name:
                    # Notify callback if provided
                    if fallback_callback:
                        fallback_callback(provider_name, fallback_name, str(e))

                    fallback_provider = self.providers[fallback_name]
                    response = await fallback_provider.generate(
                        prompt=prompt,
                        system_prompt=system_prompt
                    )
                    final_model_version = response.model
                    return response, final_model_version
            raise

    def _select_fallback_provider(self, current_provider: str) -> Optional[str]:
        """
        Select a fallback provider different from the current one.

        Prefers higher-tier models (e.g., gpt-5 over gpt-5-mini).

        Args:
            current_provider: Current provider name

        Returns:
            Fallback provider name or None
        """
        available = [k for k in self.providers.keys() if k != current_provider]
        if not available:
            return None

        # Prefer higher-tier models for fallback
        priority_order = [
            'openai-gpt-5',
            'openai-gpt-5-mini',
            'openai-gpt-5-nano',
            'openai',
            'ollama'
        ]

        for preferred in priority_order:
            if preferred in available:
                return preferred

        # Return first available as last resort
        return available[0]
