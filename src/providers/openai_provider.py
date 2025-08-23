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
        self.model_name = model_name or config.get('model', 'gpt-4o-mini')
        
        # Load model-specific configuration if available
        self.models_config = config.get('models', {})
        if self.model_name in self.models_config:
            model_config = self.models_config[self.model_name]
            self.max_tokens = model_config.get('max_tokens', 2000)
            self.temperature = model_config.get('temperature', 0.1)
            self.pricing = model_config.get('pricing', {})
            self.confidence_multiplier = model_config.get('confidence_multiplier', 1.0)
            self.complexity_threshold = model_config.get('complexity_threshold', 0.5)
        else:
            # Fallback to legacy config
            self.max_tokens = config.get('max_tokens', 2000)
            self.temperature = config.get('temperature', 0.1)
            self.pricing = {'input_per_1m': 0.0, 'output_per_1m': 0.0}  # Unknown pricing
            self.confidence_multiplier = 1.0
            self.complexity_threshold = 0.5
        
        # Cost management
        self.cost_limits = config.get('cost_limits', {})
        self.daily_cost = 0.0  # Track daily spending
        
        # Get API key from environment or config
        api_key = os.getenv('OPENAI_API_KEY') or config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment or config")
        
        # Set timeout for API calls
        timeout = config.get('timeout', 60)  # Default 60 seconds
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from OpenAI model."""
        try:
            start_time = time.time()
            
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
                "top_p": 0.9,
                "seed": 42,  # For reproducibility (when supported)
                "response_format": {"type": "json_object"} if "json" in prompt.lower() else {"type": "text"}
            }
            
            # Use correct token parameter based on model
            if self.model_name.startswith('gpt-5'):
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens
                
            response = await self.client.chat.completions.create(**api_params)
            
            processing_time = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Apply model-specific confidence multiplier
            base_confidence = 0.95
            adjusted_confidence = base_confidence * self.confidence_multiplier
            
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response.model_dump(),
                provider='openai',
                model=self.model_name,
                processing_time=processing_time,
                token_count=response.usage.total_tokens if response.usage else None,
                confidence=adjusted_confidence,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing."""
        if not self.pricing:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * self.pricing.get('input_per_1m', 0.0)
        output_cost = (output_tokens / 1_000_000) * self.pricing.get('output_per_1m', 0.0)
        total_cost = input_cost + output_cost
        
        # Update daily cost tracking
        self.daily_cost += total_cost
        
        return total_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            'model_name': self.model_name,
            'pricing': self.pricing,
            'confidence_multiplier': self.confidence_multiplier,
            'complexity_threshold': self.complexity_threshold,
            'daily_cost': self.daily_cost,
            'cost_limits': self.cost_limits
        }
    
    def can_handle_complexity(self, complexity_score: float) -> bool:
        """Check if this model can handle the given complexity level."""
        return complexity_score <= self.complexity_threshold
    
    def is_cost_within_limits(self, estimated_cost: float) -> bool:
        """Check if the estimated cost is within configured limits."""
        max_cost = self.cost_limits.get('max_cost_per_extraction', float('inf'))
        max_daily = self.cost_limits.get('max_daily_cost', float('inf'))
        
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
            api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key')
            client = openai.OpenAI(api_key=api_key)
            
            # List models to test connection
            models = client.models.list()
            return True, f"Connected to OpenAI API with model {self.model_name}"
            
        except Exception as e:
            return False, f"Cannot connect to OpenAI API: {str(e)}"
    
    def parse_structured_output(self, content: str) -> Dict[str, Any]:
        """Parse structured output from model response."""
        try:
            # OpenAI responses are usually well-formatted JSON when requested
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback to extracting JSON from text
            try:
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0].strip()
                    return json.loads(json_str)
                elif '{' in content and '}' in content:
                    start = content.index('{')
                    end = content.rindex('}') + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
            except:
                pass
            
            return {'response': content}