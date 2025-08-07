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
    """OpenAI provider for API-based LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.model = config.get('model', 'gpt-4o-mini')
        self.max_tokens = config.get('max_tokens', 2000)
        self.temperature = config.get('temperature', 0.1)
        
        # Get API key from environment or config
        api_key = os.getenv('OPENAI_API_KEY') or config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment or config")
        
        self.client = AsyncOpenAI(api_key=api_key)
    
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                seed=42,  # For reproducibility (when supported)
                response_format={"type": "json_object"} if "json" in prompt.lower() else {"type": "text"}
            )
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response.model_dump(),
                provider='openai',
                model=self.model,
                processing_time=processing_time,
                token_count=response.usage.total_tokens if response.usage else None,
                confidence=0.95  # Higher confidence for GPT-4 models
            )
            
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")
    
    def validate_connection(self) -> Tuple[bool, str]:
        """Validate connection to OpenAI API."""
        try:
            # Try a simple synchronous test
            import openai
            api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key')
            client = openai.OpenAI(api_key=api_key)
            
            # List models to test connection
            models = client.models.list()
            return True, f"Connected to OpenAI API with model {self.model}"
            
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