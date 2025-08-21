"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class LLMResponse:
    """Standard response from LLM providers."""
    content: str
    raw_response: Any
    provider: str
    model: str
    processing_time: float
    token_count: Optional[int] = None
    confidence: float = 1.0
    cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> Tuple[bool, str]:
        """Validate connection to provider."""
        pass
    
    def measure_time(self, func):
        """Decorator to measure processing time."""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return result, elapsed
        return wrapper