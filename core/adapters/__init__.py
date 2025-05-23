"""
Base adapter and common implementations
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AbstractAdapter(ABC):
    """Base class for all system adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def execute(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of inputs against the target system"""
        pass

class OpenAIAdapter(AbstractAdapter):
    """Adapter for OpenAI chat models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self._get_api_key()
            )
        except ImportError:
            raise ImportError("openai package required for OpenAIAdapter")
            
    def _get_api_key(self) -> str:
        """Get API key from config or environment"""
        if "api_key" in self.config:
            return self.config["api_key"]
        elif "api_key_env_var" in self.config:
            import os
            return os.getenv(self.config["api_key_env_var"])
        raise ValueError("No API key configured for OpenAIAdapter")
        
    def execute(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute inputs against OpenAI API"""
        model = self.config.get("model_name", "gpt-3.5-turbo")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=inputs,
                temperature=0.7
            )
            return [{
                "content": choice.message.content,
                "role": choice.message.role,
                "finish_reason": choice.finish_reason
            } for choice in response.choices]
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise 