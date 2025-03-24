"""
Core model classes for llama.cpp integration.
"""
from typing import Optional, Dict, Any
from llama_cpp import Llama

class LlamaModel:
    """Wrapper class for llama.cpp model."""
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize the model with given parameters."""
        self.model_path = model_path
        self.model_params = kwargs
        self.model = None
        
    def load(self) -> None:
        """Load the model with specified parameters."""
        self.model = Llama(
            model_path=self.model_path,
            **self.model_params
        )
        
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Generate text from the model."""
        if not self.model:
            self.load()
            
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            **kwargs
        )
        return output["choices"][0]["text"] if output else "" 