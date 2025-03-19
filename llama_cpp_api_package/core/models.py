"""
Core model implementation for the llama.cpp API.
"""

from typing import List, Optional, Dict, Any
from llama_cpp import Llama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.params = {}
        
    def load(self, 
             model_path: str,
             n_ctx: int = 2048,
             n_batch: int = 512,
             n_threads: int = 4,
             n_gpu_layers: int = 0,
             use_mlock: bool = False,
             use_mmap: bool = True,
             seed: int = -1,
             verbose: bool = True) -> None:
        """
        Load a model from the given path with specified parameters.
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mlock=use_mlock,
                use_mmap=use_mmap,
                seed=seed,
                verbose=verbose
            )
            self.model_path = model_path
            self.params = {
                'n_ctx': n_ctx,
                'n_batch': n_batch,
                'n_threads': n_threads,
                'n_gpu_layers': n_gpu_layers,
                'use_mlock': use_mlock,
                'use_mmap': use_mmap,
                'seed': seed
            }
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def unload(self) -> None:
        """
        Unload the current model and free resources.
        """
        if self.model:
            self.model = None
            self.model_path = None
            self.params = {}
            logger.info("Model unloaded")

    def generate(self,
                prompt: str,
                max_tokens: int = 100,
                temperature: float = 0.7,
                top_k: int = 40,
                top_p: float = 0.95,
                repeat_penalty: float = 1.1,
                presence_penalty: float = 0.0,
                frequency_penalty: float = 0.0,
                stop: Optional[List[str]] = None) -> str:
        """
        Generate text based on the prompt.
        """
        if not self.model:
            raise RuntimeError("No model loaded. Call load() first.")

        try:
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop or ["</s>"]
            )
            return response["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

    def chat(self,
            messages: List[Dict[str, str]],
            max_tokens: int = 100,
            temperature: float = 0.7,
            top_k: int = 40,
            top_p: float = 0.95,
            repeat_penalty: float = 1.1,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            stop: Optional[List[str]] = None) -> str:
        """
        Generate a chat response based on a list of messages.
        """
        if not self.model:
            raise RuntimeError("No model loaded. Call load() first.")

        # Format messages into a prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}</s>\n"

        # Add the final assistant prompt
        prompt += "<|assistant|>\n"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop or ["</s>", "<|user|>"]
        ) 