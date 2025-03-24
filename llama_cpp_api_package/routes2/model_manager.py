import logging
from typing import Dict, Optional
from datetime import datetime
from llama_cpp import Llama
from fastapi import HTTPException

from ..api.api_types import ModelStatus

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loaded models and their states"""
    def __init__(self):
        self.loaded_models: Dict[str, Dict] = {}
        self.model_configs: Dict[str, Dict] = {}
        
    def get_model_config(self, model_name: str, metadata: Dict) -> Dict:
        """Get model-specific configuration based on model name and metadata."""
        config = {
            "n_ctx": 2048,
            "n_batch": 512,
            "n_threads": 4,
            "n_gpu_layers": 0,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": True
        }
        
        # Get RoPE dimensions from metadata if available
        rope_dims = metadata.get("llama.rope.dimension_count", 0)
        
        # Model-specific configurations
        if "ayla" in model_name.lower():
            config.update({
                "rope_dimension_count": rope_dims or 128,
                "rope_freq_base": 1000000.0,
                "n_ctx": metadata.get("context_length", 100000),
                "tensor_split": None
            })
        elif "deepseek" in model_name.lower():
            config.update({
                "rope_freq_base": 10000.0,
                "rope_scaling_type": "linear",
                "n_ctx": 4096
            })
        elif "moe" in model_name.lower():
            config.update({
                "n_threads": 4,
                "n_batch": 256,
                "n_gpu_layers": 0,
                "tensor_split": None
            })
        elif "mistral" in model_name.lower():
            config.update({
                "n_ctx": 4096,
                "n_batch": 512
            })
        
        return config
        
    def load_model(self, model_id: str, model_path: str, metadata: Dict) -> Dict:
        """Load a model with appropriate configuration."""
        if model_id in self.loaded_models:
            raise ValueError(f"Model {model_id} is already loaded")
            
        try:
            # Get model configuration
            config = self.get_model_config(model_id, metadata)
            self.model_configs[model_id] = config
            
            # Initialize model
            llm = Llama(model_path=model_path, **config)
            
            # Store model info
            self.loaded_models[model_id] = {
                "model": llm,
                "path": model_path,
                "metadata": metadata,
                "config": config,
                "status": ModelStatus.LOADED,
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully loaded model: {model_id}")
            return {
                "id": model_id,
                "status": ModelStatus.LOADED,
                "config": config,
                "loaded_at": self.loaded_models[model_id]["loaded_at"]
            }
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def unload_model(self, model_id: str) -> bool:
        """Unload a model and free its resources."""
        if model_id not in self.loaded_models:
            return False
            
        try:
            # Clean up model resources
            del self.loaded_models[model_id]
            if model_id in self.model_configs:
                del self.model_configs[model_id]
            logger.info(f"Successfully unloaded model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
            
    def get_loaded_model(self, model_id: str) -> Optional[Dict]:
        """Get information about a loaded model."""
        return self.loaded_models.get(model_id)
        
    def get_all_loaded_models(self) -> Dict[str, Dict]:
        """Get information about all loaded models."""
        return {
            model_id: {
                "id": model_id,
                "path": info["path"],
                "status": info["status"],
                "loaded_at": info["loaded_at"]
            }
            for model_id, info in self.loaded_models.items()
        } 