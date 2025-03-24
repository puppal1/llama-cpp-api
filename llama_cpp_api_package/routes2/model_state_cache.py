import logging
import psutil
import time
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelStateCache:
    def __init__(self):
        self._loaded_models: Dict[str, Dict] = {}
        self._model_metrics: Dict[str, Dict] = {}
        self._memory_threshold = 0.9  # 90% memory usage threshold
        
    def update_model_state(self, model_id: str, state: str, info: Optional[Dict] = None) -> None:
        """Update model state in cache with memory metrics"""
        try:
            current_time = datetime.utcnow().isoformat()
            memory_info = self._get_memory_metrics()
            
            if state == "loaded":
                self._loaded_models[model_id] = {
                    "state": state,
                    "load_time": current_time,
                    "last_used": current_time,
                    "memory_used": memory_info["current_usage"],
                    "info": info or {}
                }
                logger.info(f"Model {model_id} state updated to loaded with {memory_info['current_usage']:.2f}GB memory")
            elif state == "unloaded":
                if model_id in self._loaded_models:
                    # Store metrics before removing
                    self._model_metrics[model_id] = {
                        "load_duration": (datetime.fromisoformat(current_time) - 
                                        datetime.fromisoformat(self._loaded_models[model_id]["load_time"])).total_seconds(),
                        "peak_memory": self._loaded_models[model_id].get("peak_memory", 0),
                        "last_used": self._loaded_models[model_id]["last_used"]
                    }
                    del self._loaded_models[model_id]
                    logger.info(f"Model {model_id} state updated to unloaded")
            
            # Check memory threshold
            if memory_info["usage_percent"] > self._memory_threshold:
                logger.warning(f"High memory usage detected: {memory_info['usage_percent']:.1f}%")
                
        except Exception as e:
            logger.error(f"Error updating model state cache: {e}")
            
    def get_model_state(self, model_id: str) -> Optional[Dict]:
        """Get current state of a model"""
        return self._loaded_models.get(model_id)
        
    def get_model_metrics(self, model_id: str) -> Optional[Dict]:
        """Get historical metrics for a model"""
        return self._model_metrics.get(model_id)
        
    def get_all_loaded_models(self) -> Dict[str, Dict]:
        """Get all currently loaded models"""
        return self._loaded_models.copy()
        
    def _get_memory_metrics(self) -> Dict:
        """Get current memory metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                "current_usage": memory_info.rss / (1024**3),  # GB
                "peak_usage": memory_info.vms / (1024**3),     # GB
                "usage_percent": system_memory.percent,
                "available": system_memory.available / (1024**3)  # GB
            }
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            return {
                "current_usage": 0,
                "peak_usage": 0,
                "usage_percent": 0,
                "available": 0
            }
            
    def update_model_usage(self, model_id: str) -> None:
        """Update last used timestamp for a model"""
        if model_id in self._loaded_models:
            self._loaded_models[model_id]["last_used"] = datetime.utcnow().isoformat()
            
    def get_memory_warning(self) -> Optional[str]:
        """Check if memory usage is high and return warning if needed"""
        memory_info = self._get_memory_metrics()
        if memory_info["usage_percent"] > self._memory_threshold:
            return f"High memory usage: {memory_info['usage_percent']:.1f}%"
        return None

# Global instance
model_state_cache = ModelStateCache() 