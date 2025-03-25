import psutil
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelMemoryProfile:
    base_memory_gb: float
    context_scale_factor: float = 1.0  # How much memory scales with context
    max_batch_size: int = 512
    
class MemoryManager:
    # Memory safety multiplier for development mode
    MEMORY_MULTIPLIER = 1.2  # Changed from 1.5 to 1.2 for dev settings
    
    # Maximum allowed context length
    MAX_CONTEXT_LENGTH = 4096
    
    # Model memory profiles (base memory in GB)
    MODEL_PROFILES = {
        "moe": ModelMemoryProfile(base_memory_gb=12.0, context_scale_factor=1.2),
        "ayla": ModelMemoryProfile(base_memory_gb=9.6, context_scale_factor=1.1),
        "wizardlm": ModelMemoryProfile(base_memory_gb=10.3, context_scale_factor=1.15),
        "deepseek": ModelMemoryProfile(base_memory_gb=0.5, context_scale_factor=1.0)
    }
    
    def __init__(self):
        self.model_memory_usage: Dict[str, float] = {}  # Track actual memory usage per model
        self.last_memory_check = datetime.now()
        self.memory_check_interval = 5  # seconds
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get current system memory status"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
    
    def calculate_required_memory(self, model_id: str, context_length: int) -> Tuple[float, bool]:
        """
        Calculate required memory for model with given context length
        Returns: (required_memory_gb, is_safe)
        """
        # Get model profile or use default
        model_id = model_id.lower()
        profile = self.MODEL_PROFILES.get(model_id, ModelMemoryProfile(base_memory_gb=10.0))
        
        # Calculate memory requirement
        # Base memory + Context scaling
        # For every 1024 tokens above base (2048), add 25% more memory
        base_context = 2048
        context_scaling = max(0, (context_length - base_context) / 1024) * 0.25
        
        memory_required = (profile.base_memory_gb * 
                          (1 + context_scaling) * 
                          profile.context_scale_factor * 
                          self.MEMORY_MULTIPLIER)
        
        # Check if memory requirement is safe
        system_memory = self.get_system_memory()
        is_safe = (memory_required < system_memory["available_gb"] and 
                  context_length <= self.MAX_CONTEXT_LENGTH)
        
        return memory_required, is_safe
    
    def can_load_model(self, model_id: str, context_length: int) -> Tuple[bool, str]:
        """Check if model can be safely loaded"""
        memory_required, is_safe = self.calculate_required_memory(model_id, context_length)
        system_memory = self.get_system_memory()
        
        if context_length > self.MAX_CONTEXT_LENGTH:
            return False, f"Context length {context_length} exceeds maximum allowed {self.MAX_CONTEXT_LENGTH}"
            
        if memory_required > system_memory["available_gb"]:
            return False, f"Insufficient memory. Required: {memory_required:.2f}GB, Available: {system_memory['available_gb']:.2f}GB"
            
        return True, "Model can be safely loaded"
    
    def register_model_load(self, model_id: str, memory_gb: float):
        """Register that a model has been loaded and its memory usage"""
        self.model_memory_usage[model_id] = memory_gb
    
    def unregister_model(self, model_id: str):
        """Unregister a model when it's unloaded"""
        self.model_memory_usage.pop(model_id, None)
    
    def get_memory_metrics(self) -> Dict:
        """Get current memory metrics for monitoring"""
        system_memory = self.get_system_memory()
        return {
            "system": system_memory,
            "models": self.model_memory_usage,
            "total_model_memory_gb": sum(self.model_memory_usage.values()),
            "available_for_models_gb": system_memory["available_gb"]
        }

# Create singleton instance
memory_manager = MemoryManager() 