"""
Load rules and validation for model loading.
Uses centralized configuration from model_config.py.
"""

from typing import Dict, List, Optional, Tuple
import psutil
import math
from llama_cpp_api_package.config.model_config import (
    SystemLimits,
    MemoryMultipliers,
    ModelDefaults,
    BufferSizes,
    MemoryCalculator as ConfigMemoryCalculator
)

class MemoryCalculator:
    """Memory calculation and validation"""
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """Get system memory information in MB"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024 * 1024),
            "available": memory.available / (1024 * 1024),
            "used": memory.used / (1024 * 1024)
        }

    @staticmethod
    def calculate_total_memory_required(
        model_size_mb: float,
        context_size: int,
        n_layers: int,
        safety_margin: float = SystemLimits.memory_safety_margin
    ) -> Dict[str, float]:
        """Calculate total memory requirements using configuration"""
        return ConfigMemoryCalculator.calculate_total_memory_mb(
            model_size_mb=model_size_mb,
            context_size=context_size,
            n_layers=n_layers,
            safety_margin=safety_margin
        )

class LoadRules:
    """Rules for model loading and parameter validation"""
    @staticmethod
    def get_valid_context_sizes(
        min_size: int = SystemLimits.min_context_size,
        max_size: int = 32768
    ) -> List[int]:
        """Get list of valid context sizes (powers of 2)"""
        sizes = []
        size = min_size
        while size <= max_size:
            sizes.append(size)
            size *= 2
        return sizes

    @staticmethod
    def get_recommended_thread_count() -> int:
        """Get recommended thread count based on system CPU"""
        cpu_count = psutil.cpu_count(logical=True) or ModelDefaults.default_threads
        # Use power of 2 up to system limit
        return min(
            SystemLimits.max_cpu_threads,
            2 ** math.floor(math.log2(cpu_count))
        )

    @staticmethod
    def validate_context_size(
        requested_size: int,
        model_max_context: int,
        system_max_context: int = SystemLimits.max_context_size
    ) -> Tuple[int, Optional[str]]:
        """Validate and adjust context size"""
        valid_sizes = LoadRules.get_valid_context_sizes()
        
        # Find nearest valid size
        nearest_size = min(valid_sizes, key=lambda x: abs(x - requested_size))
        
        # Apply limits
        final_size = min(nearest_size, model_max_context, system_max_context)
        
        # Generate warning if adjusted
        warning = None
        if final_size != requested_size:
            warning = f"Context size adjusted from {requested_size} to {final_size} to match valid size and limits"
        
        return final_size, warning

    @staticmethod
    def can_load_model(
        model_size_mb: float,
        context_size: int,
        n_layers: int,
        available_memory_mb: float,
        safety_margin: float = SystemLimits.memory_safety_margin
    ) -> Tuple[bool, str]:
        """Check if model can be loaded with given parameters"""
        requirements = MemoryCalculator.calculate_total_memory_required(
            model_size_mb=model_size_mb,
            context_size=context_size,
            n_layers=n_layers,
            safety_margin=safety_margin
        )
        
        if requirements["total_mb"] > available_memory_mb:
            return (False, (
                f"Insufficient memory. Required: {requirements['total_mb']:.1f}MB, "
                f"Available: {available_memory_mb:.1f}MB"
            ))
        
        return (True, "Model can be loaded with specified parameters") 