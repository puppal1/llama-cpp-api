"""
Master configuration file for model loading and memory calculations.
Contains all constants and rules used across the application.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class SystemLimits:
    """System-wide limits for model loading"""
    max_context_size: int = 4096    # Maximum context window size
    min_context_size: int = 512     # Minimum context window size
    max_cpu_threads: int = 16       # Maximum CPU threads
    memory_safety_margin: float = 0.2  # 20% safety margin for memory calculations

@dataclass
class MemoryMultipliers:
    """Memory calculation multipliers for different scenarios"""
    base_model_multiplier: float = 1.5    # Base multiplier for model size
    gpu_memory_multiplier: float = 1.8    # GPU memory multiplier
    max_memory_multiplier: float = 2.0    # Maximum memory multiplier for worst case

@dataclass
class ModelDefaults:
    """Default values for model parameters"""
    default_context_size: int = 2048      # Default context size if not specified
    default_batch_size: int = 512         # Default batch size
    default_threads: int = 4              # Default number of threads
    default_gpu_layers: int = 0           # Default number of GPU layers
    
    # RoPE parameters
    rope_freq_base: float = 10000.0       # Default RoPE frequency base
    rope_freq_scale: float = 1.0          # Default RoPE frequency scaling
    rope_scaling_type: str = "linear"     # RoPE scaling type (none, linear, yarn)
    rope_ext_factor: float = 0.0          # RoPE context extension factor
    rope_attn_factor: float = 1.0         # RoPE attention scaling factor
    rope_beta_fast: float = 32.0          # YaRN beta fast parameter
    rope_beta_slow: float = 1.0           # YaRN beta slow parameter

@dataclass
class BufferSizes:
    """Known buffer sizes from empirical measurements"""
    compute_buffer_mb: float = 164.01     # Compute buffer size in MB
    output_buffer_mb: float = 0.12        # Output buffer size in MB
    bytes_per_token: int = 2              # Bytes per token (float16)

class ModelArchitectures:
    """Known model architectures and their parameters"""
    KNOWN_ARCHITECTURES = {
        "llama": {
            "default_n_layers": 32,
            "default_n_heads": 32,
            "default_n_embd": 4096,
            "default_n_vocab": 32000,
        },
        "mistral": {
            "default_n_layers": 32,
            "default_n_heads": 32,
            "default_n_embd": 4096,
            "default_n_vocab": 32000,
        }
    }

    @classmethod
    def get_defaults(cls, model_type: str) -> Dict:
        """Get default parameters for a model architecture"""
        return cls.KNOWN_ARCHITECTURES.get(model_type.lower(), cls.KNOWN_ARCHITECTURES["llama"])

class MemoryCalculator:
    """Memory calculation rules and formulas"""
    @staticmethod
    def calculate_kv_cache_mb(
        context_size: int,
        n_layers: int,
        bytes_per_token: int = BufferSizes.bytes_per_token
    ) -> float:
        """Calculate KV cache size in MB"""
        kv_cache_bytes = 2 * context_size * bytes_per_token * n_layers
        return kv_cache_bytes / (1024 * 1024)

    @staticmethod
    def calculate_total_memory_mb(
        model_size_mb: float,
        context_size: int,
        n_layers: int,
        safety_margin: float = SystemLimits.memory_safety_margin
    ) -> Dict[str, float]:
        """Calculate total memory requirements with breakdown"""
        # Base model memory
        base_memory = model_size_mb * MemoryMultipliers.base_model_multiplier
        
        # KV cache
        kv_cache = MemoryCalculator.calculate_kv_cache_mb(context_size, n_layers)
        
        # Fixed buffers
        compute_buffer = BufferSizes.compute_buffer_mb
        output_buffer = BufferSizes.output_buffer_mb
        
        # Subtotal
        subtotal = base_memory + kv_cache + compute_buffer + output_buffer
        
        # Safety margin
        safety_buffer = subtotal * safety_margin
        total = subtotal + safety_buffer
        
        return {
            "total_mb": total,
            "breakdown": {
                "base_model_mb": base_memory,
                "kv_cache_mb": kv_cache,
                "compute_buffer_mb": compute_buffer,
                "output_buffer_mb": output_buffer,
                "safety_buffer_mb": safety_buffer
            }
        }

    @staticmethod
    def calculate_gpu_memory_mb(
        model_size_mb: float,
        n_gpu_layers: int,
        total_layers: int
    ) -> float:
        """Calculate GPU memory requirements"""
        if n_gpu_layers <= 0:
            return 0.0
        layer_size = model_size_mb / total_layers
        gpu_memory = layer_size * n_gpu_layers * MemoryMultipliers.gpu_memory_multiplier
        return gpu_memory 