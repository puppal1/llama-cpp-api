"""
Llama.cpp API Package

A self-contained API for interacting with llama.cpp models.
"""

# Package initialization
from .models.model_manager import ModelManager
from .api.api_types import ModelStatus
from .models.model_types import ModelParameters

__version__ = "0.1.0"

__all__ = ["ModelManager", "ModelStatus", "ModelParameters"] 