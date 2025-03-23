"""
Llama.cpp API Package

A self-contained API for interacting with llama.cpp models.
"""

# Package initialization
from .main import app
from .models.model_manager import model_manager
from .models.types import ModelParameters, ModelStatus

__version__ = "0.1.0" 