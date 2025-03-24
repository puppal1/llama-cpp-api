"""API-specific type definitions."""
from enum import Enum

class ModelStatus(str, Enum):
    """Model loading and operational status."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error" 