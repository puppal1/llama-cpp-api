import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import os
import glob
from datetime import datetime
import psutil
import time
from io import StringIO
from contextlib import redirect_stderr
import re
from llama_cpp import Llama
from functools import lru_cache

from ..utils.system_info import get_system_info
from ..api.api_types import ModelStatus
from ..models.model_types import ModelParameters

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for model metadata and file sizes
_model_metadata_cache = {}
_model_size_cache = {}
_known_model_files = set()  # Track known model files
_cache_initialized = False  # Track if cache has been initialized

def _get_model_files(models_dir: str) -> set:
    """Get set of model files in directory"""
    return set(glob.glob(os.path.join(models_dir, "*.gguf")))

def _initialize_model_cache(models_dir: str):
    """Initialize model cache at server startup"""
    global _cache_initialized, _known_model_files
    
    if _cache_initialized:
        return
        
    logger.info("Initializing model metadata cache")
    current_files = _get_model_files(models_dir)
    
    for model_path in current_files:
        try:
            model_id = os.path.splitext(os.path.basename(model_path))[0]
            
            # Cache file size
            _model_size_cache[model_id] = os.path.getsize(model_path) / (1024 * 1024)
            
            # Cache metadata
            _model_metadata_cache[model_id] = read_model_metadata(model_path)
            logger.info(f"Cached metadata for model: {model_id}")
                
        except Exception as e:
            logger.error(f"Error caching metadata for {model_path}: {str(e)}")
            continue
    
    _known_model_files = current_files
    _cache_initialized = True
    logger.info(f"Cache initialized with {len(_model_metadata_cache)} models")

def _check_for_new_models(models_dir: str):
    """Check for new models and update cache if needed"""
    global _known_model_files
    
    if not _cache_initialized:
        _initialize_model_cache(models_dir)
        return
        
    current_files = _get_model_files(models_dir)
    new_files = current_files - _known_model_files
    
    if not new_files:
        return  # No new files, no need to update cache
        
    logger.info(f"Found {len(new_files)} new model files, updating cache")
    
    for model_path in new_files:
        try:
            model_id = os.path.splitext(os.path.basename(model_path))[0]
            
            # Update file size cache
            _model_size_cache[model_id] = os.path.getsize(model_path) / (1024 * 1024)
            
            # Update metadata cache for new file
            _model_metadata_cache[model_id] = read_model_metadata(model_path)
            logger.info(f"Added metadata for new model: {model_id}")
                
        except Exception as e:
            logger.error(f"Error updating cache for {model_path}: {str(e)}")
            continue
    
    # Update known files set
    _known_model_files = current_files
    logger.info(f"Cache updated with {len(_model_metadata_cache)} total models")

class ModelMetrics:
    """Class to hold model metrics"""
    def __init__(self):
        self.initial_memory = 0
        self.load_memory = 0
        self.inference_memory = 0
        self.final_memory = 0
        self.load_time = 0
        self.inference_time = 0
        self.metadata = {}

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def read_model_metadata(model_path: str) -> Dict:
    """Read model metadata without full initialization"""
    try:
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            # Initialize with minimal context to quickly read metadata
            temp_model = Llama(
                model_path=model_path,
                n_ctx=8,
                n_threads=1,
                n_batch=1,
                verbose=True
            )
            del temp_model
        
        # Parse metadata from output
        metadata = {}
        model_info = stderr_buffer.getvalue()
        
        # Extract parameters
        param_patterns = {
            'architecture': r'arch\s*=\s*(\w+)',
            'num_layers': r'n_layer\s*=\s*(\d+)',
            'num_heads': r'n_head\s*=\s*(\d+)',
            'num_heads_kv': r'n_head_kv\s*=\s*(\d+)',
            'embedding_length': r'n_embd\s*=\s*(\d+)',
            'vocab_size': r'n_vocab\s*=\s*(\d+)',
            'rope_dimension_count': r'n_rot\s*=\s*(\d+)',
            'rope_freq_base': r'freq_base_train\s*=\s*(\d+\.\d+)',
            'rope_scaling': r'rope scaling\s*=\s*(\w+)',
            'context_length': r'n_ctx_train\s*=\s*(\d+)',
            'model_type': r'model type\s*=\s*(.+)',
            'model_name': r'general\.name\s*=\s*(.+)'
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, model_info)
            if match:
                val = match.group(1)
                # Convert to appropriate type
                if val.isdigit():
                    metadata[param] = int(val)
                elif '.' in val and val.replace('.', '').isdigit():
                    metadata[param] = float(val)
                else:
                    metadata[param] = val
        
        # Add architecture if not found but can be inferred
        if 'architecture' not in metadata:
            if 'model_name' in metadata:
                if 'mistral' in metadata['model_name'].lower():
                    metadata['architecture'] = 'llama'
                elif 'qwen' in metadata['model_name'].lower():
                    metadata['architecture'] = 'qwen2'
                elif 'ayla' in metadata['model_name'].lower():
                    metadata['architecture'] = 'llama'
                    # Ayla models have a much larger context length
                    metadata['context_length'] = 100000
        
        return metadata
    except Exception as e:
        logger.error(f"Error reading metadata for {model_path}: {str(e)}")
        return {}

def get_model_size(path: str) -> float:
    """Get model file size in MB"""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting model size: {e}")
        return 0.0

@router.get("/api/v2/models")
async def list_models():
    """List all available models with their metadata"""
    try:
        # Get system info
        sys_info = get_system_info()
        total_memory = sys_info.memory["total"]  # Already in MB
        
        # Check for new models
        models_dir = os.getenv("MODELS_DIR", "models")
        _check_for_new_models(models_dir)
        
        # Get list of model files
        model_files = glob.glob(os.path.join(models_dir, "*.gguf"))
        
        available_models = []
        for model_path in model_files:
            try:
                model_id = os.path.splitext(os.path.basename(model_path))[0]
                
                # Get cached file size
                size = _model_size_cache.get(model_id, 0.0)
                
                # Calculate required memory (2x model size as conservative estimate)
                required_memory = size * 2
                
                # Check if model can be loaded
                can_load = required_memory < total_memory
                
                # Get cached metadata
                metadata = _model_metadata_cache.get(model_id, {})
                
                # Extract architecture details from metadata
                architecture = {
                    "type": metadata.get("architecture", "unknown"),
                    "model_name": metadata.get("model_name", model_id),
                    "context_length": metadata.get("context_length", 2048),
                    "embedding_length": metadata.get("embedding_length", 4096),
                    "block_count": metadata.get("num_layers", 32),
                    "rope": {
                        "dimension_count": metadata.get("rope_dimension_count", 128),
                        "freq_base": metadata.get("rope_freq_base", 10000.0),
                        "freq_scale": metadata.get("rope_scaling", 1.0)
                    },
                    "attention": {
                        "head_count": metadata.get("num_heads", 32),
                        "head_count_kv": metadata.get("num_heads_kv", 32),
                        "layer_norm_rms_epsilon": 1e-5  # Default value
                    }
                }
                
                # Calculate performance characteristics
                performance = {
                    "quantization": "Q4_K_M" if "Q4_K_M" in model_id else "Q8_0" if "Q8_0" in model_id else "unknown",
                    "estimated_inference_speed": "tokens_per_second",
                    "recommended_batch_size": 256,
                    "recommended_thread_count": 4,
                    "context_utilization": 0.0,
                    "average_inference_time": 0.0,
                    "tokens_per_second": 0.0
                }
                
                model_info = {
                    "id": model_id,
                    "name": os.path.basename(model_path),
                    "path": model_path,
                    "size_mb": size,
                    "required_memory_mb": required_memory,
                    "can_load": can_load,
                    "architecture": architecture,
                    "performance": performance
                }
                
                available_models.append(model_info)
                
            except Exception as e:
                logger.error(f"Error processing model {model_path}: {str(e)}")
                continue
        
        response = {
            "models": {
                "available": available_models
            },
            "system_state": {
                "cpu": sys_info.cpu,
                "memory": {
                    "total_gb": sys_info.memory["total"] / 1024,  # Convert MB to GB
                    "used_gb": sys_info.memory["used"] / 1024,  # Convert MB to GB
                    "model_memory_gb": 0.0  # Placeholder for loaded model memory
                },
                "gpu": {
                    "available": sys_info.gpu is not None,
                    "status": sys_info.gpu.get("status", "not available") if sys_info.gpu else "not available",
                    "name": sys_info.gpu.get("name", "Unknown") if sys_info.gpu else "Unknown",
                    "memory": sys_info.gpu.get("memory") if sys_info.gpu else None
                }
            }
        }
        
        logger.info(f"Sending response with {len(available_models)} available models")
        return response
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Placeholder endpoints
@router.get("/api/v2/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    raise HTTPException(status_code=501, detail="Not implemented yet")

@router.post("/api/v2/models/{model_id}/load")
async def load_model(model_id: str, parameters: ModelParameters):
    """Load a model into memory"""
    raise HTTPException(status_code=501, detail="Not implemented yet")

@router.post("/api/v2/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    raise HTTPException(status_code=501, detail="Not implemented yet") 