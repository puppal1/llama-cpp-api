import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import os
import glob
from datetime import datetime
import psutil
import time

from ..utils.system_info import get_system_info
from ..models.model_manager import model_manager
from ..api.api_types import ModelStatus
from ..models.model_types import ModelParameters
from ..utils.model_metadata import ModelMetadataReader, metadata_cache

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model_size(path: str) -> float:
    """Get model file size in MB"""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting model size: {e}")
        return 0.0

def get_model_info(model_id: str, path: str) -> Dict:
    """Get detailed information about a model"""
    try:
        # Get file size
        size = get_model_size(path)
        
        # Get system info for memory calculations
        sys_info = get_system_info()
        total_memory = sys_info.memory["total"]  # Already in MB
        
        # Calculate required memory (2x model size as conservative estimate)
        required_memory = size * 2
        
        # Check if model can be loaded
        can_load = required_memory < total_memory
        
        # Get model status
        status = model_manager.get_model_status(model_id)
        
        # Get detailed metadata using ModelMetadataReader
        metadata = ModelMetadataReader.read_metadata(path)
        
        # Get loaded model info if available
        loaded_info = None
        if model_id in model_manager.loaded_models:
            loaded_info = model_manager.loaded_models[model_id]
        
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
                "layer_norm_rms_epsilon": 1e-5  # Default value as it's not in metadata
            }
        }
        
        # Calculate performance characteristics
        performance = {
            "quantization": "Q4_K_M" if "Q4_K_M" in model_id else "Q8_0" if "Q8_0" in model_id else "unknown",
            "estimated_inference_speed": "tokens_per_second",  # This would need to be calculated based on model size/architecture
            "recommended_batch_size": 256,
            "recommended_thread_count": 4,
            "context_utilization": 0.0,
            "average_inference_time": 0.0,
            "tokens_per_second": 0.0
        }
        
        # If model is loaded, update performance metrics
        if loaded_info:
            performance.update({
                "context_utilization": loaded_info.get("context_utilization", 0.0),
                "average_inference_time": loaded_info.get("average_inference_time", 0.0),
                "tokens_per_second": loaded_info.get("tokens_per_second", 0.0)
            })
        
        return {
            "id": model_id,
            "name": os.path.basename(path),
            "path": path,
            "size_mb": size,
            "required_memory_mb": required_memory,
            "can_load": can_load,
            "architecture": architecture,
            "performance": performance,
            "status": status.value if loaded_info else None,
            "loaded_info": loaded_info
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {
            "id": model_id,
            "name": os.path.basename(path),
            "path": path,
            "size_mb": get_model_size(path),
            "required_memory_mb": 0,
            "can_load": False,
            "error": str(e)
        }

@router.get("/api/models")
async def list_models() -> Dict:
    """List all available models and system information"""
    # Get fresh system info
    sys_info = get_system_info()
    
    # Get all model files
    models_dir = os.getenv("MODELS_DIR", "models")
    model_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    
    # Get info for each model
    available_models = []
    loaded_models = {}
    
    for path in model_files:
        model_id = os.path.splitext(os.path.basename(path))[0]
        
        # Get file size
        size = get_model_size(path)
        
        # Calculate required memory (2x model size as conservative estimate)
        required_memory = size * 2
        
        # Check if model can be loaded
        total_memory_mb = sys_info.memory["total"]
        can_load = required_memory < total_memory_mb
        
        # Get model status
        status = model_manager.get_model_status(model_id)
        
        # Get detailed metadata using ModelMetadataReader
        metadata = ModelMetadataReader.read_metadata(path)
        
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
                "layer_norm_rms_epsilon": 1e-5  # Default value as it's not in metadata
            }
        }
        
        # Calculate performance characteristics
        performance = {
            "quantization": "Q4_K_M" if "Q4_K_M" in model_id else "Q8_0" if "Q8_0" in model_id else "unknown",
            "estimated_inference_speed": "tokens_per_second",  # This would need to be calculated based on model size/architecture
            "recommended_batch_size": 256,
            "recommended_thread_count": 4,
            "context_utilization": 0.0,
            "average_inference_time": 0.0,
            "tokens_per_second": 0.0
        }
        
        # Add to available models with enhanced structure
        available_models.append({
            "id": model_id,
            "name": os.path.basename(path),
            "path": path,
            "size_mb": size,
            "required_memory_mb": required_memory,
            "can_load": can_load,
            "architecture": architecture,
            "performance": performance
        })
        
        # Add to loaded models if loaded
        if model_id in model_manager.loaded_models:
            loaded_info = model_manager.loaded_models[model_id]
            loaded_models[model_id] = {
                "status": status.value,
                "load_time": loaded_info.get("load_time"),
                "last_used": loaded_info.get("last_used"),
                "configuration": loaded_info.get("parameters", {}),
                "resources": {
                    "memory_used_mb": loaded_info.get("memory_used_mb", 0),
                    "cpu_utilization": sys_info.cpu["utilization"],
                    "gpu_utilization": sys_info.gpu.get("utilization") if sys_info.gpu else None
                },
                "performance": {
                    "average_inference_time": loaded_info.get("average_inference_time", 0),
                    "tokens_per_second": loaded_info.get("tokens_per_second", 0),
                    "context_utilization": loaded_info.get("context_utilization", 0)
                },
                "health": {
                    "is_healthy": status.value == "loaded",
                    "last_error": loaded_info.get("error"),
                    "uptime": str(datetime.now() - datetime.fromisoformat(loaded_info.get("load_time", datetime.now().isoformat())))
                }
            }
    
    # Calculate total model memory
    total_model_memory = sum(model.get("loaded_info", {}).get("memory_used_mb", 0) for model in available_models)
    
    response = {
        "models": {
            "available": available_models,
            "loaded": loaded_models
        },
        "system_state": {
            "cpu": sys_info.cpu,
            "memory": {
                "total_gb": sys_info.memory["total"] / 1024,  # Convert MB to GB
                "used_gb": sys_info.memory["used"] / 1024,  # Convert MB to GB
                "model_memory_gb": total_model_memory / 1024  # Convert MB to GB
            },
            "gpu": {
                "available": sys_info.gpu is not None,
                "status": sys_info.gpu.get("status", "not available") if sys_info.gpu else "not available",
                "name": sys_info.gpu.get("name", "Unknown") if sys_info.gpu else "Unknown",
                "memory": sys_info.gpu.get("memory") if sys_info.gpu else None
            }
        }
    }
    
    logger.info(f"Sending response: {response}")
    return response

@router.post("/api/models/{model_id}/load")
async def load_model(model_id: str, parameters: ModelParameters) -> Dict:
    """Load a model into memory"""
    try:
        # Get model path first
        models_dir = os.getenv("MODELS_DIR", "models")
        model_path = os.path.join(models_dir, f"{model_id}.gguf")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}"
            )
        
        # Get model info with path
        model_info = get_model_info(model_id, model_path)
        if not model_info["can_load"]:
            raise HTTPException(
                status_code=400,
                detail="Insufficient memory to load model"
            )
            
        # Load the model - this returns the model info dict
        loaded_info = model_manager.load_model(model_id, parameters)
        logger.info(f"Model {model_id} loaded successfully with info: {loaded_info}")
        
        return loaded_info
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model"""
    try:
        model_manager.unload_model(model_id)
        return {"status": "success", "message": f"Model {model_id} unloaded"}
    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 