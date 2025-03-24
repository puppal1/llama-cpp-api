import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import os
import glob
from datetime import datetime

from ..utils.system_info import get_system_info
from ..models.model_manager import model_manager
from ..api.api_types import ModelStatus
from ..models.model_types import ModelParameters
from ..utils.model_metadata import metadata_cache

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model_size(path: str) -> float:
    """Get model file size in MB"""
    try:
        return os.path.getsize(path) / (1024 * 1024)  # Convert to MB
    except:
        return 0

def get_model_info(model_id: str, path: str) -> Dict:
    """Get information about a model"""
    size = get_model_size(path)
    
    # Get system info lazily
    sys_info = get_system_info()
    
    # Calculate memory requirements (rough estimate)
    required_memory = size * 2  # Rough estimate: model size * 2 for loading
    
    # Check if we can load the model
    available_memory = sys_info.memory.get("available", 0)
    if available_memory is not None:
        can_load = available_memory > required_memory
    else:
        can_load = True  # If we can't determine memory, assume we can load
        
    # Get current status if model is loaded
    status = model_manager.get_model_status(model_id)
    loaded_info = model_manager.get_model_info(model_id) if status == ModelStatus.LOADED else None
    
    # Get model metadata with force_reload=True to ensure fresh data
    metadata = metadata_cache.get_metadata(path, force_reload=True)
        
    return {
        "id": model_id,
        "name": os.path.basename(path),
        "path": path,
        "size_mb": size,
        "required_memory_mb": required_memory,
        "can_load": can_load,
        "status": status.value,
        "loaded_info": loaded_info,
        "metadata": metadata
    }

@router.get("/api/models")
async def list_models() -> Dict:
    """List all available models and system information"""
    # Clear metadata cache to force refresh
    metadata_cache.clear()
    
    # Get fresh system info
    sys_info = get_system_info()
    
    # Get all model files
    models_dir = os.getenv("MODELS_DIR", "models")
    model_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    
    # Get info for each model
    available_models = []
    loaded_models = {}
    total_model_memory = 0  # Track total memory used by loaded models
    
    for path in model_files:
        model_id = os.path.splitext(os.path.basename(path))[0]
        model_info = get_model_info(model_id, path)
        
        # Convert to UI format - only include fields UI expects
        available_model = {
            "id": model_info["id"],
            "name": model_info["name"],
            "path": model_info["path"],
            "size_mb": model_info["size_mb"],
            "required_memory_mb": model_info["required_memory_mb"],
            "can_load": model_info["can_load"],
            "metadata": model_info.get("metadata", {})
        }
        
        available_models.append(available_model)
        logger.info(f"Added available model: {available_model}")
        
        if model_info["status"] == "loaded" and model_info["loaded_info"]:
            loaded_info = model_info["loaded_info"]
            memory_used = loaded_info.get("memory_used", 0)
            total_model_memory += memory_used
            loaded_models[model_id] = {
                "status": model_info["status"],
                "load_time": loaded_info.get("load_time"),
                "last_used": loaded_info.get("last_used"),
                "parameters": loaded_info.get("parameters", {}),
                "memory_used_mb": memory_used,
                "error": loaded_info.get("error")
            }
            logger.info(f"Added loaded model: {model_id}")
    
    # Convert memory values to GB and include model memory
    memory = sys_info.memory
    total_memory = memory.get("total", 0)
    used_memory = memory.get("used", 0) + total_model_memory  # Add model memory to system usage
    
    memory_gb = {
        "total_gb": total_memory / 1024 if total_memory else 0,
        "used_gb": used_memory / 1024 if used_memory else 0,
        "model_memory_gb": total_model_memory / 1024
    }
    
    # Format GPU info to match UI expectations
    gpu_info = {
        "available": sys_info.gpu_available,
        "status": "available" if sys_info.gpu_available else "not available",
        "name": sys_info.gpu_name or "Unknown",
        "memory": {
            "total_mb": sys_info.gpu.get("total"),
            "free_mb": sys_info.gpu.get("free"),
            "used_mb": sys_info.gpu.get("used")
        } if sys_info.gpu else None
    }
    
    response = {
        "models": {
            "available": available_models,
            "loaded": loaded_models
        },
        "system_state": {
            "cpu": sys_info.cpu,
            "memory": memory_gb,
            "gpu": gpu_info
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