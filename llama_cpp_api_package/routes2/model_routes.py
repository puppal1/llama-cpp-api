import logging
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Optional
import os
import glob
from datetime import datetime
import psutil
import traceback

from ..utils.system_info import get_system_info
from ..api.api_types import ModelStatus
from ..models.model_manager import model_manager
from ..models.model_types import ModelParameters
from ..utils.memory_manager import memory_manager
from .model_cache import (
    list_models,
    get_model_metadata,
    get_model_size_cached,
    initialize_cache,
    get_model_load_state,
    _initialize_or_update_cache,
    _known_models
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/models",
    tags=["v2-models"]
)

# Initialize cache at module load time
models_dir = os.getenv("MODELS_DIR")
if models_dir:
    logger.info(f"Initializing model cache from {models_dir}")
    try:
        initialize_cache(models_dir)
        logger.info(f"Model cache initialized with {len(_known_models)} models")
    except Exception as e:
        logger.error(f"Failed to initialize model cache: {e}")
else:
    logger.warning("MODELS_DIR environment variable not set, cache initialization skipped")

def get_system_info() -> Dict:
    """Get system information including memory"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_memory_gb": memory.total / (1024**3),
            "available_memory_gb": memory.available / (1024**3),
            "memory_usage_percent": memory.percent
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            "total_memory_gb": 0,
            "available_memory_gb": 0,
            "memory_usage_percent": 0
        }

@router.get("")
async def list_available_models():
    """List all available models."""
    try:
        models_dir = os.getenv("MODELS_DIR")
        if not models_dir:
            raise HTTPException(status_code=500, detail="MODELS_DIR environment variable not set")
        
        # Get models from cache
        all_models = list_models(models_dir)
        
        # Get current memory metrics
        memory_metrics = memory_manager.get_memory_metrics()
        
        # Separate into available and loaded models
        available_models = []
        loaded_models = []
        
        for model in all_models:
            # Update model ID to include .gguf extension
            if not model["id"].endswith(".gguf"):
                model["id"] = f"{model['id']}.gguf"
            
            # Check multiple ways a model could be marked as loaded
            is_loaded = False
            if model.get("metadata"):
                if model["metadata"].get("loaded", False):
                    is_loaded = True
                elif model["metadata"].get("status") == "ModelStatus.LOADED":
                    is_loaded = True
                elif model["metadata"].get("load_time") is not None:
                    is_loaded = True

                # Convert size to GB
                size_bytes = model.get("size", 0)
                model["metadata"]["size_gb"] = size_bytes / (1024 * 1024 * 1024)
                
                # Add memory requirements info
                model_id = model["id"].replace(".gguf", "").lower()
                memory_required, is_safe = memory_manager.calculate_required_memory(
                    model_id, 
                    model.get("metadata", {}).get("parameters", {}).get("context_length", 2048)
                )
                
                # Update metadata with memory information
                if "parameters" not in model["metadata"]:
                    model["metadata"]["parameters"] = {}
                
                model["metadata"].update({
                    "memory_required_gb": memory_required,
                    "is_safe_to_load": is_safe,
                    "status": "LOADED" if is_loaded else "AVAILABLE"
                })
                
                # If loaded, add current memory usage
                if is_loaded:
                    model["metadata"]["current_memory_gb"] = memory_manager.model_memory_usage.get(model_id, 0)
            
            if is_loaded:
                loaded_models.append(model)
            else:
                available_models.append(model)
        
        return {
            "available_models": available_models,
            "loaded_models": loaded_models,
            "memory_metrics": memory_metrics
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    try:
        # Ensure model_id has .gguf extension
        if not model_id.endswith(".gguf"):
            model_id = f"{model_id}.gguf"
        
        # Log the model ID
        logger.info(f"Looking up model info for model_id: {model_id}")
        logger.info(f"Known models in cache: {list(_known_models)}")
        
        # Check if model exists in known models
        if model_id not in _known_models:
            logger.error(f"Model not found in known models: {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get model metadata directly using the full filename
        metadata = get_model_metadata(model_id)
        logger.info(f"Retrieved metadata: {metadata is not None}")
        
        if not metadata:
            logger.error(f"Model metadata not found for {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get model size
        size = get_model_size_cached(model_id)
        logger.info(f"Retrieved size: {size is not None}")
        
        if size is None:
            logger.error(f"Model size not found for {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "id": model_id,
            "metadata": metadata,
            "size": size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/load")
async def load_model(model_id: str, parameters: Optional[Dict] = None):
    """Load a model into memory with optional parameters."""
    try:
        # Ensure model_id has .gguf extension
        if not model_id.endswith(".gguf"):
            model_id = f"{model_id}.gguf"
        
        logger.info(f"Requested to load model: {model_id}")
        
        # Check if model exists
        if model_id not in _known_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get model metadata
        metadata = get_model_metadata(model_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Parse parameters
        model_params = ModelParameters()
        if parameters:
            model_params = ModelParameters(**parameters)
        
        # Check memory requirements
        can_load, message = memory_manager.can_load_model(
            model_id.replace(".gguf", "").lower(), 
            model_params.n_ctx
        )
        if not can_load:
            raise HTTPException(status_code=400, detail=message)
        
        # Load model using singleton instance
        try:
            model_id_without_ext = model_id.replace(".gguf", "")
            logger.info(f"Starting model loading: {model_id_without_ext}")
            model_info = model_manager.load_model(model_id_without_ext, model_params)
            logger.info(f"Model loaded successfully: {model_id}")
            
            # Register model memory usage
            memory_required, _ = memory_manager.calculate_required_memory(
                model_id_without_ext.lower(), 
                model_params.n_ctx
            )
            memory_manager.register_model_load(model_id_without_ext.lower(), memory_required)
            
            # Update cache with load state
            _initialize_or_update_cache(
                models_dir, 
                update_only=True,
                model_ids={model_id},
                load_state={
                    "model_id": model_id,
                    "is_loaded": True,
                    "load_time": model_info.get("load_time"),
                    "memory_used": memory_required
                }
            )
            
            return {
                "status": "loaded", 
                "model_id": model_id,
                "info": model_info,
                "memory_metrics": memory_manager.get_memory_metrics()
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    try:
        if not model_id.endswith(".gguf"):
            model_id = f"{model_id}.gguf"
            
        logger.info(f"Requested to unload model: {model_id}")
        
        if model_id not in _known_models:
            logger.error(f"Model not found in known models: {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        metadata = get_model_metadata(model_id)
        if not metadata:
            logger.error(f"Model metadata not found for {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        if not metadata.get("loaded", False) and not metadata.get("status") == "ModelStatus.LOADED":
            logger.info(f"Model {model_id} is not loaded, nothing to do")
            return {"status": "not_loaded", "model_id": model_id}
        
        try:
            model_id_without_ext = model_id.replace(".gguf", "")
            model_manager.unload_model(model_id_without_ext)
            logger.info(f"Model unloaded successfully: {model_id}")
            
            # Unregister model from memory tracking
            memory_manager.unregister_model(model_id_without_ext.lower())
            
            models_dir = os.getenv("MODELS_DIR")
            if not models_dir:
                raise HTTPException(status_code=500, detail="MODELS_DIR environment variable not set")
            
            _initialize_or_update_cache(
                models_dir,
                update_only=True,
                model_ids={model_id},
                load_state={
                    "model_id": model_id,
                    "is_loaded": False
                }
            )
            
            logger.info(f"Cache updated: model {model_id} marked as unloaded")
            return {
                "status": "unloaded", 
                "model_id": model_id,
                "memory_metrics": memory_manager.get_memory_metrics()
            }
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/chat")
async def chat_with_model(model_id: str, request: Dict = Body(...)):
    """Chat with a loaded model."""
    try:
        # Ensure model_id has .gguf extension
        if not model_id.endswith(".gguf"):
            model_id = f"{model_id}.gguf"
            
        logger.info(f"Chat request for model: {model_id}")
        
        # Check if model exists and is loaded
        if model_id not in _known_models:
            logger.error(f"Model not found in known models: {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get model metadata
        metadata = get_model_metadata(model_id)
        if not metadata:
            logger.error(f"Model metadata not found for {model_id}")
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        # Verify model is loaded
        if not metadata.get("loaded", False):
            logger.error(f"Model {model_id} is not loaded")
            raise HTTPException(status_code=400, detail=f"Model {model_id} is not loaded. Please load the model first.")
        
        # Extract chat parameters
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
            
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 100)
        
        # Remove .gguf extension for the model manager
        model_id_without_ext = model_id.replace(".gguf", "")
        
        # Generate response using model manager
        try:
            response = await model_manager.chat(
                model_id=model_id_without_ext,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "model": model_id,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 