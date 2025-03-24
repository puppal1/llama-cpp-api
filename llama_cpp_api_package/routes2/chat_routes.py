import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import json

from ..models.model_manager import model_manager
from .model_cache import get_model_metadata, _known_models

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/chat",
    tags=["v2-chat"]
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 0.1
    num_ctx: Optional[int] = None  # Will use model's default context length if not specified

@router.post("/{model_id}")
async def chat(model_id: str, request: ChatRequest):
    """Chat with a specific model"""
    try:
        # Ensure model_id has .gguf extension
        if not model_id.endswith(".gguf"):
            model_id = f"{model_id}.gguf"
            
        logger.info(f"Chat request for model: {model_id}")
        
        # Check if model exists and is loaded
        if model_id not in _known_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        metadata = get_model_metadata(model_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Model {model_id} metadata not found")
            
        if not metadata.get("loaded", False):
            raise HTTPException(
                status_code=400, 
                detail=f"Model {model_id} is not loaded. Please load it first using /api/v2/models/{model_id}/load"
            )

        # Process chat request
        try:
            # Remove .gguf extension for the model manager
            model_id_without_ext = model_id.replace(".gguf", "")
            
            # If num_ctx is not specified, use the model's context length
            ctx_size = request.num_ctx
            if ctx_size is None:
                ctx_size = metadata.get("context_length", 2048)
            
            # Log the chat request parameters
            logger.info(f"Chat request parameters: temperature={request.temperature}, max_tokens={request.max_tokens}, top_k={request.top_k}, top_p={request.top_p}")
            
            response = await model_manager.chat(
                model_id=model_id_without_ext,
                messages=[msg.dict() for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_k=request.top_k,
                top_p=request.top_p,
                repeat_penalty=request.repeat_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                num_ctx=ctx_size,
                stream=request.stream
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
            logger.error(f"Error in chat generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 