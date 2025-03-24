from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, AsyncGenerator
import asyncio
import json
import logging
from ..models.model_manager import ModelManager
from ..api.api_types import ModelStatus
from ..models.model_types import ChatRequest, ChatResponse

router = APIRouter()
model_manager = ModelManager()
logger = logging.getLogger(__name__)

async def stream_response(model_id: str, stream_reader: asyncio.StreamReader) -> AsyncGenerator[str, None]:
    """Stream the response from the model"""
    try:
        while True:
            chunk = await stream_reader.readline()
            if not chunk:
                break
            try:
                # Assuming each chunk is a JSON string ending with newline
                response_chunk = {
                    "model": model_id,
                    "choices": [{
                        "delta": {
                            "content": chunk.decode('utf-8').strip()
                        }
                    }]
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in stream_response: {e}")
        yield f"data: [DONE]\n\n"

@router.post("/api/models/{model_id}/chat")
async def chat_with_model(
    model_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Chat with a specific model"""
    try:
        if model_id not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        model_info = model_manager.models[model_id]
        if model_info.status != ModelStatus.LOADED:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} is not ready (status: {model_info.status})"
            )
            
        # Update last used timestamp
        background_tasks.add_task(model_info.update_last_used)
        
        # Generate response
        response = await model_info.generate(request)
        
        if request.stream:
            return StreamingResponse(
                stream_response(model_id, response),
                media_type="text/event-stream"
            )
        else:
            return JSONResponse(content=ChatResponse(
                model=model_id,
                choices=[{
                    "message": {
                        "role": "assistant",
                        "content": response
                    }
                }]
            ).dict())
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 