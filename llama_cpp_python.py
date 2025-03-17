"""
Llama.cpp Python API using llama-cpp-python package
This avoids the need to deal with DLLs directly.

Installation:
pip install llama-cpp-python
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
import json
import os
import logging
import psutil
from datetime import datetime

# Try to import GPU libraries if available
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Llama.cpp Python API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model definitions
class ModelConfig(BaseModel):
    model_path: str
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=4, description="Number of threads to use")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU")
    verbose: bool = Field(default=False, description="Enable verbose output")

class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1)
    stop: Optional[List[str]] = Field(default=None)

# Store loaded models
models = {}

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, config: ModelConfig):
    """Load a model with the given configuration"""
    try:
        # Import here to avoid startup dependency
        from llama_cpp import Llama
        
        if model_id in models:
            return {"status": "success", "message": f"Model {model_id} already loaded"}
        
        logger.info(f"Loading model {model_id} from {config.model_path}")
        
        # Create the model
        model = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads, 
            n_gpu_layers=config.n_gpu_layers,
            verbose=config.verbose
        )
        
        # Store model along with metadata
        models[model_id] = {
            "model": model,
            "config": config,
            "load_time": datetime.now(),
            "last_used": datetime.now()
        }
        
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        # Remove model from memory
        del models[model_id]
        return {"status": "success", "message": f"Model {model_id} unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all loaded models and their status"""
    return {
        model_id: {
            "status": "loaded",
            "load_time": model_info["load_time"].isoformat(),
            "last_used": model_info["last_used"].isoformat(),
            "config": model_info["config"].dict()
        }
        for model_id, model_info in models.items()
    }

@app.post("/api/models/{model_id}/chat")
async def chat(model_id: str, request: ChatRequest):
    """Chat with a specific model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        model_info = models[model_id]
        model = model_info["model"]
        
        # Update last used time
        model_info["last_used"] = datetime.now()
        
        # Prepare prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}</s>\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}</s>\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}</s>\n"
        
        # Add the final assistant prompt
        prompt += "<|assistant|>\n"
        
        # Generate response
        response = model(
            prompt=prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or ["</s>", "<|user|>"]
        )
        
        # Extract generated text
        generated_text = response["choices"][0]["text"]
        
        return {
            "model": model_id,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                }
            }]
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Get system resource metrics"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        # Initialize GPU metrics
        gpu_usage = None
        gpu_memory_used = None
        gpu_memory_total = None
        
        # Get GPU metrics if available
        if GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    # Get GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = utilization.gpu
                    
                    # Get GPU memory
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = meminfo.used / (1024 * 1024)  # Convert to MB
                    gpu_memory_total = meminfo.total / (1024 * 1024)  # Convert to MB
            except Exception as e:
                logger.warning(f"Error getting GPU metrics: {e}")
        
        # Return metrics
        return {
            "cpu_percent": cpu_percent,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "gpu_usage": gpu_usage,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 