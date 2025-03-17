"""
Llama.cpp Python API using llama-cpp-python package
This avoids the need to deal with DLLs directly.

Installation:
pip install llama-cpp-python
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from llama_cpp import Llama
from utils.model_params import ModelParameters

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
app = FastAPI(title="Llama.cpp API")

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
    parameters: Optional[ModelParameters] = Field(default_factory=ModelParameters.default)

# Store loaded models
models: Dict[str, dict] = {}

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, params: Optional[ModelParameters] = None):
    """Load a model with given parameters"""
    try:
        if model_id in models:
            return {"status": "success", "message": f"Model {model_id} already loaded"}
        
        # Use default parameters if none provided
        if params is None:
            params = ModelParameters()
        
        logger.info(f"Loading model {model_id} with parameters: {params.dict()}")
        
        # Map model IDs to actual filenames
        model_filename_map = {
            "Phi-3-mini-4k-instruct-q4": "Phi-3-mini-4k-instruct-q4.gguf",
            "mistral-7b-instruct-v0": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        }
        
        model_filename = model_filename_map.get(model_id, f"{model_id}.gguf")
        model_path = os.path.join("models", model_filename)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        # Create the model
        model = Llama(
            model_path=model_path,
            n_ctx=params.num_ctx,
            n_batch=params.num_batch,
            n_threads=params.num_thread,
            n_gpu_layers=params.num_gpu,
            use_mlock=params.mlock,
            use_mmap=params.mmap,
            seed=params.seed if params.seed is not None else -1,
            verbose=True
        )
        
        # Store model along with metadata
        models[model_id] = {
            "model": model,
            "params": params,
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
            "parameters": model_info["params"].dict()
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
        params = request.parameters or model_info["params"]
        
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
            max_tokens=params.num_predict,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            stop=params.stop or ["</s>", "<|user|>"]
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
            "models_loaded": len(models),
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama.cpp API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port) 