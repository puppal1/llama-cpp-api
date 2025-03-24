from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
import json
from typing import Dict, Any, Optional, List
from llama_cpp import Llama
import logging
from .routes2.model_routes import router as v2_model_router
from .routes2.metrics_routes import router as v2_metrics_router
from .routes2.model_cache import initialize_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the models directory
MODELS_DIR = Path(__file__).parent.parent / "models"
if not MODELS_DIR.exists():
    # If models directory doesn't exist at parent.parent, try current directory
    MODELS_DIR = Path.cwd() / "models"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v2 routers
app.include_router(v2_model_router, prefix="/api/v2")
app.include_router(v2_metrics_router, prefix="/api/v2")

# Initialize model cache at startup
models_dir = os.getenv("MODELS_DIR", str(MODELS_DIR))
initialize_cache(models_dir)

# Global model storage
loaded_models: Dict[str, Llama] = {}

class ModelConfig(BaseModel):
    n_ctx: Optional[int] = 2048
    n_batch: Optional[int] = 512
    n_threads: Optional[int] = 4
    n_gpu_layers: Optional[int] = 0
    rope_freq_base: Optional[float] = 10000.0
    rope_freq_scale: Optional[float] = 1.0
    rope_dimension_count: Optional[int] = None
    rope_scaling_type: Optional[str] = None
    use_mmap: Optional[bool] = True
    use_mlock: Optional[bool] = False
    verbose: Optional[bool] = True
    tensor_split: Optional[List[float]] = None

def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """Read and return model metadata."""
    try:
        llm = Llama(model_path=model_path, vocab_only=True, verbose=True)
        metadata = llm.model_metadata()
        return metadata
    except Exception as e:
        logger.error(f"Error reading metadata: {str(e)}")
        return {}

def get_model_config(model_name: str, metadata: Dict[str, Any]) -> ModelConfig:
    """Get model-specific configuration based on model name and metadata."""
    config = ModelConfig()
    
    # Get RoPE dimensions from metadata if available
    rope_dims = metadata.get("llama.rope.dimension_count", 0)
    
    # Model-specific configurations
    if "ayla" in model_name.lower():
        config.rope_dimension_count = rope_dims or 128  # Use metadata or fallback to 128
        config.rope_freq_base = 1000000.0
        config.n_ctx = 2048
        config.tensor_split = None
    elif "deepseek" in model_name.lower():
        config.rope_freq_base = 10000.0
        config.rope_scaling_type = "linear"
        config.n_ctx = 4096
    elif "moe" in model_name.lower():
        config.n_threads = 4
        config.n_batch = 256
        config.n_gpu_layers = 0
        config.tensor_split = None
    elif "mistral" in model_name.lower():
        config.n_ctx = 4096
        config.n_batch = 512
    
    return config

@app.post("/api/models/{model_name}/load")
async def load_model(model_name: str, config: Optional[ModelConfig] = None):
    """Load a model with specified configuration."""
    try:
        models_dir = os.path.abspath(os.getenv("MODELS_DIR", "models"))
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")
        
        # Get model metadata
        metadata = get_model_metadata(model_path)
        
        # Get default configuration based on model type
        default_config = get_model_config(model_name, metadata)
        
        # Merge with user-provided config if any
        if config:
            # Update only non-None values from user config
            for field, value in config.dict(exclude_unset=True).items():
                setattr(default_config, field, value)
        
        # Convert config to dict and filter out None values
        config_dict = {k: v for k, v in default_config.dict().items() if v is not None}
        
        logger.info(f"Loading model {model_name} with config: {json.dumps(config_dict, indent=2)}")
        
        # Load the model
        llm = Llama(model_path=model_path, **config_dict)
        
        # Store the loaded model
        loaded_models[model_name] = llm
        
        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a model."""
    try:
        if model_name in loaded_models:
            del loaded_models[model_name]
            return {"status": "success", "message": f"Model {model_name} unloaded"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all available and loaded models."""
    try:
        available_models = []
        if os.path.exists("models"):
            available_models = [f for f in os.listdir("models") if f.endswith(".gguf")]
        
        loaded_model_names = list(loaded_models.keys())
        
        return {
            "available_models": available_models,
            "loaded_models": loaded_model_names
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False

@app.post("/api/models/{model_name}/chat")
async def chat_completion(model_name: str, request: ChatRequest):
    """Generate chat completion."""
    try:
        if model_name not in loaded_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
        
        llm = loaded_models[model_name]
        
        # Convert messages to dict format
        messages = [msg.dict() for msg in request.messages]
        
        # Generate completion
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            stream=request.stream
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 