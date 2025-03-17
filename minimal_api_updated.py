from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
import os
import ctypes
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    model_path: str
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=4, description="Number of threads to use")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU")
    use_mlock: bool = Field(default=False, description="Force system to keep model in RAM")
    use_mmap: bool = Field(default=True, description="Use memory mapping for model loading")

class LlamaModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.context = None
        self.error = None
        self.status = "unloaded"
        
    def load(self):
        """Load the model synchronously"""
        try:
            self.status = "loading"
            
            # Find and load llama.dll
            dll_path = os.path.join(os.path.dirname(__file__), "llama.dll")
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"llama.dll not found at {dll_path}")
                
            self.lib = ctypes.CDLL(dll_path)
            logger.info(f"Loaded llama.dll from {dll_path}")
            
            # Define parameter structures - updated for newer API
            class llama_model_params(ctypes.Structure):
                _fields_ = [
                    ("n_gpu_layers", ctypes.c_int),
                    ("use_mlock", ctypes.c_bool),
                    ("use_mmap", ctypes.c_bool),
                    ("vocab_only", ctypes.c_bool),
                ]
                
            class llama_context_params(ctypes.Structure):
                _fields_ = [
                    ("n_ctx", ctypes.c_int),
                    ("n_batch", ctypes.c_int),
                    ("n_threads", ctypes.c_int),
                    ("n_gpu_layers", ctypes.c_int),
                    ("use_mlock", ctypes.c_bool),
                    ("use_mmap", ctypes.c_bool),
                    ("vocab_only", ctypes.c_bool),
                ]
                
            class llama_batch(ctypes.Structure):
                _fields_ = [
                    ("n_tokens", ctypes.c_int32),
                    ("tokens", ctypes.POINTER(ctypes.c_int32)),
                    ("embd", ctypes.POINTER(ctypes.c_float)),
                    ("pos", ctypes.POINTER(ctypes.c_int32)),
                    ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
                    ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
                    ("logits", ctypes.POINTER(ctypes.c_bool)),
                ]
            
            # Set up function prototypes for newer API
            self.lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
            self.lib.llama_model_load_from_file.restype = ctypes.c_void_p
            
            self.lib.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
            self.lib.llama_init_from_model.restype = ctypes.c_void_p
            
            # Use llama_decode instead of llama_eval
            self.lib.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
            self.lib.llama_decode.restype = ctypes.c_int32
            
            # Helper function to create a batch
            self.lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
            self.lib.llama_batch_get_one.restype = llama_batch
            
            # Create parameters
            model_params = llama_model_params()
            model_params.n_gpu_layers = self.config.n_gpu_layers
            model_params.use_mlock = self.config.use_mlock
            model_params.use_mmap = self.config.use_mmap
            model_params.vocab_only = False
            
            context_params = llama_context_params()
            context_params.n_ctx = self.config.n_ctx
            context_params.n_batch = 512  # Default value
            context_params.n_threads = self.config.n_threads
            context_params.n_gpu_layers = self.config.n_gpu_layers
            context_params.use_mlock = self.config.use_mlock
            context_params.use_mmap = self.config.use_mmap
            context_params.vocab_only = False
            
            # Load the model
            model_path = self.config.model_path.encode('utf-8')
            logger.info(f"Loading model from {self.config.model_path}")
            
            # Try loading with verbose debug info
            logger.info("Creating model parameter structure")
            logger.info(f"Model params: n_gpu_layers={model_params.n_gpu_layers}, use_mlock={model_params.use_mlock}, use_mmap={model_params.use_mmap}")
            
            # Try with error handling
            try:
                logger.info("Calling llama_model_load_from_file")
                self.model = self.lib.llama_model_load_from_file(model_path, model_params)
                if not self.model:
                    logger.error("Model pointer is NULL after load attempt")
                    raise RuntimeError("Failed to load model - returned NULL pointer")
            except Exception as e:
                logger.error(f"Exception during model loading: {str(e)}")
                raise
                
            logger.info(f"Model loaded successfully, creating context (model pointer: {hex(self.model)})")
            
            # Create the context with error handling
            try:
                logger.info("Calling llama_init_from_model")
                self.context = self.lib.llama_init_from_model(self.model, context_params)
                if not self.context:
                    logger.error("Context pointer is NULL after creation attempt")
                    if hasattr(self.lib, 'llama_model_free') and self.model:
                        self.lib.llama_model_free(self.model)
                    self.model = None
                    raise RuntimeError("Failed to create context - returned NULL pointer")
            except Exception as e:
                logger.error(f"Exception during context creation: {str(e)}")
                raise
                
            logger.info(f"Context created successfully (context pointer: {hex(self.context)})")
            
            # Update status
            self.status = "loaded"
            process = psutil.Process(os.getpid())
            self.memory_used = process.memory_info().rss / (1024 * 1024)  # MB
            
            return True
            
        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def unload(self):
        """Unload the model and free resources"""
        if self.status == "loaded":
            try:
                # Free context
                if self.context and hasattr(self.lib, 'llama_free'):
                    self.lib.llama_free(self.context)
                self.context = None
                
                # Free model
                if self.model and hasattr(self.lib, 'llama_model_free'):
                    self.lib.llama_model_free(self.model)
                self.model = None
                
                self.status = "unloaded"
                logger.info(f"Model unloaded: {self.config.model_path}")
                return True
            except Exception as e:
                self.error = str(e)
                logger.error(f"Error unloading model: {str(e)}")
                raise
        return False

# Initialize FastAPI app
app = FastAPI(title="Minimal Llama.cpp API (Updated for newer API)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store loaded models
models = {}

@app.post("/api/models/{model_id}/load")
def load_model(model_id: str, config: ModelConfig):
    """Load a model with given configuration"""
    try:
        if model_id in models and models[model_id].status == "loaded":
            return {"status": "success", "message": f"Model {model_id} already loaded"}
        
        model = LlamaModel(config)
        models[model_id] = model
        model.load()
        
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/unload")
def unload_model(model_id: str):
    """Unload a specific model"""
    try:
        if model_id in models:
            models[model_id].unload()
            del models[model_id]
            return {"status": "success", "message": f"Model {model_id} unloaded successfully"}
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
def list_models():
    """List all loaded models and their status"""
    return {
        model_id: {
            "status": model.status,
            "memory_used_mb": getattr(model, 'memory_used', 0),
            "error": model.error,
            "config": model.config.dict()
        }
        for model_id, model in models.items()
    }

@app.get("/api/metrics")
def get_system_metrics():
    """Get system resource metrics"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        # Return metrics
        return {
            "cpu_percent": cpu_percent,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "timestamp": "2023-01-01T00:00:00"  # Placeholder
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082) 