from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import logging
import json
import os
import sys
import platform
from pathlib import Path
import ctypes
from enum import Enum
import asyncio
from datetime import datetime
import psutil
import math
import random

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

class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ModelConfig(BaseModel):
    model_path: str
    n_ctx: int = Field(default=2048, description="Context window size")
    n_batch: int = Field(default=512, description="Batch size for prompt processing")
    n_threads: int = Field(default=4, description="Number of threads to use")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU")
    use_mlock: bool = Field(default=False, description="Force system to keep model in RAM")
    use_mmap: bool = Field(default=True, description="Use memory mapping for model loading")
    vocab_only: bool = Field(default=False, description="Only load vocabulary, no weights")

class ChatMessage(BaseModel):
    role: ChatRole
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1)
    stop: Optional[List[str]] = Field(default=None)
    stream: bool = Field(default=False)

class LlamaModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.status = ModelStatus.UNLOADED
        self.error = None
        self.context = None
        self.model = None
        self._load_time = None
        self._last_used = None
        self._memory_used = 0

    async def load(self):
        """Load the model asynchronously"""
        try:
            self.status = ModelStatus.LOADING
            # Load llama.cpp library
            lib_path = self._find_llama_library()
            self.lib = ctypes.CDLL(lib_path)
            
            # Initialize llama.cpp model
            self._setup_library_bindings()
            params = self._create_model_params()
            
            # Perform actual model loading in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model, params)
            
            self.status = ModelStatus.LOADED
            self._load_time = datetime.now()
            self._update_memory_usage()
            logger.info(f"Model loaded successfully: {self.config.model_path}")
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error = str(e)
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _find_llama_library(self) -> str:
        """Find the llama.cpp shared library"""
        possible_paths = [
            # Local directory
            os.path.join(os.path.dirname(__file__), "llama.dll"),
            os.path.join(os.path.dirname(__file__), "libllama.so"),
            # Parent llama.cpp directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llama.cpp", "build", "bin", "Release", "llama.dll"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llama.cpp", "build", "lib", "Release", "llama.dll"),
            # System paths
            "/usr/local/lib/libllama.so",
            "/usr/lib/libllama.so",
            # Build directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llama.cpp", "build", "llama.dll"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found llama library at: {path}")
                return path
                
        # If no library is found, try downloading a pre-built one
        target_path = os.path.join(os.path.dirname(__file__), "llama.dll")
        logger.info(f"No library found. Please ensure llama.cpp is built and the DLL/SO file is in one of these locations: {possible_paths}")
        raise FileNotFoundError("llama.cpp shared library not found. Please build llama.cpp with -DBUILD_SHARED_LIBS=ON and copy the DLL/SO to the API directory.")

    def _setup_library_bindings(self):
        """Setup C function bindings for llama.cpp"""
        # Define necessary structures and function signatures
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
            
        class llama_model_params(ctypes.Structure):
            _fields_ = [
                ("n_gpu_layers", ctypes.c_int),
                ("use_mlock", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("vocab_only", ctypes.c_bool),
            ]

        class llama_token_data(ctypes.Structure):
            _fields_ = [
                ("id", ctypes.c_int),
                ("logit", ctypes.c_float),
                ("p", ctypes.c_float),
            ]

        class llama_token_data_array(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.POINTER(llama_token_data)),
                ("size", ctypes.c_size_t),
                ("sorted", ctypes.c_bool),
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

        # Setup function prototypes for newer API
        self.lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        self.lib.llama_model_load_from_file.restype = ctypes.c_void_p
        
        self.lib.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
        self.lib.llama_init_from_model.restype = ctypes.c_void_p
        
        # Token handling functions
        self.lib.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
        self.lib.llama_tokenize.restype = ctypes.c_int
        
        self.lib.llama_vocab_get_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.llama_vocab_get_text.restype = ctypes.c_char_p
        
        # Model info
        self.lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_get_vocab.restype = ctypes.c_void_p
        
        self.lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        self.lib.llama_vocab_n_tokens.restype = ctypes.c_int
        
        # Context evaluation - using llama_decode instead of llama_eval
        self.lib.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
        self.lib.llama_decode.restype = ctypes.c_int32
        
        # Helper function to create a batch
        self.lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        self.lib.llama_batch_get_one.restype = llama_batch
        
        # Token generation
        self.lib.llama_get_logits.argtypes = [ctypes.c_void_p]
        self.lib.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        
        # Sampling methods
        try:
            self.lib.llama_sample_repetition_penalty.argtypes = [ctypes.c_void_p, llama_token_data_array, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_float]
            self.lib.llama_sample_repetition_penalty.restype = None
            
            self.lib.llama_sample_frequency_and_presence_penalties.argtypes = [ctypes.c_void_p, llama_token_data_array, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
            self.lib.llama_sample_frequency_and_presence_penalties.restype = None
            
            self.lib.llama_sample_softmax.argtypes = [ctypes.c_void_p, llama_token_data_array]
            self.lib.llama_sample_softmax.restype = None
            
            self.lib.llama_sample_top_k.argtypes = [ctypes.c_void_p, llama_token_data_array, ctypes.c_int, ctypes.c_size_t]
            self.lib.llama_sample_top_k.restype = None
            
            self.lib.llama_sample_top_p.argtypes = [ctypes.c_void_p, llama_token_data_array, ctypes.c_float, ctypes.c_size_t]
            self.lib.llama_sample_top_p.restype = None
            
            self.lib.llama_sample_temperature.argtypes = [ctypes.c_void_p, llama_token_data_array, ctypes.c_float]
            self.lib.llama_sample_temperature.restype = None
        except AttributeError:
            logger.warning("Advanced sampling methods not available in this version of llama.cpp")
        
        self.lib.llama_token_eot.argtypes = [ctypes.c_void_p]
        self.lib.llama_token_eot.restype = ctypes.c_int
        
        # Store structures for later use
        self._llama_context_params = llama_context_params
        self._llama_model_params = llama_model_params
        self._llama_token_data = llama_token_data
        self._llama_token_data_array = llama_token_data_array
        self._llama_batch = llama_batch

    def _create_model_params(self) -> any:
        """Create model parameters structure"""
        model_params = self._llama_model_params()
        model_params.n_gpu_layers = self.config.n_gpu_layers
        model_params.use_mlock = self.config.use_mlock
        model_params.use_mmap = self.config.use_mmap
        model_params.vocab_only = self.config.vocab_only
        
        context_params = self._llama_context_params()
        context_params.n_ctx = self.config.n_ctx
        context_params.n_batch = self.config.n_batch
        context_params.n_threads = self.config.n_threads
        context_params.n_gpu_layers = self.config.n_gpu_layers
        context_params.use_mlock = self.config.use_mlock
        context_params.use_mmap = self.config.use_mmap
        context_params.vocab_only = self.config.vocab_only
        
        return (model_params, context_params)

    def _load_model(self, params):
        """Actually load the model (runs in thread pool)"""
        model_params, context_params = params
        model_path = self.config.model_path.encode('utf-8')
        
        # First load the model
        self.model = self.lib.llama_model_load_from_file(model_path, model_params)
        if not self.model:
            raise RuntimeError("Failed to load model")
            
        # Then initialize the context from the model
        self.context = self.lib.llama_init_from_model(self.model, context_params)
        if not self.context:
            # Free the model if context creation fails
            if hasattr(self.lib, 'llama_model_free') and self.model:
                self.lib.llama_model_free(self.model)
            self.model = None
            raise RuntimeError("Failed to create context")

    def _update_memory_usage(self):
        """Update memory usage statistics"""
        process = psutil.Process(os.getpid())
        self._memory_used = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    async def unload(self):
        """Unload the model and free resources"""
        if self.status == ModelStatus.LOADED:
            try:
                # Free context resources
                if self.context and hasattr(self.lib, 'llama_free'):
                    self.lib.llama_free(self.context)
                self.context = None
                
                # Free model resources
                if self.model and hasattr(self.lib, 'llama_model_free'):
                    self.lib.llama_model_free(self.model)
                self.model = None
                
                self.status = ModelStatus.UNLOADED
                self._memory_used = 0
                logger.info(f"Model unloaded: {self.config.model_path}")
            except Exception as e:
                logger.error(f"Error unloading model: {str(e)}")
                raise

    async def generate(self, request: ChatRequest) -> Union[str, asyncio.StreamReader]:
        """Generate response for chat request"""
        if self.status != ModelStatus.LOADED:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare input context
            prompt = self._prepare_prompt(request.messages)
            
            # Update generation parameters for this request
            self.temperature = request.temperature
            self.top_p = request.top_p
            self.max_tokens = request.max_tokens
            self.stop_strings = request.stop or []
            
            # Perform generation
            if request.stream:
                return self._stream_generate(prompt)
            else:
                return await self._batch_generate(prompt)
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

    def _prepare_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to prompt format"""
        prompt = ""
        for msg in messages:
            if msg.role == ChatRole.SYSTEM:
                prompt += f"<|system|>{msg.content}</s>"
            elif msg.role == ChatRole.USER:
                prompt += f"<|user|>{msg.content}</s>"
            elif msg.role == ChatRole.ASSISTANT:
                prompt += f"<|assistant|>{msg.content}</s>"
        return prompt

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using llama.cpp tokenizer"""
        text_bytes = text.encode('utf-8')
        max_tokens = len(text) + 100  # Allocate more than enough space
        tokens = (ctypes.c_int * max_tokens)()
        
        n_tokens = self.lib.llama_tokenize(
            self.model,
            text_bytes,
            len(text_bytes),
            tokens,
            max_tokens,
            False  # Add BOS
        )
        
        if n_tokens < 0:
            n_tokens = -n_tokens
            logger.warning(f"Tokenization failed to fit text in buffer, truncated to {n_tokens} tokens")
        
        return [tokens[i] for i in range(n_tokens)]

    async def _batch_generate(self, prompt: str) -> str:
        """Generate complete response in one batch"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._generate_text, prompt)
        return result

    def _sample_token(self, logits, temperature: float, top_p: float) -> int:
        """Sample a token from logits with temperature and top-p sampling"""
        n_vocab = self.lib.llama_vocab_n_tokens(self.lib.llama_model_get_vocab(self.model))
        
        # Create a list of (token_id, logit) pairs
        token_logits = [(i, logits[i]) for i in range(n_vocab)]
        
        # Apply temperature if not zero
        if temperature > 0:
            # Apply temperature by dividing logits by temperature
            token_logits = [(token, logit / temperature) for token, logit in token_logits]
        else:
            # If temperature is 0 or negative, use greedy sampling (argmax)
            return max(token_logits, key=lambda x: x[1])[0]
            
        # Convert logits to probabilities with softmax
        max_logit = max(logit for _, logit in token_logits)
        exp_logits = [(token, math.exp(logit - max_logit)) for token, logit in token_logits]
        total_exp_logits = sum(exp_logit for _, exp_logit in exp_logits)
        probs = [(token, exp_logit / total_exp_logits) for token, exp_logit in exp_logits]
        
        # Sort by probability in descending order
        sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
        
        # Apply top-p (nucleus) sampling
        if 0.0 < top_p < 1.0:
            cumulative_prob = 0.0
            cutoff_index = 0
            for i, (_, prob) in enumerate(sorted_probs):
                cumulative_prob += prob
                if cumulative_prob >= top_p:
                    cutoff_index = i + 1
                    break
            sorted_probs = sorted_probs[:cutoff_index]
        
        # If we have no valid tokens left (extreme top_p), just return the most likely token
        if not sorted_probs:
            return token_logits[0][0]
            
        # Sample from the filtered distribution
        tokens, probs = zip(*sorted_probs)
        token_idx = random.choices(range(len(tokens)), weights=probs, k=1)[0]
        return tokens[token_idx]
        
    def _generate_text(self, prompt: str) -> str:
        """Internal synchronous implementation of text generation"""
        # Tokenize the prompt
        input_tokens = self._tokenize(prompt)
        if not input_tokens:
            return ""
        
        # Output buffer
        output_text = ""
        
        # EOT token
        token_eot = self.lib.llama_token_eot(self.model)
        
        try:
            # Convert to ctypes array and create batch
            c_input_tokens = (ctypes.c_int32 * len(input_tokens))(*input_tokens)
            batch = self.lib.llama_batch_get_one(c_input_tokens, len(input_tokens))
            
            # Process the prompt
            if self.lib.llama_decode(self.context, batch) != 0:
                raise RuntimeError("Failed to process prompt")
            
            # Generate tokens
            max_new_tokens = self.max_tokens
            generated_count = 0
            
            while generated_count < max_new_tokens:
                # Get logits
                logits = self.lib.llama_get_logits(self.context)
                
                # Sample next token
                token = self._sample_token(logits, self.temperature, self.top_p)
                
                # Check for EOT token
                if token == token_eot:
                    break
                    
                # Convert token to text
                try:
                    vocab = self.lib.llama_model_get_vocab(self.model)
                    token_text = self.lib.llama_vocab_get_text(vocab, token)
                    if token_text:
                        token_str = token_text.decode('utf-8', errors='replace')
                        output_text += token_str
                except AttributeError:
                    # Fallback to a simple approach if the function is not available
                    logger.warning(f"llama_vocab_get_text not available, using placeholder for token {token}")
                    output_text += f"[TOKEN_{token}]"
                    
                # Check for stop strings
                should_stop = False
                for stop_str in self.stop_strings:
                    if output_text.endswith(stop_str):
                        output_text = output_text[:-len(stop_str)]
                        should_stop = True
                        break
                        
                if should_stop:
                    break
                
                # Process the new token - create a new batch with just this token
                next_token_array = (ctypes.c_int32 * 1)(token)
                next_batch = self.lib.llama_batch_get_one(next_token_array, 1)
                
                # Decode the token
                if self.lib.llama_decode(self.context, next_batch) != 0:
                    break
                    
                generated_count += 1
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error during generation: {str(e)}"
                
        return output_text

    def _stream_generate(self, prompt: str) -> asyncio.StreamReader:
        """Stream generated tokens"""
        # This is a placeholder for streaming implementation
        # For now, we'll use a simple approach that simulates streaming 
        # from the batch result
        reader, writer = asyncio.StreamReader(), asyncio.StreamReaderProtocol(asyncio.StreamReader())
        
        async def generate_stream():
            try:
                # Generate the full response
                full_response = await self._batch_generate(prompt)
                
                # Stream chunks of the response
                chunk_size = 4  # characters per chunk
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    writer.write(chunk.encode())
                    await asyncio.sleep(0.05)  # simulate delay
                    
                writer.write_eof()
            except Exception as e:
                writer.write(f"Error: {str(e)}".encode())
                writer.write_eof()
                
        asyncio.create_task(generate_stream())
        return reader

class ModelManager:
    def __init__(self):
        self.models: Dict[str, LlamaModel] = {}
        self.max_models = 2  # Maximum number of models to keep loaded
        self.cleanup_task = None

    async def load_model(self, model_id: str, config: ModelConfig):
        """Load a model with given configuration"""
        if model_id in self.models:
            if self.models[model_id].status == ModelStatus.LOADED:
                return
            
        # Check if we need to unload any models
        await self._cleanup_if_needed()
        
        # Create and load new model
        model = LlamaModel(config)
        self.models[model_id] = model
        await model.load()

    async def unload_model(self, model_id: str):
        """Unload a specific model"""
        if model_id in self.models:
            await self.models[model_id].unload()
            del self.models[model_id]

    async def _cleanup_if_needed(self):
        """Cleanup old models if we're at capacity"""
        if len(self.models) >= self.max_models:
            # Find least recently used model
            lru_model_id = min(
                self.models.keys(),
                key=lambda k: self.models[k]._last_used or datetime.min
            )
            await self.unload_model(lru_model_id)

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._cleanup_if_needed()
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

# Create API router
api_router = APIRouter()

@api_router.post("/models/{model_id}/load")
async def load_model(model_id: str, config: ModelConfig):
    """Load a model with given configuration"""
    try:
        await model_manager.load_model(model_id, config)
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model"""
    try:
        await model_manager.unload_model(model_id)
        return {"status": "success", "message": f"Model {model_id} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models")
async def list_models():
    """List all loaded models and their status"""
    return {
        model_id: {
            "status": model.status.value,
            "memory_used_mb": model._memory_used,
            "load_time": model._load_time,
            "last_used": model._last_used,
            "config": model.config.dict()
        }
        for model_id, model in model_manager.models.items()
    }

@api_router.post("/models/{model_id}/chat")
async def chat(
    model_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Chat with a specific model"""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model = model_manager.models[model_id]
    if model.status != ModelStatus.LOADED:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} is not ready (status: {model.status.value})"
        )
    
    try:
        response = await model.generate(request)
        model._last_used = datetime.now()
        
        if request.stream:
            # Return streaming response
            return response
        else:
            # Return complete response
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
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/metrics")
async def get_system_metrics():
    """Get system resource metrics"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        # Get GPU usage if available
        gpu_memory_used = None
        gpu_memory_total = None
        
        if GPU_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = meminfo.used / (1024 * 1024)
                    gpu_memory_total = meminfo.total / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Error getting GPU metrics: {e}")
        
        # Return metrics
        return {
            "cpu_percent": cpu_percent,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    await model_manager.start_cleanup_task()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await model_manager.stop_cleanup_task()
    for model_id in list(model_manager.models.keys()):
        await model_manager.unload_model(model_id)

# Mount API router
app.include_router(api_router, prefix="/api")

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081) 