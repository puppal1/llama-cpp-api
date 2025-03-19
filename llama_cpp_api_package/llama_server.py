"""
Llama.cpp Python API using llama-cpp-python package
This avoids the need to deal with DLLs directly.

Installation:
pip install llama-cpp-python
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
import json
import os
import logging
import psutil
import asyncio
from datetime import datetime
from llama_cpp import Llama
from llama_cpp_api_package.utils.model_params import ModelParameters
import re
import platform
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple GPU detection
GPU_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    GPU_AVAILABLE = device_count > 0
    if GPU_AVAILABLE:
        logger.info(f"Found {device_count} GPU device(s)")
    else:
        logger.info("No GPU devices found")
except ImportError:
    logger.info("pynvml not available - running in CPU mode")
except Exception as e:
    logger.warning(f"GPU detection failed: {e} - running in CPU mode")

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
class ModelStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    parameters: Optional[ModelParameters] = Field(default_factory=ModelParameters.default)
    stream: bool = Field(default=False, description="Whether to stream the response")

class ModelInfo:
    def __init__(self, model_path: str, params: ModelParameters):
        self.model_path = model_path
        self.params = params
        self.model = None
        self.status = ModelStatus.UNLOADED
        self.error = None
        self.load_time = None
        self.last_used = None
        self.memory_used = 0

    async def load(self):
        """Load the model asynchronously"""
        try:
            self.status = ModelStatus.LOADING
            logger.info(f"Loading model: {self.model_path}")
            
            # Create the model
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.params.num_ctx,
                n_batch=self.params.num_batch,
                n_threads=self.params.num_thread,
                n_gpu_layers=self.params.num_gpu,
                use_mlock=self.params.mlock,
                use_mmap=self.params.mmap,
                seed=self.params.seed if self.params.seed is not None else -1,
                verbose=True
            )
            
            self.status = ModelStatus.LOADED
            self.load_time = datetime.now()
            self.last_used = datetime.now()
            self._update_memory_usage()
            logger.info(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error = str(e)
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _update_memory_usage(self):
        """Update memory usage statistics"""
        process = psutil.Process(os.getpid())
        self.memory_used = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    async def unload(self):
        """Unload the model and free resources"""
        if self.status == ModelStatus.LOADED:
            try:
                # The llama-cpp-python package handles cleanup automatically
                # when the model object is deleted
                self.model = None
                self.status = ModelStatus.UNLOADED
                self.memory_used = 0
                logger.info(f"Model unloaded: {self.model_path}")
            except Exception as e:
                logger.error(f"Error unloading model: {str(e)}")
                raise

    async def generate(self, request: ChatRequest) -> Union[str, asyncio.StreamReader]:
        """Generate response for chat request"""
        if self.status != ModelStatus.LOADED:
            raise RuntimeError(f"Model not ready (status: {self.status})")

        try:
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

            # Update last used time
            self.last_used = datetime.now()
            
            if request.stream:
                return await self._stream_generate(prompt, request.parameters)
            else:
                return await self._batch_generate(prompt, request.parameters)
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

    async def _batch_generate(self, prompt: str, params: ModelParameters) -> str:
        """Generate complete response in one batch"""
        response = self.model(
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
        return response["choices"][0]["text"]

    async def _stream_generate(self, prompt: str, params: ModelParameters) -> asyncio.StreamReader:
        """Stream generated tokens"""
        reader, writer = asyncio.StreamReader(), asyncio.StreamReaderProtocol(asyncio.StreamReader())
        
        async def generate_stream():
            try:
                response = self.model(
                    prompt=prompt,
                    max_tokens=params.num_predict,
                    temperature=params.temperature,
                    top_k=params.top_k,
                    top_p=params.top_p,
                    repeat_penalty=params.repeat_penalty,
                    presence_penalty=params.presence_penalty,
                    frequency_penalty=params.frequency_penalty,
                    stop=params.stop or ["</s>", "<|user|>"],
                    stream=True
                )
                
                for chunk in response:
                    if chunk and "choices" in chunk and chunk["choices"]:
                        text = chunk["choices"][0]["text"]
                        if text:
                            writer.write(text.encode())
                            await asyncio.sleep(0.01)  # Small delay for smoother streaming
                            
                writer.write_eof()
            except Exception as e:
                writer.write(f"Error: {str(e)}".encode())
                writer.write_eof()
                
        asyncio.create_task(generate_stream())
        return reader

class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}

    async def load_model(self, model_id: str, model_path: str, params: ModelParameters):
        """Load a model with given configuration"""
        if model_id in self.models:
            if self.models[model_id].status == ModelStatus.LOADED:
                return
            elif self.models[model_id].status == ModelStatus.LOADING:
                raise HTTPException(status_code=409, detail=f"Model {model_id} is currently loading")
        
        # Create and load new model
        model_info = ModelInfo(model_path, params)
        self.models[model_id] = model_info
        await model_info.load()

    async def unload_model(self, model_id: str):
        """Unload a specific model"""
        if model_id in self.models:
            await self.models[model_id].unload()
            del self.models[model_id]

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    pass  # No background tasks needed

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    for model_id in list(model_manager.models.keys()):
        await model_manager.unload_model(model_id)

def calculate_memory_requirements(model_path: str, params: ModelParameters) -> dict:
    """Calculate detailed memory requirements for model loading."""
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    n_ctx = params.num_ctx if params else 2048
    n_batch = params.num_batch if params else 512
    n_gpu_layers = params.num_gpu if params else 0

    # Calculate components based on model architecture
    # Model buffer (mapped or loaded)
    model_buffer = model_size  # Base model size

    # KV cache size calculation
    # From logs: each layer has n_embd_k_gqa = 1024, n_embd_v_gqa = 1024
    # Size per layer = (key_size + value_size) * context_size * sizeof(float16)
    bytes_per_key_value = 2  # float16 = 2 bytes
    n_layers = 32  # From model metadata
    kv_cache = (2 * 1024 * n_ctx * bytes_per_key_value * n_layers) / (1024 * 1024)  # MB

    # Output and compute buffers
    output_buffer = 0.12  # MB (from logs)
    compute_buffer = 164.01  # MB (from logs)

    # Total CPU memory required
    total_cpu_memory = model_buffer + kv_cache + output_buffer + compute_buffer

    # GPU memory if using GPU layers
    gpu_memory = 0
    if n_gpu_layers > 0:
        # Estimate GPU memory for offloaded layers
        gpu_memory = (model_buffer * (n_gpu_layers / n_layers))
        total_cpu_memory -= gpu_memory

    return {
        "total_cpu_mb": total_cpu_memory,
        "components": {
            "model_buffer_mb": model_buffer,
            "kv_cache_mb": kv_cache,
            "output_buffer_mb": output_buffer,
            "compute_buffer_mb": compute_buffer
        },
        "gpu_memory_mb": gpu_memory if n_gpu_layers > 0 else 0
    }

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, params: Optional[ModelParameters] = None):
    """Load a model with given parameters"""
    try:
        if model_id in model_manager.models:
            model_info = model_manager.models[model_id]
            if model_info.status == ModelStatus.LOADED:
                return {"status": "success", "message": f"Model {model_id} already loaded"}
            elif model_info.status == ModelStatus.LOADING:
                return {"status": "pending", "message": f"Model {model_id} is currently loading"}
        
        # Use default parameters if none provided
        if params is None:
            params = ModelParameters()
        
        logger.info(f"Loading model {model_id} with parameters: {params.dict()}")
        
        # Find model file
        model_path = None
        model_dirs = ["models", os.path.join(os.path.dirname(__file__), "models")]
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith(".gguf"):
                        file_id = file.replace(".gguf", "")
                        file_id = re.sub(r"\.Q\d.*$", "", file_id)
                        file_id = re.sub(r"\.v\d+.*$", "", file_id)
                        
                        if file_id.lower() == model_id.lower():
                            model_path = os.path.join(model_dir, file)
                            break
                if model_path:
                    break
        
        if not model_path:
            available_files = []
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    available_files.extend([f for f in os.listdir(model_dir) if f.endswith(".gguf")])
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found for ID: {model_id}. Available models: {', '.join(available_files)}"
            )

        # Calculate memory requirements
        memory_requirements = calculate_memory_requirements(model_path, params)
        
        # Check system memory
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024 * 1024)  # MB
        available_memory = memory.available / (1024 * 1024)  # MB

        # Check GPU memory if needed
        if params.num_gpu > 0 and GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_gpu_memory = meminfo.free / (1024 * 1024)  # MB
                
                if memory_requirements["gpu_memory_mb"] > available_gpu_memory:
                    raise HTTPException(
                        status_code=507,
                        detail={
                            "message": f"Insufficient GPU memory to load model {model_id}",
                            "required_memory_mb": memory_requirements["gpu_memory_mb"],
                            "available_memory_mb": available_gpu_memory
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking GPU memory: {e}")
                params.num_gpu = 0  # Fallback to CPU-only mode
        
        # Load the model
        await model_manager.load_model(model_id, model_path, params)
        
        return {
            "status": "success",
            "message": f"Model {model_id} loaded successfully",
            "memory_info": {
                "requirements": memory_requirements,
                "system": {
                    "total_mb": total_memory,
                    "available_mb": available_memory,
                    "used_mb": total_memory - available_memory
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model"""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        await model_manager.unload_model(model_id)
        return {"status": "success", "message": f"Model {model_id} unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """Get comprehensive information about all models and system state"""
    try:
        # Get available models
        available_models_data = await list_available_models()
        
        # Get loaded models info
        loaded_models = {
            model_id: {
                "status": model_info.status,
                "load_time": model_info.load_time.isoformat() if model_info.load_time else None,
                "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
                "parameters": model_info.params.dict(),
                "memory_used_mb": model_info.memory_used,
                "error": model_info.error
            }
            for model_id, model_info in model_manager.models.items()
        }
        
        # Get system metrics
        metrics_data = await get_metrics()
        
        return {
            "models": {
                "available": available_models_data["models"],
                "loaded": loaded_models
            },
            "system_state": {
                "memory": {
                    "total_gb": metrics_data["memory_total_mb"] / 1024,
                    "used_gb": metrics_data["memory_used_mb"] / 1024
                },
                "gpu": metrics_data["gpu"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting models list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/chat")
async def chat(model_id: str, request: ChatRequest):
    """Chat with a specific model"""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model_info = model_manager.models[model_id]
    if model_info.status != ModelStatus.LOADED:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} is not ready (status: {model_info.status})"
        )
    
    try:
        response = await model_info.generate(request)
        
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
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_nvml_path():
    """Get the NVML library path based on the operating system"""
    system = platform.system().lower()
    
    if system == 'windows':
        # Common Windows paths for the NVML library
        possible_paths = [
            'nvml.dll',  # Default path
            r'C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll',  # Common NVIDIA install path
            r'C:\Windows\System32\nvml.dll',  # System32 path
            r'C:\Windows\System32\nvidia-smi\nvml.dll',  # Alternative System32 path
        ]
        
        # Add paths from NVIDIA_DRIVER_PATH environment variable if it exists
        nvidia_path = os.environ.get('NVIDIA_DRIVER_PATH')
        if nvidia_path:
            possible_paths.append(os.path.join(nvidia_path, 'nvml.dll'))
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found NVML library at: {path}")
                return path
                
        # If no path found, log the searched paths
        logger.warning(f"NVML library not found in any of these locations: {possible_paths}")
        return 'nvml.dll'  # Return default path as last resort
        
    elif system == 'linux':
        # Common Linux paths for the NVML library
        paths = [
            'libnvidia-ml.so.1',  # Try version-specific file first
            'libnvidia-ml.so',    # Then try generic name
            '/usr/lib64/libnvidia-ml.so.1',
            '/usr/lib/libnvidia-ml.so.1',
            '/usr/lib64/libnvidia-ml.so',
            '/usr/lib/libnvidia-ml.so',
        ]
        # Return the first path that exists
        for path in paths:
            if os.path.exists(path) or os.path.exists(f"/usr/lib/{path}") or os.path.exists(f"/usr/lib64/{path}"):
                logger.info(f"Found NVML library at: {path}")
                return path
        
        logger.warning(f"NVML library not found in any of these locations: {paths}")
        return 'libnvidia-ml.so'  # Default to the basic name if no paths found
    else:
        raise OSError(f"NVML not supported on: {system}")

def get_mac_gpu_info():
    """Get GPU information on macOS using system_profiler"""
    try:
        import subprocess
        import json
        
        # Use system_profiler to get GPU information
        cmd = ['system_profiler', 'SPDisplaysDataType', '-json']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        gpu_info = data.get('SPDisplaysDataType', [])
        
        if not gpu_info:
            return None
            
        gpu_data = {
            "available": True,
            "status": "GPU Available",
            "type": "Unknown",
            "memory": None,
            "details": {}
        }
        
        for gpu in gpu_info:
            gpu_model = gpu.get('sppci_model', 'Unknown GPU')
            
            # Check for Apple Silicon GPU
            if 'Apple' in gpu_model:
                gpu_data["type"] = "Apple Silicon"
                gpu_data["details"]["metal_support"] = gpu.get('spmetal_supported', 'Unknown')
                if 'vram_shared' in gpu:
                    gpu_data["memory"] = {
                        "type": "Shared Memory",
                        "size": gpu['vram_shared']
                    }
            else:
                gpu_data["type"] = "Discrete/Integrated"
                if 'sppci_memory' in gpu:
                    gpu_data["memory"] = {
                        "type": "Dedicated",
                        "size": gpu['sppci_memory']
                    }
            
            # Add additional details
            if 'metal_version' in gpu:
                gpu_data["details"]["metal_version"] = gpu['metal_version']
            
            # We only process the first GPU for now
            break
            
        return gpu_data
    except Exception as e:
        logger.warning(f"Error getting macOS GPU information: {e}")
        return None

def get_windows_gpu_info():
    """Get GPU information using Windows Management Instrumentation (WMI)"""
    try:
        import wmi
        w = wmi.WMI()
        gpu_info = w.Win32_VideoController()[0]  # Get the first GPU
        
        return {
            "available": True,
            "status": "GPU Available",
            "name": gpu_info.Name,
            "memory": {
                "total_mb": float(gpu_info.AdapterRAM) / (1024 * 1024) if hasattr(gpu_info, 'AdapterRAM') else None,
                "free_mb": None,  # WMI doesn't provide this information
                "used_mb": None   # WMI doesn't provide this information
            },
            "utilization": {
                "gpu_percent": None,  # WMI doesn't provide this information
                "memory_percent": None  # WMI doesn't provide this information
            },
            "details": {
                "driver_version": gpu_info.DriverVersion if hasattr(gpu_info, 'DriverVersion') else None,
                "video_processor": gpu_info.VideoProcessor if hasattr(gpu_info, 'VideoProcessor') else None,
                "video_memory_type": gpu_info.VideoMemoryType if hasattr(gpu_info, 'VideoMemoryType') else None,
                "video_architecture": gpu_info.VideoArchitecture if hasattr(gpu_info, 'VideoArchitecture') else None,
                "driver_date": gpu_info.DriverDate if hasattr(gpu_info, 'DriverDate') else None
            }
        }
    except Exception as e:
        logger.warning(f"Error getting Windows GPU information via WMI: {e}")
        return None

def get_gpu_info():
    """Get GPU utilization information"""
    try:
        # Load NVML library
        nvml_path = 'nvml.dll' if platform.system().lower() == 'windows' else 'libnvidia-ml.so.1'
        nvml = ctypes.CDLL(nvml_path)
        
        # Initialize NVML
        result = nvml.nvmlInit_v2()
        if result != 0:
            return {"available": False, "status": "Failed to initialize NVML", "memory": None}
        
        try:
            # Get device count
            device_count = ctypes.c_uint()
            result = nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
            if result != 0:
                return {"available": False, "status": "Failed to get device count", "memory": None}
            
            if device_count.value == 0:
                return {"available": False, "status": "No GPUs Found", "memory": None}
            
            # Get information for first GPU
            handle = ctypes.c_void_p()
            result = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
            if result != 0:
                return {"available": False, "status": "Failed to get GPU handle", "memory": None}
            
            # Get GPU name
            name_buffer = (ctypes.c_char * 96)()
            result = nvml.nvmlDeviceGetName(handle, name_buffer, 96)
            gpu_name = name_buffer.value.decode() if result == 0 else "Unknown"
            
            # Get memory info
            class c_nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)
                ]
            
            memory = c_nvmlMemory_t()
            result = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            memory_info = {
                "total_mb": memory.total / (1024 * 1024),
                "free_mb": memory.free / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024)
            } if result == 0 else None
            
            # Get utilization
            class nvmlUtilization_t(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint),
                          ("memory", ctypes.c_uint)]
            
            utilization = nvmlUtilization_t()
            result = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization))
            
            # Get temperature
            temperature = ctypes.c_uint()
            try:
                result = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temperature))
                temp_celsius = temperature.value if result == 0 else None
            except:
                temp_celsius = None
            
            # Get power usage
            power_usage = ctypes.c_uint()
            try:
                result = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power_usage))
                power_watts = power_usage.value / 1000.0 if result == 0 else None
            except:
                power_watts = None
            
            return {
                "available": True,
                "status": "GPU Available",
                "name": gpu_name,
                "memory": memory_info,
                "utilization": {
                    "gpu_percent": utilization.gpu if result == 0 else None,
                    "memory_percent": utilization.memory if result == 0 else None
                },
                "temperature_celsius": temp_celsius,
                "power_watts": power_watts
            }
            
        finally:
            try:
                nvml.nvmlShutdown()
            except:
                pass
            
    except Exception as e:
        logger.warning(f"Error getting GPU information: {e}")
        return {"available": False, "status": f"Error: {str(e)}", "memory": None}

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
        
        # Get GPU information using our cross-platform function
        gpu_info = get_gpu_info()
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_total_mb": memory_total_mb,
            "memory_used_mb": memory_used_mb,
            "gpu": gpu_info
        }
        
        # Add platform-specific information
        metrics["platform"] = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available_models")
async def list_available_models():
    """List all available model files in the models directory"""
    try:
        models_list = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dirs = [
            os.path.join(base_dir, "models"),
            os.path.join(os.path.dirname(__file__), "models")
        ]
        
        # Get current system memory
        memory = psutil.virtual_memory()
        system_memory = {
            "total": memory.total / (1024 * 1024),  # MB
            "available": memory.available / (1024 * 1024),  # MB
            "used": memory.used / (1024 * 1024)  # MB
        }
        
        # Get basic GPU memory info if available
        gpu_memory = None
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = {
                    "total": meminfo.total / (1024 * 1024),  # MB
                    "available": meminfo.free / (1024 * 1024),  # MB
                    "used": meminfo.used / (1024 * 1024)  # MB
                }
            except Exception as e:
                logger.warning(f"Error getting GPU memory info: {e}")
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith(".gguf"):
                        model_path = os.path.join(model_dir, file)
                        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                        required_memory = model_size * 2  # Estimated memory requirement
                        
                        # Convert filename to model ID
                        model_id = file.replace(".gguf", "")
                        model_id = re.sub(r"\.v\d+.*$", "", model_id)  # Remove version numbers
                        model_id = re.sub(r"\.Q\d.*$", "", model_id)   # Remove quantization suffix
                        
                        models_list.append({
                            "id": model_id,
                            "name": file,
                            "path": model_path,
                            "size_mb": model_size,
                            "required_memory_mb": required_memory,
                            "can_load": required_memory <= system_memory["available"]
                        })
        
        return {
            "models": models_list,
            "system_memory": system_memory,
            "gpu_memory": gpu_memory
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama.cpp API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port) 
