import logging
import ctypes
import os
import platform
from typing import Optional, Dict

logger = logging.getLogger(__name__)

def get_nvml_path() -> str:
    """Get the path to the NVML library"""
    if platform.system() == "Windows":
        # Try multiple possible paths for nvml.dll
        paths = [
            os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "System32", "nvml.dll"),
            os.path.join(os.environ.get("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvml.dll"),
            "nvml.dll"  # Let Windows search PATH
        ]
        
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Found NVML at: {path}")
                return path
                
        logger.warning(f"NVML not found in any of: {paths}")
        return paths[0]  # Return first path as default
    else:
        return "libnvidia-ml.so.1"

def get_gpu_memory() -> Dict[str, Optional[float]]:
    """Get GPU memory information using ctypes"""
    try:
        nvml_path = get_nvml_path()
        if not os.path.exists(nvml_path):
            logger.warning(f"NVML library not found at {nvml_path}")
            return {"available": None, "total": None, "used": None}
            
        nvml = ctypes.CDLL(nvml_path)
        
        # Initialize NVML
        result = nvml.nvmlInit_v2()
        if result != 0:
            logger.warning("Failed to initialize NVML")
            return {"available": None, "total": None, "used": None}
            
        try:
            # Get device count
            device_count = ctypes.c_uint()
            result = nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
            if result != 0:
                logger.warning("Failed to get device count")
                return {"available": None, "total": None, "used": None}
                
            if device_count.value == 0:
                logger.info("No NVIDIA GPUs found")
                return {"available": None, "total": None, "used": None}
                
            # Get first GPU handle
            handle = ctypes.c_void_p()
            result = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
            if result != 0:
                logger.warning("Failed to get GPU handle")
                return {"available": None, "total": None, "used": None}
                
            # Define memory info structure
            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)
                ]
                
            # Get memory info
            memory = nvmlMemory_t()
            result = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            if result != 0:
                logger.warning("Failed to get memory info")
                return {"available": None, "total": None, "used": None}
                
            return {
                "available": memory.free / (1024 * 1024),  # Convert to MB
                "total": memory.total / (1024 * 1024),
                "used": memory.used / (1024 * 1024)
            }
            
        finally:
            # Always try to shut down NVML
            try:
                nvml.nvmlShutdown()
            except:
                pass
                
    except Exception as e:
        logger.warning(f"Error getting GPU memory info: {str(e)}")
        return {"available": None, "total": None, "used": None}

# Initialize GPU detection on module import
try:
    gpu_info = get_gpu_memory()
    GPU_AVAILABLE = all(v is not None for v in gpu_info.values())
    if GPU_AVAILABLE:
        logger.info("GPU detected and available")
    else:
        logger.info("No GPU detected or not available")
except Exception as e:
    logger.warning(f"GPU detection failed: {e}")
    GPU_AVAILABLE = False 