import logging
import platform
import os
import ctypes
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Initialize GPU detection
GPU_AVAILABLE = False
GPU_NAME = None
GPU_INFO: Dict = {}

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    logger.info("GPU support enabled")
except ImportError:
    logger.info("pynvml not installed - running in CPU mode")
    GPU_AVAILABLE = False
except Exception as e:
    if "NVML Shared Library Not Found" in str(e):
        logger.info("Running in CPU-only mode (NVML library not found)")
    else:
        logger.warning(f"GPU detection failed: {str(e)} - running in CPU mode")
    GPU_AVAILABLE = False

def init_gpu():
    """Initialize GPU detection"""
    global GPU_AVAILABLE, GPU_NAME, GPU_INFO
    
    try:
        # Try to import NVML for NVIDIA GPUs
        import pynvml
        pynvml.nvmlInit()
        
        # Get the first GPU device
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        
        GPU_AVAILABLE = True
        GPU_NAME = name
        GPU_INFO = {
            "total": info.total / (1024 * 1024),  # Convert to MB
            "free": info.free / (1024 * 1024),
            "used": info.used / (1024 * 1024)
        }
        
        logger.info(f"GPU detected: {name}")
        
    except ImportError:
        logger.warning("pynvml not installed - GPU features will be disabled")
        GPU_AVAILABLE = False
    except Exception as e:
        if "NVML Shared Library Not Found" in str(e):
            logger.info("Running in CPU-only mode (NVML library not found)")
        else:
            logger.warning(f"GPU detection failed: {str(e)} - running in CPU mode")
        GPU_AVAILABLE = False

def detect_gpu():
    """
    Detect if GPU is available and return the number of devices.
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            logger.info(f"Found {device_count} GPU device(s)")
            return True
        else:
            logger.info("No GPU devices found")
            return False
    except ImportError:
        logger.info("pynvml not available - running in CPU mode")
        return False
    except Exception as e:
        logger.warning(f"GPU detection failed: {e} - running in CPU mode")
        return False

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
        nvml_path = get_nvml_path()
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

# Initialize GPU detection on module import
init_gpu() 