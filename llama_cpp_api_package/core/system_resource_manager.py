"""
System resource manager for handling system metrics.
"""
import logging
import platform
import os
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
import psutil
import ctypes
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Cache NVML initialization status
_nvml_initialized = False

def _init_nvml():
    """Initialize NVML with portable path handling"""
    global _nvml_initialized
    if _nvml_initialized:
        return True
        
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize NVML: {e}")
        return False

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

@dataclass
class SystemInfo:
    """System information including CPU, Memory, and GPU details"""
    cpu: Dict[str, Optional[Union[str, float, int]]]
    memory: Dict[str, Optional[float]]
    gpu: Dict[str, Optional[float]]
    gpu_name: Optional[str] = None
    gpu_available: bool = False
    gpu_layers: int = 0  # Number of layers that can be offloaded to GPU

class SystemResourceManager:
    """Manages system resources and provides real-time metrics"""
    
    def __init__(self):
        self._last_update = None
        self._cached_metrics = None
        self._cache_ttl = 1.0  # Cache TTL in seconds
        
    def get_system_metrics(self) -> Dict:
        """Get real-time system metrics"""
        current_time = datetime.now()
        
        # Return cached metrics if within TTL
        if (self._cached_metrics and self._last_update and 
            (current_time - self._last_update).total_seconds() < self._cache_ttl):
            return self._cached_metrics
            
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            
            # Get memory metrics
            memory = psutil.virtual_memory()
            total_memory = memory.total / (1024 * 1024)  # Convert to MB
            available_memory = memory.available / (1024 * 1024)
            used_memory = memory.used / (1024 * 1024)  # Convert to MB
            
            # Get GPU metrics if available
            gpu_metrics = self._get_gpu_metrics()
            
            metrics = {
                "cpu": {
                    "utilization": cpu_percent,
                    "cores": {
                        "physical": physical_cores,
                        "logical": logical_cores
                    },
                    "frequency": {
                        "current": cpu_freq.current if cpu_freq else None,
                        "min": cpu_freq.min if cpu_freq else None,
                        "max": cpu_freq.max if cpu_freq else None
                    }
                },
                "memory": {
                    "total_mb": total_memory,
                    "available_mb": available_memory,
                    "used_mb": used_memory,
                    "percent": memory.percent
                },
                "gpu": gpu_metrics,
                "timestamp": current_time.isoformat()
            }
            
            # Cache the metrics
            self._cached_metrics = metrics
            self._last_update = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            raise
            
    def _get_gpu_metrics(self) -> Dict:
        """Get GPU metrics using NVML"""
        try:
            nvml_path = get_nvml_path()
            if not os.path.exists(nvml_path):
                logger.warning(f"NVML library not found at {nvml_path}")
                return {
                    "available": False,
                    "name": None,
                    "memory": {
                        "total_mb": 0,
                        "used_mb": 0,
                        "percent": 0
                    }
                }
                
            nvml = ctypes.CDLL(nvml_path)
            
            # Initialize NVML
            result = nvml.nvmlInit_v2()
            if result != 0:
                logger.warning("Failed to initialize NVML")
                return {
                    "available": False,
                    "name": None,
                    "memory": {
                        "total_mb": 0,
                        "used_mb": 0,
                        "percent": 0
                    }
                }
                
            try:
                # Get device count
                device_count = ctypes.c_uint()
                result = nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
                if result != 0:
                    logger.warning("Failed to get device count")
                    return {
                        "available": False,
                        "name": None,
                        "memory": {
                            "total_mb": 0,
                            "used_mb": 0,
                            "percent": 0
                        }
                    }
                    
                if device_count.value == 0:
                    logger.info("No NVIDIA GPUs found")
                    return {
                        "available": False,
                        "name": None,
                        "memory": {
                            "total_mb": 0,
                            "used_mb": 0,
                            "percent": 0
                        }
                    }
                    
                # Get first GPU handle
                handle = ctypes.c_void_p()
                result = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
                if result != 0:
                    logger.warning("Failed to get GPU handle")
                    return {
                        "available": False,
                        "name": None,
                        "memory": {
                            "total_mb": 0,
                            "used_mb": 0,
                            "percent": 0
                        }
                    }
                    
                # Get GPU name
                name = ctypes.create_string_buffer(64)
                result = nvml.nvmlDeviceGetName(handle, name, ctypes.c_uint(64))
                if result != 0:
                    logger.warning("Failed to get GPU name")
                    gpu_name = None
                else:
                    gpu_name = name.value.decode('utf-8')
                    
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
                    return {
                        "available": False,
                        "name": gpu_name,
                        "memory": {
                            "total_mb": 0,
                            "used_mb": 0,
                            "percent": 0
                        }
                    }
                    
                return {
                    "available": True,
                    "name": gpu_name,
                    "memory": {
                        "total_mb": memory.total / (1024 * 1024),
                        "used_mb": memory.used / (1024 * 1024),
                        "percent": (memory.used / memory.total) * 100
                    }
                }
                
            finally:
                # Always try to shut down NVML
                try:
                    nvml.nvmlShutdown()
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            return {
                "available": False,
                "name": None,
                "memory": {
                    "total_mb": 0,
                    "used_mb": 0,
                    "percent": 0
                }
            }
            
    def check_memory_availability(self, model_size_mb: float, context_size: int) -> Tuple[bool, str]:
        """Check if there's enough memory to load a model"""
        try:
            # Get current memory state
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)
            
            # Calculate required memory (model size + context + safety margin)
            required_memory = model_size_mb * 1.5  # Base multiplier
            required_memory += (context_size * 0.1)  # Context memory (rough estimate)
            required_memory *= 1.2  # Safety margin
            
            # Check if we have enough memory
            can_load = required_memory <= available_memory_mb
            
            message = (
                f"Required: {required_memory:.1f}MB, "
                f"Available: {available_memory_mb:.1f}MB"
            )
            
            return can_load, message
            
        except Exception as e:
            logger.error(f"Error checking memory availability: {e}")
            return False, f"Error checking memory: {str(e)}"
            
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB"""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting available memory: {e}")
            return 0.0
        
    @lru_cache(maxsize=1)
    def get_system_info(self) -> SystemInfo:
        """Get system information (cached for 1 second)"""
        # Get fresh CPU and memory info
        cpu_info = self._get_cpu_info()
        memory_info = self._get_memory_info()
        gpu_info = {}
        gpu_name = None
        gpu_available = False
        gpu_layers = 0
        
        # Try to initialize GPU info
        if _init_nvml():
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info = {
                    "total": info.total / 1024 / 1024,  # Convert to MB
                    "free": info.free / 1024 / 1024,
                    "used": info.used / 1024 / 1024
                }
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_available = True
                gpu_layers = 32  # Default for NVIDIA GPUs
                logger.info(f"GPU detected: {gpu_name}")
            except Exception as e:
                logger.warning(f"Error getting GPU info via pynvml: {e}")
                gpu_available = False
        else:
            logger.warning("NVML initialization failed - GPU features will be disabled")
            
        return SystemInfo(
            cpu=cpu_info,
            memory=memory_info,
            gpu=gpu_info,
            gpu_name=gpu_name,
            gpu_available=gpu_available,
            gpu_layers=gpu_layers
        )
        
    @lru_cache(maxsize=1)
    def _get_cpu_info(self) -> Dict[str, Optional[Union[str, float, int]]]:
        """Get CPU information using psutil and platform-specific methods (cached)"""
        info = {
            "name": None,
            "architecture": platform.machine(),
            "cores_physical": None,
            "cores_logical": None,
            "features": [],
            "utilization": 0.0
        }
        
        try:
            import psutil
            
            # Get CPU info from psutil
            cpu_info = psutil.cpu_freq()
            if cpu_info:
                info["frequency"] = f"{cpu_info.current:.2f}MHz"
                
            info["cores_physical"] = psutil.cpu_count(logical=False)
            info["cores_logical"] = psutil.cpu_count(logical=True)
            
            # Get fresh CPU utilization (average across all cores)
            try:
                # Get per-CPU utilization
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
                # Calculate average
                info["utilization"] = sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0.0
                # Get per-core frequency
                freq_info = psutil.cpu_freq(percpu=True)
                if freq_info:
                    info["core_frequencies"] = [f"{f.current:.0f}MHz" for f in freq_info]
            except Exception as e:
                logger.warning(f"Could not get detailed CPU metrics: {e}")
                info["utilization"] = 0.0
                
            # Get CPU name from platform-specific methods
            system = platform.system().lower()
            if system == "windows":
                try:
                    import wmi
                    w = wmi.WMI()
                    info["name"] = w.Win32_Processor()[0].Name.strip()
                except Exception as e:
                    logger.warning(f"Could not get CPU name using WMI: {e}")
                    
            elif system == "darwin":
                try:
                    import subprocess
                    output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
                    info["name"] = output.strip()
                except Exception as e:
                    logger.warning(f"Could not get CPU name using sysctl: {e}")
                    
            elif system == "linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                info["name"] = line.split(":")[1].strip()
                                break
                except Exception as e:
                    logger.warning(f"Could not get CPU name from /proc/cpuinfo: {e}")
                    
        except ImportError:
            logger.warning("psutil not available, falling back to basic CPU info")
            # Fallback to basic platform info
            import multiprocessing
            info["cores_logical"] = multiprocessing.cpu_count()
            info["utilization"] = 0.0
            
        return info
        
    @lru_cache(maxsize=1)
    def _get_memory_info(self) -> Dict[str, Optional[float]]:
        """Get memory information using psutil (cached)"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total": vm.total / (1024 * 1024),  # Convert to MB
                "available": vm.available / (1024 * 1024),
                "used": vm.used / (1024 * 1024),
                "percent": vm.percent
            }
        except ImportError:
            logger.warning("psutil not available - cannot get detailed memory info")
            return {
                "total": None,
                "available": None,
                "used": None,
                "percent": None
            }
            
    def _get_gpu_info(self) -> Dict[str, Optional[float]]:
        """Get GPU memory information using pynvml"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Get first GPU (we currently only use one)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "total": info.total / (1024 * 1024),  # Convert to MB
                "free": info.free / (1024 * 1024),
                "used": info.used / (1024 * 1024)
            }
        except ImportError:
            logger.info("pynvml not available - checking alternative methods")
            return self._get_gpu_info_alternative()
        except Exception as e:
            logger.warning(f"Error getting GPU info via pynvml: {e}")
            return self._get_gpu_info_alternative()
            
    def _get_gpu_info_alternative(self) -> Dict[str, Optional[float]]:
        """Alternative GPU detection methods using platform-specific approaches"""
        system = platform.system().lower()
        
        if system == "windows":
            try:
                import wmi
                w = wmi.WMI()
                gpu = w.Win32_VideoController()[0]
                return {
                    "total": float(gpu.AdapterRAM) / (1024 * 1024) if hasattr(gpu, 'AdapterRAM') else None,
                    "free": None,  # WMI doesn't provide this
                    "used": None   # WMI doesn't provide this
                }
            except Exception as e:
                logger.warning(f"Could not get GPU info using WMI: {e}")
                
        elif system == "darwin":
            try:
                import subprocess
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"]).decode()
                import json
                data = json.loads(output)
                gpu_info = data.get("SPDisplaysDataType", [{}])[0]
                
                if "sppci_memory" in gpu_info:
                    memory = gpu_info["sppci_memory"].lower()
                    if "mb" in memory:
                        total = float(memory.replace("mb", "").strip())
                    elif "gb" in memory:
                        total = float(memory.replace("gb", "").strip()) * 1024
                    else:
                        total = None
                        
                    return {
                        "total": total,
                        "free": None,  # macOS doesn't provide this
                        "used": None   # macOS doesn't provide this
                    }
            except Exception as e:
                logger.warning(f"Could not get GPU info using system_profiler: {e}")
                
        return {
            "total": None,
            "free": None,
            "used": None
        } 