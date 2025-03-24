import logging
import platform
import os
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from functools import lru_cache
import psutil
import ctypes
import time

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

@dataclass
class SystemInfo:
    """System information including CPU, Memory, and GPU details"""
    cpu: Dict[str, Optional[Union[str, float, int]]]
    memory: Dict[str, Optional[float]]
    gpu: Dict[str, Optional[float]]
    gpu_name: Optional[str] = None
    gpu_available: bool = False
    gpu_layers: int = 0  # Number of layers that can be offloaded to GPU

@lru_cache(maxsize=1)
def get_system_info() -> SystemInfo:
    """Get system information (cached for 1 second)"""
    # Get fresh CPU and memory info
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
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
def get_cpu_info() -> Dict[str, Optional[Union[str, float, int]]]:
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
def get_memory_info() -> Dict[str, Optional[float]]:
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

def get_gpu_info() -> Dict[str, Optional[float]]:
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
        return get_gpu_info_alternative()
    except Exception as e:
        logger.warning(f"Error getting GPU info via pynvml: {e}")
        return get_gpu_info_alternative()

def get_gpu_info_alternative() -> Dict[str, Optional[float]]:
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

def get_gpu_name() -> Optional[str]:
    """Get GPU name using available methods"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle).decode()
    except:
        system = platform.system().lower()
        if system == "windows":
            try:
                import wmi
                w = wmi.WMI()
                return w.Win32_VideoController()[0].Name
            except:
                pass
        elif system == "darwin":
            try:
                import subprocess
                import json
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"]).decode()
                data = json.loads(output)
                return data.get("SPDisplaysDataType", [{}])[0].get("sppci_model")
            except:
                pass
    return None

def is_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except:
        # Check if we have any GPU info from alternative methods
        gpu_info = get_gpu_info_alternative()
        return gpu_info.get("total") is not None

# Don't initialize on import - let it be lazy
# The first call to get_system_info() will cache the results 