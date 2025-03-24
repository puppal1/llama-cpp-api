import logging
import platform
import os
import ctypes
import multiprocessing
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def get_cpu_info() -> Dict[str, Optional[str]]:
    """Get CPU information"""
    info = {
        "name": None,
        "architecture": platform.machine(),
        "cores_physical": None,
        "cores_logical": None,
        "features": []
    }
    
    system = platform.system().lower()
    
    try:
        info["cores_logical"] = multiprocessing.cpu_count()
    except:
        logger.warning("Could not determine logical CPU count")
    
    if system == "windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0")
            info["name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except Exception as e:
            logger.warning(f"Could not get CPU name from Windows registry: {e}")
            
    elif system == "darwin":  # macOS
        try:
            import subprocess
            output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
            info["name"] = output.strip()
            
            # Get physical core count
            output = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode()
            info["cores_physical"] = int(output.strip())
        except Exception as e:
            logger.warning(f"Could not get CPU info from sysctl: {e}")
            
    elif system == "linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                
            # Get CPU name
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    info["name"] = line.split(":")[1].strip()
                    break
                    
            # Get physical core count by counting unique core IDs
            core_ids = set()
            current_core = None
            for line in cpuinfo.split("\n"):
                if line.startswith("processor"):
                    if current_core is not None:
                        core_ids.add(tuple(current_core))
                    current_core = []
                elif current_core is not None:
                    if line.startswith("core id") or line.startswith("physical id"):
                        current_core.append(line.split(":")[1].strip())
            if current_core:
                core_ids.add(tuple(current_core))
            info["cores_physical"] = len(core_ids)
            
            # Get CPU features
            for line in cpuinfo.split("\n"):
                if "flags" in line:
                    info["features"] = line.split(":")[1].strip().split()
                    break
        except Exception as e:
            logger.warning(f"Could not read /proc/cpuinfo: {e}")
            
    return info

def get_cpu_memory() -> Dict[str, Optional[float]]:
    """Get CPU memory information in MB"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": vm.total / (1024 * 1024),  # Convert to MB
            "available": vm.available / (1024 * 1024),
            "used": vm.used / (1024 * 1024)
        }
    except ImportError:
        logger.warning("psutil not available - cannot get memory info")
        return {
            "total": None,
            "available": None,
            "used": None
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {
            "total": None,
            "available": None,
            "used": None
        }

# Initialize CPU info on module import
try:
    CPU_INFO = get_cpu_info()
    logger.info(f"CPU: {CPU_INFO['name']} ({CPU_INFO['cores_physical']} physical cores, {CPU_INFO['cores_logical']} logical cores)")
    
    MEMORY_INFO = get_cpu_memory()
    if MEMORY_INFO["total"] is not None:
        logger.info(f"Memory: {MEMORY_INFO['total']:.0f}MB total, {MEMORY_INFO['available']:.0f}MB available")
except Exception as e:
    logger.error(f"Error initializing CPU info: {e}")
    CPU_INFO = {
        "name": None,
        "architecture": platform.machine(),
        "cores_physical": None,
        "cores_logical": None,
        "features": []
    }
    MEMORY_INFO = {
        "total": None,
        "available": None,
        "used": None
    } 