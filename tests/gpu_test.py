import psutil
import sys
import os
import platform
from datetime import datetime
import ctypes
from ctypes import *

def get_size(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024

def get_mac_gpu_info():
    """Get GPU information on macOS using system_profiler"""
    try:
        import subprocess
        import json
        
        # Use system_profiler to get GPU information
        cmd = ['system_profiler', 'SPDisplaysDataType', '-json']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Failed to run system_profiler")
            
        data = json.loads(result.stdout)
        gpu_info = data.get('SPDisplaysDataType', [])
        
        if not gpu_info:
            return
            
        for gpu in gpu_info:
            print(f"\nGPU: {gpu.get('sppci_model', 'Unknown')}")
            
            # Check for Apple Silicon GPU
            if 'Apple' in gpu.get('sppci_model', ''):
                print("  Type: Apple Silicon GPU")
                metal_support = gpu.get('spmetal_supported', 'Unknown')
                print(f"  Metal Support: {metal_support}")
                if 'vram_shared' in gpu:
                    print(f"  Shared Memory: {gpu['vram_shared']}")
            else:
                # Intel/Other GPU
                if 'sppci_memory' in gpu:
                    print(f"  Memory: {gpu['sppci_memory']}")
                
            # Display additional info if available
            if 'gpu_memory' in gpu:
                print(f"  GPU Memory: {gpu['gpu_memory']}")
            if 'metal_version' in gpu:
                print(f"  Metal Version: {gpu['metal_version']}")
            
    except Exception as e:
        print(f"Error getting macOS GPU information: {e}")

def get_nvml_path():
    """Get the NVML library path based on the operating system"""
    system = platform.system().lower()
    
    if system == 'windows':
        return 'nvml.dll'
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
                return path
        return 'libnvidia-ml.so'  # Default to the basic name if no paths found
    else:
        raise OSError(f"NVML not supported on: {system}")

def get_gpu_info():
    """Get GPU utilization information"""
    print("\n3. GPU Information:")
    print("-" * 20)
    
    system = platform.system().lower()
    
    # Handle macOS separately
    if system == 'darwin':
        get_mac_gpu_info()
        return
    
    try:
        # Get the appropriate NVML library path
        nvml_path = get_nvml_path()
        print(f"Loading NVML library from: {nvml_path}")
        
        # Load the NVML library
        try:
            nvml = ctypes.CDLL(nvml_path)
        except Exception as e:
            print(f"Failed to load NVML library: {e}")
            print("GPU information not available")
            return
        
        # Initialize NVML
        result = nvml.nvmlInit_v2()
        if result != 0:
            raise Exception(f"Failed to initialize NVML: {result}")
        
        # Get device count
        device_count = c_uint()
        result = nvml.nvmlDeviceGetCount_v2(byref(device_count))
        if result != 0:
            raise Exception(f"Failed to get device count: {result}")
            
        print(f"Found {device_count.value} GPU(s)")
        
        # Get information for each GPU
        for i in range(device_count.value):
            handle = c_void_p()
            result = nvml.nvmlDeviceGetHandleByIndex_v2(i, byref(handle))
            if result != 0:
                print(f"Failed to get handle for GPU {i}")
                continue
                
            # Get memory info
            memory = c_ulonglong * 3  # total, free, used
            mem_info = memory()
            result = nvml.nvmlDeviceGetMemoryInfo(handle, byref(mem_info))
            if result == 0:
                print(f"\nGPU {i} Memory:")
                print(f"  Total: {get_size(mem_info[0])}")
                print(f"  Free:  {get_size(mem_info[1])}")
                print(f"  Used:  {get_size(mem_info[2])}")
            
            # Get utilization info
            class nvmlUtilization_t(Structure):
                _fields_ = [("gpu", c_uint),
                          ("memory", c_uint)]
            
            utilization = nvmlUtilization_t()
            result = nvml.nvmlDeviceGetUtilizationRates(handle, byref(utilization))
            if result == 0:
                print(f"  Utilization:")
                print(f"    GPU: {utilization.gpu}%")
                print(f"    Memory: {utilization.memory}%")
            
    except Exception as e:
        print(f"Error getting GPU information: {e}")
    finally:
        try:
            if 'nvml' in locals():
                nvml.nvmlShutdown()
        except:
            pass

def test_system_resources():
    print("\nSystem Resource Test Results")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # CPU Information
    print("\n1. CPU Information:")
    print("-" * 20)
    print(f"CPU Cores: {psutil.cpu_count()} (Physical: {psutil.cpu_count(logical=False)})")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # CPU Frequency
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"CPU Frequency:")
        print(f"  Current: {cpu_freq.current:.2f} MHz")
        print(f"  Min: {cpu_freq.min:.2f} MHz")
        print(f"  Max: {cpu_freq.max:.2f} MHz")
    
    # Memory Information
    print("\n2. System Memory:")
    print("-" * 20)
    vm = psutil.virtual_memory()
    print(f"Total: {get_size(vm.total)}")
    print(f"Available: {get_size(vm.available)}")
    print(f"Used: {get_size(vm.used)} ({vm.percent}%)")
    print(f"Free: {get_size(vm.free)}")
    
    # GPU Information
    get_gpu_info()
    
    # Disk Information
    print("\n4. Disk Information:")
    print("-" * 20)
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            print(f"\nDevice: {partition.device}")
            print(f"  Mountpoint: {partition.mountpoint}")
            print(f"  File System: {partition.fstype}")
            print(f"  Total: {get_size(partition_usage.total)}")
            print(f"  Used: {get_size(partition_usage.used)} ({partition_usage.percent}%)")
            print(f"  Free: {get_size(partition_usage.free)}")
        except Exception as e:
            print(f"  Error reading partition {partition.device}: {e}")

    # Network Information
    print("\n5. Network Information:")
    print("-" * 20)
    net_io = psutil.net_io_counters()
    print(f"Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Bytes Received: {get_size(net_io.bytes_recv)}")
    
    # Process Information
    print("\n6. Process Information:")
    print("-" * 20)
    process = psutil.Process()
    print(f"Current Process:")
    print(f"  CPU Usage: {process.cpu_percent()}%")
    print(f"  Memory Usage: {get_size(process.memory_info().rss)}")
    print(f"  Threads: {process.num_threads()}")

if __name__ == "__main__":
    test_system_resources() 