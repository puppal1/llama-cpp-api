from fastapi import APIRouter, HTTPException
import psutil
import logging
from ..utils.gpu import get_gpu_memory
from ..models.model_manager import ModelManager

router = APIRouter()
model_manager = ModelManager()
logger = logging.getLogger(__name__)

@router.get("/api/metrics")
async def get_system_metrics():
    """Get system resource metrics"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        # Get GPU memory if available
        gpu_memory = get_gpu_memory() if hasattr(model_manager, 'gpu_memory') else None
        
        # Calculate memory used by loaded models
        model_memory = sum(
            model.memory_used 
            for model in model_manager.models.values()
        )
        
        return {
            "cpu": {
                "percent": cpu_percent
            },
            "memory": {
                "used_mb": memory_used_mb,
                "total_mb": memory_total_mb,
                "percent": memory.percent
            },
            "gpu": {
                "memory_mb": gpu_memory if gpu_memory else 0
            },
            "models": {
                "count": len(model_manager.models),
                "memory_mb": model_memory
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 