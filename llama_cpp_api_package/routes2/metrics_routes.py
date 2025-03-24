"""
Metrics routes for system resource monitoring.
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict
from ..core.system_resource_manager import SystemResourceManager

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/metrics",
    tags=["v2-metrics"]
)

# Initialize system resource manager
system_resource_manager = SystemResourceManager()

@router.get("/")
def get_system_metrics() -> Dict:
    """Get system metrics including CPU, memory, and GPU information."""
    try:
        metrics = system_resource_manager.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get system metrics",
                "message": str(e)
            }
        ) 