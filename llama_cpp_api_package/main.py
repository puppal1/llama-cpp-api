from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import model_router, chat_router, metrics_router
from .routes2.model_routes import router as v2_model_router, _initialize_model_cache
from .utils.logging_config import setup_logging
from .utils.gpu_utils import GPU_AVAILABLE
import logging
import os
import socket
import json

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up API server...")
    models_dir = os.getenv("MODELS_DIR", "models")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Using models directory: {models_dir}")
    
    # Initialize model cache at startup
    _initialize_model_cache(models_dir)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    from .models.model_manager import model_manager
    model_manager.unload_all_models()
    logger.info("All models unloaded")

# Initialize FastAPI app
app = FastAPI(
    title="Llama CPP API",
    description="REST API for Llama.cpp models",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(v2_model_router)

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 