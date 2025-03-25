from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from llama_cpp_api_package.routes2.model_routes import router as v2_model_router
from llama_cpp_api_package.routes2.metrics_routes import router as v2_metrics_router
from llama_cpp_api_package.routes2.chat_routes import router as v2_chat_router
from llama_cpp_api_package.routes2.model_cache import initialize_cache
from llama_cpp_api_package.utils.logging_config import setup_logging
from llama_cpp_api_package.utils.gpu_utils import GPU_AVAILABLE
import logging
import os
import socket
import json
import traceback
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Get the absolute path to the models directory
MODELS_DIR = Path.cwd() / "models"
if not MODELS_DIR.exists():
    logger.error(f"Models directory not found at {MODELS_DIR}")
    raise RuntimeError(f"Models directory not found at {MODELS_DIR}")

logger.info(f"Using models directory: {MODELS_DIR}")

# Get the absolute path to the docs directory
DOCS_DIR = os.path.join(os.path.dirname(__file__), "static", "docs")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    try:
        # Startup
        logger.info("Starting up API server...")
        models_dir = os.getenv("MODELS_DIR", str(MODELS_DIR))
        logger.info(f"Using models directory: {models_dir}")
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize model cache at startup
        initialize_cache(models_dir)
        
        # Ensure docs directory exists
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down API server...")
        logger.info("All models unloaded")

# Initialize FastAPI app
app = FastAPI(
    title="LLama.cpp API",
    description="REST API for interacting with LLama.cpp models",
    version="2.0.0",
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None,  # Disable default ReDoc
    lifespan=lifespan
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    error_msg = f"Unhandled exception: {str(exc)}"
    logger.error(f"{error_msg}\nRequest path: {request.url.path}\nTraceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error": "Internal Server Error",
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "request_path": str(request.url.path),
                "request_method": request.method
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for request validation errors"""
    error_msg = f"Validation error: {str(exc)}"
    logger.error(f"{error_msg}\nRequest path: {request.url.path}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "Validation Error",
                "message": str(exc),
                "type": type(exc).__name__,
                "errors": exc.errors(),
                "request_path": str(request.url.path),
                "request_method": request.method
            }
        }
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - only using v2 API routes now
app.include_router(v2_model_router, prefix="/api/v2")
app.include_router(v2_metrics_router, prefix="/api/v2")
app.include_router(v2_chat_router, prefix="/api/v2")

# Mount the docs directory to serve static files
app.mount("/api/v2/docs", StaticFiles(directory=DOCS_DIR), name="docs")

# Serve OpenAPI spec
@app.get("/api/v2/docs/openapi.yaml")
async def get_openapi_yaml():
    yaml_path = os.path.join(DOCS_DIR, "openapi.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=404, detail=f"OpenAPI specification not found at {yaml_path}")
    return FileResponse(yaml_path, media_type="text/yaml")

# Add compatibility routes (redirect from /api to /api/v2)
@app.get("/api/models")
async def models_redirect():
    """Redirect old API endpoint to v2"""
    return {"message": "API v1 is deprecated, please use /api/v2/models", "redirect": "/api/v2/models"}

@app.get("/api/models/{model_id}")
async def model_redirect(model_id: str):
    """Redirect old API endpoint to v2"""
    return {"message": "API v1 is deprecated, please use /api/v2/models/{model_id}", "redirect": f"/api/v2/models/{model_id}"}

@app.post("/api/models/{model_id}/load")
async def load_model_redirect(model_id: str):
    """Redirect old API endpoint to v2"""
    return {"message": "API v1 is deprecated, please use /api/v2/models/{model_id}/load", "redirect": f"/api/v2/models/{model_id}/load"}

@app.get("/")
def root():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    
    # Get development mode flag from environment variable or default to True
    dev_mode = os.getenv("DEV_MODE", "true").lower() == "true"
    
    # Set port from environment variable or default to 8001
    port = int(os.getenv("SERVER_PORT", "8001"))
    
    # Run with hot reload in development mode
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="debug", 
        reload=dev_mode,
        reload_dirs=["llama_cpp_api_package"]
    ) 