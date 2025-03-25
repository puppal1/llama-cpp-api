from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Get the absolute path to the docs directory
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

app = FastAPI(
    title="LLama.cpp API",
    description="REST API for interacting with LLama.cpp models",
    version="2.0.0",
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None  # Disable default ReDoc
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the docs directory to serve static files
app.mount("/api/v2/docs", StaticFiles(directory=DOCS_DIR), name="docs")

# Serve OpenAPI spec
@app.get("/api/v2/docs/openapi.yaml")
async def get_openapi_yaml():
    yaml_path = os.path.join(DOCS_DIR, "openapi.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=404, detail=f"OpenAPI specification not found at {yaml_path}")
    return FileResponse(yaml_path, media_type="text/yaml")

# ... existing code ... 