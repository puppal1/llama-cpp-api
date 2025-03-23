import os
import pytest
from fastapi.testclient import TestClient
from llama_cpp_api_package.main import app
from llama_cpp_api_package.models.types import ModelParameters, ModelStatus

client = TestClient(app)

def test_list_models():
    """Test GET /api/models endpoint"""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "models" in data
    assert "system_state" in data
    assert "available" in data["models"]
    assert "loaded" in data["models"]
    
    # Check system state
    assert "cpu" in data["system_state"]
    assert "memory" in data["system_state"]
    assert "gpu" in data["system_state"]
    
    # Check memory info
    memory = data["system_state"]["memory"]
    assert "total_gb" in memory
    assert "used_gb" in memory
    assert "model_memory_gb" in memory
    assert memory["model_memory_gb"] >= 0
    
    # Check available models
    for model in data["models"]["available"]:
        assert "id" in model
        assert "name" in model
        assert "path" in model
        assert "size_mb" in model
        assert "required_memory_mb" in model
        assert "can_load" in model
        assert model["size_mb"] > 0
        assert model["required_memory_mb"] > 0

def test_load_nonexistent_model():
    """Test loading a model that doesn't exist"""
    params = ModelParameters(
        num_ctx=512,
        num_batch=512,
        num_thread=4,
        num_gpu=0
    )
    
    response = client.post("/api/models/nonexistent_model/load", json=params.dict())
    assert response.status_code == 404

def test_unload_nonexistent_model():
    """Test unloading a model that isn't loaded"""
    response = client.post("/api/models/nonexistent_model/unload")
    assert response.status_code == 500  # Should return error for unloading non-loaded model 