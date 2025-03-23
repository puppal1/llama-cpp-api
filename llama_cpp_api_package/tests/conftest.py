import pytest
import os

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    # Store original env vars
    original_models_dir = os.getenv("MODELS_DIR")
    
    # Set test env vars
    os.environ["MODELS_DIR"] = "models"  # Use default models directory
    
    yield
    
    # Restore original env vars
    if original_models_dir:
        os.environ["MODELS_DIR"] = original_models_dir
    else:
        del os.environ["MODELS_DIR"] 