#!/usr/bin/env python3
"""
Test runner for llama.cpp API
"""

import asyncio
import sys
from pathlib import Path
from tests.test_llama_api import test_model

async def run_tests(model_path: str):
    """Run tests with both CPU and GPU configurations"""
    print("=== Starting llama.cpp API Tests ===")
    
    # Ensure model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    try:
        # Test CPU configuration
        print("\n=== Testing CPU Configuration ===")
        await test_model(model_path, use_gpu=False)
        
        # Test GPU configuration if available
        print("\n=== Testing GPU Configuration ===")
        await test_model(model_path, use_gpu=True)
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_tests.py <path_to_model.gguf>")
        sys.exit(1)
    
    asyncio.run(run_tests(sys.argv[1])) 