"""
Simple test script to verify model loading with RoPE parameters.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import struct
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', '2025-03-23', 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_u64(f) -> int:
    """Read a 64-bit unsigned integer from a file."""
    return struct.unpack('<Q', f.read(8))[0]

def read_string(f) -> str:
    """Read a length-prefixed string from a file."""
    length = read_u64(f)
    return f.read(length).decode('utf-8')

def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """Read and return model metadata."""
    try:
        llm = Llama(model_path=model_path, vocab_only=True, verbose=True)
        metadata = llm.model_metadata()
        return metadata
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return {}

def get_model_config(model_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Get model-specific configuration based on model name and metadata."""
    config = {
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "use_mmap": True,
        "use_mlock": False,
        "verbose": True
    }
    
    # Get RoPE dimensions from metadata if available
    rope_dims = metadata.get("llama.rope.dimension_count", 0)
    
    # Model-specific configurations
    if "ayla" in model_name.lower():
        config.update({
            "rope_dimension_count": rope_dims or 128,  # Use metadata or fallback to 128
            "rope_freq_base": 1000000.0,
            "n_ctx": metadata.get("context_length", 100000),  # Use metadata context length or default to 100k
            "tensor_split": None
        })
    elif "deepseek" in model_name.lower():
        config.update({
            "rope_freq_base": 10000.0,
            "rope_scaling_type": "linear",
            "n_ctx": 4096
        })
    elif "moe" in model_name.lower():
        config.update({
            "n_threads": 4,
            "n_batch": 256,
            "n_gpu_layers": 0,
            "tensor_split": None
        })
    elif "mistral" in model_name.lower():
        config.update({
            "n_ctx": 4096,
            "n_batch": 512
        })
    
    return config

def test_model(model_path: str, verbose: bool = False) -> bool:
    """Test loading and basic inference with a model."""
    try:
        print(f"\nTesting model: {model_path}")
        
        # Read model metadata
        metadata = get_model_metadata(model_path)
        if verbose:
            print("\nModel metadata:")
            print(json.dumps(metadata, indent=2))
        
        # Get model configuration
        config = get_model_config(os.path.basename(model_path), metadata)
        if verbose:
            print("\nUsing configuration:")
            print(json.dumps(config, indent=2))
        
        # Initialize model
        llm = Llama(model_path=model_path, **config)
        
        # Test basic inference
        prompt = "Hello, I am a language model."
        print("\nTesting inference with prompt:", prompt)
        
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1
        )
        
        if output and 'choices' in output and len(output['choices']) > 0:
            print("\nModel response:", output['choices'][0]['message']['content'])
            print("\nTest completed successfully!")
            return True
        else:
            print("\nError: No valid response generated")
            return False
            
    except Exception as e:
        print(f"\nError testing model: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test GGUF model loading and inference")
    parser.add_argument("--model", type=str, help="Path to the GGUF model file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if not args.model:
        print("Error: Please specify a model path")
        sys.exit(1)
    
    success = test_model(args.model, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 