"""
Test script for model loading/unloading via API.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

from api.models import ModelLoader

def test_model(model_path: str, context_size: int = None) -> Dict[str, Any]:
    """Test loading and basic inference for a model."""
    loader = ModelLoader()
    results = {
        "model_path": str(model_path),  # Convert Path to string
        "timestamps": {},
        "memory": {},
        "metadata": None,
        "errors": None
    }
    
    try:
        # Record start time
        start_time = time.time()
        results["timestamps"]["start"] = start_time
        
        # Load model and get metadata
        print(f"\nTesting model: {Path(model_path).name}")
        print("Loading model...")
        metadata = loader.load_model(model_path, context_size=context_size)
        results["metadata"] = metadata
        
        # Record load time and memory
        load_time = time.time() - start_time
        results["timestamps"]["load_complete"] = time.time()
        results["timestamps"]["load_duration"] = load_time
        results["memory"]["after_load"] = loader.resource_manager.get_memory_stats()
        
        print(f"Model loaded in {load_time:.2f}s")
        print("Memory usage:", results["memory"]["after_load"])
        
        # Test basic inference
        print("\nTesting inference...")
        test_prompt = "Write a haiku about AI:"
        inference_start = time.time()
        
        if loader.current_model is not None:
            output = loader.current_model.create_completion(
                test_prompt,
                max_tokens=50,
                temperature=0.7,
                stop=["###"]
            )
            
            inference_time = time.time() - inference_start
            results["timestamps"]["inference_duration"] = inference_time
            results["memory"]["after_inference"] = loader.resource_manager.get_memory_stats()
            
            # Convert completion output to serializable format
            results["inference_output"] = {
                "text": output["choices"][0]["text"] if output["choices"] else "",
                "tokens_generated": len(output["choices"][0]["text"].split()) if output["choices"] else 0,
                "finish_reason": output["choices"][0].get("finish_reason", None) if output["choices"] else None
            }
            
            print(f"Inference completed in {inference_time:.2f}s")
            print("Memory usage:", results["memory"]["after_inference"])
            print("\nOutput:", results["inference_output"]["text"])
        
        # Unload model
        print("\nUnloading model...")
        unload_start = time.time()
        loader.unload_model()
        
        unload_time = time.time() - unload_start
        results["timestamps"]["unload_duration"] = unload_time
        results["memory"]["after_unload"] = loader.resource_manager.get_memory_stats()
        
        print(f"Model unloaded in {unload_time:.2f}s")
        print("Final memory usage:", results["memory"]["after_unload"])
        
    except Exception as e:
        results["errors"] = str(e)
        print(f"Error: {e}")
        loader.unload_model()
    
    return results

def main():
    # Define models to test
    models_dir = Path("models")
    test_models = [
        {
            "path": str(models_dir / "M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf"),
            "context_size": 4096
        },
        {
            "path": str(models_dir / "Ayla-Light-12B-v2.Q4_K_M.gguf"),
            "context_size": 8192
        }
    ]
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Test each model
    all_results = {}
    for model_config in test_models:
        if not Path(model_config["path"]).exists():
            print(f"Model not found: {model_config['path']}")
            continue
            
        results = test_model(
            model_config["path"],
            context_size=model_config["context_size"]
        )
        
        # Save individual model results
        model_name = Path(model_config["path"]).stem
        result_path = results_dir / f"{model_name}_api_test.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # Use str as default serializer
            
        all_results[model_name] = results
        
    # Save summary
    summary_path = results_dir / "api_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)  # Use str as default serializer
        
if __name__ == "__main__":
    main() 