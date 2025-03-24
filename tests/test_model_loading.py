import os
import psutil
import time
from llama_cpp import Llama
from contextlib import contextmanager

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

@contextmanager
def measure_time():
    """Context manager to measure execution time"""
    start = time.time()
    yield
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

def test_model_loading(model_path, n_threads=1, n_batch=512):
    """Test loading and basic inference with a model"""
    print(f"\nTesting model: {os.path.basename(model_path)}")
    print("-" * 60)
    
    # Record initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        # Load model
        print("\nLoading model...")
        with measure_time():
            model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Reasonable context size
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=False
            )
        
        # Check memory after loading
        load_memory = get_memory_usage()
        print(f"Memory usage after loading: {load_memory:.2f} MB")
        print(f"Memory increase: {load_memory - initial_memory:.2f} MB")
        
        # Test basic inference
        print("\nTesting inference...")
        test_prompt = "Write a short poem about AI."
        with measure_time():
            output = model.create_completion(
                test_prompt,
                max_tokens=50,
                stop=["</s>"],
                echo=False
            )
        
        # Check memory after inference
        infer_memory = get_memory_usage()
        print(f"Memory usage after inference: {infer_memory:.2f} MB")
        print(f"Memory increase from load: {infer_memory - load_memory:.2f} MB")
        
        # Clean up
        print("\nCleaning up...")
        del model
        time.sleep(1)  # Give some time for cleanup
        
        # Final memory check
        final_memory = get_memory_usage()
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory difference from start: {final_memory - initial_memory:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def main():
    # List of models to test
    models = [
        "models/M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf",
        "models/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_S.gguf",
        "models/Ayla-Light-12B-v2.Q4_K_M.gguf"
    ]
    
    # Test each model
    results = []
    for model_path in models:
        if os.path.exists(model_path):
            success = test_model_loading(model_path)
            results.append((os.path.basename(model_path), success))
        else:
            print(f"\nSkipping {model_path} - file not found")
            results.append((os.path.basename(model_path), False))
    
    # Print summary
    print("\nTest Summary")
    print("-" * 60)
    for model, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{model}: {status}")

if __name__ == "__main__":
    main() 