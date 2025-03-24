import os
import sys
import psutil
import time
import json
from io import StringIO
from contextlib import contextmanager, redirect_stderr
from llama_cpp import Llama
from pprint import pprint

class ModelTestMetrics:
    """Class to hold model test metrics"""
    def __init__(self):
        self.initial_memory = 0
        self.load_memory = 0
        self.inference_memory = 0
        self.final_memory = 0
        self.load_time = 0
        self.inference_time = 0
        self.metadata = {}

@contextmanager
def measure_time():
    """Context manager to measure execution time"""
    start = time.time()
    yield lambda: time.time() - start

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

class ModelTester:
    """Class to handle model testing stages"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.metrics = ModelTestMetrics()
        
    def read_metadata(self):
        """Stage 1: Read model metadata without full initialization"""
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            # Initialize with minimal context to quickly read metadata
            temp_model = Llama(
                model_path=self.model_path,
                n_ctx=8,
                n_threads=1,
                n_batch=1,
                verbose=True
            )
            del temp_model
        
        # Parse metadata from output
        metadata = {}
        model_info = stderr_buffer.getvalue()
        
        # Extract parameters
        param_patterns = {
            'arch': r'arch\s*=\s*(\w+)',
            'n_layer': r'n_layer\s*=\s*(\d+)',
            'n_head': r'n_head\s*=\s*(\d+)',
            'n_embd': r'n_embd\s*=\s*(\d+)',
            'n_vocab': r'n_vocab\s*=\s*(\d+)',
            'n_rot': r'n_rot\s*=\s*(\d+)',
            'freq_base': r'freq_base_train\s*=\s*(\d+\.\d+)',
            'rope_scaling': r'rope scaling\s*=\s*(\w+)',
            'context_length': r'n_ctx_train\s*=\s*(\d+)',
            'model_type': r'model type\s*=\s*(.+)',
            'model_name': r'general\.name\s*=\s*(.+)'
        }
        
        import re
        for param, pattern in param_patterns.items():
            match = re.search(pattern, model_info)
            if match:
                val = match.group(1)
                metadata[param] = int(val) if val.isdigit() else (float(val) if '.' in val else val)
        
        self.metrics.metadata = metadata
        return metadata
    
    def initialize_model(self, n_ctx=2048, n_threads=8, n_batch=512):
        """Stage 2: Initialize and load the model"""
        self.metrics.initial_memory = get_memory_usage()
        
        with measure_time() as get_time:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=False
            )
            self.metrics.load_time = get_time()
        
        self.metrics.load_memory = get_memory_usage()
        return self.model
    
    def test_inference(self, prompt="Write a short poem about AI.", max_tokens=50):
        """Stage 3: Test model inference"""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        with measure_time() as get_time:
            output = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>"],
                echo=False
            )
            self.metrics.inference_time = get_time()
            
        self.metrics.inference_memory = get_memory_usage()
        return output
    
    def cleanup(self):
        """Stage 4: Cleanup and memory release"""
        if self.model:
            del self.model
            self.model = None
            time.sleep(1)  # Allow time for cleanup
            self.metrics.final_memory = get_memory_usage()

def test_model(model_path):
    print(f"\nTesting model: {os.path.basename(model_path)}")
    print("-" * 80)
    
    tester = ModelTester(model_path)
    
    # Stage 1: Metadata
    print("\nStage 1: Reading Metadata")
    metadata = tester.read_metadata()
    print("\nModel Metadata:")
    pprint(metadata)
    
    # Stage 2: Model Loading
    print("\nStage 2: Model Loading")
    model = tester.initialize_model()
    print(f"Load Time: {tester.metrics.load_time:.2f}s")
    print(f"Memory after load: {tester.metrics.load_memory:.2f} MB")
    
    # Stage 3: Inference Test
    print("\nStage 3: Inference Test")
    output = tester.test_inference(
        prompt="Write a one-sentence summary of your capabilities.",
        max_tokens=100
    )
    print(f"\nInference Time: {tester.metrics.inference_time:.2f}s")
    print(f"Memory during inference: {tester.metrics.inference_memory:.2f} MB")
    print("\nModel Output:")
    print(output)
    
    # Stage 4: Cleanup
    print("\nStage 4: Cleanup")
    tester.cleanup()
    print(f"Final Memory: {tester.metrics.final_memory:.2f} MB")
    
    # Save metrics to file
    metrics_data = {
        "model_name": os.path.basename(model_path),
        "metadata": metadata,
        "performance": {
            "load_time": tester.metrics.load_time,
            "inference_time": tester.metrics.inference_time,
            "memory": {
                "initial": tester.metrics.initial_memory,
                "after_load": tester.metrics.load_memory,
                "during_inference": tester.metrics.inference_memory,
                "final": tester.metrics.final_memory
            }
        }
    }
    
    metrics_file = f"metrics_{os.path.basename(model_path).split('.')[0]}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

if __name__ == "__main__":
    models = [
        "models/M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf",
        "models/Ayla-Light-12B-v2.Q4_K_M.gguf"
    ]
    
    for model_path in models:
        if os.path.exists(model_path):
            test_model(model_path)
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"Model not found: {model_path}") 