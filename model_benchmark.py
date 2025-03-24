import os
import sys
import psutil
import time
import json
import numpy as np
from io import StringIO
from contextlib import contextmanager, redirect_stderr
from llama_cpp import Llama
from pprint import pprint
import gc

class ModelTestMetrics:
    def __init__(self):
        self.initial_memory = 0
        self.load_memory = 0
        self.inference_memory = []  # List to track memory over multiple inferences
        self.final_memory = 0
        self.load_time = 0
        self.inference_times = []  # List to track inference times
        self.metadata = {}
        self.context_window_performance = {}  # Track performance across different context sizes

@contextmanager
def measure_time():
    start = time.time()
    yield lambda: time.time() - start

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class EnhancedModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.metrics = ModelTestMetrics()
        
    def read_metadata(self):
        """Enhanced metadata reading with MOE-specific parameters"""
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            temp_model = Llama(
                model_path=self.model_path,
                n_ctx=8,
                n_threads=1,
                n_batch=1,
                verbose=True
            )
            del temp_model
        
        metadata = {}
        model_info = stderr_buffer.getvalue()
        
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
            'model_name': r'general\.name\s*=\s*(.+)',
            # MOE-specific parameters
            'n_expert': r'n_expert\s*=\s*(\d+)',
            'n_expert_used': r'n_expert_used\s*=\s*(\d+)',
            'top_k': r'top_k\s*=\s*(\d+)'
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
        gc.collect()  # Force garbage collection before loading
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
    
    def test_inference(self, prompt, max_tokens=100, num_runs=3):
        """Run multiple inference tests and collect statistics"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        for _ in range(num_runs):
            gc.collect()  # Clean up before each run
            with measure_time() as get_time:
                output = self.model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    stop=["</s>"],
                    echo=False
                )
                self.metrics.inference_times.append(get_time())
            
            self.metrics.inference_memory.append(get_memory_usage())
            
        return output
    
    def test_context_windows(self, base_prompt, sizes=[512, 1024, 2048, 4096]):
        """Test model performance with different context window sizes"""
        results = {}
        for size in sizes:
            if size > self.metrics.metadata.get('context_length', float('inf')):
                continue
                
            print(f"\nTesting context window size: {size}")
            # Reinitialize model with new context size
            self.cleanup()
            self.initialize_model(n_ctx=size)
            
            # Create a prompt that approaches the context window size
            long_prompt = base_prompt * (size // len(base_prompt))
            
            with measure_time() as get_time:
                try:
                    output = self.model.create_completion(
                        long_prompt,
                        max_tokens=50,
                        stop=["</s>"],
                        echo=False
                    )
                    results[size] = {
                        'success': True,
                        'time': get_time(),
                        'memory': get_memory_usage(),
                        'output_length': len(output['choices'][0]['text'])
                    }
                except Exception as e:
                    results[size] = {
                        'success': False,
                        'error': str(e)
                    }
                    
        self.metrics.context_window_performance = results
        return results
    
    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            time.sleep(1)
            self.metrics.final_memory = get_memory_usage()

def test_model(model_path):
    print(f"\nTesting model: {os.path.basename(model_path)}")
    print("-" * 80)
    
    tester = EnhancedModelTester(model_path)
    
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
    
    # Stage 3: Basic Inference Tests
    print("\nStage 3: Basic Inference Tests")
    prompts = [
        "Write a one-sentence summary of your capabilities.",
        "Explain the concept of quantum computing in simple terms.",
        "Write a short poem about artificial intelligence."
    ]
    
    for prompt in prompts:
        print(f"\nTesting prompt: {prompt}")
        output = tester.test_inference(prompt)
        print("\nOutput:")
        print(output['choices'][0]['text'])
    
    # Stage 4: Context Window Tests
    print("\nStage 4: Context Window Tests")
    base_prompt = "Testing the model's ability to handle different context lengths. "
    context_results = tester.test_context_windows(base_prompt)
    
    # Stage 5: Cleanup
    print("\nStage 5: Cleanup")
    tester.cleanup()
    print(f"Final Memory: {tester.metrics.final_memory:.2f} MB")
    
    # Calculate statistics
    stats = {
        "inference_times": {
            "mean": np.mean(tester.metrics.inference_times),
            "std": np.std(tester.metrics.inference_times),
            "min": np.min(tester.metrics.inference_times),
            "max": np.max(tester.metrics.inference_times)
        },
        "memory_usage": {
            "mean": np.mean(tester.metrics.inference_memory),
            "std": np.std(tester.metrics.inference_memory),
            "min": np.min(tester.metrics.inference_memory),
            "max": np.max(tester.metrics.inference_memory)
        }
    }
    
    # Save comprehensive metrics
    metrics_data = {
        "model_name": os.path.basename(model_path),
        "metadata": metadata,
        "performance": {
            "load_time": tester.metrics.load_time,
            "inference_statistics": stats,
            "context_window_performance": tester.metrics.context_window_performance,
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
        "models/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_S.gguf",
        "models/WizardLM-7B-uncensored.Q8_0.gguf"
    ]
    
    for model_path in models:
        if os.path.exists(model_path):
            test_model(model_path)
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"Model not found: {model_path}") 