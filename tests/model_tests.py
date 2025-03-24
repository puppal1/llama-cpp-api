import os
import sys
import psutil
import time
from io import StringIO
from contextlib import contextmanager, redirect_stderr
from llama_cpp import Llama
import unittest

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
    yield
    end = time.time()
    return end - start

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
    
    def initialize_model(self, n_ctx=2048, n_threads=1, n_batch=512):
        """Stage 2: Initialize and load the model"""
        self.metrics.initial_memory = get_memory_usage()
        
        with measure_time() as load_time:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=False
            )
        
        self.metrics.load_time = load_time
        self.metrics.load_memory = get_memory_usage()
        return self.model
    
    def test_inference(self, prompt="Write a short poem about AI.", max_tokens=50):
        """Stage 3: Test model inference"""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        with measure_time() as infer_time:
            output = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>"],
                echo=False
            )
            
        self.metrics.inference_time = infer_time
        self.metrics.inference_memory = get_memory_usage()
        return output
    
    def cleanup(self):
        """Stage 4: Cleanup and memory release"""
        if self.model:
            del self.model
            self.model = None
            time.sleep(1)  # Allow time for cleanup
            self.metrics.final_memory = get_memory_usage()

class TestModels(unittest.TestCase):
    """Test cases for model loading and inference"""
    
    def setUp(self):
        self.models = [
            "models/M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf",
            "models/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_S.gguf",
            "models/Ayla-Light-12B-v2.Q4_K_M.gguf"
        ]
    
    def test_model_metadata(self):
        """Test metadata reading for all models"""
        for model_path in self.models:
            if not os.path.exists(model_path):
                continue
                
            with self.subTest(model=os.path.basename(model_path)):
                tester = ModelTester(model_path)
                metadata = tester.read_metadata()
                
                # Verify essential metadata exists
                self.assertIn('arch', metadata)
                self.assertIn('n_layer', metadata)
                self.assertIn('n_head', metadata)
    
    def test_model_loading(self):
        """Test model loading and memory management"""
        for model_path in self.models:
            if not os.path.exists(model_path):
                continue
                
            with self.subTest(model=os.path.basename(model_path)):
                tester = ModelTester(model_path)
                
                # Test each stage
                metadata = tester.read_metadata()
                model = tester.initialize_model()
                self.assertIsNotNone(model)
                
                # Verify memory usage is reasonable
                self.assertLess(tester.metrics.load_memory, 1000)  # Should be under 1GB on load
                
                # Test inference
                output = tester.test_inference()
                self.assertIsNotNone(output)
                
                # Cleanup
                tester.cleanup()
                self.assertLess(
                    tester.metrics.final_memory - tester.metrics.initial_memory,
                    10  # Allow for small memory overhead
                )
    
    def test_rope_parameters(self):
        """Test specific RoPE parameters for each model"""
        for model_path in self.models:
            if not os.path.exists(model_path):
                continue
                
            with self.subTest(model=os.path.basename(model_path)):
                tester = ModelTester(model_path)
                metadata = tester.read_metadata()
                
                # Verify RoPE parameters
                self.assertIn('n_rot', metadata)
                self.assertIn('freq_base', metadata)
                self.assertIn('rope_scaling', metadata)

if __name__ == '__main__':
    unittest.main() 