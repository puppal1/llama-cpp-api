"""
Comprehensive local tests for the llama.cpp API functionality.
"""
import unittest
import time
from pathlib import Path
from typing import Dict, Any

from api.models import ModelLoader, ModelMetadata, ContextManager, ResourceManager

class TestModelAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.models_dir = Path("models")
        cls.test_models = {
            "moe": str(cls.models_dir / "M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf"),
            "ayla": str(cls.models_dir / "Ayla-Light-12B-v2.Q4_K_M.gguf")
        }
        cls.loader = ModelLoader()
        
    def setUp(self):
        """Set up each test."""
        self.loader.unload_model()  # Ensure clean state
        
    def tearDown(self):
        """Clean up after each test."""
        self.loader.unload_model()

    def test_metadata_extraction(self):
        """Test metadata extraction for each model."""
        for model_name, model_path in self.test_models.items():
            with self.subTest(model=model_name):
                metadata = ModelMetadata(model_path)
                metadata_dict = metadata.extract_metadata()
                
                # Verify basic metadata fields
                self.assertIsNotNone(metadata_dict["architecture"])
                self.assertIsNotNone(metadata_dict["embedding_dim"])
                self.assertIsNotNone(metadata_dict["n_layers"])
                self.assertIsNotNone(metadata_dict["n_heads"])
                self.assertIsNotNone(metadata_dict["vocab_size"])
                
                # Verify model-specific parameters
                if "moe" in model_name:
                    self.assertIsNotNone(metadata_dict["moe_params"])
                    self.assertIsNotNone(metadata_dict["moe_params"]["n_expert"])
                
                # Verify context parameters
                self.assertIsNotNone(metadata_dict["context_params"])
                self.assertGreater(metadata_dict["context_params"]["max_context"], 0)
                
    def test_model_loading(self):
        """Test model loading with different context sizes."""
        test_contexts = [2048, 4096]
        
        for model_name, model_path in self.test_models.items():
            for ctx_size in test_contexts:
                with self.subTest(model=model_name, context=ctx_size):
                    try:
                        metadata = self.loader.load_model(
                            model_path,
                            context_size=ctx_size
                        )
                        self.assertIsNotNone(self.loader.current_model)
                        self.assertEqual(
                            metadata["loaded_params"]["context_size"],
                            ctx_size
                        )
                    except MemoryError:
                        print(f"Skipping {model_name} with context {ctx_size} due to memory constraints")
                    finally:
                        self.loader.unload_model()
                        
    def test_inference(self):
        """Test model inference with different prompts."""
        test_prompts = [
            "Write a haiku about AI:",
            "Explain what is machine learning in one sentence:",
            "Count from 1 to 5:"
        ]
        
        for model_name, model_path in self.test_models.items():
            with self.subTest(model=model_name):
                try:
                    self.loader.load_model(model_path, context_size=2048)
                    
                    for prompt in test_prompts:
                        output = self.loader.current_model.create_completion(
                            prompt,
                            max_tokens=50,
                            temperature=0.7,
                            stop=["###"]
                        )
                        
                        self.assertIsNotNone(output)
                        self.assertIn("choices", output)
                        self.assertGreater(len(output["choices"]), 0)
                        self.assertIn("text", output["choices"][0])
                        self.assertGreater(len(output["choices"][0]["text"]), 0)
                        
                except Exception as e:
                    self.fail(f"Inference failed for {model_name}: {str(e)}")
                finally:
                    self.loader.unload_model()
                    
    def test_memory_management(self):
        """Test memory management during model operations."""
        resource_manager = ResourceManager()
        initial_memory = resource_manager.get_current_memory_usage()
        
        for model_name, model_path in self.test_models.items():
            with self.subTest(model=model_name):
                try:
                    # Check memory before load
                    pre_load_memory = resource_manager.get_current_memory_usage()
                    
                    # Load model
                    self.loader.load_model(model_path, context_size=2048)
                    post_load_memory = resource_manager.get_current_memory_usage()
                    self.assertGreater(post_load_memory, pre_load_memory)
                    
                    # Run inference
                    _ = self.loader.current_model.create_completion(
                        "Test prompt",
                        max_tokens=10
                    )
                    inference_memory = resource_manager.get_current_memory_usage()
                    
                    # Unload model
                    self.loader.unload_model()
                    post_unload_memory = resource_manager.get_current_memory_usage()
                    
                    # Verify cleanup
                    self.assertLess(post_unload_memory, inference_memory)
                    self.assertLess(
                        abs(post_unload_memory - initial_memory),
                        0.1  # Allow 100MB difference
                    )
                    
                except Exception as e:
                    self.fail(f"Memory management test failed for {model_name}: {str(e)}")
                    
    def test_context_management(self):
        """Test context window management."""
        context_manager = ContextManager()
        
        # Test context size validation
        max_size = 32768
        context_manager.set_context_params(max_size)
        
        # Test valid sizes
        self.assertEqual(
            context_manager.validate_context_size(4096),
            4096
        )
        
        # Test oversized context
        self.assertEqual(
            context_manager.validate_context_size(max_size + 1024),
            max_size
        )
        
        # Test text chunking
        long_text = "test " * 1000
        chunks = context_manager.chunk_text(long_text, None)
        self.assertGreater(len(chunks), 1)
        
        # Test optimal context size calculation
        optimal_size = context_manager.get_optimal_context_size(1000)
        self.assertGreaterEqual(optimal_size, 1024)
        self.assertLessEqual(optimal_size, max_size)
        
def run_tests():
    """Run all API tests."""
    unittest.main(verbosity=2)
    
if __name__ == "__main__":
    run_tests() 