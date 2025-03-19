from llama_cpp_api_package.llama_api import LlamaModel, ModelConfig
import pytest
import os

print("Testing llama.cpp using llama-cpp-python package with NSFW-3B model")

try:
    # Load model in vocab-only mode first
    print("\nAttempting to load model vocabulary only...")
    model_path = "models/nsfw-3b-q4_k_m.gguf"
    
    config = ModelConfig(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        vocab_only=True,
        verbose=True
    )
    model = LlamaModel(config)
    
    model.load()
    assert model.status == "loaded"
    
    print("\nModel loaded successfully!")
    print(f"Model vocabulary size: {model.n_vocab()}")
    print(f"Context size: {model.n_ctx()}")
    
    # Test tokenization
    test_text = "Hello, world!"
    tokens = model.tokenize(text=test_text.encode())
    print(f"\nTokenization test:")
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Test detokenization
    decoded = model.detokenize(tokens).decode()
    print(f"Decoded text: {decoded}")
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}")

def test_model_loading():
    config = ModelConfig(
        model_path="models/test.gguf",
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0
    )
    model = LlamaModel(config)
    
    try:
        model.load()
        assert model.status == "loaded"
    except Exception as e:
        assert "Failed to load model" in str(e) 