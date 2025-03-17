from llama_cpp import Llama
import os

def test_model(model_path):
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    try:
        # First try loading with vocab_only to quickly verify the model file
        print("Testing model vocabulary...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            vocab_only=True,
            verbose=True
        )
        print("Vocabulary test passed!")
        
        # Now try loading the full model
        print("\nTesting full model loading...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=True
        )
        print("Full model loading test passed!")
        
        # Test tokenization
        print("\nTesting tokenization...")
        text = "Hello, world!"
        tokens = llm.tokenize(text.encode())
        print(f"Input text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Test detokenization
        print("\nTesting detokenization...")
        decoded = llm.detokenize(tokens).decode()
        print(f"Decoded text: {decoded}")
        assert decoded == text, "Decoded text doesn't match input"
        print("Tokenization/detokenization test passed!")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def main():
    models = [
        "models/nsfw-flash-q4_k_m.gguf",
        "models/nsfw-3b-q4_k_m.gguf"
    ]
    
    results = []
    for model_path in models:
        if os.path.exists(model_path):
            success = test_model(model_path)
            results.append((model_path, success))
        else:
            print(f"\nError: Model file not found: {model_path}")
            results.append((model_path, False))
    
    print("\nTest Summary")
    print("-" * 50)
    for model_path, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {model_path}")

if __name__ == "__main__":
    main() 