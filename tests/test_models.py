from llama_cpp_api_package.llama_api import LlamaModel
import os

def test_model(model_path):
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    try:
        # Initialize model
        print("\nTesting model loading...")
        llm = LlamaModel()
        llm.load(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0
        )
        print("Model loading test passed!")
        
        # Test basic generation
        print("\nTesting text generation...")
        prompt = "Hello, how are you?"
        response = llm.generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.7
        )
        print(f"Input prompt: {prompt}")
        print(f"Response: {response}")
        print("Generation test passed!")
        
        # Test chat functionality
        print("\nTesting chat functionality...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        response = llm.chat(
            messages=messages,
            max_tokens=20,
            temperature=0.7
        )
        print(f"Chat response: {response}")
        print("Chat test passed!")
        
        # Unload model
        print("\nTesting model unloading...")
        llm.unload()
        print("Model unloading test passed!")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def main():
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
    if not models:
        print(f"No .gguf models found in {models_dir}")
        return
    
    results = []
    for model_file in models:
        model_path = os.path.join(models_dir, model_file)
        success = test_model(model_path)
        results.append((model_file, success))
    
    print("\nTest Summary")
    print("-" * 50)
    for model_file, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {model_file}")

if __name__ == "__main__":
    main() 