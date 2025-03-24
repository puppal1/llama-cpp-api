from llama_cpp_api_package.models.model_manager import model_manager
from llama_cpp_api_package.models.types import ModelParameters
import os

def test_model(model_path):
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    try:
        # Initialize model
        print("\nTesting model loading...")
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        
        # Create model parameters
        parameters = ModelParameters(
            model_path=model_path,
            num_ctx=2048,
            num_thread=4,
            num_gpu=0,
            rope_freq_base=1000000.0,  # From model metadata
            rope_freq_scale=1.0,
            rope_scaling=None  # Disable RoPE scaling
        )
        
        # For Ayla model specifically, adjust RoPE parameters
        if "Ayla-Light" in model_path:
            parameters.rope_freq_base = 1000000.0  # From model metadata
            parameters.rope_freq_scale = 1.0
            parameters.num_ctx = 1024000  # From model metadata
            
        # Load the model
        model_manager.load_model(
            model_id=model_id,
            parameters=parameters
        )
        print("Model loading test passed!")
        
        # Test basic generation
        print("\nTesting text generation...")
        prompt = "Hello, how are you?"
        response = model_manager.generate(
            model_id=model_id,
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
        response = model_manager.chat(
            model_id=model_id,
            messages=messages,
            max_tokens=20,
            temperature=0.7
        )
        print(f"Chat response: {response}")
        print("Chat test passed!")
        
        # Unload model
        print("\nTesting model unloading...")
        model_manager.unload_model(model_id)
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