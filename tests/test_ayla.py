import os
from llama_cpp import Llama
import traceback
import time

def test_ayla(run_number):
    """
    Test the Ayla-Light-12B model with minimal parameters.
    """
    model = None
    try:
        print(f"\n=== Test Run #{run_number} ===\n")
        
        # Model path using os.path.join for better compatibility
        model_path = os.path.join(os.getcwd(), "models", "Ayla-Light-12B-v2.Q4_K_M.gguf")
        
        # Minimal parameters based on model's metadata
        params = {
            "n_ctx": 2048,
            "n_batch": 512,
            "verbose": True
        }
        
        print(f"Loading model from: {model_path}")
        print(f"Parameters: {params}")
        
        # Initialize the model
        model = Llama(
            model_path=model_path,
            **params
        )
        
        # Test prompt with system message
        messages = [
            {
                "role": "system",
                "content": "You are Ayla, a helpful and knowledgeable AI assistant. You communicate clearly and directly."
            },
            {
                "role": "user",
                "content": "Hi Ayla! Could you tell me a bit about yourself and what makes you unique?"
            }
        ]
        
        # Generate response
        response = model.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            stop=["</s>"]
        )
        
        # Print response and token usage
        print("\nResponse from model:")
        print(response['choices'][0]['message']['content'])
        print("\nToken Usage:")
        print(f"Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"Completion tokens: {response['usage']['completion_tokens']}")
        print(f"Total tokens: {response['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())
    finally:
        if model:
            # Explicitly free the model's memory
            print("\nUnloading model...")
            del model
            # Force garbage collection to ensure memory is freed
            import gc
            gc.collect()
            time.sleep(1)  # Small delay to ensure cleanup

if __name__ == "__main__":
    num_runs = 3  # Number of test runs
    for i in range(num_runs):
        test_ayla(i + 1) 