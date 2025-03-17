import requests
import json
from utils.model_params import ModelParameters

def main():
    base_url = "http://localhost:8080/api"
    model_id = "phi-2"  # Replace with your model name
    
    # Create model parameters with custom settings
    params = ModelParameters(
        num_ctx=4096,          # Larger context window
        temperature=0.7,       # Slightly lower temperature for more focused responses
        top_p=0.9,            # Nucleus sampling
        top_k=40,             # Top-k sampling
        num_thread=8,         # Use more CPU threads
        num_gpu=32,           # Use GPU layers if available
        repeat_penalty=1.1,   # Penalize repetition
        num_predict=512,      # Generate longer responses
        presence_penalty=0.1,  # Slight penalty for token presence
        frequency_penalty=0.1  # Slight penalty for token frequency
    )
    
    # Step 1: Load the model
    print(f"\nLoading model: {model_id}...")
    try:
        response = requests.post(
            f"{base_url}/models/{model_id}/load",
            json=params.dict()
        )
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Step 2: Chat with the model
    chat_request = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a short story about a robot learning to paint."}
        ],
        "parameters": params.dict()  # You can override parameters per request
    }
    
    print("\nSending chat request...")
    try:
        response = requests.post(
            f"{base_url}/models/{model_id}/chat",
            json=chat_request
        )
        result = response.json()
        print("\nModel response:")
        print(result["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error in chat: {str(e)}")
    
    # Step 3: Get model metrics
    print("\nGetting metrics...")
    try:
        response = requests.get(f"{base_url}/metrics")
        metrics = response.json()
        print(f"System metrics: {json.dumps(metrics, indent=2)}")
    except Exception as e:
        print(f"Error getting metrics: {str(e)}")
    
    # Step 4: Unload the model
    print("\nUnloading model...")
    try:
        response = requests.post(f"{base_url}/models/{model_id}/unload")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error unloading model: {str(e)}")

if __name__ == "__main__":
    main() 