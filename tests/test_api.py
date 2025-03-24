import requests
import time
import json

def get_model_config(model_name):
    """Get model-specific configuration."""
    base_config = {
        "n_ctx": 512,  # Reduced context size
        "n_batch": 256,  # Reduced batch size
        "n_threads": 4,  # Reduced threads
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "rope_scaling_type": "linear",
        "verbose": True,
        "mlock": False,  # Don't lock memory
        "mmap": True,   # Use memory mapping
        "vocab_only": False
    }
    
    if "ayla" in model_name.lower():
        # Special config for Ayla models
        base_config.update({
            "rope_freq_base": 1000000.0,
            "rope_dimension_count": 128,
            "n_ctx": 256  # Even smaller context for this model
        })
    elif "mistral" in model_name.lower():
        # Specific config for Mistral
        base_config.update({
            "n_ctx": 512,
            "rope_freq_base": 10000.0
        })
    
    return base_config

def test_model_operations(model_name):
    base_url = "http://127.0.0.1:8000/api/models"
    config = get_model_config(model_name)
    
    # Load model
    print(f"\nTesting model: {model_name}")
    print("Loading model...")
    print(f"Using config: {json.dumps(config, indent=2)}")
    
    load_response = requests.post(
        f"{base_url}/{model_name}/load",
        json=config
    )
    print(f"Load response: {load_response.status_code}")
    try:
        print(load_response.json())
    except:
        print(f"Raw response: {load_response.text}")
    
    if load_response.status_code == 200:
        # Wait a bit to ensure model is loaded
        time.sleep(5)  # Increased wait time
        
        # Unload model
        print("\nUnloading model...")
        unload_response = requests.post(f"{base_url}/{model_name}/unload")
        print(f"Unload response: {unload_response.status_code}")
        try:
            print(unload_response.json())
        except:
            print(f"Raw response: {unload_response.text}")
        
        # Wait after unloading
        time.sleep(2)

# List of smaller models to test
models = [
    "mistral-7b-instruct-v0.2.Q4_K_M",
    "WizardLM-7B-uncensored.Q8_0"
]

# Test each model
for model in models:
    try:
        test_model_operations(model)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error testing {model}: {str(e)}")
        print("\n" + "="*50 + "\n") 