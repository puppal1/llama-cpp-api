import requests
import time
import json

def test_ayla_model():
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Send request to load model
    print("Sending request to load Ayla model...")
    url = "http://localhost:8000/api/v2/models/Ayla-Light-12B-v2.Q4_K_M.gguf/load"
    response = requests.post(url)
    
    # Print response details
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    print("\nResponse Body:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    
    # Wait for server to restart and stabilize
    print("\nWaiting for server to restart and stabilize...")
    time.sleep(10)  # Increased wait time
    
    # Check if model is loaded
    print("\nChecking model status...")
    url = "http://localhost:8000/api/v2/models/Ayla-Light-12B-v2.Q4_K_M.gguf"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    try:
        status = response.json()
        print("Model Status:", status)
    except:
        print(response.text)
    
    # Only try to unload if model is actually loaded
    # if response.status_code == 200 and status.get("status") == "loaded":
    #     # Send request to unload model
    #     print("\nSending request to unload Ayla model...")
    #     url = "http://localhost:8000/api/v2/models/Ayla-Light-12B-v2.Q4_K_M.gguf/unload"
    #     response = requests.delete(url)
        
    #     # Print response details
    #     print(f"Status Code: {response.status_code}")
    #     print("Response Headers:")
    #     for key, value in response.headers.items():
    #         print(f"  {key}: {value}")
    #     print("\nResponse Body:")
    #     try:
    #         print(json.dumps(response.json(), indent=2))
    #     except:
    #         print(response.text)
    # else:
    #     print("\nModel is not loaded, skipping unload request")

if __name__ == "__main__":
    test_ayla_model() 