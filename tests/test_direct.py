from llama_cpp import Llama
import os
import traceback

def test_ayla_model():
    print("Testing Ayla model directly...")
    model_path = "models/Ayla-Light-12B-v2.Q4_K_M.gguf"
    print(f"\nModel path: {os.path.abspath(model_path)}\n")

    params = {
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "n_rot": 160,  # Added explicit n_rot parameter
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "n_threads_batch": 4,
        "verbose": True
    }

    print("Loading model with parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    try:
        llm = Llama(model_path=model_path, **params)
        
        prompt = "Hi Ayla! Could you tell me a bit about yourself and what makes you unique?"
        print(f"\nTesting with prompt:\n{prompt}")
        
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            echo=False,
            stop=["</s>"]
        )
        
        if response and len(response) > 0:
            generated_text = response['choices'][0]['text'].strip()
            print("\nModel response:")
            print(generated_text)
        else:
            print("\nNo response generated")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_ayla_model() 