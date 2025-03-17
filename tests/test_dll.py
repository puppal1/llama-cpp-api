import ctypes
import os
import sys

def main():
    print("Testing llama.dll basic functionality")
    
    try:
        # Try to load the DLL
        print("Attempting to load llama.dll...")
        dll_path = os.path.join(os.path.dirname(__file__), "llama.dll")
        lib = ctypes.CDLL(dll_path)
        print(f"Successfully loaded DLL from {dll_path}")
        
        # Get list of available functions
        print("\nAvailable functions:")
        all_functions = []
        for item in dir(lib):
            if not item.startswith('_'):
                all_functions.append(item)
                print(f"  - {item}")
        
        print(f"\nTotal functions found: {len(all_functions)}")
        
        # Check for possible alternatives to llama_eval
        print("\nChecking for potential alternatives to llama_eval:")
        alternatives = [
            "llama_decode",
            "llama_encode",
            "llama_evaluate",
            "llama_batch_decode",
            "llama_batch_eval",
            "llama_process",
            "llama_batch_process"
        ]
        
        for func_name in alternatives:
            try:
                func = getattr(lib, func_name)
                print(f"  - {func_name}: FOUND")
            except AttributeError:
                print(f"  - {func_name}: NOT FOUND")
        
        # Try to access a few specific functions
        print("\nChecking for other key functions:")
        functions_to_check = [
            "llama_model_load_from_file", 
            "llama_init_from_model",
            "llama_eval",
            "llama_tokenize", 
            "llama_free",
            "llama_model_free",
            "llama_vocab_get_text",
            "llama_get_logits",
            "llama_token_eot"
        ]
        
        for func_name in functions_to_check:
            try:
                func = getattr(lib, func_name)
                print(f"  - {func_name}: FOUND")
            except AttributeError:
                print(f"  - {func_name}: NOT FOUND")
        
        print("\nDLL Test Complete")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 