import os
import sys
import ctypes

def main():
    print("Testing llama.dll basic functionality")

    # Get the absolute path to the DLL directory
    dll_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llama_cpp_api_package", "bin", "windows"))
    print(f"DLL directory: {dll_dir}")

    # Add DLL directory to the system's DLL search path
    os.add_dll_directory(dll_dir)

    # Construct the full path to llama.dll
    dll_path = os.path.join(dll_dir, "llama.dll")
    print(f"Attempting to load llama.dll from {dll_path}...")

    try:
        # Load the DLL
        llama = ctypes.CDLL(dll_path)
        print(f"Successfully loaded DLL from {dll_path}")
        
        # Get list of available functions
        print("\nAvailable functions:")
        all_functions = []
        for item in dir(llama):
            if not item.startswith('_'):
                all_functions.append(item)
        
        print(f"\nTotal functions found: {len(all_functions)}")
        
        # Check for potential alternatives to llama_eval
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
                func = getattr(llama, func_name)
                print(f"  - {func_name}: FOUND")
            except AttributeError:
                print(f"  - {func_name}: NOT FOUND")
        
        # Check for other key functions
        print("\nChecking for other key functions:")
        functions_to_check = [
            "llama_model_load_from_file",
            "llama_init_from_model",
            "llama_eval",
            "llama_tokenize"
        ]
        
        for func_name in functions_to_check:
            try:
                func = getattr(llama, func_name)
                print(f"  - {func_name}: FOUND")
            except AttributeError:
                print(f"  - {func_name}: NOT FOUND")
        
        print("DLL Test Complete")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 