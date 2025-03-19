import os
import sys
import ctypes

def main():
    print("Testing libllama.dylib basic functionality")

    # Get the absolute path to the dylib directory
    lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llama_cpp_api_package", "bin", "macos"))
    print(f"Library directory: {lib_dir}")

    # Add library directory to the system's library search path
    os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

    # Construct the full path to libllama.dylib
    lib_path = os.path.join(lib_dir, "libllama.dylib")
    print(f"Attempting to load libllama.dylib from {lib_path}...")

    try:
        # Load the dynamic library
        llama = ctypes.CDLL(lib_path)
        print(f"Successfully loaded library from {lib_path}")
        
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
        
        print("Library Test Complete")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 