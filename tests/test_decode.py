import ctypes
import os
from ctypes import c_int, c_float, c_char_p, c_void_p, POINTER, Structure, c_bool, c_int32, c_uint32, c_size_t, byref

print("Testing llama.dll with llama_decode instead of llama_eval")

# Load the DLL
dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama_cpp_api_package", "bin", "windows", "llamafile.exe")
print(f"Loading llama.dll...")
try:
    llama = ctypes.CDLL(dll_path)
    print(f"Successfully loaded DLL from {dll_path}\n")
except Exception as e:
    print(f"Failed to load DLL: {e}")
    exit(1)

# Set up version function signatures
llama.llama_print_system_info.restype = c_char_p

# Get system info
print("Checking llama.cpp version and system info:")
system_info = llama.llama_print_system_info().decode('utf-8')
print(system_info)

# Define necessary structures
class llama_model_params(Structure):
    _fields_ = [
        ("n_gpu_layers", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool)
    ]

class llama_context_params(Structure):
    _fields_ = [
        ("seed", c_uint32),
        ("n_ctx", c_int32),
        ("n_batch", c_int32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("rope_scaling_type", c_int32),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_int32),
        ("type_k", c_int32),
        ("type_v", c_int32),
        ("logits_all", c_bool),
        ("embedding", c_bool),
        ("offload_kqv", c_bool)
    ]

# Set up function signatures
llama.llama_model_load_from_file.argtypes = [c_char_p, POINTER(llama_model_params)]
llama.llama_model_load_from_file.restype = c_void_p

llama.llama_init_from_model.argtypes = [c_void_p, POINTER(llama_context_params)]
llama.llama_init_from_model.restype = c_void_p

llama.llama_free.argtypes = [c_void_p]
llama.llama_model_free.argtypes = [c_void_p]

# Test loading model
print("\nAttempting to load model vocabulary only...")
try:
    # Create model parameters manually
    model_params = llama_model_params(
        n_gpu_layers=0,
        main_gpu=-1,
        tensor_split=None,
        progress_callback=None,
        progress_callback_user_data=None,
        vocab_only=True,
        use_mmap=True,
        use_mlock=False
    )

    # Load model
    model_path = b"models/llama-2-7b.Q4_K_M.gguf"
    if not os.path.exists(model_path.decode()):
        print(f"Error: Model file not found at {model_path.decode()}")
        exit(1)

    print("Loading model with parameters:")
    print(f"  vocab_only: {model_params.vocab_only}")
    print(f"  n_gpu_layers: {model_params.n_gpu_layers}")
    print(f"  main_gpu: {model_params.main_gpu}")
    print(f"  use_mmap: {model_params.use_mmap}")
    print(f"  use_mlock: {model_params.use_mlock}")

    model = llama.llama_model_load_from_file(model_path, byref(model_params))
    if not model:
        print("Error: Failed to load model")
        exit(1)

    print("Successfully loaded model")

    # Create context parameters manually
    ctx_params = llama_context_params(
        seed=0,
        n_ctx=2048,
        n_batch=512,
        n_threads=4,
        n_threads_batch=4,
        rope_scaling_type=0,
        rope_freq_base=10000.0,
        rope_freq_scale=1.0,
        yarn_ext_factor=1.0,
        yarn_attn_factor=1.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_orig_ctx=0,
        type_k=1,
        type_v=1,
        logits_all=False,
        embedding=False,
        offload_kqv=False
    )

    print("\nInitializing context with parameters:")
    print(f"  n_ctx: {ctx_params.n_ctx}")
    print(f"  n_threads: {ctx_params.n_threads}")
    print(f"  embedding: {ctx_params.embedding}")
    print(f"  logits_all: {ctx_params.logits_all}")

    # Initialize context
    ctx = llama.llama_init_from_model(model, byref(ctx_params))
    if not ctx:
        print("Error: Failed to create context")
        llama.llama_model_free(model)
        exit(1)

    print("Successfully created context")

    # Clean up
    llama.llama_free(ctx)
    llama.llama_model_free(model)
    print("Successfully cleaned up resources")

except Exception as e:
    print(f"Error: {e}") 