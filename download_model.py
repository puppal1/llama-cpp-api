from huggingface_hub import hf_hub_download
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download the model
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q4_K_M.gguf",
    local_dir="models",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}") 