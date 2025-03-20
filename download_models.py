import os
import argparse
from huggingface_hub import hf_hub_download, login
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model repo (e.g., TheBloke/Llama-2-7B-GGUF)")
    parser.add_argument("--filename", type=str, required=True, help="Specific filename to download (e.g., llama-2-7b.Q4_K_M.gguf)")
    parser.add_argument("--token", type=str, help="Hugging Face token (required for gated models like Llama)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Login with token if provided (needed for Llama models)
    if args.token:
        login(args.token)
        print(f"Logged in to Hugging Face with provided token")
    
    # Determine output path
    output_path = os.path.join(args.output_dir, args.filename)
    
    # Download the model
    print(f"Downloading {args.filename} from {args.model}...")
    print(f"This may take a while depending on your internet speed and model size.")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=args.model,
            filename=args.filename,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded model to {downloaded_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        
        if "401 Client Error" in str(e) and not args.token:
            print("\nThis model requires authentication. Please run again with your Hugging Face token:")
            print(f"python download_models.py --model {args.model} --filename {args.filename} --token YOUR_HF_TOKEN")
            print("\nYou can get your token at: https://huggingface.co/settings/tokens")
        
        return 1
    
    print("\nSuggested next steps:")
    print(f"1. Start the server: python -m llama_cpp_api_package.llama_server")
    print(f"2. Load the model through the API or web interface with ID: {args.filename.split('.')[0]}")
    print(f"   Model path: {args.filename}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 