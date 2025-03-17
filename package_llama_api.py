import os
import shutil
from pathlib import Path
import sys

def create_package():
    """
    Creates a clean, organized package structure for the llama.cpp API
    in a single self-contained folder.
    """
    # Define package directory structure
    package_name = "llama_cpp_api_package"
    package_dir = Path(package_name)
    
    # Create directories
    dirs = [
        package_dir,
        package_dir / "models",
        package_dir / "static",
        package_dir / "utils",
    ]
    
    # Delete the package directory if it already exists
    if package_dir.exists():
        shutil.rmtree(package_dir)
        print(f"Removed existing directory: {package_dir}")
    
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        print(f"Created directory: {d}")
    
    # Essential files to include in the package
    files_to_copy = {
        # Core API files
        "llama_cpp_python.py": package_dir / "llama_server.py",
        "llama_api.py": package_dir / "llama_api.py",
        
        # Utils
        "download_models.py": package_dir / "utils" / "download_models.py",
        "update_web_interface.py": package_dir / "utils" / "update_web_interface.py",
        
        # Web interface
        "static/index.html": package_dir / "static" / "index.html",
        "static/monitor.html": package_dir / "static" / "monitor.html",
        
        # DLL files (these are crucial for the API to work)
        "llama.dll": package_dir / "llama.dll",
        "ggml.dll": package_dir / "ggml.dll",
        "ggml-base.dll": package_dir / "ggml-base.dll",
        "ggml-cpu.dll": package_dir / "ggml-cpu.dll",
        "llava_shared.dll": package_dir / "llava_shared.dll",
        
        # Config
        "requirements.txt": package_dir / "requirements.txt",
    }
    
    # Copy files
    for src, dest in files_to_copy.items():
        if Path(src).exists():
            shutil.copy2(src, dest)
            print(f"Copied: {src} -> {dest}")
        else:
            print(f"Warning: Source file {src} not found, skipping")
    
    # Copy existing models if they exist
    model_files = list(Path("models").glob("*.gguf"))
    if model_files:
        print("\nCopying model files...")
        for model_file in model_files:
            dest_path = package_dir / "models" / model_file.name
            shutil.copy2(model_file, dest_path)
            print(f"Copied model: {model_file} -> {dest_path}")
    
    # Create __init__.py files for proper Python package structure
    init_files = [
        package_dir / "__init__.py",
        package_dir / "utils" / "__init__.py",
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Llama.cpp API package for easy integration with llama.cpp models."""\n\n')
            if init_file.parent.name == "utils":
                f.write('# Utility functions for the llama.cpp API\n')
            else:
                f.write('# Import for convenient access\n')
                f.write('from .llama_api import LlamaModel\n')
                f.write('from .llama_server import app\n\n')
                f.write('__version__ = "0.1.0"\n')
        print(f"Created: {init_file}")
    
    # Create a README.md file
    readme_path = package_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Llama.cpp API Package

A self-contained API for interacting with llama.cpp models.

## Structure

- `llama_api.py` - Core API for interacting with llama.cpp models
- `llama_server.py` - FastAPI server for hosting the API
- `utils/` - Utility functions for downloading models and updating the web interface
- `static/` - Web interface files
- `models/` - Directory for storing model files
- Various DLL files - Required for the API to function

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models using the provided script:
   ```bash
   python download_model.py --model microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf
   ```

## Usage as a Module

```python
# Import the LlamaModel class
from llama_api import LlamaModel

# Initialize the model
model = LlamaModel()

# Load a model
model.load(
    model_path="models/your-model.gguf",
    n_ctx=2048,
    n_gpu_layers=0  # Set to higher value for GPU acceleration
)

# Generate a response
response = model.generate(
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7
)

print(response)

# Unload the model when done
model.unload()
```

## Starting the API Server

```bash
python start_server.py --host 0.0.0.0 --port 8080
```

Then access the web interface at: http://localhost:8080/static/index.html

## Downloading Models

```bash
python download_model.py --model microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf
```
""")
        print(f"Created: {readme_path}")
    
    # Update update_web_interface.py to a cleaner version for the package
    update_web_interface = package_dir / "utils" / "update_web_interface.py"
    with open(update_web_interface, 'w') as f:
        f.write("""import os
import re
from pathlib import Path

def get_model_files(models_dir="models"):
    """Get all GGUF model files in the models directory"""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory '{models_dir}' not found")
        return []
    
    return sorted([f for f in models_path.glob("*.gguf") if f.is_file()])

def update_index_html(index_path="static/index.html", models_dir="models"):
    """Update the index.html file with model options based on available models"""
    index_path = Path(index_path)
    if not index_path.exists():
        print(f"index.html not found at {index_path}")
        return False
    
    model_files = get_model_files(models_dir)
    if not model_files:
        print("No GGUF model files found in models directory")
        return False
    
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the select element for model paths
    select_pattern = r'<select class="form-select" id="modelPath" onchange="updateModelId\(\)">(.*?)</select>'
    select_match = re.search(select_pattern, content, re.DOTALL)
    
    if not select_match:
        print("Could not find model select element in index.html")
        return False
    
    # Create new options for each model
    options = []
    for model_file in model_files:
        model_name = model_file.stem
        model_id = model_name.split('.')[0]  # Use first part before dot as ID
        
        # Skip NSFW models and models with suspicious names
        if any(word in model_name.lower() for word in ["nsfw", "uncensored", "unfiltered", "dark"]):
            print(f"Skipping potentially harmful model: {model_name}")
            continue
            
        options.append(f'<option value="{models_dir}/{model_file.name}" data-id="{model_id}">{model_id}</option>')
    
    # If no safe models were found
    if not options:
        print("No safe models found (skipping potentially harmful models)")
        return False
    
    # Create the new select element content
    new_select = f'<select class="form-select" id="modelPath" onchange="updateModelId()">'
    new_select += '\n'.join(options)
    new_select += '\n</select>'
    
    # Replace the select element in the HTML
    new_content = re.sub(select_pattern, new_select, content, flags=re.DOTALL)
    
    # Save the updated HTML
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated index.html with {len(options)} model options")
    print("Models added:")
    for option in options:
        model_id = option.split('data-id=\"')[1].split('\"')[0]
        print(f"  - {model_id}")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update the web interface with available models")
    parser.add_argument("--models-dir", default="models", help="Directory containing model files")
    parser.add_argument("--index-path", default="static/index.html", help="Path to index.html file")
    args = parser.parse_args()
    
    print("Scanning for models...")
    model_files = get_model_files(args.models_dir)
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({file_size_mb:.1f} MB)")
    
    print("\nUpdating web interface...")
    if update_index_html(args.index_path, args.models_dir):
        print("\nSuccessfully updated the web interface!")
        print("Start the server with: python llama_server.py")
    else:
        print("\nFailed to update the web interface.")
    
    return 0

if __name__ == "__main__":
    exit(main())""")
        print(f"Updated: {update_web_interface}")
    
    # Create a helper script to download models
    download_helper = package_dir / "download_model.py"
    with open(download_helper, 'w') as f:
        f.write("""#!/usr/bin/env python
import argparse
from utils.download_models import download_model

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model repo (e.g., TheBloke/Llama-2-7B-GGUF)")
    parser.add_argument("--filename", type=str, required=True, help="Specific filename to download")
    parser.add_argument("--token", type=str, help="Hugging Face token (required for gated models)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    args = parser.parse_args()
    
    download_model(
        model_repo=args.model,
        filename=args.filename,
        token=args.token,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()""")
        print(f"Created: {download_helper}")
    
    # Create a helper script to start the server
    server_helper = package_dir / "start_server.py"
    with open(server_helper, 'w') as f:
        f.write("""#!/usr/bin/env python
import uvicorn
import argparse
from llama_server import app

def main():
    parser = argparse.ArgumentParser(description="Start the Llama.cpp API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()
    
    print(f"Starting Llama.cpp API server at http://{args.host}:{args.port}")
    print(f"Web interface available at http://{args.host}:{args.port}/static/index.html")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()""")
        print(f"Created: {server_helper}")
    
    # Update download_models.py to make it a proper utility function
    update_download_models = package_dir / "utils" / "download_models.py"
    with open(update_download_models, 'w') as f:
        f.write("""import os
import argparse
from huggingface_hub import hf_hub_download, login
from pathlib import Path

def download_model(model_repo, filename, token=None, output_dir="models"):
    """
    Download a model from Hugging Face.
    
    Args:
        model_repo (str): The Hugging Face model repository (e.g., 'microsoft/Phi-3-mini-4k-instruct-gguf')
        filename (str): The specific filename to download (e.g., 'Phi-3-mini-4k-instruct-q4.gguf')
        token (str, optional): Hugging Face token for gated models. Defaults to None.
        output_dir (str, optional): Directory to save the model. Defaults to "models".
    
    Returns:
        str: Path to the downloaded model or None if download failed
    """
    # Create models directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Login with token if provided (needed for Llama models)
    if token:
        login(token)
        print(f"Logged in to Hugging Face with provided token")
    
    # Determine output path
    output_path = os.path.join(output_dir, filename)
    
    # Download the model
    print(f"Downloading {filename} from {model_repo}...")
    print(f"This may take a while depending on your internet speed and model size.")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_repo,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded model to {downloaded_path}")
        
        print("\nSuggested next steps:")
        print(f"1. Start the server: python start_server.py")
        print(f"2. Load the model through the API or web interface with ID: {filename.split('.')[0]}")
        print(f"   Model path: {filename}")
        
        # Update the web interface to include the new model
        try:
            from .update_web_interface import update_index_html
            update_index_html()
            print("\n3. The web interface has been updated to include the new model.")
        except ImportError:
            print("\n3. To update the web interface, run: python -m utils.update_web_interface")
        
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        
        if "401 Client Error" in str(e) and not token:
            print("\nThis model requires authentication. Please run again with your Hugging Face token.")
            print("You can get your token at: https://huggingface.co/settings/tokens")
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model repo (e.g., TheBloke/Llama-2-7B-GGUF)")
    parser.add_argument("--filename", type=str, required=True, help="Specific filename to download")
    parser.add_argument("--token", type=str, help="Hugging Face token (required for gated models)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    args = parser.parse_args()
    
    return 0 if download_model(args.model, args.filename, args.token, args.output_dir) else 1

if __name__ == "__main__":
    exit(main())""")
        print(f"Updated: {update_download_models}")
    
    # Create a simple run script as the main entry point
    run_script = package_dir / "run.py"
    with open(run_script, 'w') as f:
        f.write("""#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Llama.cpp API Package - Main Entry Point")
    parser.add_argument("action", choices=["server", "download", "update"], 
                       help="Action to perform: 'server' to start API server, 'download' to download a model, 'update' to update web interface")
    
    # Server arguments
    parser.add_argument("--host", default="0.0.0.0", help="Server host (for 'server' action)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (for 'server' action)")
    
    # Download arguments
    parser.add_argument("--model", help="Model repository (for 'download' action, e.g., microsoft/Phi-3-mini-4k-instruct-gguf)")
    parser.add_argument("--filename", help="Model filename (for 'download' action, e.g., Phi-3-mini-4k-instruct-q4.gguf)")
    parser.add_argument("--token", help="Hugging Face token (for 'download' action, required for gated models)")
    
    args = parser.parse_args()
    
    if args.action == "server":
        # Start the server
        print(f"Starting Llama.cpp API server at http://{args.host}:{args.port}")
        print(f"Web interface available at http://{args.host}:{args.port}/static/index.html")
        
        # Use subprocess to run the server to avoid import issues
        import subprocess
        subprocess.run([sys.executable, "start_server.py", "--host", args.host, "--port", str(args.port)])
        
    elif args.action == "download":
        # Validate download arguments
        if not args.model or not args.filename:
            print("Error: Both --model and --filename are required for the 'download' action")
            return 1
        
        # Use subprocess to download the model
        cmd = [sys.executable, "download_model.py", "--model", args.model, "--filename", args.filename]
        if args.token:
            cmd.extend(["--token", args.token])
        
        import subprocess
        subprocess.run(cmd)
        
    elif args.action == "update":
        # Update the web interface
        print("Updating web interface with available models...")
        import subprocess
        subprocess.run([sys.executable, "-m", "utils.update_web_interface"])
    
    return 0

if __name__ == "__main__":
    exit(main())""")
        print(f"Created: {run_script}")
    
    print("\nPackage creation complete!")
    print(f"The package has been created at: {package_dir}")
    print("\nUsage instructions:")
    print("1. Open the package directory: cd llama_cpp_api_package")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Download models: python run.py download --model microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf")
    print("4. Start the server: python run.py server")
    print("5. Update web interface: python run.py update")

if __name__ == "__main__":
    create_package() 