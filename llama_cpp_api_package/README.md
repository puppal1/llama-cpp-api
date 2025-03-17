# Llama.cpp API Package

A self-contained API for interacting with llama.cpp models.

## Structure
- `llama_api.py` - Core API for model interaction
- `llama_server.py` - FastAPI server for hosting the API
- `utils/` - Utility scripts
- `static/` - Web interface
- `models/` - Model directory
- `run.py` - Simple command-line interface for common tasks

## Quick Start
1. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```

2. Download a model:
   ```bash
   python run.py download --model microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf
   ```

3. Start the server:
   ```bash
   python run.py server
   ```

4. Access web interface: http://localhost:8080/static/index.html

## Using the run.py Script

The `run.py` script is a simple interface for common tasks:

### Start the API server
```bash
python run.py server [--host HOST] [--port PORT]
```

### Download models
```bash
python run.py download --model MODEL_REPO --filename FILENAME [--token HUGGINGFACE_TOKEN]
```

### Update the web interface
```bash
python run.py update
```

### List available models
```bash
python run.py list
```

## Direct Usage

You can also use the individual components directly:

- Start the server: `python llama_server.py`
- Download a model: `python -m utils.download_models --model MODEL_REPO --filename FILENAME`
- Update web interface: `python -m utils.update_web_interface`

## API Integration

You can integrate the API into your own projects:

```python
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
