# LLaMA.cpp Web Interface

A web interface for interacting with LLaMA models using llama.cpp. This project provides a user-friendly way to work with various LLaMA-based models through a web browser.

## Prerequisites

### Windows
- Python 3.8 or higher
- CMake 3.21 or higher (install from https://cmake.org/download/)
- Visual Studio 2019 or higher with:
  - "Desktop development with C++" workload
  - Windows 10/11 SDK
  - MSVC v143 build tools or later
- Git (for cloning repositories)

### Linux
- Python 3.8 or higher
- Build essentials:
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential
  ```
- CMake 3.21 or higher:
  ```bash
  sudo apt-get install cmake
  # Or for latest version:
  sudo apt-get install cmake-data
  wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh
  sudo sh cmake-3.28.0-linux-x86_64.sh --prefix=/usr/local --exclude-subdir
  ```

### macOS
- Python 3.8 or higher
- Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- CMake 3.21 or higher:
  ```bash
  brew install cmake
  ```

## Features

- 🌐 Web-based interface for model interaction
- 🔄 Real-time model loading and unloading
- 📊 Performance monitoring (CPU, Memory usage)
- ⚙️ Configurable model parameters
- 🔧 Support for multiple models including MOE models
- 💻 Cross-platform support (Windows, Linux, macOS)
- 🔍 API Documentation interface
- 🚀 Streaming responses support
- 🛠️ Advanced parameter configuration

## Quick Start Guide

### 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/puppal1/llama-cpp-api
cd llama-cpp-api
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Model Setup

1. Create a models directory if it doesn't exist:
```bash
mkdir -p models
```

2. Place your GGUF model files in the models directory. Supported models include:
   - Mistral-7B-Instruct (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)
   - Ayla Light (e.g., Ayla-Light-12B-v2.Q4_K_M.gguf)
   - DeepSeek models (e.g., DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_S.gguf)
   - MOE models (e.g., M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf)
   - WizardLM (e.g., WizardLM-7B-uncensored.Q8_0.gguf)

### 3. Starting the Server

#### Windows
```bash
# Start the server with default port 8000
.\server.bat

# Or specify a custom port
.\server.bat 8080
```

#### Linux/macOS
```bash
# Make the script executable
chmod +x server.sh

# Start the server with default port 8000
./server.sh

# Or specify a custom port
./server.sh 8080
```

The server will automatically:
- Check for any existing server on the specified port and stop it
- Set up the environment and models directory
- Start a new server instance

The server will be available at `http://localhost:8000/` (or your custom port) and the API will be accessible at `http://localhost:8000/api/v2/models`.

### 4. Using the API

Once the server is running, you can interact with it using the API:

1. List available models:
```
GET http://localhost:8000/api/v2/models
```

2. Load a model:
```
POST http://localhost:8000/api/v2/models/{model_id}/load
```

3. Chat with a loaded model:
```
POST http://localhost:8000/api/v2/chat/{model_id}
```

4. Unload a model:
```
POST http://localhost:8000/api/v2/models/{model_id}/unload
```

Note: For `model_id`, you can use either the filename with or without the .gguf extension (e.g., "mistral-7b-instruct-v0.2" or "mistral-7b-instruct-v0.2.gguf").

## Model Compatibility

Different models may use varying RoPE (Rotary Position Embedding) configurations. The server automatically detects and applies the appropriate settings for:

1. **Standard Models** (Mistral, WizardLM)
   - Default parameters work out of the box

2. **Ayla Models**
   - Uses 128 RoPE dimensions
   - Automatically configured with appropriate RoPE parameters

3. **DeepSeek Models**
   - Successfully runs with context window support

4. **MOE Models**
   - Mixture of Experts models are supported
   - Expert routing is handled automatically

## Troubleshooting

1. If a model fails to load:
   - Check if the model file exists and is not corrupted
   - Verify sufficient RAM is available (models can be large)

2. If you see RoPE-related errors:
   - The server should automatically detect the correct RoPE parameters
   - If issues persist, check the model metadata

3. For best performance:
   - Use quantized models (Q4_K_M, Q5_K_S etc.)
   - Adjust the number of threads based on your CPU

## Environment Variables

- `MODELS_DIR`: Custom path to look for models
- `SERVER_PORT`: Custom port (can also be set via command line)

## Configuration

### Model Parameters
The API allows you to configure:
- Context window size: Controls the context length for the model
- Batch size: Affects processing speed and memory usage
- Temperature: Controls response creativity (0.0-1.0)
- Top-K sampling: Filters vocabulary choices
- Top-P sampling: Controls response diversity
- Repeat penalty: Prevents repetitive responses
- Number of threads: CPU thread utilization
- Stop sequences: Custom sequences to stop generation

### API Endpoints

#### Models API (v2)
- `GET /api/v2/models`: List all available models
- `GET /api/v2/models/{model_id}`: Get model information
- `POST /api/v2/models/{model_id}/load`: Load a specific model
- `POST /api/v2/models/{model_id}/unload`: Unload a model
- `POST /api/v2/chat/{model_id}`: Chat with a model
- `GET /api/v2/metrics`: Get system metrics and loaded models

> **Note**: For security reasons, model paths are not included in the API responses.

#### Example: Loading a Model
```bash
curl -X POST "http://localhost:8000/api/v2/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf/load" \
  -H "Content-Type: application/json" \
  -d '{
    "num_threads": 4,
    "num_batch": 512,
    "mlock": true
  }'
```

#### Example: Using the Chat Endpoint
```bash
curl -X POST "http://localhost:8000/api/v2/chat/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about the llama animal."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

## Advanced Usage

### Python Integration
You can use the API in your Python applications:

```python
import requests
import json

API_BASE = "http://localhost:8000/api/v2"

# List available models
response = requests.get(f"{API_BASE}/models")
models = response.json()
print(f"Available models: {models}")

# Load a model
model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
load_response = requests.post(
    f"{API_BASE}/models/{model_name}/load",
    json={"num_threads": 4}
)
print(f"Load response: {load_response.json()}")

# Chat with model
chat_response = requests.post(
    f"{API_BASE}/chat/{model_name}",
    json={
        "messages": [
            {"role": "user", "content": "Write a short poem about AI"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
)
print(chat_response.json()["response"])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Binary Files

The project requires platform-specific binary files (DLLs on Windows, .so files on Linux, .dylib files on macOS). These files are not included in the repository and need to be built from source.

### Directory Structure

Binary files should be placed in the following directory structure:

```
llama_cpp_api_package/
└── bin/
    ├── windows/  # .dll files
    ├── linux/    # .so files
    └── macos/    # .dylib files
```

### Building from Source

To build the required binary files:

1. Clone the llama.cpp repository:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   ```

2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

3. Copy the binary files to the appropriate directory:
   - Windows: Copy all .dll files from `build/bin/Release/` to `llama_cpp_api_package/bin/windows/`
   - Linux: Copy all .so files from `build/` to `llama_cpp_api_package/bin/linux/`
   - macOS: Copy all .dylib files from `build/` to `llama_cpp_api_package/bin/macos/`

4. Verify the setup by running:
   ```bash
   python tests/test_dll.py    # Windows
   python tests/test_so.py     # Linux
   python tests/test_dylib.py  # macOS
   ``` 