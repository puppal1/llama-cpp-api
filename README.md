# LLaMA.cpp Web Interface

A web interface for interacting with LLaMA models using llama.cpp. This project provides a user-friendly way to work with various LLaMA-based models through a web browser.

## Features

- 🌐 Web-based interface for model interaction
- 🔄 Real-time model loading and unloading
- 📊 Performance monitoring (CPU, Memory, GPU usage)
- ⚙️ Configurable model parameters
- 🔧 Support for multiple models
- 💻 Cross-platform support (Windows, Linux, macOS)

## Quick Start Guide

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- C++ compiler (for Linux/macOS only)

### 2. Installation

1. Clone the repository:
```bash
git clone https://github.com/puppal1/llama-cpp-api
cd llama-cpp-api
```

2. Install the package and dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

### 3. Platform-Specific Setup

#### Windows
- No additional setup required - binaries are included in `bin/windows`

#### Linux
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential cmake

# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Copy libraries (adjust paths as needed)
cp libllama.so /path/to/llama_cpp_api/llama_cpp_api_package/bin/linux/
cp libggml*.so /path/to/llama_cpp_api/llama_cpp_api_package/bin/linux/
```

#### macOS
```bash
# Install build dependencies
brew install cmake

# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Copy libraries (adjust paths as needed)
cp libllama.dylib /path/to/llama_cpp_api/llama_cpp_api_package/bin/darwin/
cp libggml*.dylib /path/to/llama_cpp_api/llama_cpp_api_package/bin/darwin/
```

### 4. Getting Started

1. Download a model (example using Phi-3):
```bash
python run.py download --model microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf
```

2. Start the server:
```bash
python run.py server
```
or use the command-line tool:
```bash
llama-cpp-api
```

3. Access the web interface:
```
http://localhost:8000
```

## Configuration

### Model Parameters
The web interface allows you to configure:
- Context window size
- Batch size
- Temperature
- Top-K sampling
- Top-P sampling
- Repeat penalty
- Number of threads
- Memory locking
- Random seed
- Maximum tokens to generate

### Environment Variables
- `LLAMA_API_HOST`: Host to bind the server (default: 0.0.0.0)
- `LLAMA_API_PORT`: Port to run the server (default: 8000)
- `LLAMA_MODEL_PATH`: Default path to look for models
- `LLAMA_NUM_THREADS`: Default number of CPU threads to use

## Advanced Usage

### API Integration
You can integrate the API into your Python projects:

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

### Useful Commands

- List available models:
```bash
python run.py list
```

- Update web interface:
```bash
python run.py update
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make

# Install the package
COPY . /app
WORKDIR /app
RUN pip install -e .

# Copy llama.cpp libraries
RUN cp /llama.cpp/libllama.so /app/llama_cpp_api_package/bin/linux/ && \
    cp /llama.cpp/libggml*.so /app/llama_cpp_api_package/bin/linux/

EXPOSE 8000
CMD ["llama-cpp-api"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-cpp-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama-cpp-api
  template:
    metadata:
      labels:
        app: llama-cpp-api
    spec:
      containers:
      - name: llama-cpp-api
        image: your-registry/llama-cpp-api:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: models
          mountPath: /app/llama_cpp_api_package/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 