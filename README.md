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
- 📊 Performance monitoring (CPU, Memory, GPU usage)
- ⚙️ Configurable model parameters
- 🔧 Support for multiple models
- 💻 Cross-platform support (Windows, Linux, macOS)
- 🔍 API Documentation interface
- 🎨 Dark theme with customizable UI
- 📈 Real-time system metrics
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

3. Install the package and dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```

### 2. Building llama.cpp

#### Windows
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Create build directory
mkdir build
cd build

# Configure with CMake (important: use -DBUILD_SHARED_LIBS=ON)
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Create directories and copy the DLLs
mkdir -p ..\..\llama_cpp_api_package\bin\windows
copy Release\*.dll ..\..\llama_cpp_api_package\bin\windows\
cd ..\..
```

#### Linux
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Create build directory
mkdir build && cd build

# Configure with CMake (important: use -DBUILD_SHARED_LIBS=ON)
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Create directories and copy libraries
mkdir -p ../../llama_cpp_api_package/bin/linux
cp lib*.so ../../llama_cpp_api_package/bin/linux/
cd ../..
```

#### macOS
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Create build directory
mkdir build && cd build

# Configure with CMake (important: use -DBUILD_SHARED_LIBS=ON)
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(sysctl -n hw.ncpu)

# Create directories and copy libraries
mkdir -p ../../llama_cpp_api_package/bin/macos
cp lib*.dylib ../../llama_cpp_api_package/bin/macos/
cd ../..
```

### 3. Verify Installation

After building and copying the libraries, verify the setup:

```bash
# Windows
python tests/test_dll.py
```

The test should show available functions and confirm successful loading of the library. Expected output should show key functions like:
- llama_decode
- llama_encode
- llama_model_load_from_file
- llama_init_from_model
- llama_tokenize

### 4. Run Model Tests

To verify model functionality:

```bash
# Test with a specific model
python run_tests.py models/your-model.gguf

# Example with Mistral model
python run_tests.py models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

The tests will verify:
- Model loading/unloading
- Basic text generation
- Chat functionality
- Memory management

### 5. Starting the Servers

#### Backend Server
Start the backend API server using one of these methods:

```bash
# Method 1: Using llama_server.py directly (recommended for development)
python llama_cpp_api_package/llama_server.py

# Method 2: Using run.py
python run.py server --host 0.0.0.0 --port 8080
```

The backend server will start at `http://localhost:8080`

You can verify the server is running by checking:
```bash
# Check server status
curl http://localhost:8080/api/metrics
```

#### Frontend Server
1. Navigate to the UI directory:
```bash
cd llama-ui
```

2. Install dependencies (first time only):
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

#### Environment Variables

1. Backend environment variables (create a `.env` file in the root directory):
```env
LLAMA_API_HOST=0.0.0.0
LLAMA_API_PORT=8080
LLAMA_MODEL_PATH=./models
LLAMA_NUM_THREADS=4
LLAMA_GPU_LAYERS=0
LLAMA_DEBUG=1
```

2. Frontend environment variables (create a `.env` file in the `llama-ui` directory):
```env
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080
```

### Troubleshooting

1. If you see deprecation warnings about `on_event` when starting the server:
```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
```
These warnings can be safely ignored for now. They're related to FastAPI's event handling and don't affect functionality.

2. If the backend server doesn't start:
   - Check if port 8080 is already in use
   - Try running the server directly using `llama_server.py`
   - Verify all DLLs are in the correct location (`llama_cpp_api_package/bin/windows/`)

3. If the frontend server doesn't start:
   - Verify you're in the `llama-ui` directory
   - Check if `node_modules` exists, if not run `npm install`
   - Try a different port if 3000 is in use

4. If model tests fail:
   - Verify the model file exists and is not corrupted
   - Check system memory availability
   - Ensure DLLs are properly loaded (run `test_dll.py` first)

### Model Compatibility

#### RoPE Parameters and Model Architecture

Different models may use varying RoPE (Rotary Position Embedding) configurations. Here's what you need to know:

1. **Standard Models**
   - Most models use standard RoPE configurations
   - Default parameters work out of the box
   - Examples: Mistral, WizardLM

2. **Custom/Hybrid Models**
   - Some models use non-standard RoPE dimensions
   - May require specific configuration
   - Examples: Ayla, MOE models

#### Known Model-Specific Configurations

1. **Ayla Models**
   - Uses 128 RoPE dimensions (non-standard)
   - Requires custom configuration:
   ```python
   {
       "rope_dimension_count": 128,
       "rope_freq_base": 1000000.0,
       "n_ctx": 2048
   }
   ```

2. **DeepSeek Models**
   - Successfully runs with default configuration
   - Optimal parameters:
   ```python
   {
       "rope_freq_base": 10000.0,
       "rope_scaling_type": "linear",
       "n_ctx": 4096
   }
   ```

3. **MOE Models**
   - May require specific tensor configurations
   - Recommended settings:
   ```python
   {
       "n_threads": 4,
       "n_batch": 256,
       "n_gpu_layers": 0
   }
   ```

#### Testing Model Compatibility

To test if your model is compatible:

```bash
# Test specific model configuration
python llama_cpp_api_package/test_models.py --model path/to/your/model.gguf

# View model metadata
python read_gguf_metadata.py path/to/your/model.gguf
```

#### Troubleshooting Model Issues

1. **RoPE Dimension Mismatch**
   - Error: "invalid n_rot: X, expected Y"
   - Solution: Use model-specific configuration with correct RoPE dimensions

2. **Memory Issues**
   - Error: "Failed to load model: insufficient memory"
   - Solution: Reduce batch size or context length, or try quantized versions

3. **Tensor Type Errors**
   - Error: "GGML_ASSERT: ..."
   - Solution: Check model compatibility and quantization format

For detailed logs and debugging:
```bash
export LLAMA_DEBUG=1
python llama_cpp_api_package/test_models.py --verbose
```

## Configuration

### Model Parameters
The web interface allows you to configure:
- Context window size (num_ctx): Controls the context length
- Batch size (num_batch): Affects processing speed
- Temperature: Controls response creativity
- Top-K sampling: Filters vocabulary choices
- Top-P sampling: Controls response diversity
- Min-P: Alternative to top-p for quality/variety balance
- Repeat penalty: Prevents repetitive responses
- Number of threads: CPU thread utilization
- Memory locking: Improves performance
- Random seed: Ensures reproducible responses
- Maximum tokens: Controls response length
- Mirostat settings: Advanced sampling control

### Environment Variables
- `LLAMA_API_HOST`: Host to bind the server (default: 0.0.0.0)
- `LLAMA_API_PORT`: Port to run the server (default: 8080)
- `LLAMA_MODEL_PATH`: Default path to look for models
- `LLAMA_NUM_THREADS`: Default number of CPU threads to use
- `LLAMA_GPU_LAYERS`: Number of layers to offload to GPU (if available)
- `LLAMA_DEBUG`: Enable debug logging (set to 1)

## API Endpoints

### Available Endpoints
- `GET /api/models`: List available models
- `POST /api/models/{model_id}/load`: Load a model
- `POST /api/models/{model_id}/unload`: Unload a model
- `POST /api/models/{model_id}/chat`: Chat with a model
- `GET /api/metrics`: Get system metrics

Detailed API documentation is available in the web interface under the "API Documentation" tab.

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

- Check system status:
```bash
python run.py status
```

- Monitor logs:
```bash
python run.py logs
```

### Troubleshooting

1. If the server fails to start:
   - Check if the required libraries are in the correct location
   - Verify port 8080 is not in use
   - Check system logs for errors

2. If a model fails to load:
   - Verify sufficient RAM is available
   - Check model file integrity
   - Ensure correct model path

3. GPU-related issues:
   - Verify CUDA installation (if using GPU)
   - Check GPU memory availability
   - Update GPU drivers

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

EXPOSE 8080
CMD ["llama-cpp-api", "--host", "0.0.0.0", "--port", "8080"]
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
        - containerPort: 8080
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