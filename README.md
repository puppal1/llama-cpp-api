# Llama.cpp API

A FastAPI-based REST API wrapper for llama.cpp, providing easy model management and inference capabilities with both CPU and GPU support.

## Features

- Load and manage multiple llama.cpp models
- CPU and GPU (CUDA) inference support
- Automatic memory management and model cleanup
- Streaming and batch inference
- REST API with OpenAPI documentation
- Async/await throughout for high performance

## Prerequisites

- Python 3.8+
- CUDA Toolkit 11.4+ (for GPU support)
- llama.cpp compiled as a shared library

### Compiling llama.cpp

1. Clone llama.cpp repository:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

2. For CPU-only build:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

3. For CUDA support:
```bash
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
```

4. Copy the compiled library:
- Windows: Copy `build/Release/llama.dll` to this project's directory
- Linux: Copy `build/libllama.so` to `/usr/local/lib/` or this project's directory

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python -m uvicorn llama_api:app --host 0.0.0.0 --port 8080
```

2. Access the OpenAPI documentation:
```
http://localhost:8080/docs
```

3. Run the test script:
```bash
python tests/test_llama_api.py
```

## API Endpoints

- `POST /models/{model_id}/load` - Load a model
- `POST /models/{model_id}/unload` - Unload a model
- `GET /models` - List loaded models
- `POST /models/{model_id}/chat` - Chat with a model

## Example Usage

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
    # Load model
    config = {
        "model_path": "models/model.gguf",
        "n_ctx": 2048,
        "n_gpu_layers": 32  # Set to 0 for CPU-only
    }
    await client.post("/models/my_model/load", json=config)
    
    # Chat
    response = await client.post("/models/my_model/chat", json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    })
    print(response.json())
```

## Configuration

### Model Parameters

- `n_ctx`: Context window size (default: 2048)
- `n_batch`: Batch size for prompt processing (default: 512)
- `n_threads`: Number of CPU threads (default: 4)
- `n_gpu_layers`: Number of layers to offload to GPU (default: 0)
- `use_mlock`: Force system to keep model in RAM (default: False)
- `use_mmap`: Use memory mapping for model loading (default: True)

### Generation Parameters

- `temperature`: Sampling temperature (0.0 - 2.0)
- `top_p`: Nucleus sampling threshold (0.0 - 1.0)
- `max_tokens`: Maximum tokens to generate
- `stop`: List of stop sequences
- `stream`: Enable token streaming

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 