# LLama.cpp API Reference

This documentation describes the LLama.cpp API endpoints. All endpoints are prefixed with `/api/v2`.
Responses are returned in JSON format unless otherwise specified.

## Table of Contents

- [Models](#models)
  - [List Available Models](#list-available-models)
  - [Get Model Information](#get-model-information)
  - [Load Model](#load-model)
  - [Unload Model](#unload-model)
- [Chat](#chat)
  - [Chat Completion](#chat-completion)
  - [Chat Completion Stream](#chat-completion-stream)
- [System](#system)
  - [System Metrics](#system-metrics)
  - [Health Check](#health-check)

## Models

### List Available Models

```http
GET /api/v2/models
```

Returns a list of all available models in the models directory and currently loaded models.

#### Response

```json
{
  "available_models": [
    {
      "id": "llama-2-7b-chat.gguf",
      "size": 4702869504,
      "modified": "2024-01-15T10:30:00Z",
      "metadata": {
        "model_type": "llama",
        "parameters": {
          "context_length": 4096,
          "embedding_length": 4096,
          "block_count": 32
        }
      }
    }
  ],
  "loaded_models": {
    "llama-2-7b-chat.gguf": {
      "status": "loaded",
      "load_time": "2024-01-15T12:00:00Z"
    }
  }
}
```

### Get Model Information

```http
GET /api/v2/models/{model_id}
```

Retrieves detailed information about a specific model.

#### Parameters

| Name     | Type   | Required | Description                    |
|----------|--------|----------|--------------------------------|
| model_id | string | Yes      | The ID (filename) of the model |

#### Response

```json
{
  "id": "llama-2-7b-chat.gguf",
  "size": 4702869504,
  "modified": "2024-01-15T10:30:00Z",
  "status": "loaded",
  "metadata": {
    "model_type": "llama",
    "parameters": {
      "context_length": 4096,
      "embedding_length": 4096,
      "block_count": 32
    }
  },
  "performance": {
    "load_time": "2024-01-15T12:00:00Z",
    "tokens_processed": 15000,
    "average_speed": 150
  }
}
```

### Load Model

```http
POST /api/v2/models/{model_id}/load
```

Loads a model into memory with specified parameters.

#### Parameters

| Name     | Type   | Required | Description                    |
|----------|--------|----------|--------------------------------|
| model_id | string | Yes      | The ID (filename) of the model |

#### Request Body

```json
{
  "n_gpu_layers": 32,
  "n_ctx": 2048,
  "n_batch": 512,
  "threads": 4,
  "use_mlock": true,
  "f16_kv": true
}
```

#### Request Body Parameters

| Name         | Type    | Required | Description                                           |
|-------------|---------|----------|-------------------------------------------------------|
| n_gpu_layers | integer | No       | Number of layers to offload to GPU                   |
| n_ctx        | integer | No       | Context window size                                  |
| n_batch      | integer | No       | Batch size for prompt processing                     |
| threads      | integer | No       | Number of threads to use for computation             |
| use_mlock    | boolean | No       | Lock memory to prevent swapping                      |
| f16_kv       | boolean | No       | Use half-precision for key/value cache              |

#### Response

```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "load_time": "2024-01-15T12:00:00Z",
  "memory_used": "4.5GB"
}
```

### Unload Model

```http
POST /api/v2/models/{model_id}/unload
```

Unloads a model from memory.

#### Parameters

| Name     | Type   | Required | Description                    |
|----------|--------|----------|--------------------------------|
| model_id | string | Yes      | The ID (filename) of the model |

#### Response

```json
{
  "status": "success",
  "message": "Model unloaded successfully",
  "unload_time": "2024-01-15T12:30:00Z"
}
```

## Chat

### Chat Completion

```http
POST /api/v2/chat/{model_id}
```

Generate a chat completion with the specified model.

#### Parameters

| Name        | Type    | Required | Description                                    |
|-------------|---------|----------|------------------------------------------------|
| model_id    | string  | Yes      | The ID of the model to use for chat           |
| messages    | array   | Yes      | Array of message objects with role and content |
| temperature | number  | No       | Sampling temperature (0.0 - 2.0)              |
| top_p       | number  | No       | Nucleus sampling parameter                     |
| n_predict   | integer | No       | Maximum number of tokens to predict            |

#### Request Body

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "n_predict": 100
}
```

#### Response

```json
{
  "id": "chat-12345",
  "object": "chat.completion",
  "created": 1705320000,
  "model": "llama-2-7b-chat.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 33,
    "completion_tokens": 7,
    "total_tokens": 40
  }
}
```

### Chat Completion Stream

```http
POST /api/v2/chat/{model_id}/stream
```

Generate a streaming chat completion with the specified model. Returns Server-Sent Events (SSE).

#### Parameters

| Name        | Type    | Required | Description                                    |
|-------------|---------|----------|------------------------------------------------|
| model_id    | string  | Yes      | The ID of the model to use for chat           |
| messages    | array   | Yes      | Array of message objects with role and content |
| temperature | number  | No       | Sampling temperature (0.0 - 2.0)              |
| top_p       | number  | No       | Nucleus sampling parameter                     |

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Tell me a story"
    }
  ],
  "temperature": 0.8,
  "top_p": 0.95
}
```

#### Response (Server-Sent Events)

```
data: {"id":"chat-67890","object":"chat.completion.chunk","created":1705320001,"model":"llama-2-7b-chat.gguf","choices":[{"index":0,"delta":{"role":"assistant","content":"Once"},"finish_reason":null}]}

data: {"id":"chat-67890","object":"chat.completion.chunk","created":1705320001,"model":"llama-2-7b-chat.gguf","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chat-67890","object":"chat.completion.chunk","created":1705320001,"model":"llama-2-7b-chat.gguf","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

data: {"id":"chat-67890","object":"chat.completion.chunk","created":1705320001,"model":"llama-2-7b-chat.gguf","choices":[{"index":0,"delta":{"content":" time..."},"finish_reason":"stop"}]}

data: [DONE]
```

## System

### System Metrics

```http
GET /api/v2/metrics
```

Get current system metrics including CPU, memory, and GPU usage.

#### Response

```json
{
  "cpu": {
    "utilization": 45.2,
    "cores": {
      "physical": 8,
      "logical": 16
    },
    "frequency": {
      "current": 3600,
      "min": 2200,
      "max": 4100
    }
  },
  "memory": {
    "total_mb": 32768,
    "available_mb": 24576,
    "used_mb": 8192,
    "percent": 25
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA RTX 4090",
    "memory": {
      "total_mb": 24576,
      "used_mb": 4096,
      "percent": 16.7
    }
  },
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### Health Check

```http
GET /api/v2/health
```

Check the health status of the API server.

#### Response

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": "2d 3h 45m",
  "last_error": null
}
``` 