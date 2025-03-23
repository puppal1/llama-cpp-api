import React from 'react';

interface Endpoint {
  method: string;
  path: string;
  description: string;
  requestBody?: any;
  responseBody?: any;
}

const endpoints: Endpoint[] = [
  {
    method: 'GET',
    path: '/api/models',
    description: 'List all available models',
    responseBody: {
      "status": "success",
      "timestamp": "2024-03-21T12:00:00Z",
      "data": {
        "available": ["model1", "model2"],
        "loaded": ["model1"]
      },
      "server_info": {
        "version": "1.0.0",
        "uptime": 3600
      }
    }
  },
  {
    method: 'POST',
    path: '/api/models/{model_id}/load',
    description: 'Load a specific model into memory',
    requestBody: {
      "n_ctx": 2048,
      "n_batch": 512,
      "n_gpu_layers": 0
    },
    responseBody: {
      "status": "success",
      "message": "Model {model_id} loaded successfully"
    }
  },
  {
    method: 'POST',
    path: '/api/models/{model_id}/chat',
    description: 'Chat with a specific model',
    requestBody: {
      "messages": [
        {
          "role": "user",
          "content": "Hello!"
        }
      ],
      "stream": false
    },
    responseBody: {
      "model": "model_id",
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
          }
        }
      ]
    }
  },
  {
    method: 'POST',
    path: '/api/models/{model_id}/unload',
    description: 'Unload a specific model',
    responseBody: {
      "status": "success",
      "message": "Model {model_id} unloaded successfully"
    }
  },
  {
    method: 'POST',
    path: '/completions',
    description: 'Generate text completions',
    requestBody: {
      "prompt": "Once upon a time",
      "max_tokens": 100,
      "temperature": 0.7
    },
    responseBody: {
      "choices": [
        {
          "text": "there was a magical kingdom...",
          "finish_reason": "length"
        }
      ]
    }
  },
  {
    method: 'POST',
    path: '/chat/completions',
    description: 'Generate chat completions with support for conversation history',
    requestBody: {
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Hello!"
        }
      ]
    },
    responseBody: {
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
          }
        }
      ]
    }
  },
  {
    method: 'POST',
    path: '/embeddings',
    description: 'Generate embeddings for given text',
    requestBody: {
      "input": "The quick brown fox jumps over the lazy dog"
    },
    responseBody: {
      "data": [
        {
          "embedding": [0.1, 0.2, 0.3],
          "index": 0
        }
      ]
    }
  },
  {
    method: 'POST',
    path: '/rerank',
    description: 'Rerank documents according to relevance to a query',
    requestBody: {
      "query": "What is machine learning?",
      "documents": [
        "Machine learning is a field of AI...",
        "Biology is the study of life...",
        "Chemistry deals with matter..."
      ]
    },
    responseBody: {
      "model": "reranker-model",
      "object": "list",
      "results": [
        {
          "index": 0,
          "relevance_score": 0.95
        },
        {
          "index": 1,
          "relevance_score": 0.3
        }
      ],
      "usage": {
        "prompt_tokens": 50,
        "total_tokens": 50
      }
    }
  },
  {
    method: 'POST',
    path: '/tokenize',
    description: 'Convert text into tokens',
    requestBody: {
      "content": "Hello world!"
    },
    responseBody: {
      "tokens": [
        {"id": 123, "piece": "Hello"},
        {"id": 456, "piece": "world"},
        {"id": 789, "piece": "!"}
      ]
    }
  },
  {
    method: 'POST',
    path: '/detokenize',
    description: 'Convert tokens back into text',
    requestBody: {
      "tokens": [
        {"id": 123, "piece": "Hello"},
        {"id": 456, "piece": "world"}
      ]
    },
    responseBody: {
      "content": "Hello world"
    }
  },
  {
    method: 'GET',
    path: '/lora-adapters',
    description: 'Get list of all LoRA adapters',
    responseBody: [
      {
        "id": 0,
        "path": "adapter1.gguf",
        "scale": 1.0
      },
      {
        "id": 1,
        "path": "adapter2.gguf",
        "scale": 0.0
      }
    ]
  },
  {
    method: 'POST',
    path: '/lora-adapters',
    description: 'Set LoRA adapter scales',
    requestBody: [
      {
        "id": 0,
        "scale": 0.5
      },
      {
        "id": 1,
        "scale": 1.0
      }
    ],
    responseBody: {
      "status": "success"
    }
  }
];

const ApiDocs: React.FC = () => {
  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">API Documentation</h2>
      <div className="space-y-6">
        {endpoints.map((endpoint, index) => (
          <div key={index} className="border rounded-lg p-4 bg-white dark:bg-gray-800">
            <div className="flex items-center gap-2 mb-2">
              <span className={`px-2 py-1 rounded text-sm font-mono
                ${endpoint.method === 'GET' ? 'bg-green-100 text-green-800' : 
                  endpoint.method === 'POST' ? 'bg-blue-100 text-blue-800' : 
                  endpoint.method === 'PUT' ? 'bg-yellow-100 text-yellow-800' : 
                  'bg-red-100 text-red-800'}`}>
                {endpoint.method}
              </span>
              <span className="font-mono text-sm">{endpoint.path}</span>
            </div>
            <p className="text-gray-600 dark:text-gray-300 mb-4">{endpoint.description}</p>
            
            {endpoint.requestBody && (
              <div className="mb-4">
                <h4 className="font-semibold mb-2">Request Body:</h4>
                <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded-lg overflow-x-auto">
                  {JSON.stringify(endpoint.requestBody, null, 2)}
                </pre>
              </div>
            )}
            
            {endpoint.responseBody && (
              <div>
                <h4 className="font-semibold mb-2">Response Body:</h4>
                <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded-lg overflow-x-auto">
                  {JSON.stringify(endpoint.responseBody, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ApiDocs; 