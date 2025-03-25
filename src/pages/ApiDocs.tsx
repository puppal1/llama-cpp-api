import React from 'react';
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { TbApi, TbChevronDown, TbChevronUp, TbCode } from 'react-icons/tb';

type Endpoint = {
  title: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  requestParams?: Record<string, {
    type: string;
    description: string;
    required: boolean;
  }>;
  requestBody?: string;
  responseBody: string;
  expanded: boolean;
};

const ApiDocs = () => {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([
    {
      title: 'List Models',
      method: 'GET',
      path: '/api/v2/models',
      description: 'Returns a list of available models and currently loaded models.',
      responseBody: `{
  "available_models": [
    {
      "id": "model1.gguf",
      "metadata": {
        "architecture": "LLaMA",
        "parameters": {
          "context_length": 4096
        }
      },
      "size": 4294967296
    }
  ],
  "loaded_models": []
}`,
      expanded: false
    },
    {
      title: 'Get Model Info',
      method: 'GET',
      path: '/api/v2/models/{model_id}',
      description: 'Get detailed information about a specific model.',
      responseBody: `{
  "id": "model1.gguf",
  "metadata": {
    "architecture": "LLaMA",
    "parameters": {
      "context_length": 4096,
      "gpu_layers": 0
    },
    "loaded": false
  },
  "size": 4294967296
}`,
      expanded: false
    },
    {
      title: 'Load Model',
      method: 'POST',
      path: '/api/v2/models/{model_id}/load',
      description: 'Load a model into memory with optional parameters.',
      requestBody: `{
  "n_gpu_layers": 0,
  "n_ctx": 2048,
  "n_batch": 512,
  "threads": 4,
  "use_mlock": true,
  "f16_kv": true
}`,
      responseBody: `{
  "status": "success",
  "message": "Model loaded successfully",
  "model_id": "model1.gguf"
}`,
      expanded: false
    },
    {
      title: 'Unload Model',
      method: 'POST',
      path: '/api/v2/models/{model_id}/unload',
      description: 'Unload a model from memory.',
      responseBody: `{
  "status": "success",
  "message": "Model unloaded successfully",
  "model_id": "model1.gguf"
}`,
      expanded: false
    },
    {
      title: 'Chat',
      method: 'POST',
      path: '/api/v2/chat/{model_id}',
      description: 'Chat with a specific model.',
      requestBody: `{
  "messages": [
    {
      "role": "user",
      "content": "Hello, who are you?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false,
  "top_k": 40,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "presence_penalty": 0.1,
  "frequency_penalty": 0.1,
  "num_ctx": 2048
}`,
      responseBody: `{
  "model": "model1.gguf",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "I am an AI assistant powered by the LLaMA model..."
      }
    }
  ]
}`,
      expanded: false
    },
    {
      title: 'System Metrics',
      method: 'GET',
      path: '/api/v2/metrics',
      description: 'Get system metrics including CPU, memory, and GPU information.',
      responseBody: `{
  "memory": {
    "total_memory_gb": 32.0,
    "available_memory_gb": 16.5,
    "memory_usage_percent": 48.4
  },
  "cpu": {
    "cpu_count": 12,
    "cpu_usage_percent": 25.8
  },
  "gpu": [
    {
      "name": "NVIDIA GeForce RTX 3080",
      "memory_total_mb": 10240,
      "memory_used_mb": 2048,
      "memory_free_mb": 8192,
      "gpu_utilization_percent": 30.5
    }
  ]
}`,
      expanded: false
    }
  ]);

  const toggleEndpoint = (index: number) => {
    setEndpoints(prevEndpoints => {
      const newEndpoints = [...prevEndpoints];
      newEndpoints[index].expanded = !newEndpoints[index].expanded;
      return newEndpoints;
    });
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET':
        return 'bg-blue-700 text-white';
      case 'POST':
        return 'bg-green-700 text-white';
      case 'PUT':
        return 'bg-yellow-700 text-black';
      case 'DELETE':
        return 'bg-red-700 text-white';
      default:
        return 'bg-gray-700 text-white';
    }
  };

  return (
    <div className="mt-6">
      <h2 className="text-xl text-terminal-green mb-2 flex items-center">
        <TbApi className="mr-2" size={24} />
        API Documentation
      </h2>
      <p className="text-terminal-dimmed mb-6">
        Complete reference for the LLama.cpp API endpoints. Click on each endpoint to see detailed information.
      </p>
      
      <div className="space-y-4">
        {endpoints.map((endpoint, index) => (
          <div key={index} className="terminal-window">
            <div 
              className="flex items-center cursor-pointer p-2"
              onClick={() => toggleEndpoint(index)}
            >
              <div className={`${getMethodColor(endpoint.method)} px-2 py-1 rounded-md text-xs font-bold mr-3`}>
                {endpoint.method}
              </div>
              <div className="flex-1 font-mono">{endpoint.path}</div>
              <div className="text-terminal-dimmed mr-2">{endpoint.title}</div>
              {endpoint.expanded ? <TbChevronUp /> : <TbChevronDown />}
            </div>
            
            {endpoint.expanded && (
              <div className="p-4 border-t border-terminal-dimmed/30 mt-2">
                <div className="mb-4">
                  <h4 className="text-terminal-green mb-1">Description</h4>
                  <p>{endpoint.description}</p>
                </div>
                
                {endpoint.requestBody && (
                  <div className="mb-4">
                    <h4 className="text-terminal-green mb-1 flex items-center">
                      <TbCode className="mr-1" />
                      Request Body
                    </h4>
                    <SyntaxHighlighter 
                      language="json" 
                      style={vscDarkPlus}
                      customStyle={{
                        backgroundColor: '#0C0C0C',
                        borderRadius: '0.375rem',
                        fontSize: '0.875rem'
                      }}
                    >
                      {endpoint.requestBody}
                    </SyntaxHighlighter>
                  </div>
                )}
                
                <div>
                  <h4 className="text-terminal-green mb-1 flex items-center">
                    <TbCode className="mr-1" />
                    Response Body
                  </h4>
                  <SyntaxHighlighter 
                    language="json" 
                    style={vscDarkPlus}
                    customStyle={{
                      backgroundColor: '#0C0C0C',
                      borderRadius: '0.375rem',
                      fontSize: '0.875rem'
                    }}
                  >
                    {endpoint.responseBody}
                  </SyntaxHighlighter>
                </div>
                
                <div className="mt-4 text-terminal-dimmed text-sm">
                  <p>Try this endpoint using curl:</p>
                  <SyntaxHighlighter 
                    language="bash" 
                    style={vscDarkPlus}
                    customStyle={{
                      backgroundColor: '#0C0C0C',
                      borderRadius: '0.375rem',
                      fontSize: '0.875rem'
                    }}
                  >
                    {endpoint.method === 'GET' 
                      ? `curl -X ${endpoint.method} 'http://localhost:8000${endpoint.path.replace(/{([^}]+)}/g, 'example_$1')}'`
                      : `curl -X ${endpoint.method} 'http://localhost:8000${endpoint.path.replace(/{([^}]+)}/g, 'example_$1')}' \\
  -H 'Content-Type: application/json' \\
  -d '${endpoint.requestBody ? endpoint.requestBody.replace(/\n/g, '\n  ') : '{}'}'`
                    }
                  </SyntaxHighlighter>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ApiDocs; 