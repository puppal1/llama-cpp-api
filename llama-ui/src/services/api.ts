import axios from 'axios';

const BASE_URL = 'http://localhost:8080';

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ModelParameters {
  num_ctx: number;
  num_batch: number;
  num_thread: number;
  num_gpu: number;
  mlock: boolean;
  mmap: boolean;
  seed?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  num_predict?: number;
  stop?: string[];
}

export interface LoadedModel {
  status: 'loaded' | 'unloaded' | 'loading' | 'error';
  load_time: string | null;
  last_used: string | null;
  parameters: ModelParameters;
  memory_used_mb: number;
  error: string | null;
}

export interface AvailableModel {
  id: string;
  name: string;
  path: string;
  size_mb: number;
  required_memory_mb: number;
  can_load: boolean;
}

export interface GpuInfo {
  available: boolean;
  status: string;
  name: string;
  memory?: {
    total_mb: number;
    free_mb: number;
    used_mb: number;
  };
  utilization?: {
    gpu_percent: number;
    memory_percent: number;
  };
  temperature_celsius?: number;
  power_watts?: number;
}

export interface ModelsResponse {
  models: {
    available: AvailableModel[];
    loaded: Record<string, LoadedModel>;
  };
  system_state: {
    memory: {
      total_gb: number;
      used_gb: number;
    };
    gpu: GpuInfo;
  };
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  parameters?: ModelParameters;
  stream?: boolean;
}

export interface ChatResponse {
  model: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
    };
  }>;
}

export const modelApi = {
  getModels: () => api.get<ModelsResponse>('/api/models').then(res => res.data),
  
  loadModel: (modelId: string, parameters: ModelParameters) => 
    api.post(`/api/models/${encodeURIComponent(modelId)}/load`, { parameters }),
  
  unloadModel: (modelId: string) => 
    api.post(`/api/models/${encodeURIComponent(modelId)}/unload`),
};

export const chatApi = {
  sendMessage: (modelId: string, request: ChatRequest) =>
    api.post<ChatResponse>(`/api/models/${encodeURIComponent(modelId)}/chat`, request),
};

// Add error interceptor
api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api; 