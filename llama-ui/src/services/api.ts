import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

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

export interface CpuInfo {
  name: string;
  cores: {
    physical: number;
    logical: number;
  };
  frequency?: string;
  architecture: string;
  utilization: number;
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

export interface MemoryInfo {
    total_gb: number;
    used_gb: number;
    model_memory_gb: number;
}

export interface SystemState {
    cpu: CpuInfo;
    memory: MemoryInfo;
    gpu: GpuInfo;
}

export interface ModelsResponse {
  models: {
    available: Array<{
      id: string;
      name: string;
      path: string;
      size_mb: number;
      required_memory_mb: number;
      can_load: boolean;
    }>;
    loaded: Record<string, {
      status: string;
      load_time?: string;
      last_used?: string;
      parameters?: ModelParameters;
      memory_used_mb?: number;
      error?: string;
    }>;
  };
  system_state: SystemState;
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
  getModels: async () => {
    try {
      const res = await api.get<ModelsResponse>('/api/models');
      return res.data;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      throw new Error('Failed to fetch model information');
    }
  },
  
  loadModel: async (modelId: string, parameters: ModelParameters) => {
    try {
      const res = await api.post(`/api/models/${encodeURIComponent(modelId)}/load`, { parameters });
      return res.data;
    } catch (error: any) {
      console.error('Failed to load model:', error);
      throw new Error(error.response?.data?.detail || 'Failed to load model');
    }
  },
  
  unloadModel: async (modelId: string) => {
    try {
      const res = await api.post(`/api/models/${encodeURIComponent(modelId)}/unload`);
      return res.data;
    } catch (error: any) {
      console.error('Failed to unload model:', error);
      throw new Error(error.response?.data?.detail || 'Failed to unload model');
    }
  },
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