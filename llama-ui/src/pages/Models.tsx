import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { TbLoader, TbBrandOpenai, TbCircleCheck, TbX } from 'react-icons/tb';

type Model = {
  id: string;
  metadata: {
    parameters?: {
      name?: string;
      context_length?: number;
      gpu_layers?: number;
    };
    layers?: number;
    status?: string;
    loaded?: boolean;
    load_time?: string | null;
    architecture?: string;
  };
  size: number;
};

type ModelParams = {
  n_gpu_layers?: number;
  n_ctx?: number;
  n_batch?: number;
  threads?: number;
  use_mlock?: boolean;
  f16_kv?: boolean;
};

const Models = () => {
  const queryClient = useQueryClient();
  const [modelParams, setModelParams] = useState<ModelParams>({
    n_gpu_layers: 0,
    n_ctx: 2048,
    n_batch: 512,
    threads: 4,
    use_mlock: true,
    f16_kv: true,
  });
  const [showParams, setShowParams] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Fetch models
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get('/api/v2/models');
      return response.data;
    },
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Load model mutation
  const loadModelMutation = useMutation({
    mutationFn: async ({ modelId, params }: { modelId: string, params: ModelParams }) => {
      await axios.post(`/api/v2/models/${modelId}/load`, params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  // Unload model mutation
  const unloadModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      await axios.post(`/api/v2/models/${modelId}/unload`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleLoadModel = (modelId: string) => {
    loadModelMutation.mutate({ modelId, params: modelParams });
  };

  const handleUnloadModel = (modelId: string) => {
    unloadModelMutation.mutate(modelId);
  };

  return (
    <div className="mt-6">
      <h2 className="text-xl text-terminal-green mb-4">LLM Models</h2>
      
      {/* Model Parameters */}
      <div className="mb-6">
        <button 
          className="text-terminal-green border border-terminal-green px-3 py-1 rounded-md hover:bg-terminal-green/20 transition-colors"
          onClick={() => setShowParams(!showParams)}
        >
          {showParams ? 'Hide Parameters' : 'Show Parameters'}
        </button>
        
        {showParams && (
          <div className="terminal-window p-4 mt-3">
            <h3 className="text-terminal-green mb-3">Model Loading Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm mb-1">GPU Layers</label>
                <input 
                  type="number" 
                  value={modelParams.n_gpu_layers}
                  onChange={(e) => setModelParams({...modelParams, n_gpu_layers: parseInt(e.target.value) || 0})}
                  className="bg-terminal-background border border-terminal-green p-2 w-full rounded-md text-terminal-foreground"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">Context Length</label>
                <input 
                  type="number" 
                  value={modelParams.n_ctx}
                  onChange={(e) => setModelParams({...modelParams, n_ctx: parseInt(e.target.value) || 0})}
                  className="bg-terminal-background border border-terminal-green p-2 w-full rounded-md text-terminal-foreground"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">Batch Size</label>
                <input 
                  type="number" 
                  value={modelParams.n_batch}
                  onChange={(e) => setModelParams({...modelParams, n_batch: parseInt(e.target.value) || 0})}
                  className="bg-terminal-background border border-terminal-green p-2 w-full rounded-md text-terminal-foreground"
                />
              </div>
              <div>
                <label className="block text-sm mb-1">Threads</label>
                <input 
                  type="number" 
                  value={modelParams.threads}
                  onChange={(e) => setModelParams({...modelParams, threads: parseInt(e.target.value) || 0})}
                  className="bg-terminal-background border border-terminal-green p-2 w-full rounded-md text-terminal-foreground"
                />
              </div>
              <div className="flex items-center space-x-4 mt-6">
                <div className="flex items-center">
                  <input 
                    type="checkbox" 
                    id="use_mlock" 
                    checked={modelParams.use_mlock}
                    onChange={(e) => setModelParams({...modelParams, use_mlock: e.target.checked})}
                    className="mr-2"
                  />
                  <label htmlFor="use_mlock">MLock</label>
                </div>
                <div className="flex items-center">
                  <input 
                    type="checkbox" 
                    id="f16_kv" 
                    checked={modelParams.f16_kv}
                    onChange={(e) => setModelParams({...modelParams, f16_kv: e.target.checked})}
                    className="mr-2"
                  />
                  <label htmlFor="f16_kv">f16_kv</label>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Available Models */}
      <div className="terminal-window p-4 mb-6">
        <h3 className="text-terminal-green mb-3">Available Models</h3>
        
        {isLoadingModels ? (
          <div className="text-terminal-dimmed flex items-center">
            <TbLoader className="animate-spin mr-2" />
            Loading models...
          </div>
        ) : !modelsData?.available_models || modelsData.available_models.length === 0 ? (
          <div className="text-terminal-dimmed">No available models found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-terminal-dimmed/40">
                  <th className="text-left py-2 px-3">Model ID</th>
                  <th className="text-left py-2 px-3">Size</th>
                  <th className="text-left py-2 px-3">Architecture</th>
                  <th className="text-left py-2 px-3">Context</th>
                  <th className="text-left py-2 px-3">Action</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.available_models.map((model: Model) => (
                  <tr key={model.id} className="border-b border-terminal-dimmed/20 hover:bg-terminal-highlight/10">
                    <td className="py-2 px-3">
                      <div className="flex items-center">
                        <TbBrandOpenai className="mr-2 text-terminal-green" />
                        {model.id.replace('.gguf', '')}
                      </div>
                    </td>
                    <td className="py-2 px-3">{formatBytes(model.size)}</td>
                    <td className="py-2 px-3">{model.metadata?.architecture || 'Unknown'}</td>
                    <td className="py-2 px-3">{model.metadata?.parameters?.context_length || 'Default'}</td>
                    <td className="py-2 px-3">
                      <button 
                        className="text-terminal-green border border-terminal-green px-2 py-1 rounded-md hover:bg-terminal-green/20 transition-colors"
                        onClick={() => handleLoadModel(model.id)}
                        disabled={loadModelMutation.isPending && selectedModelId === model.id}
                      >
                        {loadModelMutation.isPending && selectedModelId === model.id ? 
                          <span className="flex items-center"><TbLoader className="animate-spin mr-1" /> Loading...</span> : 
                          'Load Model'
                        }
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      
      {/* Loaded Models */}
      <div className="terminal-window p-4">
        <h3 className="text-terminal-green mb-3">Loaded Models</h3>
        
        {isLoadingModels ? (
          <div className="text-terminal-dimmed flex items-center">
            <TbLoader className="animate-spin mr-2" />
            Loading models...
          </div>
        ) : !modelsData?.loaded_models || modelsData.loaded_models.length === 0 ? (
          <div className="text-terminal-dimmed">No loaded models</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-terminal-dimmed/40">
                  <th className="text-left py-2 px-3">Model ID</th>
                  <th className="text-left py-2 px-3">Size</th>
                  <th className="text-left py-2 px-3">Status</th>
                  <th className="text-left py-2 px-3">Loaded At</th>
                  <th className="text-left py-2 px-3">Action</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.loaded_models.map((model: Model) => (
                  <tr key={model.id} className="border-b border-terminal-dimmed/20 hover:bg-terminal-highlight/10">
                    <td className="py-2 px-3">
                      <div className="flex items-center">
                        <TbBrandOpenai className="mr-2 text-terminal-green" />
                        {model.id.replace('.gguf', '')}
                      </div>
                    </td>
                    <td className="py-2 px-3">{formatBytes(model.size)}</td>
                    <td className="py-2 px-3">
                      <div className="flex items-center">
                        <TbCircleCheck className="mr-1 text-green-500" />
                        Active
                      </div>
                    </td>
                    <td className="py-2 px-3">{model.metadata?.load_time ? new Date(model.metadata.load_time).toLocaleString() : 'Unknown'}</td>
                    <td className="py-2 px-3">
                      <button 
                        className="text-red-400 border border-red-400 px-2 py-1 rounded-md hover:bg-red-400/20 transition-colors flex items-center"
                        onClick={() => handleUnloadModel(model.id)}
                        disabled={unloadModelMutation.isPending && selectedModelId === model.id}
                      >
                        {unloadModelMutation.isPending && selectedModelId === model.id ? 
                          <span className="flex items-center"><TbLoader className="animate-spin mr-1" /> Unloading...</span> : 
                          <><TbX className="mr-1" /> Unload</>
                        }
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Models; 