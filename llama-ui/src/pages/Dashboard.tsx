import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { TbServer, TbCpu, TbBrandOpenai, TbLoader, TbTrash } from 'react-icons/tb';

type SystemMetrics = {
  cpu: {
    utilization: number;
    cores: {
      physical: number;
      logical: number;
    };
    frequency: {
      current: number;
      min: number;
      max: number;
    };
  };
  memory: {
    total_mb: number;
    available_mb: number;
    used_mb: number;
    percent: number;
  };
  gpu: {
    available: boolean;
    name: string;
    memory: {
      total_mb: number;
      used_mb: number;
      percent: number;
    };
  };
  timestamp: string;
};

type ModelParameters = {
  n_gpu_layers: number;
  n_ctx: number;
  n_batch: number;
  threads: number;
  use_mlock: boolean;
  f16_kv: boolean;
};

const Dashboard = () => {
  const [timeConnected, setTimeConnected] = useState(0);
  const [smoothMetrics, setSmoothMetrics] = useState<SystemMetrics | null>(null);
  const previousMetrics = useRef<SystemMetrics | null>(null);
  const frameRef = useRef<number>();
  const queryClient = useQueryClient();

  // Fetch system metrics
  const { data: systemMetrics, isLoading: isLoadingMetrics } = useQuery<SystemMetrics>({
    queryKey: ['systemMetrics'],
    queryFn: async () => {
      const response = await axios.get('/api/v2/metrics');
      return response.data;
    },
    refetchInterval: 2000,
  });

  // Fetch models data
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get('/api/v2/models');
      return response.data;
    },
    refetchInterval: 5000,
  });

  // Model operations mutations
  const loadModelMutation = useMutation({
    mutationFn: async ({ modelId, parameters }: { modelId: string; parameters: ModelParameters }) => {
      await axios.post(`/api/v2/models/${modelId}/load`, parameters);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  const unloadModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      await axios.post(`/api/v2/models/${modelId}/unload`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });

  // Smooth transition effect for metrics
  useEffect(() => {
    if (!systemMetrics) return;

    if (!smoothMetrics) {
      setSmoothMetrics(systemMetrics);
      previousMetrics.current = systemMetrics;
      return;
    }

    previousMetrics.current = smoothMetrics;
    const startTime = performance.now();
    const duration = 1000;

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
      const ease = easeOut(progress);
      const interpolate = (prev: number, next: number) => prev + (next - prev) * ease;

      const newMetrics: SystemMetrics = {
        ...systemMetrics,
        cpu: {
          ...systemMetrics.cpu,
          utilization: interpolate(previousMetrics.current!.cpu.utilization, systemMetrics.cpu.utilization),
        },
        memory: {
          ...systemMetrics.memory,
          percent: interpolate(previousMetrics.current!.memory.percent, systemMetrics.memory.percent),
          used_mb: interpolate(previousMetrics.current!.memory.used_mb, systemMetrics.memory.used_mb),
        },
        gpu: systemMetrics.gpu && previousMetrics.current?.gpu ? {
          ...systemMetrics.gpu,
          memory: {
            ...systemMetrics.gpu.memory,
            percent: interpolate(previousMetrics.current.gpu.memory.percent, systemMetrics.gpu.memory.percent),
            used_mb: interpolate(previousMetrics.current.gpu.memory.used_mb, systemMetrics.gpu.memory.used_mb),
          },
        } : systemMetrics.gpu,
      };

      setSmoothMetrics(newMetrics);

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      }
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [systemMetrics]);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeConnected(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatNumber = (value: number | undefined, decimals = 1) => {
    if (typeof value === 'undefined' || value === null) return '0.0';
    return value.toFixed(decimals);
  };

  const handleLoadModel = async (modelId: string) => {
    const parameters: ModelParameters = {
      n_gpu_layers: smoothMetrics?.gpu?.available ? 32 : 0,
      n_ctx: 2048,
      n_batch: 512,
      threads: smoothMetrics?.cpu?.cores?.physical || 4,
      use_mlock: true,
      f16_kv: true,
    };
    
    try {
      await loadModelMutation.mutateAsync({ modelId, parameters });
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  };

  const handleUnloadModel = async (modelId: string) => {
    try {
      await unloadModelMutation.mutateAsync(modelId);
    } catch (error) {
      console.error('Failed to unload model:', error);
    }
  };

  return (
    <div className="mt-6 space-y-6">
      <h2 className="text-xl text-terminal-green mb-4">System Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* System Status Card */}
        <div className="terminal-window p-4">
          <div className="flex items-center mb-2">
            <TbServer className="mr-2 text-terminal-green" size={20} />
            <h3 className="text-lg text-terminal-green">System Status</h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="text-green-400 relative">
                Online
                <span className="absolute -right-3 top-1/2 -translate-y-1/2 w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              </span>
            </div>
            <div className="flex justify-between">
              <span>Connected:</span>
              <span>{formatTime(timeConnected)}</span>
            </div>
            <div className="flex justify-between">
              <span>API Version:</span>
              <span>v2.0.0</span>
            </div>
          </div>
        </div>
        
        {/* Resources Card */}
        <div className="terminal-window p-4">
          <div className="flex items-center mb-2">
            <TbCpu className="mr-2 text-terminal-green" size={20} />
            <h3 className="text-lg text-terminal-green">Resources</h3>
          </div>
          {isLoadingMetrics ? (
            <div className="text-terminal-dimmed">Loading metrics...</div>
          ) : smoothMetrics ? (
            <div className="space-y-4 text-sm">
              <div>
                <div className="flex justify-between mb-1">
                  <span>Memory Usage:</span>
                  <span className="tabular-nums">{formatNumber(smoothMetrics.memory.percent)}%</span>
                </div>
                <div className="w-full bg-terminal-dimmed/30 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-terminal-green h-full rounded-full transition-all duration-200 ease-out"
                    style={{ width: `${smoothMetrics.memory.percent}%` }}
                  ></div>
                </div>
                <div className="text-xs text-terminal-dimmed mt-1 tabular-nums">
                  {(smoothMetrics.memory.used_mb / 1024).toFixed(1)} GB / {(smoothMetrics.memory.total_mb / 1024).toFixed(1)} GB
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span>CPU Usage:</span>
                  <span className="tabular-nums">{formatNumber(smoothMetrics.cpu.utilization)}%</span>
                </div>
                <div className="w-full bg-terminal-dimmed/30 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-terminal-green h-full rounded-full transition-all duration-200 ease-out"
                    style={{ width: `${smoothMetrics.cpu.utilization}%` }}
                  ></div>
                </div>
                <div className="text-xs text-terminal-dimmed mt-1">
                  {smoothMetrics.cpu.cores.physical} Cores @ {smoothMetrics.cpu.frequency.current} MHz
                </div>
              </div>
              
              {smoothMetrics.gpu?.available && (
                <div>
                  <div className="flex justify-between mb-1">
                    <span>GPU Memory:</span>
                    <span className="tabular-nums">{formatNumber(smoothMetrics.gpu.memory.percent)}%</span>
                  </div>
                  <div className="w-full bg-terminal-dimmed/30 h-2 rounded-full overflow-hidden">
                    <div 
                      className="bg-terminal-green h-full rounded-full transition-all duration-200 ease-out"
                      style={{ width: `${smoothMetrics.gpu.memory.percent}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-terminal-dimmed mt-1 tabular-nums">
                    {smoothMetrics.gpu.name} - {(smoothMetrics.gpu.memory.used_mb / 1024).toFixed(1)} GB / {(smoothMetrics.gpu.memory.total_mb / 1024).toFixed(1)} GB
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-terminal-dimmed">No metrics available</div>
          )}
        </div>

        {/* Models Overview Card */}
        <div className="terminal-window p-4">
          <div className="flex items-center mb-2">
            <TbBrandOpenai className="mr-2 text-terminal-green" size={20} />
            <h3 className="text-lg text-terminal-green">Models Overview</h3>
          </div>
          {isLoadingModels ? (
            <div className="text-terminal-dimmed">Loading models...</div>
          ) : modelsData ? (
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Available Models:</span>
                <span>{modelsData.available_models?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Loaded Models:</span>
                <span>{Object.keys(modelsData.loaded_models || {}).length}</span>
              </div>
              <div className="text-terminal-dimmed mt-4 mb-2">Active Models:</div>
              <div className="space-y-2">
                {Object.entries(modelsData.loaded_models || {}).map(([modelId]) => (
                  <div key={modelId} className="text-xs flex items-center justify-between">
                    <span className="font-mono">{modelId.replace('.gguf', '')}</span>
                    <button
                      onClick={() => handleUnloadModel(modelId)}
                      className="text-red-500 hover:text-red-400 transition-colors"
                      title="Unload Model"
                    >
                      <TbTrash size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-terminal-dimmed">No model data available</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 