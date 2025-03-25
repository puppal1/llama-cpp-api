import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Grid,
  Paper,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { TbLoader, TbBrandOpenai, TbCircleCheck, TbX } from 'react-icons/tb';

interface ModelMetadata {
  model_type: string;
  size_gb: number;
  memory_required_gb: number;
  is_safe_to_load: boolean;
  current_memory_gb?: number;
  status: 'AVAILABLE' | 'LOADING' | 'LOADED' | 'ERROR';
  parameters: {
    context_length: number;
    n_gpu_layers: number;
  };
}

interface Model {
  id: string;
  size: number;
  metadata: ModelMetadata;
}

// Types for memory metrics
interface MemoryMetrics {
  system: {
    total_gb: number;
    available_gb: number;
    used_gb: number;
    percent_used: number;
  };
  models: Record<string, number>;
  total_model_memory_gb: number;
  available_for_models_gb: number;
}

interface ModelParams {
  n_gpu_layers: number;
  n_ctx: number;
  n_batch: number;
  threads: number;
  use_mlock: boolean;
  f16_kv: boolean;
}

interface ModelResponse {
  available_models: Model[];
  loaded_models: Model[];
  memory_metrics: MemoryMetrics;
}

const StyledCard = styled(Card)(({ theme }) => ({
  margin: theme.spacing(2),
  position: 'relative',
  backgroundColor: theme.palette.mode === 'dark' ? '#0C0C0C' : '#fff',
  color: theme.palette.mode === 'dark' ? '#f8f8f2' : '#000',
  border: `1px solid ${theme.palette.mode === 'dark' ? '#333' : '#e0e0e0'}`,
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? '#0C0C0C' : '#fff',
  color: theme.palette.mode === 'dark' ? '#f8f8f2' : '#000',
  border: `1px solid ${theme.palette.mode === 'dark' ? '#333' : '#e0e0e0'}`,
  padding: theme.spacing(2),
}));

const MemoryIndicator = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(2),
  padding: theme.spacing(1),
  '& .MuiLinearProgress-root': {
    backgroundColor: theme.palette.mode === 'dark' ? '#333' : '#e0e0e0',
    '& .MuiLinearProgress-bar': {
      backgroundColor: theme.palette.success.main,
    },
    '&.unsafe .MuiLinearProgress-bar': {
      backgroundColor: theme.palette.error.main,
    }
  }
}));

const DEFAULT_MODEL_PARAMS: ModelParams = {
  n_gpu_layers: 0,
  n_ctx: 2048,
  n_batch: 512,
  threads: 4,
  use_mlock: true,
  f16_kv: true,
};

const Models = () => {
  const queryClient = useQueryClient();
  const [modelParams, setModelParams] = useState<ModelParams>(DEFAULT_MODEL_PARAMS);
  const [showParams, setShowParams] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Fetch models
  const { data: modelsData, isLoading: isLoadingModels } = useQuery<ModelResponse>({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get('/api/v2/models');
      return response.data;
    },
    refetchInterval: 5000, // Refetch every 5 seconds for memory updates
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

  const formatBytes = (gb: number) => {
    return `${gb.toFixed(2)} GB`;
  };

  const handleLoadModel = (modelId: string) => {
    loadModelMutation.mutate({ modelId, params: modelParams });
  };

  const handleUnloadModel = (modelId: string) => {
    unloadModelMutation.mutate(modelId);
  };

  const handleContextChange = (_event: Event, newValue: number | number[]) => {
    setModelParams(prev => ({
      ...prev,
      n_ctx: Array.isArray(newValue) ? newValue[0] : newValue
    }));
  };

  if (isLoadingModels) {
    return <LinearProgress />;
  }

  const memoryMetrics: MemoryMetrics = modelsData?.memory_metrics || {
    system: { total_gb: 0, available_gb: 0, used_gb: 0, percent_used: 0 },
    models: {},
    total_model_memory_gb: 0,
    available_for_models_gb: 0
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Models
      </Typography>

      {/* System Memory Overview */}
      <StyledPaper sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ color: 'inherit' }}>
          System Memory Overview
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Total System Memory: {formatBytes(memoryMetrics.system.total_gb)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={(memoryMetrics.total_model_memory_gb / memoryMetrics.system.total_gb) * 100}
              sx={{
                my: 1,
                height: 8,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: (theme) => theme.palette.success.main
                }
              }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Model Memory Usage: {formatBytes(memoryMetrics.total_model_memory_gb)}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Available for Models: {formatBytes(memoryMetrics.available_for_models_gb)}
            </Typography>
          </Grid>
        </Grid>
      </StyledPaper>

      {/* Model Parameters */}
      <StyledPaper sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ color: 'inherit' }}>
          Model Parameters
        </Typography>
        <Box sx={{ width: '100%', mt: 2 }}>
          <Typography gutterBottom sx={{ color: 'text.secondary' }}>
            Context Length: {modelParams.n_ctx}
          </Typography>
          <Slider
            value={modelParams.n_ctx}
            onChange={handleContextChange}
            min={512}
            max={4096}
            step={512}
            marks={[
              { value: 512, label: '512' },
              { value: 2048, label: '2048' },
              { value: 4096, label: '4096' },
            ]}
            valueLabelDisplay="auto"
            sx={{
              '& .MuiSlider-rail': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              },
              '& .MuiSlider-track': {
                backgroundColor: 'primary.main',
              },
              '& .MuiSlider-thumb': {
                backgroundColor: 'primary.main',
              },
              '& .MuiSlider-mark': {
                backgroundColor: 'rgba(255, 255, 255, 0.3)',
              },
              '& .MuiSlider-markLabel': {
                color: 'text.secondary',
              },
            }}
          />
        </Box>
      </StyledPaper>

      {/* Available Models */}
      <Typography variant="h6" gutterBottom sx={{ color: 'inherit', mb: 2 }}>
        Available Models
      </Typography>
      <Grid container spacing={2}>
        {modelsData?.available_models.map((model) => (
          <Grid item xs={12} key={model.id}>
            <StyledCard>
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" sx={{ color: 'inherit' }}>
                      {model.id}
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Type: {model.metadata.model_type}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Context Length: {model.metadata.parameters?.context_length || 2048}
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      GPU Layers: {model.metadata.parameters?.n_gpu_layers || 0}
                    </Typography>
                  </Grid>
                </Grid>
                
                {/* Memory Requirements */}
                <MemoryIndicator>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    Required Memory: {formatBytes(model.metadata.memory_required_gb)}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(model.metadata.memory_required_gb / memoryMetrics.system.total_gb) * 100}
                    className={!model.metadata.is_safe_to_load ? 'unsafe' : ''}
                    sx={{ 
                      my: 1, 
                      height: 8,
                    }}
                  />
                  {!model.metadata.is_safe_to_load && (
                    <Alert severity="error" sx={{ mt: 1 }}>
                      Insufficient memory to load this model with current parameters
                    </Alert>
                  )}
                </MemoryIndicator>

                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleLoadModel(model.id)}
                    disabled={!model.metadata.is_safe_to_load || loadModelMutation.isPending}
                    startIcon={loadModelMutation.isPending ? <TbLoader className="animate-spin" /> : <TbBrandOpenai />}
                  >
                    Load Model
                  </Button>
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        ))}
      </Grid>

      {/* Loaded Models */}
      {modelsData?.loaded_models && modelsData.loaded_models.length > 0 && (
        <>
          <Typography variant="h6" gutterBottom sx={{ color: 'inherit', mt: 4, mb: 2 }}>
            Loaded Models
          </Typography>
          <Grid container spacing={2}>
            {modelsData.loaded_models.map((model) => (
              <Grid item xs={12} key={model.id}>
                <StyledCard>
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="h6" sx={{ color: 'inherit', display: 'flex', alignItems: 'center', gap: 1 }}>
                          {model.id}
                          <TbCircleCheck className="text-green-500" />
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                          Current Memory: {formatBytes(model.metadata.current_memory_gb || 0)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                          Context Length: {model.metadata.parameters?.context_length || 2048}
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                          GPU Layers: {model.metadata.parameters?.n_gpu_layers || 0}
                        </Typography>
                      </Grid>
                    </Grid>

                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button
                        variant="outlined"
                        color="error"
                        onClick={() => handleUnloadModel(model.id)}
                        disabled={unloadModelMutation.isPending}
                        startIcon={unloadModelMutation.isPending ? <TbLoader className="animate-spin" /> : <TbX />}
                      >
                        Unload Model
                      </Button>
                    </Box>
                  </CardContent>
                </StyledCard>
              </Grid>
            ))}
          </Grid>
        </>
      )}
    </Box>
  );
};

export default Models; 