import { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Alert,
  Snackbar,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
  Tooltip,
  IconButton,
  ListItemIcon,
  alpha,
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi, ModelsResponse, ModelParameters } from '../services/api';
import InfoIcon from '@mui/icons-material/Info';
import MemoryIcon from '@mui/icons-material/Memory';
import { useTheme } from '@mui/material/styles';

function ModelManager() {
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const theme = useTheme();

  const { data, isLoading } = useQuery<ModelsResponse>({
    queryKey: ['models'],
    queryFn: modelApi.getModels,
    staleTime: Infinity, // Only refetch when explicitly invalidated
  });

  const loadModelMutation = useMutation({
    mutationFn: ({ modelId, parameters }: { modelId: string; parameters: ModelParameters }) =>
      modelApi.loadModel(modelId, parameters),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      setError(null);
    },
    onError: (err: Error) => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      setError(err.message);
    },
  });

  const unloadModelMutation = useMutation({
    mutationFn: modelApi.unloadModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      setError(null);
    },
    onError: (err: Error) => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      setError(err.message);
    },
  });

  const handleLoadModel = (modelId: string) => {
    const parameters: ModelParameters = {
      num_ctx: 2048,
      num_batch: 512,
      num_thread: 4,
      num_gpu: data?.system_state.gpu.available ? 1 : 0,
      mlock: false,
      mmap: true,
    };
    loadModelMutation.mutate({ modelId, parameters });
  };

  const handleCloseError = () => {
    setError(null);
  };

  if (isLoading || !data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  const formatMemory = (mb: number | null | undefined) => {
    if (mb == null) return 'N/A';
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  const memoryUsagePercent = (data.system_state.memory.used_gb / data.system_state.memory.total_gb) * 100;
  const modelMemoryGB = data.system_state.memory.model_memory_gb;
  const systemMemoryGB = data.system_state.memory.used_gb - modelMemoryGB;

  const getMemoryExplanation = (model: any) => {
    const modelSizeGB = (model.size_mb / 1024).toFixed(1);
    const requiredMemoryGB = (model.required_memory_mb / 1024).toFixed(1);
    const overheadGB = ((model.required_memory_mb - model.size_mb) / 1024).toFixed(1);
    
    return `Model file: ${modelSizeGB} GB
Working memory: ${overheadGB} GB
Total required: ${requiredMemoryGB} GB

The model requires additional memory for context window, 
key/value caches, and temporary computations.`;
  };

  return (
    <Box>
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError}>
        <Alert onClose={handleCloseError} severity="error">
          {error}
        </Alert>
      </Snackbar>

      {/* System Resources Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>System Resources</Typography>
          
          {/* CPU Info */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>CPU</Typography>
            <Typography variant="body2" gutterBottom>
              {data.system_state.cpu?.name || 'Unknown CPU'}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Cores: {data.system_state.cpu?.cores?.physical || 0} Physical / {data.system_state.cpu?.cores?.logical || 0} Logical
            </Typography>
            {data.system_state.cpu?.frequency && (
              <Typography variant="body2" gutterBottom>
                Frequency: {data.system_state.cpu.frequency}
              </Typography>
            )}
            <Box sx={{ mt: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">
                  CPU Usage: {(data.system_state.cpu?.utilization ?? 0).toFixed(1)}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={data.system_state.cpu?.utilization ?? 0} 
                color={(data.system_state.cpu?.utilization ?? 0) > 90 ? "error" : (data.system_state.cpu?.utilization ?? 0) > 70 ? "warning" : "primary"}
                sx={{ height: 8, borderRadius: 1 }}
              />
            </Box>
          </Box>
          
          {/* RAM Usage */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>RAM Usage</Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <MemoryIcon />
                </ListItemIcon>
                <ListItemText 
                  primary={
                    <Box>
                      <Typography variant="body1">
                        Memory Usage: {memoryUsagePercent.toFixed(1)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={memoryUsagePercent}
                        sx={{
                          marginTop: 1,
                          marginBottom: 1,
                          backgroundColor: alpha(theme.palette.primary.main, 0.1),
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: 
                              memoryUsagePercent > 90 ? theme.palette.error.main :
                              memoryUsagePercent > 70 ? theme.palette.warning.main :
                              theme.palette.success.main
                          }
                        }}
                      />
                    </Box>
                  }
                  secondary={
                    <>
                      <Typography variant="body2" component="span">
                        Total: {data.system_state.memory.total_gb.toFixed(1)} GB | 
                      </Typography>
                      <Typography variant="body2" component="span" color="primary">
                        System: {systemMemoryGB.toFixed(1)} GB | 
                      </Typography>
                      <Typography variant="body2" component="span" color="secondary">
                        Models: {modelMemoryGB.toFixed(1)} GB
                      </Typography>
                    </>
                  }
                />
              </ListItem>
            </List>
          </Box>

          {/* GPU Info */}
          {data.system_state.gpu.available && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                GPU: {data.system_state.gpu.name || 'Unknown'}
              </Typography>
              {data.system_state.gpu.memory && (
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">
                      Used: {formatMemory(data.system_state.gpu.memory.used_mb)}
                    </Typography>
                    <Typography variant="body2">
                      Available: {formatMemory(data.system_state.gpu.memory.free_mb)}
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={data.system_state.gpu.memory.total_mb ? ((data.system_state.gpu.memory.used_mb || 0) / data.system_state.gpu.memory.total_mb) * 100 : 0}
                    color="secondary"
                    sx={{ height: 8, borderRadius: 1 }}
                  />
                </Box>
              )}
              {data.system_state.gpu.utilization && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>GPU Utilization: {data.system_state.gpu.utilization.gpu_percent || 0}%</Typography>
                  <Typography variant="body2" gutterBottom>Memory Utilization: {data.system_state.gpu.utilization.memory_percent || 0}%</Typography>
                </Box>
              )}
              {data.system_state.gpu.temperature_celsius && (
                <Typography variant="body2" gutterBottom>
                  Temperature: {data.system_state.gpu.temperature_celsius}°C
                </Typography>
              )}
              {data.system_state.gpu.power_watts && (
                <Typography variant="body2" gutterBottom>
                  Power: {data.system_state.gpu.power_watts.toFixed(1)}W
                </Typography>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Available Models */}
      <Typography variant="h6" gutterBottom>Available Models</Typography>
      <Grid container spacing={2}>
        {data.models.available.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {model.id}
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="Size"
                      secondary={formatMemory(model.size_mb)}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          Required Memory
                          <Tooltip title={getMemoryExplanation(model)} placement="right">
                            <IconButton size="small" sx={{ ml: 0.5 }}>
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      }
                      secondary={formatMemory(model.required_memory_mb)}
                    />
                  </ListItem>
                </List>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={() => handleLoadModel(model.id)}
                  disabled={
                    loadModelMutation.isPending || 
                    !model.can_load || 
                    Object.keys(data.models.loaded || {}).includes(model.id)
                  }
                  color={model.can_load ? "primary" : "error"}
                >
                  {!model.can_load ? 'Insufficient Memory' :
                   Object.keys(data.models.loaded || {}).includes(model.id) ? 'Already Loaded' :
                   loadModelMutation.isPending ? 'Loading...' : 'Load Model'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Loaded Models */}
      {Object.keys(data.models.loaded || {}).length > 0 && (
        <>
          <Divider sx={{ my: 4 }} />
          <Typography variant="h6" gutterBottom>Loaded Models</Typography>
          <Grid container spacing={2}>
            {Object.entries(data.models.loaded || {}).map(([modelId, model]) => (
              <Grid item xs={12} sm={6} md={4} key={modelId}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {modelId}
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Memory Used"
                          secondary={formatMemory(model.memory_used_mb)}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Context Window"
                          secondary={`${model.parameters?.num_ctx || 'N/A'} tokens`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="GPU Layers"
                          secondary={model.parameters?.num_gpu && model.parameters.num_gpu > 0 
                            ? `${model.parameters.num_gpu} layers` 
                            : 'CPU only'}
                        />
                      </ListItem>
                      {model.last_used && (
                        <ListItem>
                          <ListItemText 
                            primary="Last Used"
                            secondary={new Date(model.last_used).toLocaleString()}
                          />
                        </ListItem>
                      )}
                    </List>
                    <Button
                      fullWidth
                      variant="outlined"
                      color="secondary"
                      onClick={() => unloadModelMutation.mutate(modelId)}
                      disabled={unloadModelMutation.isPending || model.status !== 'loaded'}
                    >
                      {unloadModelMutation.isPending ? 'Unloading...' : 'Unload'}
                    </Button>
                    {model.error && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {model.error}
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}
    </Box>
  );
}

export default ModelManager; 