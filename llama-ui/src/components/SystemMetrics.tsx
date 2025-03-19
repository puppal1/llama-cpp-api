import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  LinearProgress,
  Chip,
  Stack,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';

interface GpuMetrics {
  available: boolean;
  status: string;
  name: string;
  memory: {
    total_mb: number;
    free_mb: number;
    used_mb: number;
  };
  utilization: {
    gpu_percent: number;
    memory_percent: number;
  };
  temperature_celsius?: number;
  power_watts?: number;
}

interface SystemMetricsData {
  cpu_percent: number;
  memory_total_mb: number;
  memory_used_mb: number;
  gpu: GpuMetrics;
}

function SystemMetrics() {
  // Fetch metrics data directly from /api/metrics endpoint
  const { data: metricsData, isLoading: isMetricsLoading } = useQuery<SystemMetricsData>({
    queryKey: ['metrics'],
    queryFn: () => fetch('http://localhost:8080/api/metrics').then(res => res.json()),
    refetchInterval: 1000, // Refresh every second for more real-time updates
  });

  if (isMetricsLoading || !metricsData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  const formatMemory = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  const memoryUsagePercent = (metricsData.memory_used_mb / metricsData.memory_total_mb) * 100;
  const cpuPercent = metricsData.cpu_percent;
  
  return (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>CPU Usage</Typography>
              <Typography variant="h4" gutterBottom>{cpuPercent.toFixed(1)}%</Typography>
              <LinearProgress 
                variant="determinate" 
                value={cpuPercent} 
                color={cpuPercent > 90 ? "error" : cpuPercent > 70 ? "warning" : "primary"}
                sx={{ height: 8, borderRadius: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Memory Usage</Typography>
              <Typography variant="h4" gutterBottom>{memoryUsagePercent.toFixed(1)}%</Typography>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                  {formatMemory(metricsData.memory_used_mb)} / {formatMemory(metricsData.memory_total_mb)}
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={memoryUsagePercent} 
                color={memoryUsagePercent > 90 ? "error" : memoryUsagePercent > 70 ? "warning" : "primary"}
                sx={{ height: 8, borderRadius: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Status
                <Chip 
                  label={metricsData.gpu.status}
                  color={metricsData.gpu.available ? "success" : "default"}
                  size="small"
                  sx={{ ml: 1 }}
                />
              </Typography>
              
              {metricsData.gpu.available ? (
                <>
                  {/* GPU Name and Basic Info */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2">
                      {metricsData.gpu.name}
                    </Typography>
                    {metricsData.gpu.temperature_celsius && (
                      <Typography variant="body2" color="textSecondary">
                        Temperature: {metricsData.gpu.temperature_celsius}°C
                      </Typography>
                    )}
                    {metricsData.gpu.power_watts && (
                      <Typography variant="body2" color="textSecondary">
                        Power: {metricsData.gpu.power_watts.toFixed(1)}W
                      </Typography>
                    )}
                  </Box>

                  {/* GPU Utilization */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>GPU Utilization</Typography>
                    <Stack spacing={1}>
                      <Box>
                        <Typography variant="body2">Compute: {metricsData.gpu.utilization.gpu_percent}%</Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={metricsData.gpu.utilization.gpu_percent}
                          color="secondary"
                          sx={{ height: 8, borderRadius: 1 }}
                        />
                      </Box>
                      <Box>
                        <Typography variant="body2">Memory: {metricsData.gpu.utilization.memory_percent}%</Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={metricsData.gpu.utilization.memory_percent}
                          color="secondary"
                          sx={{ height: 8, borderRadius: 1 }}
                        />
                      </Box>
                    </Stack>
                  </Box>

                  {/* GPU Memory */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>GPU Memory</Typography>
                    <Typography variant="body2">
                      {formatMemory(metricsData.gpu.memory.used_mb)} / {formatMemory(metricsData.gpu.memory.total_mb)}
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={(metricsData.gpu.memory.used_mb / metricsData.gpu.memory.total_mb) * 100}
                      color="secondary"
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>
                </>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  {metricsData.gpu.status}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default SystemMetrics; 