import React, { useState, useEffect } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Card,
  CardContent,
  Slider,
  Typography,
  Switch,
  FormControlLabel,
  Tooltip,
  IconButton,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import { ModelParameters } from '../services/api';

interface ModelParametersCardProps {
  parameters: ModelParameters;
  onParameterChange: (param: keyof ModelParameters, value: any) => void;
  modelMetadata?: {
    maxContext: number;
    modelType: string;
    requiredMemoryMb: number;
  };
}

export function ModelParametersCard({ parameters, onParameterChange, modelMetadata }: ModelParametersCardProps) {
  const [estimatedMemoryMb, setEstimatedMemoryMb] = useState<number>(0);
  const [memoryWarning, setMemoryWarning] = useState<string | null>(null);

  // Calculate memory requirements based on context size and model size
  const calculateMemoryRequirements = (contextSize: number) => {
    if (!modelMetadata) return 0;
    
    // Base model memory (from metadata)
    const baseMemoryMb = modelMetadata.requiredMemoryMb;
    
    // KV cache memory calculation (simplified)
    // Memory = 2 * context_size * bytes_per_token * num_layers
    // Assuming 16-bit (2 bytes) per token and estimating num_layers based on model size
    const bytesPerToken = 2;
    const estimatedLayers = Math.ceil(modelMetadata.requiredMemoryMb / (1024 * 2)); // Rough estimate
    const kvCacheMb = (2 * contextSize * bytesPerToken * estimatedLayers) / (1024 * 1024);
    
    // Total memory estimate
    return Math.ceil(baseMemoryMb + kvCacheMb);
  };

  // Check system constraints whenever parameters change
  useEffect(() => {
    if (!modelMetadata) return;

    const totalMemoryMb = calculateMemoryRequirements(parameters.num_ctx);
    setEstimatedMemoryMb(totalMemoryMb);

    // Get available system memory (you might want to pass this as a prop)
    const systemMemoryMb = 16 * 1024; // Example: 16GB system memory
    
    if (totalMemoryMb > systemMemoryMb) {
      setMemoryWarning(`Warning: Selected parameters may require ${(totalMemoryMb / 1024).toFixed(1)} GB of memory, which exceeds available system memory.`);
    } else if (totalMemoryMb > modelMetadata.requiredMemoryMb * 2) {
      setMemoryWarning(`Note: Selected context size will use ${(totalMemoryMb / 1024).toFixed(1)} GB of memory (${((totalMemoryMb - modelMetadata.requiredMemoryMb) / 1024).toFixed(1)} GB more than base model).`);
    } else {
      setMemoryWarning(null);
    }
  }, [parameters.num_ctx, parameters.num_thread, modelMetadata]);

  // Helper function to get context window marks
  const getContextMarks = (min: number, max: number) => {
    const marks = [];
    let size = min;
    while (size <= max) {
      marks.push({ 
        value: size, 
        label: size.toString()
      });
      size *= 2;
    }
    return marks;
  };

  // Helper function to snap context window to nearest valid size
  const snapToValidContextSize = (value: number) => {
    const validSizes = [512, 1024, 2048, 4096, 8192, 16384, 32768];
    return validSizes.reduce((prev, curr) => {
      return Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev;
    });
  };

  // Helper function for sliders
  const SliderWithTooltip = ({ 
    label, 
    tooltip, 
    value, 
    onChange, 
    min, 
    max, 
    step = 1,
    marks,
    isContext = false
  }: { 
    label: string;
    tooltip: string;
    value: number;
    onChange: (value: number) => void;
    min: number;
    max: number;
    step?: number | null;
    marks?: { value: number; label: string; }[];
    isContext?: boolean;
  }) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="body2">{label}</Typography>
        <Tooltip title={tooltip} placement="right">
          <IconButton size="small" sx={{ ml: 0.5 }}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Typography variant="body2" sx={{ ml: 'auto', color: 'text.secondary' }}>
          {value}
          {isContext && <span> tokens</span>}
        </Typography>
      </Box>
      <Slider
        value={value}
        onChange={(_, newValue) => {
          // Allow continuous movement
          onChange(newValue as number);
        }}
        onChangeCommitted={(_, newValue) => {
          if (isContext) {
            // Snap to nearest valid size on release
            const snapped = snapToValidContextSize(newValue as number);
            onChange(snapped);
          }
        }}
        min={min}
        max={max}
        step={step}
        valueLabelDisplay="auto"
        marks={marks}
        sx={{
          '& .MuiSlider-mark': {
            height: isContext ? 8 : 2,
            backgroundColor: isContext ? '#666' : undefined,
            width: isContext ? 2 : undefined,
          },
          '& .MuiSlider-rail': {
            opacity: 0.5,
            height: 4,
          },
          '& .MuiSlider-track': {
            height: 4,
          },
          '& .MuiSlider-thumb': {
            width: 16,
            height: 16,
            '&:hover, &.Mui-focusVisible': {
              boxShadow: '0 0 0 8px rgba(25, 118, 210, 0.16)',
            },
          }
        }}
      />
    </Box>
  );

  // Get max context size (default 4096 if not provided)
  const maxContext = Math.min(
    modelMetadata?.maxContext || 4096,
    32768 // Hard maximum
  );

  // Generate valid context sizes up to maxContext
  const validContextSizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    .filter(size => size <= maxContext)
    .map(size => ({
      value: size,
      label: size >= 1024 ? `${(size/1024)}K` : size.toString()
    }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Parameters
        </Typography>

        {/* Memory Warning */}
        {memoryWarning && (
          <Alert severity={memoryWarning.startsWith('Warning') ? 'warning' : 'info'} sx={{ mb: 2 }}>
            {memoryWarning}
          </Alert>
        )}

        {/* Basic Parameters - Always Visible */}
        <Box sx={{ mb: 3 }}>
          <SliderWithTooltip
            label="Context Window"
            tooltip={`Number of tokens the model can process at once. Current setting will use approximately ${(estimatedMemoryMb / 1024).toFixed(1)} GB of memory.`}
            value={parameters.num_ctx}
            onChange={(value) => onParameterChange('num_ctx', value)}
            min={512}
            max={maxContext}
            step={null}
            marks={[
              { value: 512, label: '512' },
              { value: 1024, label: '1K' },
              { value: 2048, label: '2K' },
              { value: 4096, label: '4K' },
              { value: 8192, label: '8K' },
              { value: 16384, label: '16K' },
              { value: 32768, label: '32K' }
            ].filter(mark => mark.value <= maxContext)}
            isContext={true}
          />
          
          <SliderWithTooltip
            label="Temperature"
            tooltip="Controls randomness in responses. Higher values make the output more creative but less focused"
            value={parameters.temperature || 0.8}
            onChange={(value) => onParameterChange('temperature', value)}
            min={0}
            max={2}
            step={0.05}
          />
        </Box>

        {/* Advanced Parameters - In Accordion */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Performance Settings</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <SliderWithTooltip
              label="CPU Threads"
              tooltip={`Number of CPU threads to use for computation. More threads may increase memory usage but improve performance. Recommended: ${Math.min(16, navigator.hardwareConcurrency || 4)} for your system.`}
              value={parameters.num_thread}
              onChange={(value) => onParameterChange('num_thread', value)}
              min={1}
              max={Math.min(16, navigator.hardwareConcurrency || 16)}
              marks={[1, 2, 4, 8, Math.min(16, navigator.hardwareConcurrency || 16)].map(v => ({ value: v, label: v.toString() }))}
            />

            <SliderWithTooltip
              label="GPU Layers"
              tooltip="Number of layers to offload to GPU (if available)"
              value={parameters.num_gpu}
              onChange={(value) => onParameterChange('num_gpu', value)}
              min={0}
              max={100}
              step={5}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={parameters.mlock}
                  onChange={(e) => onParameterChange('mlock', e.target.checked)}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="body2">Lock in Memory</Typography>
                  <Tooltip title="Keep model in RAM" placement="right">
                    <IconButton size="small" sx={{ ml: 0.5 }}>
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              }
            />
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Generation Settings</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <SliderWithTooltip
              label="Top K"
              tooltip="Limits token selection to K most probable tokens"
              value={parameters.top_k || 40}
              onChange={(value) => onParameterChange('top_k', value)}
              min={0}
              max={100}
              step={5}
            />

            <SliderWithTooltip
              label="Top P"
              tooltip="Cumulative probability threshold for token selection"
              value={parameters.top_p || 0.9}
              onChange={(value) => onParameterChange('top_p', value)}
              min={0}
              max={1}
              step={0.05}
            />

            <SliderWithTooltip
              label="Repeat Penalty"
              tooltip="Penalty for repeating tokens"
              value={parameters.repeat_penalty || 1.1}
              onChange={(value) => onParameterChange('repeat_penalty', value)}
              min={1}
              max={2}
              step={0.05}
            />
          </AccordionDetails>
        </Accordion>

        {/* Model Info - If Available */}
        {modelMetadata && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Model Type: {modelMetadata.modelType}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Base Memory: {(modelMetadata.requiredMemoryMb / 1024).toFixed(1)} GB
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Estimated Total Memory: {(estimatedMemoryMb / 1024).toFixed(1)} GB
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
} 