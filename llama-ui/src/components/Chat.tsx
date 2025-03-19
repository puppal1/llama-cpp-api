import { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { useMutation, useQuery } from '@tanstack/react-query';
import SendIcon from '@mui/icons-material/Send';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TuneIcon from '@mui/icons-material/Tune';
import DeleteIcon from '@mui/icons-material/Delete';
import { chatApi, modelApi, ChatRequest } from '../services/api';
import { ChatDocs } from './ChatDocs';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`chat-tabpanel-${index}`}
      aria-labelledby={`chat-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

interface ChatParameters {
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  repeat_penalty: number;
  repeat_last_n: number;
  presence_penalty: number;
  frequency_penalty: number;
  num_predict: number;
  stream: boolean;
}

const defaultParameters: ChatParameters = {
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  min_p: 0.05,
  repeat_penalty: 1.1,
  repeat_last_n: 64,
  presence_penalty: 0,
  frequency_penalty: 0,
  num_predict: 128,
  stream: false,
};

// Add theme constants
const THEME = {
  background: '#1a1a1a',
  text: '#33ff00', // Unix green
  gold: {
    light: '#ffd700',
    dark: '#b8860b',
  },
  paper: '#2d2d2d',
  border: 'linear-gradient(90deg, #ffd700 0%, #b8860b 100%)',
};

// Add storage keys
const STORAGE_KEYS = {
  MESSAGES: 'chat_messages',
  PARAMETERS: 'chat_parameters',
  SELECTED_MODEL: 'selected_model',
};

function Chat() {
  // Initialize state from localStorage if available
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.MESSAGES);
    return saved ? JSON.parse(saved) : [];
  });

  const [input, setInput] = useState('');
  
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    return localStorage.getItem(STORAGE_KEYS.SELECTED_MODEL) || '';
  });
  
  const [tabValue, setTabValue] = useState(0);
  
  const [parameters, setParameters] = useState<ChatParameters>(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.PARAMETERS);
    return saved ? JSON.parse(saved) : defaultParameters;
  });

  // Save to localStorage whenever these values change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.PARAMETERS, JSON.stringify(parameters));
  }, [parameters]);

  useEffect(() => {
    if (selectedModel) {
      localStorage.setItem(STORAGE_KEYS.SELECTED_MODEL, selectedModel);
    }
  }, [selectedModel]);

  // Add clear chat history function
  const clearChatHistory = () => {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEYS.MESSAGES);
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const { data, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: modelApi.getModels,
  });

  const chatMutation = useMutation({
    mutationFn: ({ message, modelId }: { message: string; modelId: string }) => {
      const chatRequest: ChatRequest = {
        messages: [{ role: 'user', content: message }],
        parameters: {
          ...parameters,
          num_ctx: 2048,
          num_batch: 512,
          num_thread: 4,
          num_gpu: 1,
          mlock: false,
          mmap: true,
        }
      };
      return chatApi.sendMessage(modelId, chatRequest);
    },
    onSuccess: (data) => {
      if (data.data && data.data.choices && data.data.choices.length > 0) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.data.choices[0].message.content },
        ]);
      }
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && selectedModel) {
      setMessages((prev) => [...prev, { role: 'user', content: input }]);
      chatMutation.mutate({ message: input, modelId: selectedModel });
      setInput('');
    }
  };

  const handleParameterChange = (param: keyof ChatParameters) => (
    _event: Event | React.ChangeEvent<HTMLInputElement>,
    newValue: number | number[] | boolean
  ) => {
    setParameters(prev => ({
      ...prev,
      [param]: newValue
    }));
  };

  if (isLoading || !data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  // Get loaded models from the API response
  const loadedModels = Object.entries(data.models.loaded)
    .filter(([_, model]) => model.status === 'loaded')
    .map(([id]) => id);

  if (loadedModels.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        Please load a model from the Models tab before starting a chat.
      </Alert>
    );
  }

  const renderParameterControls = () => (
    <Accordion 
      sx={{
        bgcolor: THEME.paper,
        backgroundImage: 'none',
        border: '1px solid transparent',
        borderImage: THEME.border,
        borderImageSlice: 1,
        '& .MuiAccordionSummary-root': {
          color: THEME.text,
        },
        '& .MuiAccordionDetails-root': {
          bgcolor: THEME.paper,
        },
      }}
    >
      <AccordionSummary 
        expandIcon={<ExpandMoreIcon sx={{ color: THEME.text }} />}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TuneIcon sx={{ color: THEME.text }} />
          <Typography sx={{ color: THEME.text }}>Generation Parameters</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: 2,
          '& .MuiTypography-root': {
            color: THEME.text,
          },
          '& .MuiSlider-root': {
            color: THEME.gold.light,
            '& .MuiSlider-rail': {
              background: THEME.gold.dark,
            },
          },
          '& .MuiSwitch-root': {
            '& .MuiSwitch-track': {
              backgroundColor: THEME.gold.dark,
            },
            '& .Mui-checked': {
              color: THEME.gold.light,
              '& + .MuiSwitch-track': {
                backgroundColor: THEME.gold.light,
              },
            },
          },
        }}>
          <Box>
            <Typography gutterBottom>Temperature ({parameters.temperature})</Typography>
            <Slider
              value={parameters.temperature}
              onChange={handleParameterChange('temperature')}
              min={0}
              max={2}
              step={0.1}
              marks={[{ value: 0.7, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Top K ({parameters.top_k})</Typography>
            <Slider
              value={parameters.top_k}
              onChange={handleParameterChange('top_k')}
              min={0}
              max={100}
              marks={[{ value: 40, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Top P ({parameters.top_p})</Typography>
            <Slider
              value={parameters.top_p}
              onChange={handleParameterChange('top_p')}
              min={0}
              max={1}
              step={0.05}
              marks={[{ value: 0.95, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Min P ({parameters.min_p})</Typography>
            <Slider
              value={parameters.min_p}
              onChange={handleParameterChange('min_p')}
              min={0}
              max={1}
              step={0.05}
              marks={[{ value: 0.05, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Repeat Penalty ({parameters.repeat_penalty})</Typography>
            <Slider
              value={parameters.repeat_penalty}
              onChange={handleParameterChange('repeat_penalty')}
              min={1}
              max={2}
              step={0.1}
              marks={[{ value: 1.1, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Repeat Last N ({parameters.repeat_last_n})</Typography>
            <Slider
              value={parameters.repeat_last_n}
              onChange={handleParameterChange('repeat_last_n')}
              min={0}
              max={256}
              marks={[{ value: 64, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Presence Penalty ({parameters.presence_penalty})</Typography>
            <Slider
              value={parameters.presence_penalty}
              onChange={handleParameterChange('presence_penalty')}
              min={-2}
              max={2}
              step={0.1}
              marks={[{ value: 0, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Frequency Penalty ({parameters.frequency_penalty})</Typography>
            <Slider
              value={parameters.frequency_penalty}
              onChange={handleParameterChange('frequency_penalty')}
              min={-2}
              max={2}
              step={0.1}
              marks={[{ value: 0, label: 'Default' }]}
            />
          </Box>
          <Box>
            <Typography gutterBottom>Max Tokens ({parameters.num_predict})</Typography>
            <Slider
              value={parameters.num_predict}
              onChange={handleParameterChange('num_predict')}
              min={-1}
              max={2048}
              marks={[{ value: 128, label: 'Default' }]}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <FormControlLabel
              control={
                <Switch
                  checked={parameters.stream}
                  onChange={(e) => handleParameterChange('stream')(e, e.target.checked)}
                />
              }
              label="Stream Tokens"
            />
          </Box>
        </Box>
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => setParameters(defaultParameters)}
            sx={{
              color: THEME.text,
              borderColor: THEME.gold.light,
              '&:hover': {
                borderColor: THEME.gold.dark,
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
              },
            }}
          >
            Reset to Defaults
          </Button>
        </Box>
      </AccordionDetails>
    </Accordion>
  );

  return (
    <Box sx={{ 
      width: '100%',
      bgcolor: THEME.background,
      minHeight: '100vh',
      color: THEME.text,
    }}>
      <Box sx={{ 
        borderBottom: '1px solid transparent',
        borderImage: THEME.border,
        borderImageSlice: 1,
        mb: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        px: 2,
      }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="chat tabs">
          <Tab label="Chat" id="chat-tab-0" aria-controls="chat-tabpanel-0" />
          <Tab label="Documentation" id="chat-tab-1" aria-controls="chat-tabpanel-1" />
        </Tabs>
        
        {/* Add clear chat button */}
        {messages.length > 0 && (
          <Button
            onClick={clearChatHistory}
            startIcon={<DeleteIcon />}
            sx={{
              color: THEME.text,
              borderColor: THEME.gold.light,
              '&:hover': {
                borderColor: THEME.gold.dark,
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
              },
            }}
          >
            Clear Chat
          </Button>
        )}
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Box sx={{ height: '80vh', display: 'flex', flexDirection: 'column' }}>
          <FormControl sx={{ 
            mb: 2,
            '& .MuiInputLabel-root': {
              color: THEME.text,
            },
            '& .MuiOutlinedInput-root': {
              color: THEME.text,
              '& fieldset': {
                borderColor: THEME.gold.dark,
              },
              '&:hover fieldset': {
                borderColor: THEME.gold.light,
              },
              '&.Mui-focused fieldset': {
                borderColor: THEME.gold.light,
              },
            },
            '& .MuiSelect-icon': {
              color: THEME.text,
            },
            '& .MuiMenuItem-root': {
              color: THEME.text,
              '&:hover': {
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
              },
            },
          }}>
            <InputLabel>Select Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              label="Select Model"
              MenuProps={{
                PaperProps: {
                  sx: {
                    bgcolor: THEME.paper,
                  },
                },
              }}
            >
              {loadedModels.map((modelId) => (
                <MenuItem key={modelId} value={modelId}>
                  {modelId}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {renderParameterControls()}

          <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2, mt: 2 }}>
            {messages.map((message, index) => (
              <Paper
                key={index}
                sx={{
                  p: 2,
                  mb: 2,
                  ml: message.role === 'assistant' ? 0 : 'auto',
                  mr: message.role === 'user' ? 0 : 'auto',
                  maxWidth: '70%',
                  bgcolor: THEME.paper,
                  color: THEME.text,
                  border: '1px solid transparent',
                  borderImage: THEME.border,
                  borderImageSlice: 1,
                  boxShadow: `0 0 10px ${message.role === 'user' ? THEME.gold.light : THEME.gold.dark}`,
                }}
              >
                <Typography sx={{ color: THEME.text }}>{message.content}</Typography>
              </Paper>
            ))}
            {chatMutation.isPending && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                <CircularProgress sx={{ color: THEME.gold.light }} size={20} />
              </Box>
            )}
          </Box>

          <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={!selectedModel || chatMutation.isPending}
              sx={{
                '& .MuiOutlinedInput-root': {
                  color: THEME.text,
                  backgroundColor: THEME.paper,
                  '& fieldset': {
                    borderColor: THEME.gold.dark,
                  },
                  '&:hover fieldset': {
                    borderColor: THEME.gold.light,
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: THEME.gold.light,
                  },
                },
                '& .MuiInputBase-input::placeholder': {
                  color: 'rgba(51, 255, 0, 0.5)',
                },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              endIcon={<SendIcon />}
              disabled={!selectedModel || !input.trim() || chatMutation.isPending}
              sx={{
                background: THEME.border,
                color: THEME.background,
                '&:hover': {
                  background: `linear-gradient(90deg, ${THEME.gold.dark} 0%, ${THEME.gold.light} 100%)`,
                },
                '&.Mui-disabled': {
                  background: 'rgba(255, 215, 0, 0.3)',
                },
              }}
            >
              Send
            </Button>
          </Box>
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <ChatDocs />
      </TabPanel>
    </Box>
  );
}

export default Chat;