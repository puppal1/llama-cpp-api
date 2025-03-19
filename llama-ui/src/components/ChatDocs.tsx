import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export function ChatDocs() {
  const parameterDocs = [
    {
      category: 'Generation Control',
      parameters: [
        {
          name: 'temperature',
          type: 'float',
          range: '0.0-2.0',
          default: '0.8',
          description: 'Controls randomness in responses. Higher = more creative, lower = more focused'
        },
        {
          name: 'top_k',
          type: 'integer',
          range: '≥0',
          default: '40',
          description: 'Limits token selection to top K most probable tokens'
        },
        {
          name: 'top_p',
          type: 'float',
          range: '0.0-1.0',
          default: '0.9',
          description: 'Nucleus sampling threshold. Selects from smallest set of tokens whose cumulative probability exceeds P'
        },
        {
          name: 'min_p',
          type: 'float',
          range: '0.0-1.0',
          default: '0.05',
          description: 'Minimum probability threshold for token consideration'
        }
      ]
    },
    {
      category: 'Repetition Control',
      parameters: [
        {
          name: 'repeat_penalty',
          type: 'float',
          range: '≥1.0',
          default: '1.1',
          description: 'Penalizes repetition of tokens. Higher values reduce repetition'
        },
        {
          name: 'repeat_last_n',
          type: 'integer',
          range: '≥0',
          default: '64',
          description: 'Number of tokens to consider for repetition penalty'
        },
        {
          name: 'presence_penalty',
          type: 'float',
          range: '-2.0 to 2.0',
          default: '0.0',
          description: 'Penalizes tokens based on their presence in the prompt'
        },
        {
          name: 'frequency_penalty',
          type: 'float',
          range: '-2.0 to 2.0',
          default: '0.0',
          description: 'Penalizes tokens based on their frequency in generated text'
        }
      ]
    },
    {
      category: 'Message Structure',
      parameters: [
        {
          name: 'system',
          type: 'message',
          range: 'optional',
          default: 'none',
          description: 'Initial system message to set context/behavior (must be first)'
        },
        {
          name: 'user/assistant',
          type: 'messages',
          range: 'required',
          default: 'n/a',
          description: 'Must alternate user/assistant messages after system message'
        },
        {
          name: 'tool_calls',
          type: 'special',
          range: 'optional',
          default: 'none',
          description: 'Function calls with 9-character IDs'
        },
        {
          name: 'tool_results',
          type: 'special',
          range: 'optional',
          default: 'none',
          description: 'Results from tool calls with matching IDs'
        }
      ]
    },
    {
      category: 'System Parameters',
      parameters: [
        {
          name: 'num_predict',
          type: 'integer',
          range: '≥-1',
          default: '128',
          description: 'Maximum number of tokens to generate (-1 for unlimited)'
        },
        {
          name: 'stream',
          type: 'boolean',
          range: 'true/false',
          default: 'false',
          description: 'Stream tokens as they are generated'
        },
        {
          name: 'stop',
          type: 'string[]',
          range: 'optional',
          default: 'none',
          description: 'Array of sequences where generation should stop'
        }
      ]
    }
  ];

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>Chat API Documentation</Typography>
      
      {parameterDocs.map((category) => (
        <Accordion key={category.category}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">{category.category}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Parameter</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Range</TableCell>
                    <TableCell>Default</TableCell>
                    <TableCell>Description</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {category.parameters.map((param) => (
                    <TableRow key={param.name}>
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {param.name}
                        </Typography>
                      </TableCell>
                      <TableCell>{param.type}</TableCell>
                      <TableCell>{param.range}</TableCell>
                      <TableCell>{param.default}</TableCell>
                      <TableCell>{param.description}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      ))}

      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>Example Usage</Typography>
        <Paper sx={{ p: 2 }}>
          <pre style={{ margin: 0, overflow: 'auto' }}>
{`// Example chat request
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 256
  },
  "stream": true
}`}
          </pre>
        </Paper>
      </Box>
    </Box>
  );
}

export default ChatDocs; 