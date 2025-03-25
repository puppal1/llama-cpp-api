import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';
import { TbSend, TbLoader, TbBrandOpenai } from 'react-icons/tb';

type Message = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

type ChatParams = {
  temperature?: number;
  max_tokens?: number;
  top_k?: number;
  top_p?: number;
  repeat_penalty?: number;
};

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [chatParams, setChatParams] = useState<ChatParams>({
    temperature: 0.7,
    max_tokens: 500,
    top_k: 40,
    top_p: 0.9,
    repeat_penalty: 1.1
  });
  const [showParams, setShowParams] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch loaded models
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get('/api/v2/models');
      return response.data;
    }
  });

  // Send chat message
  const chatMutation = useMutation({
    mutationFn: async ({ modelId, messages, params }: { modelId: string, messages: Message[], params: ChatParams }) => {
      const response = await axios.post(`/api/v2/chat/${modelId}`, {
        messages,
        ...params
      });
      return response.data;
    }
  });

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle form submit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || !selectedModel) return;
    
    // Add user message
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    
    // Send to API
    chatMutation.mutate(
      { 
        modelId: selectedModel, 
        messages: [...messages, userMessage],
        params: chatParams
      },
      {
        onSuccess: (data) => {
          // Add assistant response
          if (data.choices && data.choices.length > 0) {
            setMessages(prev => [...prev, data.choices[0].message]);
          }
        }
      }
    );
  };

  return (
    <div className="mt-6">
      <h2 className="text-xl text-terminal-green mb-4">Chat Interface</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Sidebar */}
        <div className="md:col-span-1">
          <div className="terminal-window mb-4">
            <h3 className="text-terminal-green mb-3 border-b border-terminal-dimmed pb-2">Models</h3>
            {isLoadingModels ? (
              <div className="text-terminal-dimmed flex items-center">
                <TbLoader className="animate-spin mr-2" />
                Loading models...
              </div>
            ) : !modelsData?.loaded_models || modelsData.loaded_models.length === 0 ? (
              <div className="text-terminal-dimmed text-sm">
                <p>No loaded models found.</p>
                <p className="mt-2">Please load a model from the Models page before chatting.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {modelsData.loaded_models.map((model: any) => (
                  <div 
                    key={model.id} 
                    className={`p-2 rounded-md cursor-pointer flex items-center ${
                      selectedModel === model.id ? 
                      'bg-terminal-highlight/30 text-terminal-green' : 
                      'hover:bg-terminal-highlight/20 text-terminal-dimmed hover:text-terminal-green'
                    }`}
                    onClick={() => setSelectedModel(model.id)}
                  >
                    <TbBrandOpenai className="mr-2" />
                    <span className="truncate">{model.id.replace('.gguf', '')}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          
          {/* Parameters */}
          <div className="terminal-window">
            <div 
              className="flex justify-between items-center cursor-pointer border-b border-terminal-dimmed pb-2 mb-3"
              onClick={() => setShowParams(!showParams)}
            >
              <h3 className="text-terminal-green">Parameters</h3>
              <span>{showParams ? '−' : '+'}</span>
            </div>
            
            {showParams && (
              <div className="space-y-3">
                <div>
                  <label className="block text-sm mb-1">Temperature</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="2" 
                    step="0.1"
                    value={chatParams.temperature || 0.7}
                    onChange={(e) => setChatParams({...chatParams, temperature: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-terminal-dimmed">
                    <span>0</span>
                    <span>{chatParams.temperature}</span>
                    <span>2</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm mb-1">Max Tokens</label>
                  <input 
                    type="number" 
                    value={chatParams.max_tokens || 500}
                    onChange={(e) => setChatParams({...chatParams, max_tokens: parseInt(e.target.value)})}
                    className="bg-terminal-background border border-terminal-green p-1 w-full rounded-md text-terminal-foreground"
                  />
                </div>
                
                <div>
                  <label className="block text-sm mb-1">Top K</label>
                  <input 
                    type="number" 
                    value={chatParams.top_k || 40}
                    onChange={(e) => setChatParams({...chatParams, top_k: parseInt(e.target.value)})}
                    className="bg-terminal-background border border-terminal-green p-1 w-full rounded-md text-terminal-foreground"
                  />
                </div>
                
                <div>
                  <label className="block text-sm mb-1">Top P</label>
                  <input 
                    type="range" 
                    min="0" 
                    max="1" 
                    step="0.05"
                    value={chatParams.top_p || 0.9}
                    onChange={(e) => setChatParams({...chatParams, top_p: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-terminal-dimmed">
                    <span>0</span>
                    <span>{chatParams.top_p}</span>
                    <span>1</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm mb-1">Repeat Penalty</label>
                  <input 
                    type="range" 
                    min="1" 
                    max="2" 
                    step="0.05"
                    value={chatParams.repeat_penalty || 1.1}
                    onChange={(e) => setChatParams({...chatParams, repeat_penalty: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-terminal-dimmed">
                    <span>1</span>
                    <span>{chatParams.repeat_penalty}</span>
                    <span>2</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Chat Area */}
        <div className="terminal-window md:col-span-3 min-h-[600px] flex flex-col">
          <div className="flex-1 overflow-y-auto mb-4">
            {messages.length === 0 ? (
              <div className="text-center text-terminal-dimmed p-6">
                <div className="text-xl mb-2">Welcome to LLama Chat</div>
                <div>Select a model and start chatting!</div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={`p-3 rounded-lg ${
                      msg.role === 'user' ? 
                      'bg-terminal-highlight/20 ml-8' : 
                      'border border-terminal-green/50 mr-8'
                    }`}
                  >
                    <div className="text-xs text-terminal-dimmed mb-1">
                      {msg.role === 'user' ? 'You' : 'AI'}
                    </div>
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                ))}
                {chatMutation.isPending && (
                  <div className="border border-terminal-green/50 p-3 rounded-lg mr-8">
                    <div className="text-xs text-terminal-dimmed mb-1">AI</div>
                    <div className="flex items-center">
                      <TbLoader className="animate-spin mr-2" />
                      <span>Generating response...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={selectedModel ? "Type your message..." : "Please select a model first"}
              disabled={!selectedModel || chatMutation.isPending}
              className="bg-terminal-background border border-terminal-green/50 p-2 rounded-md flex-1 focus:border-terminal-green focus:outline-none"
            />
            <button 
              type="submit" 
              disabled={!selectedModel || !input.trim() || chatMutation.isPending}
              className="bg-terminal-green/20 border border-terminal-green rounded-md px-4 flex items-center gap-1 hover:bg-terminal-green/30 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <TbSend size={18} />
              <span>Send</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat; 