import { useState, useRef, useEffect } from 'react';
import { TbTerminal2 } from 'react-icons/tb';

type CommandOutput = {
  id: number;
  command: string;
  output: React.ReactNode;
};

const Terminal = () => {
  const [commandHistory, setCommandHistory] = useState<CommandOutput[]>([
    {
      id: 0,
      command: 'help',
      output: (
        <div className="text-terminal-green">
          <p>Available commands:</p>
          <ul className="list-disc pl-6">
            <li><span className="text-terminal-yellow">help</span> - Display this help message</li>
            <li><span className="text-terminal-yellow">about</span> - Show information about LLama.cpp API</li>
            <li><span className="text-terminal-yellow">models</span> - List available models</li>
            <li><span className="text-terminal-yellow">clear</span> - Clear the terminal</li>
            <li><span className="text-terminal-yellow">status</span> - Show system status</li>
          </ul>
          <p className="mt-2">Navigate the UI using the sidebar menu.</p>
        </div>
      )
    }
  ]);
  const [currentCommand, setCurrentCommand] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to bottom of terminal when command history changes
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [commandHistory]);

  const handleCommandSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentCommand.trim()) return;
    
    // Process command
    const command = currentCommand.trim().toLowerCase();
    let output: React.ReactNode = 'Command not recognized. Type "help" for available commands.';
    
    // Command processing logic
    if (command === 'help') {
      output = (
        <div className="text-terminal-green">
          <p>Available commands:</p>
          <ul className="list-disc pl-6">
            <li><span className="text-terminal-yellow">help</span> - Display this help message</li>
            <li><span className="text-terminal-yellow">about</span> - Show information about LLama.cpp API</li>
            <li><span className="text-terminal-yellow">models</span> - List available models</li>
            <li><span className="text-terminal-yellow">clear</span> - Clear the terminal</li>
            <li><span className="text-terminal-yellow">status</span> - Show system status</li>
          </ul>
          <p className="mt-2">Navigate the UI using the sidebar menu.</p>
        </div>
      );
    } else if (command === 'about') {
      output = (
        <div>
          <p className="text-terminal-green">LLama.cpp API Terminal v2.0.0</p>
          <p>A modern interface for the LLama.cpp API server.</p>
          <p className="text-terminal-dimmed">© 2023 LLama.cpp Project</p>
        </div>
      );
    } else if (command === 'clear') {
      setCommandHistory([]);
      setCurrentCommand('');
      return;
    } else if (command === 'models') {
      output = (
        <div>
          <p>Fetching models...</p>
          <p className="text-terminal-dimmed">Please use the Models tab for a better view.</p>
        </div>
      );
    } else if (command === 'status') {
      output = (
        <div>
          <p className="text-terminal-green">System Status: Online</p>
          <p>API Version: 2.0.0</p>
          <p>Server: Running</p>
          <p>Memory Usage: 25%</p>
          <p>CPU Usage: 10%</p>
        </div>
      );
    }
    
    // Add to command history
    setCommandHistory(prev => [...prev, { id: prev.length, command, output }]);
    setCurrentCommand('');
  };

  const focusInput = () => {
    inputRef.current?.focus();
  };

  return (
    <div 
      className="min-h-[200px] max-h-[400px] overflow-y-auto mb-6 font-mono" 
      onClick={focusInput}
    >
      {/* Command History */}
      {commandHistory.map((item) => (
        <div key={item.id} className="mb-4">
          <div className="flex items-center text-terminal-green">
            <span className="text-terminal-commandPrompt mr-2">$</span>
            <span>{item.command}</span>
          </div>
          <div className="ml-6 mt-1">{item.output}</div>
        </div>
      ))}
      
      {/* Current Command Line */}
      <form onSubmit={handleCommandSubmit} className="flex items-center">
        <span className="text-terminal-commandPrompt mr-2">$</span>
        <input
          ref={inputRef}
          type="text"
          value={currentCommand}
          onChange={(e) => setCurrentCommand(e.target.value)}
          className="command-input flex-1"
          autoFocus
          placeholder="Type a command (e.g., 'help')"
        />
      </form>
      
      {/* Auto-scroll anchor */}
      <div ref={terminalEndRef} />
    </div>
  );
};

export default Terminal; 