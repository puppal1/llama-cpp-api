import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

const Layout = () => {
  return (
    <div className="flex h-screen bg-terminal-background text-terminal-foreground overflow-hidden">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <main className="flex-1 p-4 overflow-auto">
        <div className="terminal-window min-h-full">
          <div className="terminal-header">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            <div className="terminal-title">LLama.cpp API Terminal</div>
            <div className="text-xs text-terminal-dimmed">v2.0.0</div>
          </div>
          
          <div className="terminal-content p-4">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Layout; 