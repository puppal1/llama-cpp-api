import { NavLink } from 'react-router-dom';
import { 
  TbDashboard, 
  TbBrandOpenai, 
  TbMessage, 
  TbApi, 
  TbServer, 
  TbTerminal2
} from 'react-icons/tb';

const Sidebar = () => {
  const navItems = [
    { path: '/', label: 'Dashboard', icon: <TbDashboard size={20} /> },
    { path: '/models', label: 'Models', icon: <TbBrandOpenai size={20} /> },
    { path: '/chat', label: 'Chat', icon: <TbMessage size={20} /> },
    { path: '/api-docs', label: 'API Docs', icon: <TbApi size={20} /> },
  ];

  return (
    <div className="w-64 h-full border-r border-terminal-green flex flex-col">
      <div className="p-4 border-b border-terminal-green flex items-center gap-2">
        <TbTerminal2 className="text-terminal-green" size={24} />
        <h1 className="text-xl font-bold text-terminal-green">LLama-CLI</h1>
      </div>
      
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink 
                to={item.path} 
                className={({ isActive }) => 
                  `flex items-center p-2 rounded-md transition-colors ${
                    isActive 
                      ? 'bg-terminal-highlight/30 text-terminal-green' 
                      : 'text-terminal-dimmed hover:bg-terminal-highlight/20 hover:text-terminal-green'
                  }`
                }
              >
                <span className="mr-3">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-terminal-green">
        <div className="flex items-center gap-2 text-terminal-dimmed">
          <TbServer size={18} />
          <div className="text-xs">
            <div>System Status</div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              <span>Online</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 