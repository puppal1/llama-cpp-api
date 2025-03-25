import { Link } from 'react-router-dom';
import { TbError404, TbArrowLeft } from 'react-icons/tb';

const NotFound = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[500px] text-center">
      <TbError404 size={80} className="text-terminal-green mb-4" />
      <h1 className="text-4xl text-terminal-green mb-2">404</h1>
      <h2 className="text-2xl mb-4">Page Not Found</h2>
      <p className="text-terminal-dimmed mb-6 max-w-md">
        The page you are looking for might have been removed, 
        had its name changed, or is temporarily unavailable.
      </p>
      <Link 
        to="/" 
        className="flex items-center gap-2 px-4 py-2 bg-terminal-green/20 border border-terminal-green rounded-md hover:bg-terminal-green/30 transition-colors"
      >
        <TbArrowLeft />
        <span>Back to Dashboard</span>
      </Link>
    </div>
  );
};

export default NotFound; 