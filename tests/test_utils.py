"""
Utility functions for test result handling and metrics collection.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class TestResultManager:
    def __init__(self, base_dir: str = "test_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def save_test_result(self, 
                        test_name: str, 
                        results: Dict[str, Any], 
                        model_name: Optional[str] = None) -> Path:
        """Save test results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        if model_name:
            filename = f"{model_name}_{filename}"
            
        result_path = self.base_dir / filename
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return result_path
    
    def save_metrics(self, 
                    metrics: Dict[str, Any], 
                    category: str = "general") -> Path:
        """Save performance metrics to a JSON file."""
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_{timestamp}.json"
        metrics_path = metrics_dir / filename
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        return metrics_path
    
    def get_latest_result(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent test result for a given test name."""
        pattern = f"{test_name}_*.json"
        result_files = list(self.base_dir.glob(pattern))
        if not result_files:
            return None
            
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def get_latest_metrics(self, category: str = "general") -> Optional[Dict[str, Any]]:
        """Get the most recent metrics for a given category."""
        metrics_dir = self.base_dir / "metrics"
        if not metrics_dir.exists():
            return None
            
        pattern = f"{category}_*.json"
        metric_files = list(metrics_dir.glob(pattern))
        if not metric_files:
            return None
            
        latest_file = max(metric_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)

def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper 