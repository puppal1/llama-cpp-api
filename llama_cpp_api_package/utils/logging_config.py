import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import pathlib

_is_logging_setup = False

def setup_logging():
    global _is_logging_setup
    if _is_logging_setup:
        return
    
    # Create logs directory in the project root
    project_root = pathlib.Path(__file__).parent.parent.parent
    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = project_root / 'logs' / today
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log files with today's date
    api_log_file = log_dir / 'api.log'
    error_log_file = log_dir / 'error.log'
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'Exception:\n%(exc_info)s\n'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup API file handler with daily rotation
    api_file_handler = TimedRotatingFileHandler(
        filename=api_log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep logs for 30 days
        encoding='utf-8'
    )
    api_file_handler.setFormatter(file_formatter)
    api_file_handler.setLevel(logging.INFO)
    
    # Setup error file handler with daily rotation
    error_file_handler = TimedRotatingFileHandler(
        filename=error_log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep logs for 30 days
        encoding='utf-8'
    )
    error_file_handler.setFormatter(error_formatter)
    error_file_handler.setLevel(logging.ERROR)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Get root logger and llama_cpp logger
    root_logger = logging.getLogger()
    llama_logger = logging.getLogger('llama_cpp')
    
    # Remove any existing handlers
    root_logger.handlers = []
    llama_logger.handlers = []
    
    # Set levels
    root_logger.setLevel(logging.INFO)
    llama_logger.setLevel(logging.INFO)
    
    # Add handlers to both loggers
    root_logger.addHandler(api_file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(console_handler)
    llama_logger.addHandler(api_file_handler)
    llama_logger.addHandler(error_file_handler)
    llama_logger.addHandler(console_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {api_log_file}")
    
    _is_logging_setup = True 