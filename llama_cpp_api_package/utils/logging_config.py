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
    
    # Create logs directory with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = pathlib.Path('logs') / today
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with today's date
    log_file = log_dir / 'api.log'
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup file handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep logs for 30 days
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
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
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    llama_logger.addHandler(file_handler)
    llama_logger.addHandler(console_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    _is_logging_setup = True 