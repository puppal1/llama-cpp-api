import logging
import os
import glob
import re
from typing import Dict, Optional, Set, List
from io import StringIO
from contextlib import redirect_stderr
from llama_cpp import Llama
from datetime import datetime

logger = logging.getLogger(__name__)

# Cache data structures (private to this module)
_model_metadata_cache: Dict[str, Dict] = {}
_model_size_cache: Dict[str, int] = {}
_known_models: Set[str] = set()

def _get_model_files(models_dir: str) -> Set[str]:
    """Get set of model files in directory."""
    models_dir = os.path.abspath(models_dir)
    return {
        os.path.basename(f) for f in glob.glob(os.path.join(models_dir, "*.gguf"))
    }

def _get_model_id_from_filename(filename: str) -> str:
    """Convert a model filename to a model ID by removing the .gguf extension."""
    return os.path.splitext(filename)[0]

def _get_filename_from_model_id(model_id: str) -> str:
    """Convert a model ID to a filename by ensuring it has .gguf extension."""
    if not model_id.endswith(".gguf"):
        return f"{model_id}.gguf"
    return model_id

def _convert_metadata_value(val: str) -> any:
    """Convert metadata value to appropriate type."""
    if val.isdigit():
        return int(val)
    elif '.' in val and val.replace('.', '').isdigit():
        return float(val)
    return val

def _extract_model_metadata(model_path: str) -> Dict:
    """Extract metadata from model file using llama.cpp's metadata extraction."""
    try:
        # Get basic file info
        file_size = os.path.getsize(model_path)
        file_name = os.path.basename(model_path)
        
        # Initialize metadata with basic info
        metadata = {
            'file_size': file_size,
            'file_name': file_name
        }
        
        # Use llama.cpp to extract metadata
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            # Initialize with minimal context to quickly read metadata
            temp_model = Llama(
                model_path=model_path,
                n_ctx=8,  # Minimal context for quick metadata read
                n_threads=1,
                n_batch=1,
                verbose=True
            )
            del temp_model
        
        # Parse metadata from stderr output
        model_info = stderr_buffer.getvalue()
        
        # Extract parameters using patterns
        param_patterns = {
            'architecture': r'arch\s*=\s*(\w+)',
            'num_layers': r'n_layer\s*=\s*(\d+)',
            'num_heads': r'n_head\s*=\s*(\d+)',
            'num_heads_kv': r'n_head_kv\s*=\s*(\d+)',
            'embedding_length': r'n_embd\s*=\s*(\d+)',
            'context_length': r'n_ctx_train\s*=\s*(\d+)',
            'rope_dimension_count': r'n_rot\s*=\s*(\d+)',
            'rope_freq_base': r'freq_base_train\s*=\s*(\d+\.\d+)',
            'rope_scaling': r'rope scaling\s*=\s*(\w+)',
            'model_type': r'model type\s*=\s*(.+)',
            'model_name': r'general\.name\s*=\s*(.+)',
            'n_expert': r'n_expert\s*=\s*(\d+)',
            'n_expert_used': r'n_expert_used\s*=\s*(\d+)',
            'top_k': r'top_k\s*=\s*(\d+)'
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, model_info)
            if match:
                val = match.group(1)
                # Convert to appropriate type
                if val.isdigit():
                    metadata[param] = int(val)
                elif '.' in val and val.replace('.', '').isdigit():
                    metadata[param] = float(val)
                else:
                    metadata[param] = val
        
        # Add default parameters for model loading
        metadata.update({
            'default_parameters': {
                'num_batch': 512,          # Default batch size
                'num_thread': 4,           # Default number of threads
                'num_gpu': 0,              # CPU only
                'mlock': False,            # Don't lock memory
                'mmap': True,              # Use memory mapping for better performance
                'seed': -1,                # Random seed
                'vocab_only': False,       # Load full model
                'rope_freq_scale': 1.0     # Default RoPE frequency scale
            }
        })
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {model_path}: {e}")
        return {}

def _check_for_new_models(models_dir: str) -> Optional[Set[str]]:
    """Check for changes in model directory.
    Returns set of model IDs if changes detected, None otherwise."""
    try:
        current_models = _get_model_files(models_dir)
        
        # Check for new or deleted models
        new_models = current_models - _known_models
        deleted_models = _known_models - current_models
        
        if new_models or deleted_models:
            logger.info(f"Model changes detected: {len(new_models)} new, {len(deleted_models)} deleted")
            return current_models
            
        return None
        
    except Exception as e:
        logger.error(f"Error checking for new models: {e}")
        return None

def _initialize_or_update_cache(
    models_dir: str,
    update_only: bool = False,
    model_ids: Optional[Set[str]] = None,
    load_state: Optional[Dict] = None
) -> None:
    """Initialize or update the model cache with metadata and file sizes"""
    try:
        logger.info(f"{'Updating' if update_only else 'Initializing'} cache from directory: {models_dir}")
        
        # Get all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        logger.info(f"Found {len(model_files)} model files: {model_files}")
        
        # If update_only and model_ids provided, only process those models
        if update_only and model_ids:
            model_files = [f for f in model_files if f in model_ids]
            logger.info(f"Filtered to {len(model_files)} models: {model_files}")
        
        # Update known models set
        _known_models.update(model_files)
        logger.info(f"Known models updated: {_known_models}")
        
        # Process each model file
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            logger.info(f"Processing model file: {model_file}")
            
            # Skip if file doesn't exist
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                if model_file in _model_metadata_cache:
                    del _model_metadata_cache[model_file]
                if model_file in _model_size_cache:
                    del _model_size_cache[model_file]
                _known_models.discard(model_file)
                continue
            
            # Update file size cache
            _model_size_cache[model_file] = os.path.getsize(model_path)
            logger.info(f"Updated size cache for {model_file}: {_model_size_cache[model_file]} bytes")
            
            # Update metadata cache
            if model_file not in _model_metadata_cache:
                _model_metadata_cache[model_file] = {}
            
            # Update basic metadata
            _model_metadata_cache[model_file].update({
                "file_size": _model_size_cache[model_file],
                "file_name": model_file,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            })
            logger.info(f"Updated basic metadata for {model_file}")
            
            # Update load state if provided
            if load_state and load_state.get("model_id") == model_file:
                logger.info(f"Updating load state for {model_file}: {load_state}")
                if load_state.get("is_loaded"):
                    # Add load state information
                    _model_metadata_cache[model_file]["loaded"] = True
                    _model_metadata_cache[model_file]["load_time"] = load_state.get("load_time")
                    _model_metadata_cache[model_file]["memory_used"] = load_state.get("memory_used")
                else:
                    # Remove all load state information
                    _model_metadata_cache[model_file].pop("loaded", None)
                    _model_metadata_cache[model_file].pop("load_time", None)
                    _model_metadata_cache[model_file].pop("memory_used", None)
            
            # Get model info if not in cache or if file was modified
            if not update_only or model_file not in _model_metadata_cache or "architecture" not in _model_metadata_cache[model_file]:
                try:
                    logger.info(f"Extracting metadata for {model_file}")
                    model_info = _extract_model_metadata(model_path)
                    if model_info:
                        _model_metadata_cache[model_file].update(model_info)
                        logger.info(f"Updated metadata for {model_file}")
                except Exception as e:
                    logger.error(f"Error getting model info for {model_file}: {e}")
                    continue
        
        logger.info(f"Cache {'updated' if update_only else 'initialized'} with {len(_model_metadata_cache)} models")
        logger.info(f"Metadata cache: {_model_metadata_cache}")
        logger.info(f"Size cache: {_model_size_cache}")
        logger.info(f"Known models: {_known_models}")
        
    except Exception as e:
        logger.error(f"Error {'updating' if update_only else 'initializing'} cache: {e}")
        raise

def initialize_cache(models_dir: str) -> None:
    """Initialize cache at server startup."""
    if not models_dir:
        logger.error("MODELS_DIR environment variable not set")
        return
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory {models_dir} does not exist")
        return
    
    logger.info(f"Initializing model metadata cache from {models_dir}...")
    _initialize_or_update_cache(models_dir, update_only=False)
    logger.info(f"Cache initialized with {len(_known_models)} models")

# Public read-only interface
def get_model_metadata(model_id: str) -> Optional[Dict]:
    """Pure read operation - returns cached metadata."""
    # Convert model_id to filename for cache lookup
    filename = _get_filename_from_model_id(model_id)
    metadata = _model_metadata_cache.get(filename)
    
    # Remove path from metadata if it exists
    if metadata and 'path' in metadata:
        metadata = {k: v for k, v in metadata.items() if k != 'path'}
        
    return metadata

def get_model_size_cached(model_id: str) -> Optional[int]:
    """Pure read operation - returns cached size."""
    # Convert model_id to filename for cache lookup
    filename = _get_filename_from_model_id(model_id)
    return _model_size_cache.get(filename)

def get_known_models() -> Set[str]:
    """Pure read operation - returns set of known model IDs."""
    return {_get_model_id_from_filename(f) for f in _known_models}

def list_models(models_dir: str) -> List[Dict]:
    """List all available models with their metadata."""
    try:
        # Initialize cache if needed
        if not _model_metadata_cache:
            initialize_cache(models_dir)
        
        # Convert cache to list format
        models = []
        for filename, metadata in _model_metadata_cache.items():
            model_id = _get_model_id_from_filename(filename)
            
            # Create a copy of metadata without the path
            metadata_clean = {k: v for k, v in metadata.items() if k != 'path'}
            
            model_info = {
                "id": model_id,
                "metadata": metadata_clean,
                "size": _model_size_cache.get(filename)
            }
            models.append(model_info)
        
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise

def get_model_load_state(model_id: str) -> bool:
    """Get model load state from cache"""
    return _model_metadata_cache.get(model_id, {}).get("loaded", False) 