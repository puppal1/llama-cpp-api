"""
Model metadata cache manager to avoid repeated model loading
"""

import os
import json
import logging
import struct
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from llama_cpp import Llama
from llama_cpp_api_package.config.model_config import ModelArchitectures
import requests

# Configure logging
logger = logging.getLogger(__name__)

# Define relevant metadata keys to keep
RELEVANT_METADATA_KEYS = {
    # General model info
    'general.architecture',
    'general.name',
    
    # Common model architecture fields
    'llama.context_length',
    'llama.embedding_length',
    'llama.block_count',
    'llama.head_count',
    'llama.rope.dimension_count',
    'llama.rope.freq_base',
    'llama.rope.scale',
    'llama.attention.head_count',
    'llama.attention.head_count_kv',
    'llama.attention.layer_norm_rms_epsilon',
    'llama.rope.orig_ctx',
    
    # Qwen specific fields
    'qwen2.context_length',
    'qwen2.embedding_length',
    'qwen2.block_count',
    'qwen2.head_count',
    'qwen2.rope.dimension_count',
    'qwen2.rope.freq_base',
    'qwen2.rope.scale',
    'qwen2.attention.head_count',
    'qwen2.attention.head_count_kv',
    'qwen2.attention.layer_norm_rms_epsilon',
    'qwen2.rope.orig_ctx',
    
    # Mistral specific fields
    'mistral.context_length',
    'mistral.embedding_length',
    'mistral.block_count',
    'mistral.head_count',
    'mistral.rope.dimension_count',
    'mistral.rope.freq_base',
    'mistral.rope.scale',
    'mistral.attention.head_count',
    'mistral.attention.head_count_kv',
    'mistral.attention.layer_norm_rms_epsilon',
    'mistral.rope.orig_ctx',
    
    # Yi specific fields
    'yi.context_length',
    'yi.embedding_length',
    'yi.block_count',
    'yi.head_count',
    'yi.rope.dimension_count',
    'yi.rope.freq_base',
    'yi.rope.scale',
    'yi.attention.head_count',
    'yi.attention.head_count_kv',
    'yi.attention.layer_norm_rms_epsilon',
    'yi.rope.orig_ctx',
    
    # Phi-2 specific fields
    'phi2.context_length',
    'phi2.embedding_length',
    'phi2.block_count',
    'phi2.head_count',
    'phi2.rope.dimension_count',
    'phi2.rope.freq_base',
    'phi2.rope.scale',
    'phi2.attention.head_count',
    'phi2.attention.head_count_kv',
    'phi2.attention.layer_norm_rms_epsilon',
    'phi2.rope.orig_ctx',
    
    # Generic fields that might be present
    'rope.dimension_count',
    'rope.freq_base',
    'rope.scale',
    'rope.orig_ctx',
    'attention.head_count',
    'attention.head_count_kv',
    'attention.layer_norm_rms_epsilon',
    'context_length',
    'embedding_length',
    'block_count',
    'head_count'
}

# Skip these keys that contain large arrays or unnecessary data
SKIP_METADATA_KEYS = {
    'tokenizer.ggml.tokens',
    'tokenizer.ggml.scores',
    'tokenizer.ggml.token_type',
    'tokenizer.chat_template',
    'tokenizer.ggml.bos_token_id',
    'tokenizer.ggml.eos_token_id',
    'tokenizer.ggml.padding_token_id',
    'tokenizer.ggml.add_bos_token',
    'tokenizer.ggml.add_eos_token',
    'tokenizer.ggml.model_max_length',
    'tokenizer.ggml.clean_up_tokenization_spaces',
}

def _get_huggingface_context_length(org: str, model: str) -> Optional[int]:
    """Get context length from Hugging Face API."""
    url = f"https://huggingface.co/api/models/{org}/{model}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'config' in data:
                config = data['config']
                # Check various possible keys for context length
                for key in ['max_position_embeddings', 'max_sequence_length', 'max_seq_len', 'context_length']:
                    if key in config:
                        return config[key]
    except Exception as e:
        logger.warning(f"Failed to get context length from Hugging Face for {org}/{model}: {e}")
    return None

def _read_gguf_metadata(file_path: str) -> Dict[str, Any]:
    """Read metadata from a GGUF file."""
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            # Read file header
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Not a GGUF file")
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            metadata['gguf_version'] = version
            
            # Read tensor count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata count
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata
            for _ in range(metadata_count):
                try:
                    # Read key length
                    key_len_bytes = f.read(8)
                    if not key_len_bytes:
                        break
                    key_len = struct.unpack('<Q', key_len_bytes)[0]
                    
                    if key_len > 1024:  # Sanity check - keys shouldn't be huge
                        logger.warning(f"Skipping entry with suspiciously long key length: {key_len}")
                        f.seek(key_len, 1)  # Skip the key
                        continue
                    
                    # Read key
                    key_bytes = f.read(key_len)
                    if not key_bytes:
                        break
                    key = key_bytes.decode('utf-8')
                    
                    # Skip if key is not relevant or in skip list
                    if key not in RELEVANT_METADATA_KEYS or key in SKIP_METADATA_KEYS:
                        # Read value type
                        value_type_bytes = f.read(4)
                        if not value_type_bytes:
                            break
                        value_type = struct.unpack('<I', value_type_bytes)[0]
                        
                        # Skip array data
                        if value_type == 9:  # Array
                            f.read(4)  # array type
                            array_length = struct.unpack('<Q', f.read(8))[0]
                            # Skip array data based on type and length
                            # This is a simplified skip - you might need to adjust based on actual data types
                            f.seek(array_length * 4, 1)
                        else:
                            # Skip single value data
                            f.seek(4, 1)
                        continue

                    # Read value type
                    value_type_bytes = f.read(4)
                    if not value_type_bytes:
                        break
                    value_type = struct.unpack('<I', value_type_bytes)[0]
                    
                    # Read value based on type
                    if value_type == 9:  # Array
                        array_type_bytes = f.read(4)
                        if not array_type_bytes:
                            break
                        array_type = struct.unpack('<I', array_type_bytes)[0]
                        
                        array_length_bytes = f.read(8)
                        if not array_length_bytes:
                            break
                        array_length = struct.unpack('<Q', array_length_bytes)[0]
                        
                        if array_length > 1024000:  # Sanity check - 1MB worth of elements
                            logger.warning(f"Skipping array with suspiciously large length: {array_length}")
                            continue
                        
                        values = []
                        for _ in range(array_length):
                            if array_type in [0, 8]:  # String
                                str_len_bytes = f.read(8)
                                if not str_len_bytes:
                                    break
                                str_len = struct.unpack('<Q', str_len_bytes)[0]
                                
                                if str_len > 1024000:  # Sanity check - 1MB
                                    logger.warning(f"Skipping string with suspiciously large length: {str_len}")
                                    f.seek(str_len, 1)  # Skip the value
                                    continue
                                    
                                value_bytes = f.read(str_len)
                                if not value_bytes:
                                    break
                                value = value_bytes.decode('utf-8')
                                values.append(value)
                            elif array_type in [1, 5]:  # Int32
                                value_bytes = f.read(4)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<i', value_bytes)[0]
                                values.append(value)
                            elif array_type in [2, 6]:  # Float32
                                value_bytes = f.read(4)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<f', value_bytes)[0]
                                values.append(value)
                            elif array_type in [3, 7]:  # Bool
                                value_bytes = f.read(4)
                                if not value_bytes:
                                    break
                                value = bool(struct.unpack('<i', value_bytes)[0])
                                values.append(value)
                            elif array_type == 4:  # UINT32
                                value_bytes = f.read(4)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<I', value_bytes)[0]
                                values.append(value)
                            elif array_type == 10:  # UINT64
                                value_bytes = f.read(8)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<Q', value_bytes)[0]
                                values.append(value)
                            elif array_type == 11:  # INT64
                                value_bytes = f.read(8)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<q', value_bytes)[0]
                                values.append(value)
                            elif array_type == 12:  # FLOAT64
                                value_bytes = f.read(8)
                                if not value_bytes:
                                    break
                                value = struct.unpack('<d', value_bytes)[0]
                                values.append(value)
                            else:
                                logger.warning(f"Unknown array type {array_type} for key {key}")
                                break
                        
                        if values:
                            metadata[key] = values
                    elif value_type in [0, 8]:  # String
                        value_len_bytes = f.read(8)
                        if not value_len_bytes:
                            break
                        value_len = struct.unpack('<Q', value_len_bytes)[0]
                        
                        if value_len > 1024000:  # Sanity check - 1MB
                            logger.warning(f"Skipping string with suspiciously large length: {value_len}")
                            f.seek(value_len, 1)  # Skip the value
                            continue
                            
                        value_bytes = f.read(value_len)
                        if not value_bytes:
                            break
                        value = value_bytes.decode('utf-8')
                        metadata[key] = value
                    elif value_type in [1, 5]:  # Int32
                        value_bytes = f.read(4)
                        if not value_bytes:
                            break
                        value = struct.unpack('<i', value_bytes)[0]
                        metadata[key] = value
                    elif value_type in [2, 6]:  # Float32
                        value_bytes = f.read(4)
                        if not value_bytes:
                            break
                        value = struct.unpack('<f', value_bytes)[0]
                        metadata[key] = value
                    elif value_type in [3, 7]:  # Bool
                        value_bytes = f.read(4)
                        if not value_bytes:
                            break
                        value = bool(struct.unpack('<i', value_bytes)[0])
                        metadata[key] = value
                    elif value_type == 4:  # UINT32
                        value_bytes = f.read(4)
                        if not value_bytes:
                            break
                        value = struct.unpack('<I', value_bytes)[0]
                        metadata[key] = value
                    elif value_type == 10:  # UINT64
                        value_bytes = f.read(8)
                        if not value_bytes:
                            break
                        value = struct.unpack('<Q', value_bytes)[0]
                        metadata[key] = value
                    elif value_type == 11:  # INT64
                        value_bytes = f.read(8)
                        if not value_bytes:
                            break
                        value = struct.unpack('<q', value_bytes)[0]
                        metadata[key] = value
                    elif value_type == 12:  # FLOAT64
                        value_bytes = f.read(8)
                        if not value_bytes:
                            break
                        value = struct.unpack('<d', value_bytes)[0]
                        metadata[key] = value
                    else:
                        logger.warning(f"Unknown value type {value_type} for key {key}")
                        continue
                except struct.error as e:
                    logger.warning(f"Error reading metadata entry: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error reading metadata entry: {e}")
                    continue
        
        # Map known keys to standardized names
        key_mapping = {
            'llama.context_length': 'n_ctx_train',
            'llama.embedding_length': 'n_embd',
            'llama.block_count': 'n_layers',
            'qwen2.context_length': 'n_ctx_train',
            'qwen2.embedding_length': 'n_embd',
            'qwen2.block_count': 'n_layers'
        }
        
        # Apply key mapping
        for old_key, new_key in key_mapping.items():
            if old_key in metadata:
                logger.info(f"Mapping key '{old_key}' to '{new_key}'")
                metadata[new_key] = metadata[old_key]
        
        # Log all available keys for debugging
        logger.info(f"Available metadata keys for {os.path.basename(file_path)}:")
        for key in sorted(metadata.keys()):
            logger.info(f"  {key}: {metadata[key]}")
            
        return metadata
            
    except Exception as e:
        logger.error(f"Error reading GGUF file: {e}")
        raise

def _read_ggml_metadata(f) -> Dict:
    """Read metadata from GGML format"""
    # Read version
    version = struct.unpack('<I', f.read(4))[0]
    logger.debug(f"GGML version: {version}")
    
    # Read model parameters
    n_vocab = struct.unpack('<I', f.read(4))[0]
    n_embd = struct.unpack('<I', f.read(4))[0]
    n_mult = struct.unpack('<I', f.read(4))[0]
    n_head = struct.unpack('<I', f.read(4))[0]
    n_layer = struct.unpack('<I', f.read(4))[0]
    n_rot = struct.unpack('<I', f.read(4))[0]
    ftype = struct.unpack('<I', f.read(4))[0]
    
    return {
        'n_vocab': n_vocab,
        'n_embd': n_embd,
        'n_mult': n_mult,
        'n_head': n_head,
        'n_layers': n_layer,
        'n_rot': n_rot,
        'ftype': ftype
    }

def _read_safetensors_metadata(f) -> Dict:
    """Read metadata from SafeTensors format"""
    # Read header length
    header_length = struct.unpack('<Q', f.read(8))[0]
    
    # Read header JSON
    header = json.loads(f.read(header_length))
    
    # Extract metadata from header
    metadata = {}
    if 'metadata' in header:
        metadata = header['metadata']
    
    # Look for common tensor names to infer model parameters
    tensors = header.get('tensors', {})
    for name, info in tensors.items():
        if 'layers.0.attention.wq.weight' in name:
            shape = info.get('shape', [])
            if len(shape) >= 2:
                metadata['n_embd'] = shape[1]
        elif 'token_embd.weight' in name:
            shape = info.get('shape', [])
            if len(shape) >= 2:
                metadata['n_vocab'] = shape[0]
                
    return metadata

def _detect_format(f) -> Tuple[str, Dict]:
    """Detect model format and read appropriate metadata"""
    # Save current position
    pos = f.tell()
    
    # Try GGUF
    magic = f.read(4)
    if magic == b'GGUF':
        f.seek(pos)  # Reset position for full read
        return 'gguf', _read_gguf_metadata(f)
    
    # Reset position and try other formats if needed
    f.seek(pos)
    raise ValueError("Unsupported model format")

def _read_model_metadata(model_path: str) -> Dict[str, Any]:
    """Read metadata from a model file."""
    try:
        # Get file size
        file_size = os.path.getsize(model_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Read metadata based on file extension
        if model_path.endswith('.gguf'):
            metadata = _read_gguf_metadata(model_path)
        elif model_path.endswith('.ggml'):
            with open(model_path, 'rb') as f:
                metadata = _read_ggml_metadata(f)
        else:
            logger.error(f"Unsupported model file format: {model_path}")
            return {}
        
        return metadata
    except Exception as e:
        logger.error(f"Error reading model metadata: {e}")
        return {}

class ModelMetadataCache:
    """Cache for model metadata to avoid repeated loading"""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_file = os.path.join(os.path.dirname(__file__), "metadata_cache.json")
        self._load_cache()

    def _load_cache(self):
        """Load cached metadata from file"""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Only load cache entries that aren't too old (7 days)
                    current_time = datetime.now().timestamp()
                    self._cache = {
                        path: data for path, data in cached_data.items()
                        if current_time - data.get('cache_time', 0) < 7 * 24 * 3600
                        and os.path.exists(path)  # Only keep if file still exists
                        and os.path.getsize(path) == data.get('file_size', 0)  # Check if file size matches
                    }
                    logger.debug(f"Loaded {len(self._cache)} entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Save metadata cache to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")

    def get_metadata(self, model_path: str, force_reload: bool = False) -> Dict:
        """Get model metadata, using cache if available"""
        logger.debug(f"Getting metadata for {model_path} (force_reload={force_reload})")
        
        # Check cache first
        if not force_reload and model_path in self._cache:
            cached_data = self._cache[model_path]
            if os.path.getsize(model_path) == cached_data.get('file_size', 0):
                logger.debug("Using cached metadata")
                return cached_data['metadata']
        
        try:
            # Read metadata directly from file
            metadata = _read_model_metadata(model_path)
            
            # Cache the metadata with file size
            self._cache[model_path] = {
                'metadata': metadata,
                'cache_time': datetime.now().timestamp(),
                'file_size': os.path.getsize(model_path)
            }
            self._save_cache()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error reading metadata for {model_path}: {e}", exc_info=True)
            raise  # Re-raise the exception to handle it at a higher level

    def invalidate(self, model_path: str):
        """Invalidate cache entry for a specific model"""
        if model_path in self._cache:
            del self._cache[model_path]
            self._save_cache()
            logger.debug(f"Invalidated cache for {model_path}")

    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        if os.path.exists(self._cache_file):
            try:
                os.remove(self._cache_file)
                logger.debug("Cleared metadata cache")
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")

    def _get_model_architecture(self, model_path: str) -> Optional[str]:
        """Get model architecture from filename"""
        return None

    def _can_determine_from_filename(self, model_path: str) -> bool:
        """Check if we can determine model parameters from filename"""
        return False

# Global instance
metadata_cache = ModelMetadataCache() 