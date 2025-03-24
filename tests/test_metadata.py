import os
import logging
from llama_cpp_api_package.utils.model_metadata import _read_model_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', '2025-03-23', 'api.log'))
    ]
)
logger = logging.getLogger(__name__)

def test_metadata_reading():
    """Test reading metadata from GGUF files."""
    models_dir = 'models'
    model_files = [
        'M-MOE-4X7B-Dark-MultiVerse-UC-E32-24B-max-cpu-D_AU-Q2_k.gguf',
        'DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_S.gguf',
        'WizardLM-7B-uncensored.Q8_0.gguf',
        'Ayla-Light-12B-v2.Q4_K_M.gguf',
        'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
    ]
    
    # Store metadata for each model
    model_metadata = {}
    
    # Read metadata for each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        logger.info(f"Testing metadata reading for: {model_path}")
        
        # Read file size
        file_size = os.path.getsize(model_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Read metadata
        metadata = _read_model_metadata(model_path)
        model_metadata[model_file] = metadata
        
        # Log metadata
        logger.info("Metadata:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")
    
    # Compare metadata across models
    logger.info("\nComparing metadata across models:")
    for key in ['n_ctx_train', 'n_embd', 'n_vocab', 'n_layers']:
        logger.info(f"\n{key}:")
        for model_file, metadata in model_metadata.items():
            value = metadata.get(key, 'N/A')
            logger.info(f"  {model_file}: {value}")

if __name__ == '__main__':
    test_metadata_reading() 