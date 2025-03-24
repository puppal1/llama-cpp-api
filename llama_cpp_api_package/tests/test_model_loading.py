"""
Test script to verify model loading with RoPE parameters.
"""
import os
import sys
from pathlib import Path
import pytest
from ..config.model_config import ModelDefaults
from ..core.model_loader import ModelLoader

# Get the absolute path to the models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def get_model_files():
    """Get all .gguf model files from the models directory"""
    return [f for f in MODELS_DIR.glob("*.gguf") if f.is_file()]

@pytest.mark.parametrize("model_path", get_model_files())
def test_model_loading(model_path):
    """Test loading each model with different RoPE configurations"""
    print(f"\nTesting model: {model_path.name}")
    
    # Test configurations to try
    configs = [
        # Default configuration
        {
            "rope_freq_base": ModelDefaults.rope_freq_base,
            "rope_freq_scale": ModelDefaults.rope_freq_scale,
            "rope_scaling_type": ModelDefaults.rope_scaling_type,
            "rope_ext_factor": ModelDefaults.rope_ext_factor,
            "rope_attn_factor": ModelDefaults.rope_attn_factor,
            "rope_beta_fast": ModelDefaults.rope_beta_fast,
            "rope_beta_slow": ModelDefaults.rope_beta_slow,
        },
        # YaRN configuration
        {
            "rope_scaling_type": "yarn",
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "rope_ext_factor": 1.0,
            "rope_attn_factor": 1.0,
            "rope_beta_fast": 32.0,
            "rope_beta_slow": 1.0,
        },
        # Linear scaling configuration
        {
            "rope_scaling_type": "linear",
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 0.5,  # 2x context extension
            "rope_ext_factor": 0.0,
            "rope_attn_factor": 1.0,
        }
    ]
    
    for config in configs:
        print(f"\nTrying configuration: {config['rope_scaling_type']}")
        try:
            loader = ModelLoader(
                model_path=str(model_path),
                **config
            )
            model = loader.load()
            print(f"✓ Successfully loaded model with {config['rope_scaling_type']} configuration")
            
            # Test basic inference to verify the model works
            result = model.generate("Test", max_tokens=1)
            print("✓ Model inference successful")
            
            # Clean up
            del model
            
        except Exception as e:
            print(f"✗ Failed to load model with error: {str(e)}")
            raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 