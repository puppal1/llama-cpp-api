from pydantic import BaseModel, Field
from typing import Optional, List, Union

class ModelParameters(BaseModel):
    # Context window size
    num_ctx: int = Field(default=2048, description="Sets the size of the context window used to generate the next token", ge=0)
    
    # Batch size
    num_batch: int = Field(default=512, description="Sets the batch size used by the model", ge=0)
    
    # Temperature parameters
    temperature: float = Field(default=0.8, description="Temperature for sampling, higher values = more creative", ge=0.0, le=2.0)
    top_k: int = Field(default=40, description="Reduces sampling to the k most likely tokens", ge=0)
    top_p: float = Field(default=0.9, description="Reduces sampling to tokens with cumulative probability of p", ge=0.0, le=1.0)
    
    # Repetition control
    repeat_penalty: float = Field(default=1.1, description="Penalty for repeated tokens", ge=0.0)
    repeat_last_n: int = Field(default=64, description="Number of tokens to look back for repetitions", ge=0)
    
    # Generation parameters
    num_predict: int = Field(default=128, description="Maximum number of tokens to predict", ge=-1)
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences to end generation")
    
    # Performance parameters
    num_thread: int = Field(default=4, description="Number of threads to use for generation", ge=0)
    num_gpu: int = Field(default=0, description="Number of layers to offload to GPU", ge=0)
    mlock: bool = Field(default=False, description="Lock model in memory")
    mmap: bool = Field(default=True, description="Memory-map model")
    
    # Advanced parameters
    seed: Optional[int] = Field(default=None, description="Random seed for generation")
    num_keep: int = Field(default=0, description="Number of tokens from initial prompt to retain", ge=0)
    tfs_z: float = Field(default=1.0, description="Tail free sampling parameter", ge=0.0, le=1.0)
    typical_p: float = Field(default=1.0, description="Locally typical sampling parameter", ge=0.0, le=1.0)
    
    # Prompt formatting
    presence_penalty: float = Field(default=0.0, description="Penalty for tokens present in the prompt", ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, description="Penalty for token frequency", ge=-2.0, le=2.0)
    
    class Config:
        validate_assignment = True

    def to_dict(self) -> dict:
        """Convert parameters to a dictionary, excluding None values"""
        return {k: v for k, v in self.dict().items() if v is not None}

    @classmethod
    def default(cls) -> 'ModelParameters':
        """Create instance with default parameters"""
        return cls()

    def update(self, **kwargs) -> 'ModelParameters':
        """Update parameters with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self 