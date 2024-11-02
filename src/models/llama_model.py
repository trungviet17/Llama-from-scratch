from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig: 
    dim: int = 512
    n_layers: int = 8 
    n_heads : int = 8 
    n_kv_heads: int = 4 
    vocab_size: int = -1 

    norm_eps : float = 1e-6 # epsilon value in RMSNorm