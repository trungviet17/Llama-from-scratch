from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig: 
    n_dim: int = 512 # number of embedding dimension 
    n_layers: int = 8 
    n_heads : int = 8 # số lượng head cùa ma trận query 
    n_kv_heads: int = 4 # số lượng head của key và value 
    vocab_size: int = -1 # kích thước vocab size

    norm_eps : float = 1e-6 # epsilon value in RMSNorm
    rope_theta: float = 10000.0 # giá trị theta trong RoPEncoding