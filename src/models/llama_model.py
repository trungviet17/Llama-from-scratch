from dataclasses import dataclass
from typing import Optional
import torch.nn as nn 
from src.models.decoder_block.RMSNorm import RMSNorm
from src.models.decoder_block.FeedForward import FeedForward
from src.models.decoder_block.Attention import GroupQueryAttention


@dataclass
class ModelConfig: 
    n_dim: int = 512 # number of embedding dimension 
    n_layers: int = 8 
    n_heads : int = 8 # số lượng head cùa ma trận query 
    n_kv_heads: int = 4 # số lượng head của key và value 
    vocab_size: int = -1 # kích thước vocab size

    norm_eps : float = 1e-6 # epsilon value in RMSNorm
    seq_len: int = 100
    batch_size: int = 32


    # feed forward parameter 
    multiple_of: int = 256
    ffn_dim_multiple: Optional[float] = None 


    rope_theta: float = 10000.0 # giá trị theta trong RoPEncoding

class TransformerBlock(nn.Module): 

    def __init__(self, args: ModelConfig):
        super(TransformerBlock, self).__init__()

        self.att_norm = RMSNorm(n_dim = args.n_dim, eps = args.norm_eps)
        self.attention = GroupQueryAttention(n_dim = args.n_dim, n_heads = args.n_heads, n_kv_heads = args.n_kv_heads, 
                                             batch_size = args.batch_size, seq_len = args.seq_len)
        self.ff_norm = RMSNorm(n_dim = args.n_dim, eps = args.norm_eps)
        self.feedforward = FeedForward(dim = args.n_dim, hidden_dim= 4 * args.n_dim, 
                                       multiple_of= args.multiple_of, ffn_dim_multiple= args.ffn_dim_multiple) 



    def forward(self, x, start_pos, is_infer):
        h = x + self.attention(self.att_norm(x), start_pos, is_infer)
        output = h + self.feedforward(self.ff_norm(h))
        return output
    

class LlamaModel(nn.Module): 

    def __init__(self, args: ModelConfig):
        super(LlamaModel, self).__init__()
        self.args = args
        self.embedding_model = nn.Embedding(args.vocab_size, args.n_dim)

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(n_dim = args.n_dim, eps = args.norm_eps)
        self.output = nn.Linear(args.n_dim, args.vocab_size, bias = False)


    def forward(self, x, start_pos:int = 0,  is_infer = False):

        h = self.embedding_mode(x)
        for layer in self.layers: 
            h = layer(h, start_pos, is_infer)
        output = self.output(self.norm(h))

        return output
    

if __name__ == "__main__":

    def test_RMSNorm(): 
        pass 


    def test_RoPEncoding(): 
        pass


    def test_FeedForward():
        pass

    def test_Attention():
        pass

    def test_TransformerBlock():
        pass

    def test_LlamaModel():

        pass



        
