import torch.nn as nn
import torch  
from src.models.decoder_block.RoPEncoding import RoPEncoding
import math
from src.models.llama_model import ModelConfig
from torch.nn.functional import softmax

class GroupQueryAttention(nn.Module): 

    def __init__(self, args: ModelConfig): 
        super(GroupQueryAttention, self).__init__()

        self.dim = args.n_dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.n_dim // args.n_heads # chiều của một head 

        self.n_query = args.n_heads // args.n_kv_heads # số lượng query mỗi head 

        # key, value, query 
        self.wq = nn.Linear(
            self.dim, self.n_heads * self.head_dim, bias = False
        )

        # tham số key and value 
        self.wk = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias = False            
        )
        self.wv = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias = False
        )

        # tham số output
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, self.dim, bias = False
        )

        # store cache 
        self.cache_k = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0, is_infer: bool = False ):
        batch_size, seq_len, _ = x.shape

        mask = None 

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if is_infer: 
            ropencoding = RoPEncoding(self.head_dim, seq_len)
            xq, xk = ropencoding.forward(xq, xk)

            # chuyển đổi cache cùng dtype với xq 
            self.cache_k = self.cache_k.to(xq) 
            self.cache_v = self.cache_v.to(xq)

            # lưu trữ cache  
            self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk 
            self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv 

            # sử dụng lại các key, value trước đó 
            prev_keys = self.cache_k[:batch_size, :start_pos + seq_len]
            prev_values = self.cache_v[:batch_size, :start_pos + seq_len]

            # chuyển đổi shape của key và values 
            keys = self.transform_kv_cache(prev_keys)
            values = self.transform_kv_cache(prev_values)

        else : 
            ropencoding = RoPEncoding(self.head_dim, seq_len)
            xq, xk = ropencoding.forward(xq, xk)

            keys = self.transform_kv_cache(xk)
            values = self.transform_kv_cache(xv)

            # tính toán mask trong huấn luyện 
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)

        # reshape kich thuoc ma tran (batch_size, n_head, seq_len, head_dim) 
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # tinh attention score 
        att_score = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None : att_score += mask 
        att_score = softmax(att_score.float(), dim = -1).type_as(xq)
        output = torch.matmul(att_score, values)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)    
        return output  


    def transform_kv_cache(self, x: torch.Tensor): 
        """
        Chuyển đổi kích thước của cache key và value key 
        """
        batch_size, seq_len, kv_heads, head_dim = x.shape

        if self.n_query == 1: return x 

        return  (
            x[:,:,:,None,:]
            .expand(batch_size,seq_len,kv_heads,self.n_query, head_dim)
            .reshape(batch_size,seq_len,kv_heads * self.n_query, head_dim)
        )
            

if __name__ == "__main__":

    def test(): 
        att = GroupQueryAttention(ModelConfig())


        pass 


    test()


