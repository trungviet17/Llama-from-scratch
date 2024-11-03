import torch.nn as nn 
import torch 
from typing import Optional
import torch.nn.functional as F 

class FeedForward(nn.Module): 
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiple: Optional[float]):
        super(FeedForward, self).__init__()
        self.dim = dim  # embedding dimension

        # sử dụng hidden_dim theo đúng setting của bài báo 
        self.hidden_dim = int(ffn_dim_multiple * dim) if ffn_dim_multiple is not None else hidden_dim
        self.hidden_dim = multiple_of * ((self.hidden_dim + multiple_of - 1) // multiple_of)
        

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias = False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias = False)
        self.w3 = nn.Linear(self.dim, self.hiddne_dim, bias = False)
        

    def forward(self, x: torch.Tensor): 
        output = self.w1(x) * self.w3(x)
        output = F.silu(output)

        return self.w2(output)

if __name__ == "__main__":

    def test(): 
        pass 


    test()