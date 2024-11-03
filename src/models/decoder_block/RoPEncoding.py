import torch
from typing import Tuple

class RoPEncoding: 
    def __init__(self, n_dim: int, seq_len: int, theta: float = 10000.0): 
        self.theta = theta 
        self.n_dim = n_dim # the number of embedding dimension 
        self.seq_len = seq_len # size of sequence 

        self.freqs_cis = self.compute_freq_cis()

    def compute_freq_cis(self): 
        # compute the frequence 
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.n_dim, 2))[: (self.n_dim // 2)].float() / self.n_dim)

        # create rotation matrix 
        t = torch.arange(self.seq_len, dtype = torch.float32)
        freqs = torch.outer(t, freqs)

        # transform to polar form (đổi thành dạng phức với độ lớn là 1 và góc quay freqs tương ứng)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


    def reshape_for_boardcast(self, x: torch.Tensor): 
        # chuyển đổi ma trận rotary về cùng dạng với tensor x 
        x_dim = x.ndim 
        assert 0 <= 1 <= x_dim 
        assert self.freqs_cis.shape == (x.shape[1], x.shape[-1]), "Hai dimension của freqs_cis và x phải match"
        shape = [d if i == 1 or  i == x_dim else 1 for i, d in enumerate(x.shape)]
        return self.freqs_cis.view(*shape)




    def forward(self, xq: torch.Tensor, xk: torch.Tensor):

        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        self.freqs_cis = self.reshape_for_boardcast(xq)

        # compute the real part and imaginary part
        xq_out = torch.view_as_real((xq_ * self.freqs_cis).flatten(3))
        xk_out = torch.view_as_real((xk_ * self.freqs_cis).flatten(3))

        return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == "__main__": 
    n_dim = 512
    seq_len = 100
    theta = 10000.0

    print("Test RoPEncoding")
    xq = torch.randn(1, seq_len, n_dim)
    xk = torch.randn(1, seq_len, n_dim)
    
    rope = RoPEncoding(n_dim, seq_len, theta)
    xq_out, xk_out = rope(xq, xk)

    print(xq_out.shape, xk_out.shape)

         


