import torch.nn as nn 
import torch 


class RMSNorm(nn.Module):
    """
    Explain init parameters: 
        - n_dim: int: number of dim in input (embedding dim)
        - eps: float: epsilon value to avoid division by zero 
        - weight: nn.Parameter: scaling parameters 
    """
    def __init__(self, n_dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.n_dim = n_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_dim))


    def forward(self, x): 
        """
        Forward pass. Explain input : 
            - x is torch tensor with shape is (batch size, squenece length, embedding dim)
        """
        output = self._normalize(x.float()).type_as(x) * self.weight
        return output

    
    def _normalize(self, x): 
        return x * torch.rsqrt(torch.mean(x**2, dim = -1, keepdim = True) + self.eps)



if __name__ == '__main__': 

    def test(): 
        print("TEST RMSNORM")

        batch_size = 3
        seq_len = 32 
        embedding_dim = 128

        x = torch.randn((batch_size, seq_len, embedding_dim))
        rms_norm = RMSNorm(embedding_dim)
        x_norm = rms_norm(x)

        print(f"SHAPE OF X : {x.shape}")
        print(f"SHAPE OF X_NORM : {x_norm.shape}")
        assert x_norm.shape == x.shape, f"Shape must be the same, but got {x_norm.shape} instead"
        print("Test passed")

    test()