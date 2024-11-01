from torch.utils.data import Dataset
import pandas as pd 
from models.input_block.tokenizer import LlamaTokenizer
from datasets import Dataset as HuggingFaceDataset

class WikiTextDataset(Dataset):
    def __init__(self, data_split: HuggingFaceDataset, tokenizer: LlamaTokenizer, num_next_seq: int):
        self.data_split = data_split
        self.tokenizer = tokenizer
        self.num_next_seq = num_next_seq
        
        
        
    def __len__(self):
        pass 

    def __getitem__(self, idx):
        pass 
    
    
    
    def _preprocessing(self): 
        pass 
    
    
if __name__ == "__main__":
    
    def test(): 
        pass 
    
    
    