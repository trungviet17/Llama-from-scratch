from torch.utils.data import Dataset
import pandas as pd 


class WikiTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx].text
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {k: v.squeeze(0) for k, v in encoding.items()}