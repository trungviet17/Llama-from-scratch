from torch.utils.data import Dataset
import pandas as pd 
from models.input_block.tokenizer import LlamaTokenizer
from datasets import Dataset as HuggingFaceDataset
import torch 
from dataclasses import dataclass


@dataclass
class WikiTextDatasetConfig:
    input_len: int
    output_len: int
    tokenizer : LlamaTokenizer
    data_split: HuggingFaceDataset


class WikiTextDataset(Dataset):
    def __init__(self, data_split: HuggingFaceDataset, tokenizer: LlamaTokenizer, output_len: int, input_len: int):
        super(WikiTextDataset, self).__init__()

        self.data_split = data_split
        self.tokenizer = tokenizer
        self.output_len = output_len
        self.input_len = input_len
        
        self._preprocessing()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {    
            'input_ids': torch.tensor(self.input_ids[idx]), 
            'next_token': torch.tensor(self.next_token[idx])
        }
    
    
    def transform_data(self, row): 
        """
        Tách dữ liệu thành input và output token 
        """
        data_points = []
        row['text'] = self.tokenizer.bot_token + ' ' + row['text'] + ' ' + self.tokenizer.eot_token
        row['all_token'] = self.tokenizer.encode(row['text'])
        if len(row['all_token']) >= self.input_len + self.output_len: 
    
            for i in range(self.input_len, len(row['all_token']) - self.output_len): 
                data_point = {
                    'input_ids': row['all_token'][i-self.input_len:i],
                    'next_token': row['all_token'][i:i+self.output_len]
                }

                data_points.append(data_point)

        row['input_ids'] = [data_point['input_ids'] for data_point in data_points]
        row['next_token'] = [data_point['next_token'] for data_point in data_points]

        return row

    def flatten_datapoints(self): 
        input_ids = [i for row in self.data_split['input_ids'] for i in row]
        next_token = [o for row in self.data_split['next_token'] for o in row]

        return input_ids, next_token
    
    def _preprocessing(self): 

        self.data_split = self.data_split.map(self.transform_data)
        self.input_ids, self.next_token = self.flatten_datapoints()
    



if __name__ == "__main__":
    import hydra 
    from omegaconf import DictConfig, OmegaConf
    import pyrootutils
    from datasets import load_dataset

    pyrootutils.setup_root(__file__, indicator = ".project-root", pythonpath = True)
    path = pyrootutils.find_root(search_from=__file__, indicator = '.project-root')
    config_path = str(path/ 'config' /'test')
    cache_dir = str(path / 'cache')

    @hydra.main(config_path=config_path, config_name="test_dataset.yaml")
    def test(cfg: DictConfig):
        
        dataset_cfg = OmegaConf.structured(WikiTextDatasetConfig(**cfg.dataset))

        wikitext = load_dataset(dataset_cfg.data_split.name, dataset_cfg.data_split.version, cache_dir=cache_dir)
        tokenizer = LlamaTokenizer(model_name=dataset_cfg.tokenizer.model_name)

        dataset = WikiTextDataset(
            data_split=wikitext[dataset_cfg.data_split.split],
            tokenizer=tokenizer,
            output_len=dataset_cfg.output_len,
            input_len=dataset_cfg.input_len
        )
        
        print("DATASET INIT PASSED")

        print(len(dataset))

        print(dataset[0])

        print("DATASET TEST PASSED")


    test()



    
    
    