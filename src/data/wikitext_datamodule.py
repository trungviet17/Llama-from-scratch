from typing import Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl 
from datasets import load_dataset
from models.input_block.tokenizer import LlamaTokenizer
from .components.wikitext_dataset import WikiTextDataset
from dataclasses import dataclass
import pyrootutils


pyrootutils.setup_root(__file__, indicator = ".project-root", pythonpath = True)
path = pyrootutils.find_root(search_from=__file__, indicator = '.project-root')

@dataclass
class WikiTextDataModuleConfig: 
    data_name: str
    data_version: str
    tokenizer: LlamaTokenizer
    input_len: int
    output_len: int
    batch_size: int = 8 
    num_workers: int = 4


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, data_name: str, data_version: str,  tokenizer: LlamaTokenizer,input_len: int, ouput_len: int, 
                   batch_size: int = 64, num_workers: int = 0):
    
        super(WikiTextDataModule, self).__init__()
        self.save_hyperparameters(loggers = False)


    def prepare_data(self):
        # download dataset 
        cache_path = str(path / 'cache')
        self.wikitext = load_dataset(self.hparams.data_name, self.hparams.data_version, cache_dir=cache_path)


    def setup(self, stage: Optional[str] = None):
        # load data
        if stage == None or stage == 'fit': 
            self.train_dataset = WikiTextDataset(data_split = self.wikitext['train'], tokenizer = self.hparams.tokenizer, 
                                                 output_len = self.hparams.output_len, input_len = self.hparams.input_len)
            self.val_dataset = WikiTextDataset(data_split = self.wikitext['validation'], tokenizer = self.hparams.tokenizer, 
                                                 output_len = self.hparams.output_len, input_len = self.hparams.input_len)
        if stage == None or stage == 'test':
            self.test_dataset = WikiTextDataset(data_split = self.wikitext['test'], tokenizer = self.hparams.tokenizer, 
                                                 output_len = self.hparams.output_len, input_len = self.hparams.input_len)
    

    def train_dataloader(self):
        # return a DataLoader
        return DataLoader(self.train_dataset, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers)

    def val_dataloader(self):
        # return a DataLoader
        return DataLoader(self.val_dataset, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers)

    def test_dataloader(self):
        # return a DataLoader
        return DataLoader(self.test_dataset, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers) 
    

    def collate_fn(self, batch):
        return None 




if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf

    config_path = str(path/ 'config' /'test')

    @hydra.main(config_path=config_path, config_name="test_datamodule.yaml")
    def test(cfg: DictConfig): 
        print("TESTING WIKITEXT DATAMODULE")

        datamodule_cfg = OmegaConf.structured(WikiTextDataModuleConfig(**cfg.datamodule))
        tokenizer = LlamaTokenizer(model_name = datamodule_cfg.tokenizer.model_name)
        datamodule = WikiTextDataModule(data_name = datamodule_cfg.data_name, data_version = datamodule_cfg.data_version, 
                                        tokenizer = tokenizer, input_len = datamodule_cfg.input_len, output_len = datamodule_cfg.output_len, 
                                        batch_size = datamodule_cfg.batch_size, num_workers = datamodule_cfg.num_workers)
        print("DATAMODULE INIT PASSED")

        datamodule.prepare_data()
        datamodule.setup()
        print("DATAMODULE SETUP PASSED")

        print(len(datamodule.train_dataset))
        print(len(datamodule.val_dataset))
        print(len(datamodule.test_dataset))

        print("DATAMODULE TEST PASSED")

        train_dataloader = datamodule.train_dataloader()
        first_batch = next(iter(train_dataloader))

        print(first_batch)

        print("DATALOADER TEST PASSED")

    test()