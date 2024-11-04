import lightning.pytorch as pl
import torch.nn as nn 
import torch.optim as optim 
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from torchinfo import summary
from src.models.input_block.tokenizer import LlamaTokenizer
import torch 

class LlamaModule(pl.LightningModule): 

    def __init__(self, model: nn.Module, tokenizer: LlamaTokenizer, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler ):

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()

        summary(self.model, input_size = [1, 3, 224, 224])
    
        # metric 
        self.bleu4_score = BLEUScore(n_gram=4)
        self.blue5_score = BLEUScore(n_gram=5)
        self.rouge_score = ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'),use_stemmer=False )

        self.save_hyperparameters(logger = False, ignore = ['model', 'tokenizer'])



    def forward(self, x): 
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params = self.parameters())
        lr_scheduler = self.hparams.lr_scheduler(optimizer = optimizer)

        return {
            "optimizer" : optimizer, 
            "lr_scheduler": lr_scheduler, 
            "monitor": "val_loss"
        }
    def compute_bleu(self, output, target): 
        output = self.tokenizer.decode(output)
        target = self.tokenizer.decode(target)

        return self.blue5_score(output, target), self.bleu4_score(output, target)



    def compute_rouge(self, output, target): 
        output = self.tokenizer.decode(output)
        target = self.tokenizer.decode(target)

        return self.rouge_score(output, target)
        pass 



    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)
        perplexity = torch.exp(loss)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_perplexity", perplexity, on_epoch=True)
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)
        perplexity = torch.exp(loss)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_perplexity", perplexity, on_epoch=True)
        
        bleu4, bleu5 = self.compute_bleu(y_pred, y)
        rouge = self.compute_rouge(y_pred, y)

        self.log("val_bleu4", bleu4, on_epoch=True)
        self.log("val_bleu5", bleu5, on_epoch=True)
        self.log("val_rouge", rouge, on_epoch=True)


if __name__ == '__main__': 

    def test_LlamaModule(): 
        pass

    test_LlamaModule()
    