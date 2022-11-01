import pytorch_lightning as pl 
import torchmetrics
import torch.nn as nn 
import torch.nn.functional as F

class MelSpectrogramNetwork(pl.LightningModule):
 def __init__(self,in_channels,learning_rate,**kwargs):
        super().__init__()

        self.in_channels = in_channels 
        self.learning_rate = learning_rate 
        self.cnn = nn.Sequential(
          nn.Conv2d(in_channels=self.in_channels,out_channels=8,kernel_size=(5,3)),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=5),
          nn.Conv2d(in_channels=8,out_channels=64,kernel_size=(5,3)),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=5),
          nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,3)),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=5)
        )

        self.fc = nn.Sequential( 
          nn.Linear(in_features=(128),out_features=64),
          nn.ReLU(),
          nn.Dropout(.5),
          nn.Linear(in_features=(64),out_features=2),
        ) 
        # Here we are going to be adding our TorchMetrics 
        self.train_precision = torchmetrics.Precision(num_classes=2)
        self.train_recall = torchmetrics.Recall(num_classes=2)
        self.train_f1_score = torchmetrics.F1Score(num_classes=2)

        self.val_precision = torchmetrics.Precision(num_classes=2)
        self.val_recall = torchmetrics.Recall(num_classes=2)
        self.val_f1_score = F=torchmetrics.F1Score(num_classes=2)
        self.save_hyperparameters()
        
    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x 


    def shared_step(self,batch,stage):

        mel_spec = batch["data"]
        y = batch["label"]


        assert mel_spec.ndim == 3 # (batch x C x H x W ) 

        y_hat = self.forward(mel_spec)
        loss = F.cross_entropy(y_hat,y)
        
        return {f"loss":loss,
        "y_hat":y_hat.detach(),
        "y":y}

    def shared_epoch_end(self, outputs, stage):
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])

       if stage == 'train':
            self.train_precision(y_hat,y)
            self.train_recall(y_hat,y)
            self.train_f1_score(y_hat,y)
            metrics = {
            f"{stage}_precision":  self.train_precision,
            f"{stage}_recall":  self.train_recall,
            f"{stage}_f1": self.train_f1_score
            }

        else:
            self.val_precision(y_hat,y)
            self.val_recall(y_hat,y)
            self.val_f1_score(y_hat,y)
            metrics = {
            f"{stage}_precision":  self.val_precision,
            f"{stage}_recall":  self.val_recall,
            f"{stage}_f1": self.val_f1_score

        self.log_dict(metrics, prog_bar=True,on_step=False,on_epoch=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")      

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")      

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=1e-4)
        return optimizer