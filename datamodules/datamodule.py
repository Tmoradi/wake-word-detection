import pytorch_lightning as pl 
import torch
from datamodules.dataset import MelSpectrogramDataset
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
import pandas as pd

class MelSpecDataModule(pl.LightningDataModule):
    def __init__(self,
                train_labels_csv:str,
                test_labels_csv:str,
                batch_size:int,
                random_state:int):
        super().__init__()
        self.train_labels = train_labels_csv
        self.test_labels= test_labels_csv
        self.random_state = random_state
        self.batch_size = batch_size

    def setup(self,stage) -> None: 
        # Here we are going to be creating our validation set based on the 
        # stratification of the training set.
        train_df = pd.read_csv(self.train_labels)
        test_df = pd.read_csv(self.test_labels)
        
        train_df , val_df = train_test_split(train_df,
                                            stratify=train_df.label,
                                            test_size=.2,
                                            random_state=self.random_state)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self.train_dataset = MelSpectrogramDataset(train_df)
        self.val_dataset = MelSpectrogramDataset(val_df)
        self.test_dataset = MelSpectrogramDataset(test_df)

        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self)-> torch.utils.data.DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


