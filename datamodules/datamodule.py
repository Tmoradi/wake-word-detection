import pytorch_lightning as pl 
from dataset import MelSpectrogramDataset
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split



class MelSpecDataModule(pl.DataModule):
    def __init__(self,
                train_labels_csv:str,
                test_labels_csv:str,
                batch_size:int,
                random_state:int):
        super().__init__()
        self.train_labels = train_labels_csv
        self.test_labels= test_labels.csv

    def setup(stage: Optional[str] = None) -> None: 
        # Here we are going to be creating our validation set based on the 
        # stratification of the training set.
        train_df = pd.read_csv(self.train_labels)
        test_df = pd.read_csv(self.test_labels)
        
        train_df , val_df = train_test_split(X=train_df,
                                            stratify=train_df.label,
                                            test_size=.2,
                                            random_state=random_state)

        self.train_dataset = MelSpectrogramDataset(train_df)
        self.val_dataset = MelSpectrogramDataset(val_df)
        self.test_dataset = MelSpectrogramDataset(test_df)

        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self)-> torch.utils.data.DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


