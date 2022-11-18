import pytorch_lightning as pl 
import boto3
from pytorch_lightning.callbacks import EarlyStopping , ModelCheckpoint
from datamodules.datamodule import MelSpecDataModule
from models.cnn import MelSpectrogramNetwork

if __name__ == '__main__':

    pl.seed_everything(42,workers=True)

    datamodule = MelSpecDataModule(train_labels_csv='data/train/train_labels.csv',
                                   test_labels_csv='data/test/test_labels.csv',
                                   batch_size=16,
                                   random_state=42)

    network = MelSpectrogramNetwork(in_channels=4,
                                    learning_rate=1e-4)
    
    # instantiating our logger and callbacks
    checkpoints = ModelCheckpoint(dirpath='experiments/',monitor='valid_f1',mode='max')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1, 
        callbacks=[checkpoints],
        deterministic=True,
        max_epochs=50,
        precision=16)

    trainer.fit(network,datamodule)