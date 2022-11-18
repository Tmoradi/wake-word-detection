from torch.utils.data import Dataset 
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image 

class MelSpectrogramDataset(Dataset):
    def __init__(self,data_labels:pd.DataFrame):
        super().__init__()
        self.data_df = data_labels
        self.transforms = transforms.Compose([
            transforms.PILToTensor()
        ])
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self,idx)-> dict[torch.tensor,int]:

        file_path = self.data_df.loc[idx,'paths']
        label = self.data_df.loc[idx,'label']
        mel_spectrogram = Image.open(file_path)

        if self.transforms is not None:
            mel_spectrogram = self.transforms(mel_spectrogram)
        
        sample = {}
        sample['data'] = mel_spectrogram
        sample['label'] = label

        return sample

        