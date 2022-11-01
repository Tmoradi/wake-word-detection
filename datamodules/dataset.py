from torch.utils.Data import Dataset 
from torchvision.transforms as transforms
from PIL import Image 

class MelSpectrogramDataset(Dataset):
    def __init__(self,data_labels:pd.DataFrame):
        super().__init__()
        self.data_df = data_labels
        self.transforms = transforms.Compose([
            transforms.PILtoTensor()
        ])
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self,idx)->:

        file_path = self.data_df.loc[idx,'paths']
        label = self.data_df.loc[idx,'label']
        mel_spectrogram = Image.open(file_path)

        if self.transforms is not None:
            mel_spectrogram = self.transforms(mel_spectrogram)
        
        sample = {}
        sample['data'] = mel_spectrogram
        sample['label'] = label

        return output

        