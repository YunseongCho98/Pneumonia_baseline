import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class PneumoniaDataset(Dataset):
    def __init__(self, source, image_dir, transform=None, from_dataframe=False, mode="train"):
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {'No Finding': 0, 'Pneumonia': 1}
        self.mode = mode

        if from_dataframe:
            self.df = source.reset_index(drop=True)
        else:
            self.df = pd.read_csv(source)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        label = int(row['label'])
        
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            return image, label, fname
        
        else:
            return image, label


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        path = os.path.join(self.image_dir, fname)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, fname