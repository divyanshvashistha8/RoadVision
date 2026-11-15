from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"]), row["image"]

class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["image"]
