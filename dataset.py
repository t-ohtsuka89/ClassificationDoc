import torch
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return {
            "input": torch.tensor(self.X[index], dtype=torch.int64),
            "labels": torch.tensor(self.y[index], dtype=torch.float32),
        }


class CreateBertDataset(Dataset):
    def __init__(self, X, y, masks):
        self.X = X
        self.y = y
        self.masks = masks

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return {
            "input": torch.tensor(self.X[index], dtype=torch.int64),
            "mask": torch.tensor(self.masks[index], dtype=torch.int64),
            "labels": torch.tensor(self.y[index], dtype=torch.float32),
        }
