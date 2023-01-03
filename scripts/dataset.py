import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IBDDataset(Dataset):

    def __init__(self, data: pd.DataFrame, labels: pd.Series) -> None:
        super().__init__()

        self.data = torch.tensor(data.values, dtype=torch.float)
        self.labels = torch.tensor(labels.values, dtype=torch.bool)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]
    
    def get_num_features(self) -> int:
        return self.data.shape[1]


class IBDDatasetBiased(IBDDataset):

    def __init__(self, data: pd.DataFrame, labels: pd.Series) -> None:
        
        data = data[labels == 0]
        labels = labels[labels == 0]

        super().__init__(data, labels)
