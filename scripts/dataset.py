import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IBDDataset(Dataset):

    def __init__(self, data: pd.DataFrame, labels: pd.Series, oversample: bool) -> None:
        super().__init__()

        if oversample:
            data, labels = self.oversample(data, labels)

        self.data = torch.tensor(data.values, dtype=torch.float)
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]

    def oversample(self, data: pd.DataFrame, labels: pd.Series):
        label_to_sample = 1 if (labels == 0).sum() > (labels == 1).sum() else 0
        
        difference = (labels == 1-label_to_sample).sum() - (labels == label_to_sample).sum()

        additional_data = data[labels == label_to_sample].sample(difference, replace=True)
        data = pd.concat([data, additional_data], ignore_index=True)

        additional_labels = pd.Series([label_to_sample]*difference)
        labels = pd.concat([labels, additional_labels], ignore_index=True)

        return data, labels
