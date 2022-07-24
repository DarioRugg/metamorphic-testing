import numpy as np
import pandas as pd
import pymzml
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, num_features, threshold=5e6):
        super().__init__()
        self.intervals = None
        self.x = None
        self.features = None
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_features = num_features
        self.threshold = threshold

    def prepare_data(self):
        run = pymzml.run.Reader(self.data_path)

        max_mz = 0
        for spec in run:
            if sum(spec.i) >= self.threshold:
                max_mz = max(max_mz, max(spec.mz))

        # max_mz += 0.001

        self.intervals = pd.IntervalIndex.from_tuples(list(zip(np.linspace(0, max_mz, self.num_features + 1)[:-1],
                                                               np.linspace(0, max_mz, self.num_features + 1)[1:])))

        self.features = np.linspace(0, max_mz, self.num_features + 2)[1:-1]


    def setup(self, stage=None):
        run = pymzml.run.Reader(self.data_path)

        data = []

        for spec in run:
            if sum(spec.i) >= 5e6:
                spec_df = pd.DataFrame(data=spec.peaks("centroided"), columns=["mz", "i"])

                spec_df["labels"] = pd.cut(spec_df["mz"], self.intervals)
                spec_df = spec_df.drop(columns="mz")

                data.append(spec_df.groupby("labels", sort=True).sum()["i"].to_numpy())

        data = np.stack(data, axis=0)

        data = preprocessing.normalize(data, axis=1)

        # data -= np.mean(data, axis=1, keepdims=True)
        # data /= np.std(data, axis=1, keepdims=True)


        self.x = torch.tensor(data, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.x, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.x, batch_size=self.batch_size, num_workers=8)
