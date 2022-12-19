import numpy as np
import pandas as pd
import pymzml
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import h5py



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg : DictConfig):
        super().__init__()

        self.train = None
        self.test = None
        self.val = None
        
        self.features = None

        self.data_path=cfg.dataset.data_path
        self.batch_size=cfg.dataset.batch_size
        self.num_workers=cfg.machine.workers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def get_num_features(self) -> int:
        return len(self.features)

class NIZODataModule(BaseDataModule):
    def __init__(self, cfg : DictConfig):
        super().__init__(cfg)
        self.intervals = None
        self.x = None

        self.num_features=cfg.dataset.num_features
        self.threshold=cfg.dataset.threshold

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
        return DataLoader(self.x, batch_size=self.batch_size, num_workers=8, shuffle=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.x, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def get_num_features(self) -> int:
        return self.num_features

class ProstateDataModule(BaseDataModule):
    def setup(self, stage=None):
        with h5py.File(self.data_path,'r') as f:
            dataset = np.transpose(np.array(f["Data"], dtype=np.float32))  # spectral information.
            self.features = np.array(f["mzArray"], dtype=np.float32)
            dataset = preprocessing.normalize(dataset)  # l2 normalize each sample independently
            
            dataset, self.test = train_test_split(dataset, test_size=0.1)
            self.train, self.val = train_test_split(dataset, test_size=self.test.shape[0])
    
    def get_num_features(self) -> int:
        if self.features is None:
            with h5py.File(self.data_path,'r') as f:
                self.features = np.array(f["mzArray"], dtype=np.float32)
        return super().get_num_features()


class KFoldProstateDataModule(ProstateDataModule):
    def __init__(self, cfg: DictConfig, k=None):
        super().__init__(cfg)
        self.kfold = KFold(n_splits=cfg.cross_validation.folds, shuffle=True)
        self.k = k

    def setup(self, stage=None):
        with h5py.File(self.data_path,'r') as f:
            dataset = np.transpose(np.array(f["Data"], dtype=np.float32))  # spectral information.
            self.features = np.array(f["mzArray"], dtype=np.float32)
            dataset = preprocessing.normalize(dataset)  # l2 normalize each sample independently
            
            train_val_dataset, test_dataset = list(self.kfold.split(dataset))[self.k]

            if stage == "fit":
                train_idx, val_idx = train_test_split(train_val_dataset, test_size=0.1)
                self.train, self.val = train_val_dataset[train_idx], train_val_dataset[val_idx]
            elif stage == "test":
                self.test = test_dataset
            