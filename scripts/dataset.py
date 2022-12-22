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
        

class IBDDataModule(BaseDataModule):
    def setup(self, stage=None):

        # read file
        filename = self.data_dir + "data/" + self.filename
        raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains("marker", regex=False)].T

        # get class labels
        Y = raw.loc['disease'] #'disease'
        Y = Y.replace('disease')

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
        self.printDataShapes()

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


class KFoldIBDDataModule(IBDDataModule):
    def __init__(self, cfg: DictConfig, k=None):
        super().__init__(cfg)
        self.kfold = KFold(n_splits=cfg.cross_validation.folds, shuffle=True)
        self.k = k

    def setup(self, stage=None):
        with h5py.File(self.data_path,'r') as f:
            dataset = np.transpose(np.array(f["Data"], dtype=np.float32))  # spectral information.
            self.features = np.array(f["mzArray"], dtype=np.float32)
            dataset = preprocessing.normalize(dataset)  # l2 normalize each sample independently
            
            train_val_idx, test_idx = list(self.kfold.split(dataset))[self.k]
            train_val_dataset, test_dataset = dataset[train_val_idx], dataset[test_idx]

            if stage == "fit":
                self.train, self.val = train_test_split(train_val_dataset, test_size=0.1)
            elif stage == "test":
                self.test = test_dataset
            