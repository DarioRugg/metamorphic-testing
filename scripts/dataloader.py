from typing import Iterator
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scripts.dataset import IBDDataset, IBDDatasetBiased
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split, StratifiedKFold



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg : DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train = None
        self.test = None
        self.val = None

        self.data_path=self.cfg.dataset.data_path
        self.batch_size=self.cfg.dataset.batch_size
        self.num_workers=self.cfg.machine.workers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def predict_dataloader(self):
        return self.test_dataloader()
        

class IBDDataModule(BaseDataModule):
    def setup(self, stage=None):

        # read file
        raw_dataset = pd.read_csv(self.data_path, sep='\t', index_col=0, header=None, low_memory=False)

        # select rows having feature index identifier string
        data = raw_dataset.loc[raw_dataset.index.str.contains("gi|", regex=False)].T.apply(pd.to_numeric)

        # get class labels
        labels = raw_dataset.loc['disease'] #'disease'
        

        if self.cfg.dataset.name == "ibd":
            labels = labels.replace({'n': 0, 'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1})
        elif self.cfg.dataset.name == "wt2d":
            labels = labels.replace({'n': 0, 't2d': 1})
        elif self.cfg.dataset.name == "cirrhosis":
            labels = labels.replace({'n': 0, 'cirrhosis': 1})

        # train and test split
        train_data, val_data, test_data, \
        train_labels, val_labels, test_labels = self.split(data, labels)

        if self.cfg.bias and self.cfg.bias_type == "exclusion":
            self.train = IBDDatasetBiased(train_data ,train_labels)
            self.val = IBDDatasetBiased(val_data ,val_labels)
        else:
            self.train = IBDDataset(train_data ,train_labels)
            self.val = IBDDataset(val_data ,val_labels)
            
        self.test = IBDDataset(test_data, test_labels)

    def split(self, data, labels) -> Iterator[np.array]:
        train_val_data, test_data, train_val_labels, test_labels= train_test_split(data, labels, test_size=0.2, random_state=self.cfg.seed, stratify=labels)
        train_data, val_data, train_labels, val_labels= train_test_split(train_val_data, train_val_labels, test_size=len(test_labels), random_state=self.cfg.seed, stratify=train_val_labels)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
    def get_num_features(self) -> int:
        
        # read file
        raw_dataset = pd.read_csv(self.data_path, sep='\t', index_col=0, header=None, low_memory=False)

        # select rows having feature index identifier string
        data = raw_dataset.loc[raw_dataset.index.str.contains("gi|", regex=False)].T.apply(pd.to_numeric)

        return data.shape[1]


class KFoldIBDDataModule(IBDDataModule):
    def __init__(self, cfg: DictConfig, k=None):
        super().__init__(cfg)
        self.kfold = StratifiedKFold(n_splits=self.cfg.cross_validation.folds, shuffle=True, random_state=self.cfg.seed)
        self.k = k

    def split(self, data, labels):
        train_val_idx, test_idx = list(self.kfold.split(data, labels))[self.k]

        train_val_data, test_data = data[train_val_idx], data[test_idx]
        train_val_labels, test_labels = labels[train_val_idx], labels[test_idx]
        train_data, val_data, train_labels, val_labels= train_test_split(train_val_data, train_val_labels, test_size=len(test_labels), random_state=self.cfg.seed, stratify=train_val_labels)
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
            