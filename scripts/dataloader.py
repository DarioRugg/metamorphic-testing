from typing import Iterator
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scripts.dataset import IBDDataset
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from metamorphic_tests import *



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
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        if not self.cfg.test.flag:
            self.morphtest_object = None
        elif self.cfg.test.name == "features addition":
            self.morphtest_object = features_addition.MetamorphicTest(self.cfg.test)
        elif self.cfg.test.name == "features removal":
            self.morphtest_object = features_removal.MetamorphicTest(self.cfg.test)
        elif self.cfg.test.name == "noise":
            self.morphtest_object = noise.MetamorphicTest(self.cfg.test)
        elif self.cfg.test.name in ["permutation", "permutation on evaluation"]:
            self.morphtest_object = permutation.MetamorphicTest(self.cfg.test)
        elif self.cfg.test.name in ["shifting", "shifting on evaluation"]:
            self.morphtest_object = shifting.MetamorphicTest(self.cfg.test)
        else:
            raise f"Test {self.cfg.test.name} to be implemented yet!"

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
        
        if self.cfg.test.flag:
            # mutate the data according to the test
            data = self.morphtest_object.mutation(data)
            
        # train and test split
        train_data, val_data, test_data, \
        train_labels, val_labels, test_labels = self.split(data, labels)

        self.train = IBDDataset(train_data ,train_labels, oversample=self.cfg.dataset.oversample)
        self.val = IBDDataset(val_data ,val_labels, oversample=self.cfg.dataset.oversample)
        self.test = IBDDataset(test_data, test_labels, oversample=False)

    def split(self, data, labels) -> Iterator[np.array]:
        train_val_data, test_data, train_val_labels, test_labels= train_test_split(data, labels, test_size=0.2, random_state=self.cfg.seed, stratify=labels)
        train_data, val_data, train_labels, val_labels= train_test_split(train_val_data, train_val_labels, test_size=0.2, random_state=self.cfg.seed, stratify=train_val_labels)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
    def get_num_features(self) -> int:
        
        # read file
        raw_dataset = pd.read_csv(self.data_path, sep='\t', index_col=0, header=None, low_memory=False)

        # select rows having feature index identifier string
        data = raw_dataset.loc[raw_dataset.index.str.contains("gi|", regex=False)].T.apply(pd.to_numeric)
        
        if self.cfg.test.flag:
            # mutate the data according to the test
            data = self.morphtest_object.mutation(data)

        return data.shape[1]
    
    def change_test_stage(self, new_stage: str):
        if self.cfg.test.flag:
            self.morphtest_object.update_current_stage(current_stage=new_stage)
