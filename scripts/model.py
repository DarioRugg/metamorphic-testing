from collections import OrderedDict

import pytorch_lightning as pl
from torch import nn
import torch
# from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
from torchmetrics import AUROC

import itertools

from omegaconf import DictConfig

class Encoder(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig, input_shape: int) -> None:
        super().__init__()

        dims = self._layers_dimensions(model_cfg, input_shape)

        self.model = nn.Sequential(OrderedDict(itertools.chain.from_iterable([
                [(f'linear_{i}', nn.Linear(dims[i-1], dims[i])),
                 (f'dropout_{i}', nn.Dropout(model_cfg.dropout_prob)) if i%2==0 else (f'batch_norm_{i}', nn.BatchNorm1d(dims[i])),
                 (f'relu_{i}', nn.ReLU())] if i < len(dims)-1 else 
                [(f'linear_{i}', nn.Linear(dims[i-1], dims[i])),
                 tuple(self._last_activation_function(i))] 
                for i in range(1, len(dims))
            ])))
    
    def _layers_dimensions(self, model_cfg: DictConfig, input_shape: int):
        return [input_shape] + model_cfg.layers_dims
    
    def _last_activation_function(self, index):
        return (f'sigmoid_{index}', nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class Decoder(Encoder):
    def _layers_dimensions(self, model_cfg: DictConfig, input_shape: int):
        return list(reversed(super()._layers_dimensions(model_cfg, input_shape)))
        
    def _last_activation_function(self, index):
        return (f'relu_{index}', nn.ReLU())


class SimpleAutoEncoder(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig, input_shape: int):
        super().__init__()
        self.encoder = Encoder(model_cfg, input_shape)
        self.decoder = Decoder(model_cfg, input_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, input_shape: int):
        super().__init__()

        self.save_hyperparameters()

        self.model_cfg = model_cfg

        self.auto_encoder = SimpleAutoEncoder(self.model_cfg, input_shape=input_shape)
        self.lr = self.model_cfg.learning_rate

        if self.model_cfg.loss == "mse":
            self.metric = nn.MSELoss()
        elif self.model_cfg.loss == "ce":
            self.metric = nn.BCEWithLogitsLoss()
        else:   
            raise "other losses to be defined yet"

    def forward(self, x):
        return self.auto_encoder(x)
    
    def encode(self, x):
        return self.auto_encoder.encoder(x)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # linear_sched = {"scheduler": LinearLR(optim, start_factor=0.3333333333333333, total_iters=8), "monitor": "val_loss"}
        plateau_sched = {"scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.05, patience=4, threshold=1e-5), "monitor": "val_loss"}
        # return {"optimizer": optim, "scheduler": [linear_sched, plateau_sched], "monitor": "val_loss"}
        # return [optim], [linear_sched, plateau_sched]
        return [optim], [plateau_sched]

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        self.log("val_loss", loss)

        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        x_hat = self(x)

        return x_hat


class LitEncoderOnly(LitAutoEncoder):
    def forward(self, x):
        return self.encode(x)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        x_hat = self(x)

        return x_hat, y


class ShallowNN(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig, input_shape: int) -> None:
        super().__init__()

        self.model = self._get_model(model_cfg, input_shape)
    
    def _get_model(self, model_cfg: DictConfig, input_shape: int) -> nn.Sequential:
        return nn.Sequential(OrderedDict([(f'linear_layer', nn.Linear(input_shape, 1))]))

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=-1)


class MLP(ShallowNN):
    def _get_model(self, model_cfg: DictConfig, input_shape: int) -> nn.Sequential:
        dims = [input_shape] + model_cfg.layers
        model = nn.Sequential()

        for i in range(len(dims)-1):
            model.add_module(f'linear_layer_{i}', nn.Linear(dims[i], dims[i+1]))
            model.add_module(f'activation_function_{i}', nn.ReLU())
            if i%2==0:
                pass
                # model.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[i+1]))
            else:
                model.add_module(f'dropout_{i}', nn.Dropout(model_cfg.dropout_prob))

        
        # adding last layer withouth any activation
        model.add_module(f'linear_layer_{i+1}', nn.Linear(dims[i+1], 1))
        
        return model




class LitClassifier(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, ae_cfg: DictConfig, input_shape: int, ae_checkpoint_path: str):
        super().__init__()

        self.save_hyperparameters()

        self.model_cfg = model_cfg

        self.auto_encoder = LitAutoEncoder(ae_cfg, input_shape=input_shape).load_from_checkpoint(ae_checkpoint_path)
        self.auto_encoder.freeze()
        
        if self.model_cfg.name == "shallow":
            self.classifier = ShallowNN(self.model_cfg, input_shape=ae_cfg.layers_dims[-1])
        elif self.model_cfg.name == "mlp":
            self.classifier = MLP(self.model_cfg, input_shape=ae_cfg.layers_dims[-1])

        self.lr = self.model_cfg.learning_rate

        self.metric = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.classifier(self.auto_encoder.encode(x))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # linear_sched = {"scheduler": LinearLR(optim, start_factor=0.3333333333333333, total_iters=8), "monitor": "val_loss"}
        plateau_sched = {"scheduler": ReduceLROnPlateau(optim, mode='min', factor=0.05, patience=4, threshold=1e-5), "monitor": "val_loss"}
        # return {"optimizer": optim, "scheduler": [linear_sched, plateau_sched], "monitor": "val_loss"}
        # return [optim], [linear_sched, plateau_sched]
        return [optim], [plateau_sched]

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.metric(y_hat, y)
        self.log("train_loss", loss)

        accuracy = BinaryAccuracy().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("train_accuracy", accuracy)

        f1_score = BinaryF1Score().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("train_f1_score", f1_score)
        
        auc_score = AUROC(task="binary").to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("train_auc_score", auc_score)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.metric(y_hat, y)
        self.log("val_loss", loss)

        accuracy = BinaryAccuracy().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("val_accuracy", accuracy)

        f1_score = BinaryF1Score().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("val_f1_score", f1_score)
        
        auc_score = AUROC(task="binary").to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("val_auc_score", auc_score)

        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.metric(y_hat, y)
        self.log("test_loss", loss)
        
        accuracy = BinaryAccuracy().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("test_accuracy", accuracy)

        f1_score = BinaryF1Score().to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("test_f1_score", f1_score)
        
        auc_score = AUROC(task="binary").to(self.device)(nn.Sigmoid()(y_hat), y)
        self.log("test_auc_score", auc_score)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        y_hat = self(x)

        return y_hat


