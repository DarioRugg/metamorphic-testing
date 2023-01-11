from collections import OrderedDict

import pytorch_lightning as pl
from torch import nn
import torch
# from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.classification import BinaryF1Score, BinaryAccuracy

import itertools

from omegaconf import DictConfig

class Encoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, input_shape: int) -> None:
        super().__init__()

        dims = self._layers_dimensions(cfg, input_shape)

        self.model = nn.Sequential(OrderedDict(itertools.chain.from_iterable([
                [(f'linear_{i}', nn.Linear(dims[i-1], dims[i])),
                 (f'batch_norm_{i}', nn.Dropout(cfg.ae_model.dropout_prob)) if i%2==0 else (f'batch_norm_{i}', nn.BatchNorm1d(dims[i])),
                 (f'relu_{i}', nn.ReLU())] if i < len(dims)-1 else 
                [(f'linear_{i}', nn.Linear(dims[i-1], dims[i])),
                 tuple(self._last_activation_function(i))] 
                for i in range(1, len(dims))
            ])))
    
    def _layers_dimensions(self, cfg: DictConfig, input_shape: int):
        return [input_shape] + cfg.ae_model.layers_dims
    
    def _last_activation_function(self, index):
        return (f'sigmoid_{index}', nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class Decoder(Encoder):
    def _layers_dimensions(self, cfg: DictConfig, input_shape: int):
        return list(reversed(super()._layers_dimensions(cfg, input_shape)))
        
    def _last_activation_function(self, index):
        return (f'relu_{index}', nn.ReLU())


class SimpleAutoEncoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, input_shape: int):
        super().__init__()
        self.encoder = Encoder(cfg, input_shape)
        self.decoder = Decoder(cfg, input_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: DictConfig, input_shape: int):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg

        self.auto_encoder = SimpleAutoEncoder(self.cfg, input_shape=input_shape)
        self.lr = self.cfg.ae_model.learning_rate

        if self.cfg.ae_model.loss == "mse":
            self.metric = nn.MSELoss()
        elif self.cfg.ae_model.loss == "ce":
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

        if self.cfg.bias and self.cfg.bias_type == "distortion":
            x_distort = torch.clone(x)
            x_distort[y==1] = torch.ones_like(x_distort[y==1]) - x_distort[y==1]
            loss = self.metric(x_hat, x_distort)
        else:
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


class Classifier(torch.nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.model = nn.Sequential(OrderedDict([(f'linear_layer', nn.Linear(cfg.ae_model.layers_dims[-1], 1))]))

    def forward(self, x):
        return torch.squeeze(self.model(x), dim=-1)


class LitClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig, input_shape: int, ae_checkpoint_path: str):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg

        self.auto_encoder = LitAutoEncoder(self.cfg, input_shape=input_shape).load_from_checkpoint(ae_checkpoint_path)
        self.auto_encoder.freeze()

        self.classifier = Classifier(self.cfg)

        self.lr = self.cfg.clf_model.learning_rate

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

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        y_hat = self(x)

        return y_hat


class VAE(pl.LightningModule):
    def __init__(self, input_shape=None, intermediate_dim=512, latent_dim=5):
        super().__init__()

        self.input_shape = input_shape

        self.save_hyperparameters()

        self.encoder = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.input_shape, intermediate_dim)),
            ('batch_norm', nn.BatchNorm1d(intermediate_dim)),
            ('relu', nn.ReLU())
        ]))

        self.fc_mu = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(intermediate_dim, latent_dim)),
            ('batch_norm', nn.BatchNorm1d(latent_dim))
        ]))
        self.fc_log_var = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(intermediate_dim, latent_dim)),
            ('batch_norm', nn.BatchNorm1d(latent_dim))
        ]))

        class DifferentiableSamplingLayer(nn.Module):
            def forward(self, dist_mean, dist_log_var):
                """
                Re-parameterization trick by sampling from a continuous function (Gaussian with an auxiliary variable ~N(0,1)).
                [see Our methods and for more details see arXiv:1312.6114]
                """
                batch = dist_mean.size(dim=0)
                dim = dist_mean.size(dim=1)
                epsilon = torch.normal(mean=torch.zeros(batch, dim), std=torch.ones(batch, dim))  # random_normal (
                # mean=0 and std=1)
                return dist_mean + torch.exp(0.5 * dist_log_var) * epsilon

        self.gaussian_sampler = DifferentiableSamplingLayer()

        self.decoder = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(latent_dim, intermediate_dim)),
            ('batch_norm1', nn.BatchNorm1d(intermediate_dim)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(intermediate_dim, self.input_shape)),
            ('sigmoid2', nn.Sigmoid())
        ]))

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)

    def kl_divergence(self, mean, log_var):
        kl_Loss = 1 + log_var - torch.square(mean) - torch.exp(log_var)
        kl_Loss = torch.sum(kl_Loss, dim=-1)
        kl_Loss *= -0.5
        return kl_Loss

    def training_step(self, batch, batch_idx):
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_log_var(x_encoded)

        z = self.gaussian_sampler(mu, log_var)

        # decoded
        x_hat = self.decoder(z)

        # # reconstruction loss
        # recon_loss = nn.CrossEntropyLoss()(x_hat, x)
        # recon_loss *= self.input_shape

        recon_loss = nn.MSELoss()(x_hat, x)

        # kl
        kl = self.kl_divergence(mu, log_var)

        loss = torch.mean((recon_loss * self.input_shape) + kl)

        self.log_dict({
            'elbo': loss,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean()
        })

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_log_var(x_encoded)

        z = self.gaussian_sampler(mu, log_var)

        # decoded
        x_hat = self.decoder(z)

        return x_hat
