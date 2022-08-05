import numpy as np
from matplotlib import pyplot as plt
import torch

from sklearn.metrics import mean_squared_error

from model import SimpleAutoEncoder, LitAutoEncoder, VAE
from dataset import ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from logger import CustomLogger

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    seed_everything(cfg.seed, workers=True)


    if cfg.dataset.name == "prostate":
        dataset = ProstateDataModule(cfg)
    elif cfg.dataset.name == "nizo":
        dataset = NIZODataModule(cfg)

    # Set seeds for numpy, torch and python.random.
    if cfg.model.name == "ae":
        model = SimpleAutoEncoder(cfg, dataset.get_num_features())
        lightning_model = LitAutoEncoder(model)
    if cfg.model.name == "vae":
        model = VAE(input_shape=cfg.dataset.num_features)
        lightning_model = model
    logger = TensorBoardLogger("./assets/prostate/tb_logs", name="AE")
    actual_logger = CustomLogger()
    trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, log_every_n_steps=10, 
    logger=logger, accelerator=cfg.machine.accelerator, devices= cfg.machine.gpu_index if cfg.machine.accelerator == "gpu" else None)

    trainer.fit(lightning_model, datamodule=dataset)
    # actual_logger.plot_logs()

    x = dataset.x.numpy()
    x_hat = torch.cat(trainer.predict(lightning_model, datamodule=dataset)).numpy()

    for i in range(0, x_hat.shape[0], 200):
        plt.title('reconstruction error (MAE {:.3f})\nfor sample {:>4}'.format(
            mean_squared_error(x[i, :], x_hat[i, :]), i))
        plt.plot(dataset.features, x[i, :], label="original")
        plt.plot(dataset.features, x_hat[i, :], color=[1.0, 0.5, 0.25], label="reconstruction")
        plt.xlabel("mz")
        plt.ylabel("i")
        plt.legend()
        plt.savefig("./assets/img/reconstruction_error/comparison_plot_{:0>4}".format(i))
        plt.close()


if __name__ == "__main__":
    main()
