import numpy as np
from matplotlib import pyplot as plt
import torch

from sklearn.metrics import mean_squared_error

from model import VAE
from dataset import DataModule
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from logger import CustomLogger

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    seed_everything(cfg.seed, workers=True)



    # Set seeds for numpy, torch and python.random.
    model = VAE(input_shape=cfg.model.num_features)
    logger = TensorBoardLogger("./assets/tb_logs", name="VAE")
    actual_logger = CustomLogger()
    trainer = Trainer(deterministic=True, max_epochs=epochs, log_every_n_steps=10, logger=logger, callbacks=actual_logger)

    dataset = DataModule(data_path="assets/data/TQ8_201023SJ01_0103.mzML",
                        batch_size=cfg.dataset.batch_size,
                        num_features=cfg.model.num_features)

    trainer.fit(model, dataset)
    actual_logger.plot_logs()

    x = dataset.x.numpy()
    x_hat = torch.cat(trainer.predict(model, dataset)).numpy()

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
