import os
import numpy as np
from matplotlib import pyplot as plt

from model import SimpleAutoEncoder, LitAutoEncoder, VAE
from dataset import ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from logger import CustomLogger

import hydra
from omegaconf import DictConfig, OmegaConf
os.environ["HYDRA_FULL_ERROR"] = "1"

from clearml import Task

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.machine.gpu_index

    task = Task.init(project_name='e-muse/DeepMS', task_name='test')

    seed_everything(cfg.seed, workers=True)


    if cfg.dataset.name == "prostate":
        dataset = ProstateDataModule(cfg)
    elif cfg.dataset.name == "nizo":
        dataset = NIZODataModule(cfg)
    
    if cfg.model.name == "ae":
        model = SimpleAutoEncoder(cfg, dataset.get_num_features())
        print(model)
        lightning_model = LitAutoEncoder(cfg, model, dataset)
    if cfg.model.name == "vae":
        model = VAE(input_shape=cfg.dataset.num_features)
        lightning_model = model
    # logger = TensorBoardLogger("./assets/prostate/tb_logs", name="AE")
    # actual_logger = CustomLogger()
    trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, log_every_n_steps=10,
    # accelerator=cfg.machine.accelerator, devices= cfg.machine.gpu_index if cfg.machine.accelerator == "gpu" else None)
    accelerator=cfg.machine.accelerator)

    trainer.fit(lightning_model, datamodule=dataset)
    # actual_logger.plot_logs()

    trainer.test(lightning_model, datamodule=dataset)


if __name__ == "__main__":
    main()
