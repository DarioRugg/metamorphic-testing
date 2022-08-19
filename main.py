import os

from scripts.model import SimpleAutoEncoder, LitAutoEncoder, VAE
from scripts.dataset import ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
from omegaconf import DictConfig, OmegaConf
os.environ["HYDRA_FULL_ERROR"] = "1"

from clearml import Task
from scripts.utils import adjust_paths, connect_hyperparameters

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.machine.gpu_index
    adjust_paths(cfg=cfg)

    task = Task.init(project_name='e-muse/DeepMS', task_name='test')
    connect_hyperparameters(clearml_task=task, cfg=cfg)

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
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-8, patience=5)

    trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, callbacks=[early_stop_callback], 
                        log_every_n_steps=10, accelerator=cfg.machine.accelerator)
    
    if cfg.train:
        trainer.fit(lightning_model, datamodule=dataset)

    if cfg.test:
        trainer.test(lightning_model, datamodule=dataset)


if __name__ == "__main__":
    main()
