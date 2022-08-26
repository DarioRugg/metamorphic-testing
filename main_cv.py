import os
from subprocess import call
from scripts.logger import BestValLossLogger

from scripts.model import SimpleAutoEncoder, LitAutoEncoder, VAE
from scripts.dataset import KFoldProstateDataModule, ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
from omegaconf import DictConfig, OmegaConf
os.environ["HYDRA_FULL_ERROR"] = "1"

from clearml import Task
from scripts.utils import adjust_paths, connect_hyperparameters, calculate_layers_dims

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.machine.gpu_index
    adjust_paths(cfg=cfg)

    task = Task.init(project_name='e-muse/DeepMS', task_name='test')
    connect_hyperparameters(clearml_task=task, cfg=cfg)
    calculate_layers_dims(cfg=cfg)

    print(OmegaConf.to_yaml(cfg))
    assert "cross_validation" in cfg.keys(), "this main_cv.py is for cross-validation training"

    seed_everything(cfg.seed, workers=True)


    if cfg.dataset.name == "prostate":
        dataset = KFoldProstateDataModule(cfg)
    elif cfg.dataset.name == "nizo":
        dataset = NIZODataModule(cfg)
    
    if cfg.model.name == "ae":
        model = SimpleAutoEncoder(cfg, dataset.get_num_features())
        print(model)
        lightning_model = LitAutoEncoder(cfg, model, dataset)
    if cfg.model.name == "vae":
        model = VAE(input_shape=cfg.dataset.num_features)
        lightning_model = model
    
    callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10)]

    best_loss_logger=BestValLossLogger()
    callbacks.append(best_loss_logger)

    trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, callbacks=callbacks, 
                        log_every_n_steps=10, accelerator=cfg.machine.accelerator)
    
    if cfg.train:
        cross_validation_loss = 0
        for k in range(cfg.cross_validation.folds):
            if cfg.dataset.name == "prostate":
                dataset = KFoldProstateDataModule(cfg)
            elif cfg.dataset.name == "nizo":
                dataset = NIZODataModule(cfg)
            
            if cfg.model.name == "ae":
                model = SimpleAutoEncoder(cfg, dataset.get_num_features())
                print(model)
                lightning_model = LitAutoEncoder(cfg, model, dataset)
            if cfg.model.name == "vae":
                model = VAE(input_shape=cfg.dataset.num_features)
                lightning_model = model
            
            callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10)]

            best_loss_logger=BestValLossLogger()
            callbacks.append(best_loss_logger)

            trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, callbacks=callbacks, 
                                log_every_n_steps=10, accelerator=cfg.machine.accelerator)
                        
            dataset.set_current_fold(k)

            trainer.fit(lightning_model, datamodule=dataset)

            cross_validation_loss += best_loss_logger.get_best_val_loss()/cfg.cross_validation.folds
        
        print(f"\n\n ---> KFold Cross-Validation Loss: {cross_validation_loss}\n\n")

    if cfg.test:
        trainer.test(lightning_model, datamodule=dataset)


if __name__ == "__main__":
    main()
