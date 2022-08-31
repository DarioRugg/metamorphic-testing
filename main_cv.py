import os
import random
from subprocess import call
from scripts.logger import BestValLossLogger

from scripts.model import SimpleAutoEncoder, LitAutoEncoder, VAE
from scripts.dataset import KFoldProstateDataModule, ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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
    
    if cfg.train:
        cross_validation_loss = 0
        for k in range(cfg.cross_validation.folds):
            
            best_loss_logger=BestValLossLogger()
            callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10),
                         best_loss_logger]

            dataset = KFoldProstateDataModule(cfg, k)
            
            if cfg.model.name == "ae":
                model = SimpleAutoEncoder(cfg, dataset.get_num_features())
                print(model)
                lightning_model = LitAutoEncoder(cfg, model, dataset)
            if cfg.model.name == "vae":
                model = VAE(input_shape=cfg.dataset.num_features)
                lightning_model = model

            trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, callbacks=callbacks, 
                                log_every_n_steps=10, accelerator=cfg.machine.accelerator)

            trainer.fit(lightning_model, datamodule=dataset)

            cross_validation_loss += best_loss_logger.get_best_val_loss()/cfg.cross_validation.folds
        
        print(f"\n\n ---> KFold Cross-Validation Loss: {cross_validation_loss}\n\n")

    # no point in testing the last model, we should test one picked at random
    if cfg.test:

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_weights_only=True)
        callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10),
                    checkpoint_callback]

        random_fold = random.randint(0, cfg.cross_validation.folds)
        dataset = KFoldProstateDataModule(cfg, k=random_fold)
        
        if cfg.model.name == "ae":
            model = SimpleAutoEncoder(cfg, dataset.get_num_features())
            print(model)
            lightning_model = LitAutoEncoder(cfg, model, dataset)
        if cfg.model.name == "vae":
            model = VAE(input_shape=cfg.dataset.num_features)
            lightning_model = model
        

        trainer = Trainer(deterministic=True, max_epochs=cfg.model.epochs, callbacks=callbacks, 
                            log_every_n_steps=10, accelerator=cfg.machine.accelerator)

        trainer.fit(lightning_model, datamodule=dataset)

        lightning_model.load_from_checkpoint(checkpoint_callback.best_model_path)

        trainer.test(lightning_model, datamodule=dataset)


if __name__ == "__main__":
    main()
