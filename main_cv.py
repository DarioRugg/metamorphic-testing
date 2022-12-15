import random
from subprocess import call

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from scripts.model import SimpleAutoEncoder, LitAutoEncoder, VAE
from scripts.dataset import KFoldProstateDataModule, ProstateDataModule, NIZODataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig, OmegaConf

from clearml import Task
from scripts.logger import PixelsPlotter
from scripts.utils import adjust_paths, connect_hyperparameters, calculate_layers_dims, dev_test_param_overwrite

import torch

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    task = Task.init(project_name='e-muse/DeepMS', task_name=cfg.task_name)

    task.set_base_docker(
        docker_image='rugg/deepms:latest',
        docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --env CLEARML_AGENT_GIT_USER=ruggeri\
                          --env CLEARML_AGENT_GIT_PASS={access_token}\
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/DeepMS/,target=/data/ \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/access_tokens/,target=/tokens/ \
                          --rm --ipc=host'.format(access_token=open("/tokens/gitlab_access_token.txt", "r").read())
    )

    task.execute_remotely(queue_name=f"rgai-gpu-01-2080ti:{cfg.machine.gpu_index}")

    connect_hyperparameters(clearml_task=task, cfg=cfg)
    if cfg.fast_dev_run: dev_test_param_overwrite(cfg=cfg)

    calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)
    
    assert "cross_validation" in cfg.keys(), "this main_cv.py is for cross-validation training"

    seed_everything(cfg.seed, workers=True)

    if cfg.train:
        cross_validation_losses = []
        for k in range(cfg.cross_validation.folds):
            
            dataset = KFoldProstateDataModule(cfg, k)

            checkpoint_callback = ModelCheckpoint(dirpath="assets/weights/cross_validation", filename=f"best_model_fold_{k}", monitor="val_loss", save_weights_only=True)
            callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10),
                         checkpoint_callback, PixelsPlotter(cfg=cfg, dataset=dataset, task=task)]
            
            if cfg.model.name == "ae":
                lightning_model = LitAutoEncoder(cfg, dataset.get_num_features())
            if cfg.model.name == "vae":
                lightning_model = VAE(input_shape=cfg.dataset.num_features)

            trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks, #  devices=[cfg.machine.gpu_index],
                                log_every_n_steps=10, accelerator=cfg.machine.accelerator, fast_dev_run=cfg.fast_dev_run)

            trainer.fit(lightning_model, datamodule=dataset)

            cross_validation_losses.append(checkpoint_callback.best_model_score.cpu().item())


        fig = plt.figure()
        sns.kdeplot(np.array(cross_validation_losses), bw=0.5)
        task.get_logger().report_matplotlib_figure(title="Cross-validation losses distribution", series="losses distribution", figure=fig)
        fig.clear()

        
        task.get_logger().report_single_value("cross-validation loss", np.mean(cross_validation_losses))
        
        print(f"\n\n ---> KFold Cross-Validation Loss: {np.mean(cross_validation_losses)}\n\n")

    # no point in testing the last model, we should test one picked at random
    if cfg.test:

        random_fold = random.choice(range(cfg.cross_validation.folds))
        
        dataset = KFoldProstateDataModule(cfg, k=random_fold)

        checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
        callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10),
                    checkpoint_callback, PixelsPlotter(cfg=cfg, dataset=dataset, task=task)]

        if cfg.model.name == "ae":
            lightning_model = LitAutoEncoder(cfg, dataset.get_num_features())
        if cfg.model.name == "vae":
            lightning_model = VAE(input_shape=cfg.dataset.num_features)
        

        trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks, #  devices=[cfg.machine.gpu_index],
                            log_every_n_steps=10, accelerator=cfg.machine.accelerator, fast_dev_run=cfg.fast_dev_run)

        trainer.fit(lightning_model, datamodule=dataset)

        lightning_model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(lightning_model, datamodule=dataset)


if __name__ == "__main__":
    main()
