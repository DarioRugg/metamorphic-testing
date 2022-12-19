import random
from subprocess import call

from matplotlib import pyplot as plt
import pandas as pd
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
from scripts.utils import adjust_paths, connect_confiuration, calculate_layers_dims, dev_test_param_overwrite

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

    cfg = connect_confiuration(clearml_task=task, configuration=cfg)
    if cfg.fast_dev_run: dev_test_param_overwrite(cfg=cfg)

    calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)

    seed_everything(cfg.seed, workers=True)

    cv_df = pd.DataFrame()
    for k in range(cfg.cross_validation.folds):
        
        dataset = KFoldProstateDataModule(cfg, k)

        checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
        callbacks = [EarlyStopping(monitor="val_loss", min_delta=3e-7, patience=10),
                        checkpoint_callback] #, PixelsPlotter(cfg=cfg, dataset=dataset, task=task)]
        
        if cfg.model.name == "ae":
            lightning_model = LitAutoEncoder(cfg, dataset.get_num_features())
        if cfg.model.name == "vae":
            lightning_model = VAE(input_shape=cfg.dataset.num_features)

        trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks, #  devices=[cfg.machine.gpu_index],
                            log_every_n_steps=10, accelerator=cfg.machine.accelerator, fast_dev_run=cfg.fast_dev_run, gpus=[0])

        trainer.fit(lightning_model, datamodule=dataset)

        lightning_model.load_from_checkpoint(checkpoint_callback.best_model_path)

        val_metrics = trainer.validate(lightning_model, datamodule=dataset)
        val_loss = val_metrics[0]["val_loss"]
        test_metrics = trainer.test(lightning_model, datamodule=dataset)
        test_loss = test_metrics[0]["test_loss"]

        cv_df = pd.concat([cv_df, pd.DataFrame.from_dict({"fold": [k], "val_loss": [val_loss], "test_loss": [test_loss]})])

    print(cv_df)
    sns.kdeplot(cv_df["val_loss"], legend="Validation", fill=True)
    sns.kdeplot(cv_df["test_loss"], legend="Test", fill=True)
    plt.legend()
    plt.title("Cross-Validation Losses distribution", fontdict={"size":15})
    plt.xlabel("Reconstruction loss")
    task.get_logger().report_matplotlib_figure(title="Cross-Validation Losses distribution", series="Cross-Validation Losses distribution", figure=plt.gcf())
    plt.close()

    task.get_logger().report_vector(title='Cross-Validation Losses', series='Cross-Validation Losses', values=cv_df[["val_loss", "test_loss"]], labels=["Validation", "Test"], xlabels=cv_df["fold"], xaxis='Cross-Validation Folds', yaxis='Loss')
    task.get_logger().report_table(title="Cross-Validation metrics", series="Cross-Validation metrics", iteration=0, table_plot=cv_df.set_index("fold", drop=True))

    task.get_logger().report_single_value("test average loss", np.mean(cv_df["test_loss"]))


if __name__ == "__main__":
    main()
