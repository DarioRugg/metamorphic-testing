from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from scripts.model import LitAutoEncoder
from scripts.dataloader import KFoldIBDDataModule, IBDDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

import hydra
from omegaconf import DictConfig, OmegaConf

from clearml import Task
from scripts.utils import adjust_paths, connect_confiuration, calculate_layers_dims, dev_test_param_overwrite


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    task = Task.init(project_name='e-muse/PartialTraining', task_name=cfg.task_name)

    task.set_base_docker(
        docker_image='rugg/aebias:latest',
        docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --env CLEARML_AGENT_GIT_USER=ruggeri\
                          --env CLEARML_AGENT_GIT_PASS={access_token}\
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/IBD/,target=/data/ \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/access_tokens/,target=/tokens/ \
                          --rm --ipc=host'.format(access_token=open("/tokens/gitlab_access_token.txt", "r").read())
    )

    cfg = connect_confiuration(clearml_task=task, configuration=cfg)

    # task.execute_remotely()
    # task.execute_remotely(queue_name=f"rgai-gpu-01-2080ti:{cfg.machine.gpu_index}")

    if cfg.fast_dev_run: dev_test_param_overwrite(cfg=cfg)

    calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)

    seed_everything(cfg.seed, workers=True)

    cv_df = pd.DataFrame()
    for k in range(cfg.cross_validation.folds) if cfg.cross_validation.flag else [0]:
        
        dataset = KFoldIBDDataModule(cfg, k) if cfg.cross_validation.flag else IBDDataModule(cfg)

        checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
        callbacks = [EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10),
                        checkpoint_callback]
        
        if cfg.model.name == "ae":
            lightning_model = LitAutoEncoder(cfg, dataset.get_num_features())

        trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks, accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index],
                          log_every_n_steps=10, fast_dev_run=cfg.fast_dev_run)

        trainer.fit(lightning_model, datamodule=dataset)

        lightning_model.load_from_checkpoint(checkpoint_callback.best_model_path)

        val_metrics = trainer.validate(lightning_model, datamodule=dataset)
        val_loss = val_metrics[0]["val_loss"]
        test_metrics = trainer.test(lightning_model, datamodule=dataset)
        test_loss = test_metrics[0]["test_loss"]

        cv_df = pd.concat([cv_df, pd.DataFrame.from_dict({"fold": [k], "val_loss": [val_loss], "test_loss": [test_loss]})])

        test_x_hat = torch.cat(trainer.predict(lightning_model, datamodule=dataset), dim=0)


        test_x, test_y = dataset.test[:]
        loss = torch.mean(nn.MSELoss(reduction='none')(test_x, test_x_hat), dim=1)

        dist_df = pd.DataFrame.from_dict({"label": test_y.numpy(), "loss": loss.numpy()})

        sns.kdeplot(dist_df[dist_df["label"]==0]["loss"], label="control", fill=True, color="green")
        sns.kdeplot(dist_df[dist_df["label"]==1]["loss"], label="test", fill=True, color="red")
        plt.legend()
        plt.title("Reconstruction loss distribution per class", fontdict={"size":15})
        plt.xlabel("Reconstruction loss")
        task.get_logger().report_matplotlib_figure(title="Losses distribution test set", series="Losses distribution test set", figure=plt.gcf())
        plt.close()


    if cfg.cross_validation.flag:
        sns.kdeplot(cv_df["val_loss"], label="Validation", fill=True)
        sns.kdeplot(cv_df["test_loss"], labels="Test", fill=True)
        plt.legend()
        plt.title("Cross-Validation Losses distribution", fontdict={"size":15})
        plt.xlabel("Reconstruction loss")
        task.get_logger().report_matplotlib_figure(title="Cross-Validation Losses distribution", series="Cross-Validation Losses distribution", figure=plt.gcf())
        plt.close()

        task.get_logger().report_vector(title='Cross-Validation Losses', series='Cross-Validation Losses', values=cv_df[["val_loss", "test_loss"]].values.transpose(), labels=["Validation", "Test"], xlabels=cv_df["fold"], xaxis='Cross-Validation Folds', yaxis='Loss')
        task.get_logger().report_table(title="Cross-Validation metrics", series="Cross-Validation metrics", iteration=0, table_plot=cv_df.set_index("fold", drop=True))

        task.get_logger().report_scalar(title='cross-validation average loss', series='validation', value=cv_df["val_loss"].mean(), iteration=0)
        task.get_logger().report_scalar(title='cross-validation average loss', series='test', value=cv_df["test_loss"].mean(), iteration=0)
    
    else:
        task.get_logger().report_vector(title='Losses', series='Losses', values=cv_df[["val_loss", "test_loss"]].values, xlabels=["Validation", "Test"], yaxis='Loss')
        task.get_logger().report_table(title="Metrics", series="Metrics", iteration=0, table_plot=cv_df.drop(columns="fold"))
        
        task.get_logger().report_scalar(title='loss', series='validation', value=cv_df["val_loss"], iteration=0)
        task.get_logger().report_scalar(title='loss', series='test', value=cv_df["test_loss"], iteration=0)
    

if __name__ == "__main__":
    main()
