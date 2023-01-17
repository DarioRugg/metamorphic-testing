from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from scripts.model import LitAutoEncoder, LitClassifier
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

    if cfg.machine.execution_type == "draft":
        task.execute_remotely()
    elif cfg.machine.execution_type == "remote" and cfg.machine.accelerator == "gpu":
        task.execute_remotely(queue_name=f"rgai-gpu-01-2080ti:{cfg.machine.gpu_index}")

    if cfg.fast_dev_run: dev_test_param_overwrite(cfg=cfg)

    calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)

    seed_everything(cfg.seed, workers=True)

    cv_df = pd.DataFrame()
    for k in range(cfg.cross_validation.folds) if cfg.cross_validation.flag else [0]:
        
        # preparing dataset:
        dataset = KFoldIBDDataModule(cfg, k) if cfg.cross_validation.flag else IBDDataModule(cfg)


        # training aeuto-encoder:
        checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
        callbacks = [EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10),
                        checkpoint_callback]
        
        if cfg.model.name == "ae":
            lightning_model = LitAutoEncoder(cfg, dataset.get_num_features())

        trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks,
                          log_every_n_steps=10, fast_dev_run=cfg.fast_dev_run, 
                          accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index] if cfg.machine.execution_type == "local" and cfg.machine.accelerator == "gpu" else None)

        trainer.fit(lightning_model, datamodule=dataset)
        
        
        # we don't want the bias during classification
        if cfg.bias:
            dataset.suppress_training_bias()

        # training classifier:
        clf_checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
        clf_callbacks = [EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10),
                         clf_checkpoint_callback]

        lightning_classifier = LitClassifier(cfg, dataset.get_num_features(), ae_checkpoint_path=checkpoint_callback.best_model_path)

        clf_trainer = Trainer(max_epochs=cfg.clf_model.epochs, callbacks=clf_callbacks,
                              log_every_n_steps=10, fast_dev_run=cfg.fast_dev_run, 
                              accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index] if cfg.machine.execution_type == "local" and cfg.machine.accelerator == "gpu" else None)

        clf_trainer.fit(lightning_classifier, datamodule=dataset)


        # loading best models:
        lightning_model.load_from_checkpoint(checkpoint_callback.best_model_path)
        lightning_classifier.load_from_checkpoint(clf_checkpoint_callback.best_model_path)


        # reporting auto-encoder results:
        test_x_hat = torch.cat(trainer.predict(lightning_model, datamodule=dataset), dim=0)

        test_x, test_y = dataset.test[:]

        mse_loss = torch.mean(nn.MSELoss(reduction='none')(test_x, test_x_hat), dim=1)
        ce_loss = torch.mean(nn.BCEWithLogitsLoss(reduction='none')(test_x, test_x_hat), dim=1)

        dist_df = pd.DataFrame.from_dict({"label": test_y.numpy(), "mse": mse_loss.numpy(), "ce": ce_loss.numpy()})

        task.upload_artifact(name="losses dataframe", artifact_object=dist_df)

        # reporting classifier results:
        # test_y_hat = torch.squeeze(torch.cat(clf_trainer.predict(lightning_classifier, datamodule=dataset), dim=0), dim=1)
        test_y_hat = torch.cat(clf_trainer.predict(lightning_classifier, datamodule=dataset), dim=0)

        pred_df = pd.DataFrame.from_dict({"label": test_y.numpy(), "logits": test_y_hat.numpy(), "predictions": torch.round(nn.Sigmoid()(test_y_hat)).numpy()})

        task.upload_artifact(name="predictions dataframe", artifact_object=pred_df)


        # aggregate cross-validation results:
        val_metrics = trainer.validate(lightning_model, datamodule=dataset)
        val_loss = val_metrics[0]["val_loss"]
        test_metrics = trainer.test(lightning_model, datamodule=dataset)
        test_loss = test_metrics[0]["test_loss"]
        
        # val_metrics = clf_trainer.validate(lightning_classifier, datamodule=dataset)
        # val_loss = val_metrics[0]["val_loss"]
        # test_metrics = clf_trainer.test(lightning_classifier, datamodule=dataset)
        # test_loss = test_metrics[0]["test_loss"]
        
        cv_df = pd.concat([cv_df, pd.DataFrame.from_dict({"fold": [k], "val_loss": [val_loss], "test_loss": [test_loss]})])


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
