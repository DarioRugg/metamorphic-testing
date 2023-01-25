from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from scripts.model import LitAutoEncoder
from scripts.dataloader import IBDDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

import hydra
from omegaconf import DictConfig, OmegaConf

from clearml import OutputModel, Task
from scripts.utils import adjust_paths, connect_confiuration, calculate_layers_dims, dev_test_param_overwrite


@hydra.main(version_base=None, config_path="conf", config_name="ae_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/metamorphic-testing', task_name=cfg.task_name,
                     auto_connect_frameworks={"pytorch": False})

    task.set_base_docker(
        docker_image='rugg/metamorphic:latest',
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

    results_df = pd.DataFrame()
        
    # preparing dataset:
    dataset = IBDDataModule(cfg)


    # training aeuto-encoder:
    checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="best weights", save_weights_only=True)
    callbacks = [EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10),
                    checkpoint_callback]
    
    lightning_model = LitAutoEncoder(cfg.model, dataset.get_num_features())
    clearml_model_instance = OutputModel(task=task, name=f"{cfg.task_name} - best weights", 
                                            config_dict=OmegaConf.to_yaml(OmegaConf.create({"ae_model": cfg.model, "test": cfg.test})),
                                            comment="Standard training" if not cfg.test.flag else f"Test training\nWith {cfg.test.name} test type")
    task.connect(clearml_model_instance, name="best weights")

    trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks,
                        log_every_n_steps=10, fast_dev_run=cfg.fast_dev_run, 
                        accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index] if cfg.machine.execution_type == "local" and cfg.machine.accelerator == "gpu" else None)

    trainer.fit(lightning_model, datamodule=dataset)

    clearml_model_instance.update_weights(weights_filename=checkpoint_callback.best_model_path)
    # using the artifact to store the weights since there is the ClearML bug when downloading the model
    task.upload_artifact(name="auto-encoder model weights", artifact_object=clearml_model_instance.get_weights())

    # loading best models:
    lightning_model.load_from_checkpoint(clearml_model_instance.get_weights())

    # activate test tests:
    dataset.change_test_stage("test")

    # reporting auto-encoder results:
    test_x_hat = torch.cat(trainer.predict(lightning_model, datamodule=dataset), dim=0)

    test_x, test_y = dataset.test[:]

    mse_loss = torch.mean(nn.MSELoss(reduction='none')(test_x, test_x_hat), dim=1)
    ce_loss = torch.mean(nn.BCEWithLogitsLoss(reduction='none')(test_x, test_x_hat), dim=1)

    dist_df = pd.DataFrame.from_dict({"label": test_y.numpy(), "mse": mse_loss.numpy(), "ce": ce_loss.numpy()})

    task.upload_artifact(name="losses dataframe", artifact_object=dist_df)


    # aggregate cross-validation results:
    val_metrics = trainer.validate(lightning_model, datamodule=dataset)
    val_loss = val_metrics[0]["val_loss"]
    test_metrics = trainer.test(lightning_model, datamodule=dataset)
    test_loss = test_metrics[0]["test_loss"]
    
    results_df = pd.concat([results_df, pd.DataFrame.from_dict({"val_loss": [val_loss], "test_loss": [test_loss]})], ignore_index=True)


    task.get_logger().report_vector(title='Losses', series='Losses', values=results_df[["val_loss", "test_loss"]].values, xlabels=["Validation", "Test"], yaxis='Loss')
    task.get_logger().report_table(title="Metrics", series="Metrics", iteration=0, table_plot=results_df.drop(columns="fold"))
    
    task.get_logger().report_scalar(title='loss', series='validation', value=results_df["val_loss"], iteration=0)
    task.get_logger().report_scalar(title='loss', series='test', value=results_df["test_loss"], iteration=0)
    

if __name__ == "__main__":
    main()
