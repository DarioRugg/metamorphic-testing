from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from scripts.model import LitClassifier
from scripts.dataloader import IBDDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from clearml import InputModel, OutputModel, Task
from scripts.utils import adjust_paths, connect_confiuration, dev_test_param_overwrite


@hydra.main(version_base=None, config_path="conf", config_name="clf_config")
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

    # calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)

    seed_everything(cfg.seed, workers=True)

    reslts_df = pd.DataFrame()
    
    # preparing dataset:
    dataset = IBDDataModule(cfg)

    # training classifier:
    checkpoint_callback = ModelCheckpoint(dirpath="assets/weights", filename="model_to_test", save_weights_only=True)
    callbacks = [EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10),
                        checkpoint_callback]

    clearml_ae_model_instance = InputModel(Task.get_task(task_id=cfg.ae_task_id).get_models()["output"][0].id, 
                                            name="pretrained auto-encoder")
    task.connect(clearml_ae_model_instance, name="pretrained auto-encoder")
    cfg = OmegaConf.merge(cfg, {"ae_model": OmegaConf.create(clearml_ae_model_instance.config_text).ae_model})
    
    # checking that AE and classifier have the same test type
    assert cfg.test == OmegaConf.create(clearml_ae_model_instance.config_text).test, "AE and classifier must have the same test types"

    # using the artifact to load the weights since there is the ClearML bug when downloading the model
    lightning_model = LitClassifier(model_cfg=cfg.model, ae_cfg=cfg.ae_model, input_shape=dataset.get_num_features(), ae_checkpoint_path=Task.get_task(task_id=cfg.ae_task_id).artifacts["auto-encoder model weights"].get_local_copy())
    clearml_model_instance = OutputModel(task=task, name=f"{cfg.task_name} - best weights", 
                                            config_dict=OmegaConf.to_yaml(OmegaConf.merge(OmegaConf.create({"clf_model": cfg.model}), OmegaConf.create(clearml_ae_model_instance.config_text))),
                                            comment=clearml_ae_model_instance.comment)
    task.connect(clearml_model_instance, name="best weights")

    trainer = Trainer(max_epochs=cfg.model.epochs, callbacks=callbacks,
                            log_every_n_steps=10, fast_dev_run=cfg.fast_dev_run, 
                            accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index] if cfg.machine.execution_type == "local" and cfg.machine.accelerator == "gpu" else None)

    trainer.fit(lightning_model, datamodule=dataset)


    # loading best models:
    clearml_model_instance.update_weights(weights_filename=checkpoint_callback.best_model_path)
    lightning_model.load_from_checkpoint(clearml_model_instance.get_weights())

    # activate test tests:
    dataset.change_test_stage("test")


    # reporting classifier results:
    test_x, test_y = dataset.test[:]

    test_y_hat = torch.cat(trainer.predict(lightning_model, datamodule=dataset), dim=0)

    pred_df = pd.DataFrame.from_dict({"label": test_y.numpy(), "logits": test_y_hat.numpy(), "probabilities": nn.Sigmoid()(test_y_hat).numpy(), "predictions": torch.round(nn.Sigmoid()(test_y_hat)).numpy()})

    task.upload_artifact(name="predictions dataframe", artifact_object=pred_df)


    # aggregate cross-validation results:
    val_metrics = dict(map(lambda item: (item[0], [item[1]]), trainer.validate(lightning_model, datamodule=dataset)[0].items()))
    test_metrics = dict(map(lambda item: (item[0], [item[1]]), trainer.test(lightning_model, datamodule=dataset)[0].items()))

    reslts_df = pd.concat([reslts_df, pd.DataFrame.from_dict(val_metrics | test_metrics)], ignore_index=True)


    task.get_logger().report_vector(title='Losses', series='Losses', values=reslts_df[["val_loss", "test_loss"]].values, xlabels=["Validation", "Test"], yaxis='Loss')
    task.get_logger().report_table(title="Metrics", series="Metrics", iteration=0, table_plot=reslts_df)
    
    task.get_logger().report_scalar(title='loss', series='validation', value=reslts_df["val_loss"], iteration=0)
    task.get_logger().report_scalar(title='loss', series='test', value=reslts_df["test_loss"], iteration=0)
    

if __name__ == "__main__":
    main()
