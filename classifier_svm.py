from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch

from scripts.model import LitEncoderOnly
from scripts.dataloader import KFoldIBDDataModule, IBDDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from sklearn import svm

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from clearml import InputModel, OutputModel, Task
from scripts.utils import adjust_paths, connect_confiuration, calculate_layers_dims, dev_test_param_overwrite


@hydra.main(version_base=None, config_path="conf", config_name="clf_svm_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/PartialTraining', task_name=cfg.task_name,
                     auto_connect_frameworks={"pytorch": False})

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

    # calculate_layers_dims(cfg=cfg)
    print(OmegaConf.to_yaml(cfg))
    adjust_paths(cfg=cfg)

    seed_everything(cfg.seed, workers=True)

    cv_df = pd.DataFrame()
    for k in range(cfg.cross_validation.folds) if cfg.cross_validation.flag else [0]:
        
        # preparing dataset:
        dataset = KFoldIBDDataModule(cfg, k) if cfg.cross_validation.flag else IBDDataModule(cfg)

        # getting the AE model from the previous task 
        assert len(Task.get_task(task_id=cfg.ae_task_id).get_models()["output"]) == cfg.cross_validation.folds if cfg.cross_validation.flag else 1, \
            "either there are multiple models when cross validation is false or there is only one model when cross validation is true"
        clearml_ae_model_instance = InputModel(Task.get_task(task_id=cfg.ae_task_id).get_models()["output"][k if cfg.cross_validation.flag else 0].id, 
                                               name="pretrained auto-encoder - fold {k}" if cfg.cross_validation.flag else "pretrained auto-encoder")
        task.connect(clearml_ae_model_instance, name="pretrained auto-encoder - fold {k}" if cfg.cross_validation.flag else "pretrained auto-encoder")
        cfg = OmegaConf.merge(cfg, {"ae_model": OmegaConf.create(clearml_ae_model_instance.config_text).ae_model})

        # using the artifact to load the weights since there is the ClearML bug when downloading the model
        lightning_ae = LitEncoderOnly(cfg.ae_model, input_shape=dataset.get_num_features(), distortion=False).load_from_checkpoint(Task.get_task(task_id=cfg.ae_task_id).artifacts["auto-encoder model weights"].get_local_copy())
        
        lightning_ae.freeze()

        trainer = Trainer(max_epochs=cfg.model.epochs, fast_dev_run=cfg.fast_dev_run, 
                              accelerator=cfg.machine.accelerator, gpus=[cfg.machine.gpu_index] if cfg.machine.execution_type == "local" and cfg.machine.accelerator == "gpu" else None)
        
        # reporting classifier results:
        encoded_train, y_train = torch.cat(trainer.predict(lightning_ae, dataloaders=dataset.train_dataloader()), dim=0).numpy(), dataset.train[:][1].numpy()
        encoded_val, y_val = torch.cat(trainer.predict(lightning_ae, dataloaders=dataset.val_dataloader()), dim=0).numpy(), dataset.val[:][1].numpy()
        encoded_test, y_test = torch.cat(trainer.predict(lightning_ae, dataloaders=dataset.test_dataloader()), dim=0).numpy(), dataset.test[:][1].numpy()

        clf = svm.SVC()
        clf.fit(encoded_train, y_train)

        yhat_val = clf.predict(encoded_val)
        yhat_test = clf.predict(encoded_test)
        
        proba_val = clf.predict_proba(encoded_val)
        proba_test = clf.predict_proba(encoded_test)

        if k==0:
            pred_df = pd.DataFrame.from_dict({"label": y_test, "predictions": yhat_test, "probabilities": proba_test})
            task.upload_artifact(name="predictions dataframe", artifact_object=pred_df)
        
        cv_df = pd.concat([cv_df, pd.DataFrame.from_dict({"fold": [k], "val_accuracy": accuracy_score(y_val, yhat_val), "test_accuracy": accuracy_score(y_test, yhat_test),
                                                                       "val_auc": roc_auc_score(y_val, proba_val), "test_auc": roc_auc_score(y_test, proba_test),
                                                                       "val_f1": f1_score(y_val, yhat_val), "test_f1": f1_score(y_test, yhat_test)})], ignore_index=True)


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
