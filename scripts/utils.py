from clearml import Task
from omegaconf import OmegaConf
from pathlib import Path

from sqlalchemy import false

def adjust_paths(cfg: OmegaConf):
    cfg.dataset.data_path = Path(cfg.dataset.data_path)

def connect_hyperparameters(clearml_task: Task, cfg: OmegaConf):
    hyperparams = {
            "num_layers": cfg.model.num_layers,
            "start_dim": cfg.model.start_dim,
            "dropout_prob": cfg.model.dropout_prob,
            "learning_rate": cfg.model.learning_rate,
            "loss": cfg.model.loss,
            "batch_size": cfg.dataset.batch_size
        }

    hyperparams = clearml_task.connect(hyperparams)

    cfg.model.num_layers = hyperparams["num_layers"]
    cfg.model.start_dim = hyperparams["start_dim"]
    cfg.model.dropout_prob = hyperparams["dropout_prob"]
    cfg.model.learning_rate = hyperparams["learning_rate"]
    cfg.model.loss = hyperparams["loss"]
    cfg.dataset.batch_size = hyperparams["batch_size"]

    # parameters used to speedup the first execution, then the agent will set the actual value.
    dev_params = {
            "fast_dev_run": cfg.fast_dev_run,
            "gpu_index": cfg.machine.gpu_index
        }

    dev_params = clearml_task.connect(dev_params, name="dev_config")

    cfg.fast_dev_run = dev_params["fast_dev_run"]
    cfg.machine.gpu_index = dev_params["gpu_index"]

def dev_test_param_overwrite(cfg: OmegaConf):
    cfg.test=False
    cfg.model.num_layers=1
    cfg.model.start_dim=32
    cfg.cross_validation.folds=2

def calculate_layers_dims(cfg: OmegaConf):
    if cfg.model.num_layers == 1:
        cfg.model.layers_dims = [cfg.model.start_dim]
    else:
        cfg.model.layers_dims = [cfg.model.start_dim]*round(cfg.model.num_layers/3) + \
                                [int(cfg.model.start_dim/2) if cfg.model.start_dim/2 >= 16 else 16]*round(cfg.model.num_layers/3) + \
                                [int(cfg.model.start_dim/4) if cfg.model.start_dim/4 >= 8 else 8]*(cfg.model.num_layers-round(cfg.model.num_layers/3)*2)