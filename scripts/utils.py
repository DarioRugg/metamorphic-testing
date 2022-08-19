from clearml import Task
from omegaconf import OmegaConf
from pathlib import Path

def adjust_paths(cfg: OmegaConf):
    cfg.dataset.data_path = Path(cfg.dataset.data_path)

def connect_hyperparameters(clearml_task: Task, cfg: OmegaConf):
    hyperparams = {
            "layers_dims": cfg.model.layers_dims,
            "dropout_prob": cfg.model.dropout_prob,
            "learning_rate": cfg.model.learning_rate,
            "loss": cfg.model.loss,
            "batch_size": cfg.dataset.batch_size
        }

    clearml_task.connect(hyperparams)

    cfg.model.layers_dims = hyperparams["layers_dims"]
    cfg.model.dropout_prob = hyperparams["dropout_prob"]
    cfg.model.learning_rate = hyperparams["learning_rate"]
    cfg.model.loss = hyperparams["loss"]
    cfg.dataset.batch_size = hyperparams["batch_size"]
