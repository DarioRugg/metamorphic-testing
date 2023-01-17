from clearml import Task
from omegaconf import OmegaConf
from pathlib import Path

from sqlalchemy import false

def adjust_paths(cfg: OmegaConf):
    cfg.dataset.data_path = Path(cfg.dataset.data_path)

def connect_confiuration(clearml_task: Task, configuration: OmegaConf) -> OmegaConf:
    return OmegaConf.create(str(clearml_task.connect(OmegaConf.to_object(configuration), name="hydra_config")))

def dev_test_param_overwrite(cfg: OmegaConf):
    cfg.model.num_layers=1
    cfg.model.start_dim=32
    cfg.cross_validation.folds=2
    cfg.machine.execution_type="local"

def calculate_layers_dims(cfg: OmegaConf):
    if cfg.model.num_layers == 1:
        cfg.model.layers_dims = [cfg.model.start_dim]
    else:
        cfg.model.layers_dims = [cfg.model.start_dim]*round(cfg.model.num_layers/3) + \
                                [int(cfg.model.start_dim/2) if cfg.model.start_dim/2 >= 16 else 16]*round(cfg.model.num_layers/3) + \
                                [int(cfg.model.start_dim/4) if cfg.model.start_dim/4 >= 8 else 8]*(cfg.model.num_layers-round(cfg.model.num_layers/3)*2)