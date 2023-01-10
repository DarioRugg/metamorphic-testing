from clearml import Task
from omegaconf import OmegaConf
from pathlib import Path

from sqlalchemy import false

def adjust_paths(cfg: OmegaConf):
    cfg.dataset.data_path = Path(cfg.dataset.data_path)

def connect_confiuration(clearml_task: Task, configuration: OmegaConf) -> OmegaConf:
    return OmegaConf.create(str(clearml_task.connect(OmegaConf.to_object(configuration), name="hydra_config")))

def dev_test_param_overwrite(cfg: OmegaConf):
    cfg.ae_model.num_layers=1
    cfg.ae_model.start_dim=32
    cfg.cross_validation.folds=2
    cfg.machine.execution_type="local"

def calculate_layers_dims(cfg: OmegaConf):
    if cfg.ae_model.num_layers == 1:
        cfg.ae_model.layers_dims = [cfg.ae_model.start_dim]
    else:
        cfg.ae_model.layers_dims = [cfg.ae_model.start_dim]*round(cfg.ae_model.num_layers/3) + \
                                [int(cfg.ae_model.start_dim/2) if cfg.ae_model.start_dim/2 >= 16 else 16]*round(cfg.ae_model.num_layers/3) + \
                                [int(cfg.ae_model.start_dim/4) if cfg.ae_model.start_dim/4 >= 8 else 8]*(cfg.ae_model.num_layers-round(cfg.ae_model.num_layers/3)*2)