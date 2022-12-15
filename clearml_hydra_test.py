from clearml import Task
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg : DictConfig) -> None:
    task = Task.init(project_name='e-muse/DeepMS', task_name="Hydra logging Test", tags=["hydra test"])

    task.get_logger().report_text(OmegaConf.to_yaml(cfg))
    

    task.execute_remotely()

    task.get_logger().report_text(OmegaConf.to_yaml(cfg))