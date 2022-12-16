from clearml import Task
import clearml
import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.utils import connect_confiuration


task = Task.init(project_name='e-muse/DeepMS', task_name="Hydra logging Test", tags=["hydra test"])

task.set_base_docker(
    docker_image='rugg/deepms:latest',
    docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                        --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                        --rm'
)

@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg : DictConfig) -> None:

    task.get_logger().report_text(OmegaConf.to_yaml(cfg))

    cfg = connect_confiuration(task, cfg)
    
    task.get_logger().report_text(OmegaConf.to_yaml(cfg))
    
    task.execute_remotely()

    task.get_logger().report_text(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
