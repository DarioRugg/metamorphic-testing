from clearml import Task
from clearml.automation import PipelineController

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="pipeline_config")
def main(cfg : DictConfig) -> None:
    
    task = Task.init(project_name='e-muse/PartialTraining',
                     task_name='twin pipeline',
                     task_type=Task.TaskTypes.controller)
    
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

    pipe = PipelineController(name="Twin pipeline",
        project="e-muse/PartialTraining",
        version="0.0.1"
    )

    task.execute_remotely(queue_name="services")


    pipe.add_step(name="standard training", 
                  base_task_project="e-muse/PartialTraining", 
                  base_task_name=cfg.base_task_name, 
                  parameter_override={"hydra_config/biased":False},
                  execution_queue=f'rgai-gpu-01-2080ti:{cfg.gpus_index[0]}')

    pipe.add_step(name="biased training", 
                  base_task_project="e-muse/PartialTraining", 
                  base_task_name=cfg.base_task_name, 
                  parameter_override={"hydra_config/biased":True},
                  execution_queue=f'rgai-gpu-01-2080ti:{cfg.gpus_index[0]}')

    pipe.add_step(name="experiments aggregator", 
                  base_task_project="e-muse/PartialTraining", 
                  base_task_name=cfg.aggregator_task_name,
                  parameter_override={"hydra_config/standard_task_id": "${standard training.id}", 
                                      "hydra_config/biased_task_id": "${biased training.id}"},
                  parents=["${standard training}", "${biased training}"],
                  execution_queue="services")

    pipe.start()
    pipe.wait()
    pipe.stop()
    

if __name__ == "__main__":
    main()
