import os
import sys
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformParameterRange, LogUniformParameterRange,
    UniformIntegerParameterRange, ParameterSet)
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf/hpo", config_name="clf_hyperoptim_config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    execution_queue = f'rgai-gpu-01-2080ti:{cfg.machine.gpu_index}'

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(
        project_name='e-muse/PartialTraining',
        task_name='Classifier Hyper-Parameter Optimization',
        task_type=Task.TaskTypes.optimizer, 
        reuse_last_task_id=False
    )
    
    task.set_base_docker(
        docker_image='rugg/aebias:latest',
        docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --env CLEARML_AGENT_GIT_USER=ruggeri\
                          --env CLEARML_AGENT_GIT_PASS={access_token}\
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/DeepMS/,target=/data/ \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/access_tokens/,target=/tokens/ \
                          --rm --ipc=host'.format(access_token=open("/tokens/gitlab_access_token.txt", "r").read())
    )

    task.execute_remotely(queue_name="services")

    hyper_parameters=[
            UniformIntegerParameterRange('hydra_config/model/num_layers', min_value=2, max_value=9),
            UniformIntegerParameterRange('hydra_config/model/start_dim', min_value=16, max_value=96, step_size=4),
            UniformParameterRange('hydra_config/model/dropout_prob', min_value=0, max_value=0.5),
            LogUniformParameterRange('hydra_config/model/learning_rate', min_value=1e-9, max_value=1e-3),
            # DiscreteParameterRange('hydra_config/model/loss', values=["mse"]),
            # DiscreteParameterRange('hydra_config/dataset/batch_size', values=[16, 32, 64]),
            DiscreteParameterRange("hydra_config/model/layer_dims", [(32), (64), (64, 32), (64, 64, 32, 16), (64, 64, 32, 32), (64, 64, 32, 32, 16), (64, 64, 32, 32, 32)])
        ]


    # Example use case:
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=Task.get_task(project_name='e-muse/PartialTraining', task_name=cfg.task_name).id,
        hyper_parameters=hyper_parameters,
        
        objective_metric_title='val_loss',
        objective_metric_series='val_loss',
        objective_metric_sign='min',

        execution_queue=execution_queue,

        # setting optimizer 
        optimizer_class=OptimizerBOHB,

        max_number_of_concurrent_tasks=2,  
        optimization_time_limit=cfg.optimization.time_limit, 
        total_max_jobs=cfg.optimization.trials,

        min_iteration_per_job=3000,
        max_iteration_per_job=3000,

        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5
    )
    

    # start the optimization process
    optimizer.start()
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()

    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()


if __name__ == "__main__":
    main()
