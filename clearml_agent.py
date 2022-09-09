import os
import sys
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformParameterRange, LogUniformParameterRange,
    UniformIntegerParameterRange)
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="hyperoptim_config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    execution_queue = f'rgai-gpu-01-2080ti:{cfg.machine.gpu_index}'

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(
        project_name='e-muse/DeepMS',
        task_name='Hyper-Parameter Optimization',
        task_type=Task.TaskTypes.optimizer
    )

    hyper_parameters=[
            UniformIntegerParameterRange('General/num_layers', min_value=2, max_value=9),
            UniformIntegerParameterRange('General/start_dim', min_value=16, max_value=96, step_size=4),
            UniformParameterRange('General/dropout_prob', min_value=0, max_value=0.08),
            LogUniformParameterRange('General/learning_rate', min_value=1e-6, max_value=1e-3),
            DiscreteParameterRange('General/loss', values=["mse"]),
            DiscreteParameterRange('General/batch_size', values=[32, 64, 128, 256])
        ]
    
    # ensuring "gpu_index" is False
    hyper_parameters.append(DiscreteParameterRange('dev_config/fast_dev_run', values=[False]))
    hyper_parameters.append(DiscreteParameterRange('dev_config/gpu_index', values=cfg.machine.gpu_index))


    # Example use case:
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=cfg.task_id,
        hyper_parameters=hyper_parameters,
        
        objective_metric_title='summary',
        objective_metric_series='cross-validation loss',
        objective_metric_sign='min',

        execution_queue=execution_queue,

        # setting optimizer 
        optimizer_class=OptimizerOptuna,

        max_number_of_concurrent_tasks=cfg.machine.workers,  
        optimization_time_limit=cfg.optimization.time_limit, 
        total_max_jobs=cfg.optimization.trials,  

        # iterations for early stopping to act:
        min_iteration_per_job=300,  
        max_iteration_per_job=3000, 

        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5,
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
