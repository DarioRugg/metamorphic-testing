from clearml import Task
from clearml.automation import PipelineController

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="pipeline_config")
def main(cfg : DictConfig) -> None:
    
    task = Task.init(project_name='e-muse/metamorphic-testing',
                     task_name=f'pipeline {cfg.test.name} test',
                     task_type=Task.TaskTypes.controller)
    
    task.set_base_docker(
        docker_image='rugg/metamorphic:latest',
        docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/IBD/,target=/data/ \
                          --rm --ipc=host'
    )

    
    print(OmegaConf.to_yaml(cfg))

    pipe = PipelineController(name="Twin pipeline",
        project="e-muse/metamorphic-testing",
        version="0.0.3"
    )

    parents_to_wait = []

    if not cfg.past_tasks.use or cfg.past_tasks.standard.ae == "":
        # standard training:
        pipe.add_step(name="auto-encoder standard training", 
                    base_task_project="e-muse/metamorphic-testing", 
                    base_task_name=cfg.ae_base_task_name, 
                    parameter_override={"hydra_config/seed":cfg.seed,
                                        "hydra_config/test/flag":cfg.no_test.flag,
                                        "hydra_config/test/name":cfg.no_test.name,
                                        "hydra_config/test/stage":cfg.no_test.stage,
                                        "hydra_config/test/param/num_features":cfg.no_test.param.num_features,
                                        "hydra_config/test/param/mu":cfg.no_test.param.mu,
                                        "hydra_config/test/param/std":cfg.no_test.param.std,
                                        "hydra_config/test/threshold/ae/equal":cfg.no_test.threshold.ae.equal,
                                        "hydra_config/test/threshold/ae/different":cfg.no_test.threshold.ae.different,
                                        "hydra_config/test/threshold/clf/equal":cfg.no_test.threshold.clf.equal,
                                        "hydra_config/test/threshold/clf/different":cfg.no_test.threshold.clf.different,
                                        "hydra_config/dataset/name": cfg.dataset.name,
                                        "hydra_config/dataset/data_path": cfg.dataset.data_path,
                                        "hydra_config/dataset/batch_size": cfg.dataset.batch_size},
                    execution_queue=f'rgai-gpu-01-2080ti:{cfg.machine.gpus_index[0]}')
        parents_to_wait.append("auto-encoder standard training")

    if not cfg.past_tasks.use or cfg.past_tasks.standard.clf == "":
        pipe.add_step(name="classifier standard training", 
                    base_task_project="e-muse/metamorphic-testing", 
                    base_task_name=cfg.clf_base_task_name, 
                    parameter_override={"hydra_config/seed":cfg.seed,
                                        "hydra_config/test/flag":cfg.no_test.flag,
                                        "hydra_config/test/name":cfg.no_test.name,
                                        "hydra_config/test/stage":cfg.no_test.stage,
                                        "hydra_config/test/param/num_features":cfg.no_test.param.num_features,
                                        "hydra_config/test/param/mu":cfg.no_test.param.mu,
                                        "hydra_config/test/param/std":cfg.no_test.param.std,
                                        "hydra_config/test/threshold/ae/equal":cfg.no_test.threshold.ae.equal,
                                        "hydra_config/test/threshold/ae/different":cfg.no_test.threshold.ae.different,
                                        "hydra_config/test/threshold/clf/equal":cfg.no_test.threshold.clf.equal,
                                        "hydra_config/test/threshold/clf/different":cfg.no_test.threshold.clf.different,
                                        "hydra_config/dataset/name": cfg.dataset.name,
                                        "hydra_config/dataset/data_path": cfg.dataset.data_path,
                                        "hydra_config/dataset/batch_size": cfg.dataset.batch_size,
                                        "hydra_config/ae_task_id": "${auto-encoder standard training.id}" if not cfg.past_tasks.use or cfg.past_tasks.standard.ae == "" else cfg.past_tasks.standard.ae},
                    parents=["auto-encoder standard training"] if not cfg.past_tasks.use or cfg.past_tasks.standard.ae == "" else None,
                    execution_queue=f'rgai-gpu-01-2080ti:{cfg.machine.gpus_index[0]}')
        parents_to_wait.append("classifier standard training")

    if not cfg.past_tasks.use or cfg.past_tasks.test.ae == "":
        # test training
        pipe.add_step(name="auto-encoder test training", 
                    base_task_project="e-muse/metamorphic-testing", 
                    base_task_name=cfg.ae_base_task_name, 
                    parameter_override={"hydra_config/seed":cfg.seed,
                                        "hydra_config/test/flag":cfg.test.flag,
                                        "hydra_config/test/name":cfg.test.name,
                                        "hydra_config/test/stage":cfg.test.stage,
                                        "hydra_config/test/param/num_features":cfg.test.param.num_features,
                                        "hydra_config/test/param/mu":cfg.test.param.mu,
                                        "hydra_config/test/param/std":cfg.test.param.std,
                                        "hydra_config/test/threshold/ae/equal":cfg.test.threshold.ae.equal,
                                        "hydra_config/test/threshold/ae/different":cfg.test.threshold.ae.different,
                                        "hydra_config/test/threshold/clf/equal":cfg.test.threshold.clf.equal,
                                        "hydra_config/test/threshold/clf/different":cfg.test.threshold.clf.different,
                                        "hydra_config/dataset/name": cfg.dataset.name,
                                        "hydra_config/dataset/data_path": cfg.dataset.data_path,
                                        "hydra_config/dataset/batch_size": cfg.dataset.batch_size},
                    execution_queue=f'rgai-gpu-01-2080ti:{cfg.machine.gpus_index[1]}')
        parents_to_wait.append("auto-encoder test training")
    
    if not cfg.past_tasks.use or cfg.past_tasks.test.clf == "":
        pipe.add_step(name="classifier test training", 
                    base_task_project="e-muse/metamorphic-testing", 
                    base_task_name=cfg.clf_base_task_name, 
                    parameter_override={"hydra_config/seed":cfg.seed,
                                        "hydra_config/test/flag":cfg.test.flag,
                                        "hydra_config/test/name":cfg.test.name,
                                        "hydra_config/test/stage":cfg.test.stage,
                                        "hydra_config/test/param/num_features":cfg.test.param.num_features,
                                        "hydra_config/test/param/mu":cfg.test.param.mu,
                                        "hydra_config/test/param/std":cfg.test.param.std,
                                        "hydra_config/test/threshold/ae/equal":cfg.test.threshold.ae.equal,
                                        "hydra_config/test/threshold/ae/different":cfg.test.threshold.ae.different,
                                        "hydra_config/test/threshold/clf/equal":cfg.test.threshold.clf.equal,
                                        "hydra_config/test/threshold/clf/different":cfg.test.threshold.clf.different,
                                        "hydra_config/dataset/name": cfg.dataset.name,
                                        "hydra_config/dataset/data_path": cfg.dataset.data_path,
                                        "hydra_config/dataset/batch_size": cfg.dataset.batch_size,
                                        "hydra_config/ae_task_id": "${auto-encoder test training.id}" if not cfg.past_tasks.use or cfg.past_tasks.test.ae == "" else cfg.past_tasks.test.ae},
                    parents=["auto-encoder test training"] if not cfg.past_tasks.use or cfg.past_tasks.test.ae == "" else None,
                    execution_queue=f'rgai-gpu-01-2080ti:{cfg.machine.gpus_index[1]}')
        parents_to_wait.append("classifier test training")
    
    # results aggregation
    pipe.add_step(name=f"experiments aggregator ({cfg.test.name} test)", 
                base_task_project="e-muse/metamorphic-testing", 
                base_task_name=cfg.aggregator_task_name,
                parameter_override={"hydra_config/test/flag":cfg.test.flag,
                                    "hydra_config/test/name":cfg.test.name,
                                    "hydra_config/test/stage":cfg.test.stage,
                                    "hydra_config/test/param/num_features":cfg.test.param.num_features,
                                    "hydra_config/test/param/mu":cfg.test.param.mu,
                                    "hydra_config/test/param/std":cfg.test.param.std,
                                    "hydra_config/test/threshold/ae/equal":cfg.test.threshold.ae.equal,
                                    "hydra_config/test/threshold/ae/different":cfg.test.threshold.ae.different,
                                    "hydra_config/test/threshold/clf/equal":cfg.test.threshold.clf.equal,
                                    "hydra_config/test/threshold/clf/different":cfg.test.threshold.clf.different,
                                    "hydra_config/dataset": cfg.dataset.name, 
                                    "hydra_config/standard_task_id/ae": "${auto-encoder standard training.id}" if not cfg.past_tasks.use or cfg.past_tasks.standard.ae == "" else cfg.past_tasks.standard.ae,
                                    "hydra_config/standard_task_id/clf": "${classifier standard training.id}" if not cfg.past_tasks.use or cfg.past_tasks.standard.clf == "" else cfg.past_tasks.standard.clf,
                                    "hydra_config/test_task_id/ae": "${auto-encoder test training.id}" if not cfg.past_tasks.use or cfg.past_tasks.test.ae == "" else cfg.past_tasks.test.ae,
                                    "hydra_config/test_task_id/clf": "${classifier test training.id}" if not cfg.past_tasks.use or cfg.past_tasks.test.clf == "" else cfg.past_tasks.test.clf},
                parents=parents_to_wait,
                execution_queue="rgai-gpu-01-cpu:1")

    pipe.start()
    pipe.wait()
    pipe.stop()
    

if __name__ == "__main__":
    main()
