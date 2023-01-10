from clearml import Task
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.utils import connect_confiuration


@hydra.main(version_base=None, config_path="conf", config_name="aggregator_config")
def main(cfg : DictConfig) -> None:
    task = Task.init(project_name='e-muse/PartialTraining',
                    task_name=cfg.task_name,
                    task_type=Task.TaskTypes.monitor)

    task.set_base_docker(
        docker_image='rugg/aebias:latest',
        docker_arguments='--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
                          --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
                          --env CLEARML_AGENT_GIT_USER=ruggeri\
                          --env CLEARML_AGENT_GIT_PASS={access_token}\
                          --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/IBD/,target=/data/ \
                          --mount type=bind,source=/srv/nfs-data/ruggeri/access_tokens/,target=/tokens/ \
                          --rm'.format(access_token=open("/tokens/gitlab_access_token.txt", "r").read())
    )

    cfg = connect_confiuration(clearml_task=task, configuration=cfg)

    task.execute_remotely()
    
    print(OmegaConf.to_yaml(cfg))
    
    standard_loss_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    biased_loss_df = pd.read_csv(Task.get_task(task_id=cfg.biased_task_id).artifacts["losses dataframe"].get_local_copy(), index_col=0)

    for loss_name, loss in [("MSE", "mse"), ("Cross-Entropy", "ce")]:

        for model_name, model_results in [("standard", standard_loss_df), ("biased", biased_loss_df)]:
            sns.kdeplot(model_results[model_results["label"]==0][loss], label="control", fill=True, color="green" if model_name=="standard" else "purple")
            sns.kdeplot(model_results[model_results["label"]==1][loss], label="test", fill=True, color="red" if model_name=="standard" else "yellow")
            plt.legend()
            plt.title(f"{loss_name} losses distribution {model_name} experiment ({cfg.dataset} dataset)", fontdict={"size":15})
            plt.xlabel("Reconstruction loss")
            task.get_logger().report_matplotlib_figure(title=f"Losses distribution {model_name} experiment {loss_name}", series="test set", figure=plt.gcf())
            plt.close()
        
        sns.kdeplot(standard_loss_df[standard_loss_df["label"]==0][loss], label="control standard", fill=True, color="green")
        sns.kdeplot(standard_loss_df[standard_loss_df["label"]==1][loss], label="test standard", fill=True, color="red")
        sns.kdeplot(biased_loss_df[biased_loss_df["label"]==0][loss], label="control biased", fill=True, color="purple")
        sns.kdeplot(biased_loss_df[biased_loss_df["label"]==1][loss], label="test biased", fill=True, color="yellow")
        plt.legend()
        plt.title(f"{loss_name} losses distribution both experiments ({cfg.dataset} dataset)", fontdict={"size":15})
        plt.xlabel("Reconstruction loss")
        task.get_logger().report_matplotlib_figure(title=f"Losses distribution both experiments {loss_name}", series="test set", figure=plt.gcf())
        plt.close()

        difference_df = pd.DataFrame.from_dict({"label": biased_loss_df["label"], "difference": (biased_loss_df[loss] - standard_loss_df[loss]).abs()})
        sns.kdeplot(difference_df[difference_df["label"]==0]["difference"], label="control", fill=True)
        sns.kdeplot(difference_df[difference_df["label"]==1]["difference"], label="test", fill=True)
        plt.legend()
        plt.title(f"{loss_name} absolute difference on reconstruction ({cfg.dataset} dataset)", fontdict={"size":15})
        plt.xlabel("Reconstruction difference")
        task.get_logger().report_matplotlib_figure(title=f"Absolute difference on reconstruction {loss_name}", series="test set", figure=plt.gcf())
        plt.close()


if __name__ == '__main__':
    main()
