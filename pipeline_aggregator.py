from clearml import Task
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import hydra
from omegaconf import DictConfig

from clearml import Task
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

    standard_loss_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    biased_loss_df = pd.read_csv(Task.get_task(task_id=cfg.biased_task_id).artifacts["losses dataframe"].get_local_copy(), index_col=0)

    sns.kdeplot(standard_loss_df[standard_loss_df["label"]==0]["loss"], label="control", fill=True, color="green")
    sns.kdeplot(standard_loss_df[standard_loss_df["label"]==1]["loss"], label="test", fill=True, color="red")
    plt.legend()
    plt.title("Reconstruction loss distribution per class", fontdict={"size":15})
    plt.xlabel("Reconstruction loss")
    task.get_logger().report_matplotlib_figure(title="Losses distribution standard experiment", series="test set", figure=plt.gcf())
    plt.close()

    
    sns.kdeplot(biased_loss_df[biased_loss_df["label"]==0]["loss"], label="control", fill=True, color="purple")
    sns.kdeplot(biased_loss_df[biased_loss_df["label"]==1]["loss"], label="test", fill=True, color="yellow")
    plt.legend()
    plt.title("Reconstruction loss distribution per class", fontdict={"size":15})
    plt.xlabel("Reconstruction loss")
    task.get_logger().report_matplotlib_figure(title="Losses distribution standard experiment", series="test set", figure=plt.gcf())
    plt.close()

    
    sns.kdeplot(biased_loss_df[biased_loss_df["label"]==0]["loss"], label="control biased", fill=True, color="purple")
    sns.kdeplot(biased_loss_df[biased_loss_df["label"]==1]["loss"], label="test biased", fill=True, color="yellow")
    sns.kdeplot(standard_loss_df[standard_loss_df["label"]==0]["loss"], label="control standard", fill=True, color="green")
    sns.kdeplot(standard_loss_df[standard_loss_df["label"]==1]["loss"], label="test standard", fill=True, color="red")
    plt.legend()
    plt.title("Reconstruction loss distribution per class", fontdict={"size":15})
    plt.xlabel("Reconstruction loss")
    task.get_logger().report_matplotlib_figure(title="Losses distribution both experiments", series="test set", figure=plt.gcf())
    plt.close()


if __name__ == '__main__':
    main()
