from clearml import Task
from matplotlib import patches, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve

from scripts.utils import connect_confiuration


def get_data(cfg: OmegaConf) -> list[pd.DataFrame, pd.DataFrame]:
    
    ae_standard_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    ae_biased_df = pd.read_csv(Task.get_task(task_id=cfg.biased_task_id.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    clf_standard_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_standard_df["absolute_error"] = clf_standard_df["label"] - clf_standard_df["probabilities"]
    clf_biased_df = pd.read_csv(Task.get_task(task_id=cfg.biased_task_id.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_biased_df["absolute_error"] = np.abs(clf_biased_df["label"] - clf_biased_df["probabilities"])

    ae_df = pd.concat([ae_standard_df.drop(columns="label").add_suffix("_standard"), ae_biased_df.drop(columns="label").add_suffix("_biased")], axis=1)
    ae_df["label"] = ae_standard_df["label"]
    clf_df = pd.concat([clf_standard_df.drop(columns="label").add_suffix("_standard"), clf_biased_df.drop(columns="label").add_suffix("_biased")], axis=1)
    clf_df["label"] = clf_standard_df["label"]
    
    return ae_df, clf_df


@hydra.main(version_base=None, config_path="conf", config_name="aggregator_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/metamorphic-testing',
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

    
    if cfg.execution_type == "draft":
        task.execute_remotely()
    elif cfg.execution_type == "remote":
        task.execute_remotely(queue_name="rgai-gpu-01-cpu:1")
    
    print(OmegaConf.to_yaml(cfg))

    ae_df, clf_df = get_data(cfg)
    
    # auto-encoder results reporting
    for loss_name, loss in [("MSE", "mse"), ("Cross-Entropy", "ce")]:

        for training_startegy in ["standard", "biased"]:
            sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_{training_startegy}"], label="healthy", fill=True, color="green" if training_startegy=="standard" else "purple")
            sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_{training_startegy}"], label="pathological", fill=True, color="red" if training_startegy=="standard" else "yellow")
            plt.legend()
            plt.title(f"Auto-Encoder - {loss_name} losses distribution {training_startegy} experiment ({cfg.dataset} dataset)", fontdict={"size":15})
            plt.xlabel("Reconstruction loss")
            task.get_logger().report_matplotlib_figure(title=f"Auto-Encoder - Losses distribution {training_startegy} experiment", series=f"{loss_name} loss", figure=plt.gcf())
            plt.close()
        
        sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_standard"], label="healthy standard", fill=True, color="green")
        sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_standard"], label="pathological standard", fill=True, color="red")
        sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_biased"], label="healthy biased", fill=True, color="purple")
        sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_biased"], label="pathological biased", fill=True, color="yellow")
        plt.legend()
        plt.title(f"Auto-Encoder - {loss_name} losses distribution both experiments ({cfg.dataset} dataset)", fontdict={"size":15})
        plt.xlabel("Reconstruction loss")
        task.get_logger().report_matplotlib_figure(title="Auto-Encoder - Losses distribution both experiments", series=f"{loss_name} loss", figure=plt.gcf())
        plt.close()

        if np.sum(np.abs(ae_df[f"{loss}_standard"] - ae_df[f"{loss}_biased"])) > 0:
            difference_df = pd.DataFrame.from_dict({"label": ae_df["label"], "difference": np.abs(ae_df[f"{loss}_standard"] - ae_df[f"{loss}_biased"])})
            sns.kdeplot(difference_df[difference_df["label"]==0]["difference"], label="healthy", fill=True)
            sns.kdeplot(difference_df[difference_df["label"]==1]["difference"], label="pathological", fill=True)
            plt.legend()
            plt.title(f"Auto-Encoder - {loss_name} absolute difference on reconstruction ({cfg.dataset} dataset)", fontdict={"size":15})
            plt.xlabel("Reconstruction difference")
            task.get_logger().report_matplotlib_figure(title="Auto-Encoder - Absolute difference on reconstruction", series=f"{loss_name} loss", figure=plt.gcf())
            plt.close()
    
    # classifier results reporting
    for training_startegy in ["standard", "biased"]:
        fpr, tpr, thresholds = roc_curve(clf_df["label"], clf_df[f"probabilities_{training_startegy}"])
        plt.plot(fpr, tpr, label=f"{training_startegy.capitalize()} training")
    plt.title(f"Classifier - ROC curve ({cfg.dataset} dataset)", fontdict={"size":15})
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    task.get_logger().report_matplotlib_figure(title="Classifier - ROC curve", series="", figure=plt.gcf())
    plt.close()

    plt.scatter(clf_df[clf_df["label"]==0][f"probabilities_standard"], ae_df[ae_df["label"]==0][f"mse_standard"], marker="o", c="forestgreen")
    plt.scatter(clf_df[clf_df["label"]==1][f"probabilities_standard"], ae_df[ae_df["label"]==1][f"mse_standard"], marker="o", c="tomato")
    plt.scatter(clf_df[clf_df["label"]==0][f"probabilities_biased"], ae_df[ae_df["label"]==0][f"mse_biased"], marker="^", c="darkgreen")
    plt.scatter(clf_df[clf_df["label"]==1][f"probabilities_biased"], ae_df[ae_df["label"]==1][f"mse_biased"], marker="^", c="red")
    plt.title(f"Classifier - scatterplot ({cfg.dataset} dataset)", fontdict={"size":15})
    plt.legend(handles=[patches.Patch(color='green', label='healthy'),
                patches.Patch(color='red', label='pathological'),
                Line2D([0], [0], marker='o', markerfacecolor='grey', color='w', markersize=10, label='standard training'),
                Line2D([0], [0], marker='^', markerfacecolor='grey', color='w', markersize=10, label='biased training')])
    plt.xlabel('Classifier squared error')
    plt.ylabel(f'Auto-Encoder MSE')
    task.get_logger().report_matplotlib_figure(title="Classifier - scatterplot", series="", figure=plt.gcf(), report_interactive=False)
    plt.close()


if __name__ == '__main__':
    main()
