from clearml import Task
from matplotlib import patches, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve

from scripts.utils import connect_confiuration

from metamorphic_tests import *


def get_data(standard_task_cfg: OmegaConf, test_task_cfg: OmegaConf) -> list[pd.DataFrame, pd.DataFrame]:
    
    ae_standard_df = pd.read_csv(Task.get_task(task_id=standard_task_cfg.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    ae_test_df = pd.read_csv(Task.get_task(task_id=test_task_cfg.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    clf_standard_df = pd.read_csv(Task.get_task(task_id=standard_task_cfg.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_standard_df["absolute_error"] = clf_standard_df["label"] - clf_standard_df["probabilities"]
    clf_test_df = pd.read_csv(Task.get_task(task_id=test_task_cfg.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_test_df["absolute_error"] = np.abs(clf_test_df["label"] - clf_test_df["probabilities"])

    ae_df = pd.concat([ae_standard_df.drop(columns="label").add_suffix("_standard"), ae_test_df.drop(columns="label").add_suffix("_test")], axis=1)
    ae_df["label"] = ae_standard_df["label"]
    clf_df = pd.concat([clf_standard_df.drop(columns="label").add_suffix("_standard"), clf_test_df.drop(columns="label").add_suffix("_test")], axis=1)
    clf_df["label"] = clf_standard_df["label"]
    
    return ae_df, clf_df


@hydra.main(version_base=None, config_path="conf", config_name="aggregator_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/metamorphic-testing',
                    task_name=cfg.task_name,
                    task_type=Task.TaskTypes.monitor,
                    reuse_last_task_id=False)

    task.set_base_docker(
        docker_image='rugg/metamorphic:latest',
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

    ae_df, clf_df = get_data(cfg.standard_task_id, cfg.test_task_id)
    
    # auto-encoder results reporting
    for loss_name, loss in [("MSE", "mse"), ("Cross-Entropy", "ce")]:

        for training_startegy in ["standard", "test"]:
            sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_{training_startegy}"], label="healthy", fill=True, color="green" if training_startegy=="standard" else "purple")
            sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_{training_startegy}"], label="pathological", fill=True, color="red" if training_startegy=="standard" else "yellow")
            plt.legend()
            plt.title(f"Auto-Encoder - {loss_name} losses distribution {training_startegy} experiment ({cfg.dataset} dataset)", fontdict={"size":15})
            plt.xlabel("Reconstruction loss")
            task.get_logger().report_matplotlib_figure(title=f"Auto-Encoder - Losses distribution {training_startegy} experiment", series=f"{loss_name} loss", figure=plt.gcf())
            plt.close()
        
        sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_standard"], label="healthy standard", fill=True, color="green")
        sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_standard"], label="pathological standard", fill=True, color="red")
        sns.kdeplot(ae_df[ae_df["label"]==0][f"{loss}_test"], label="healthy test", fill=True, color="purple")
        sns.kdeplot(ae_df[ae_df["label"]==1][f"{loss}_test"], label="pathological test", fill=True, color="yellow")
        plt.legend()
        plt.title(f"Auto-Encoder - {loss_name} losses distribution both experiments ({cfg.dataset} dataset)", fontdict={"size":15})
        plt.xlabel("Reconstruction loss")
        task.get_logger().report_matplotlib_figure(title="Auto-Encoder - Losses distribution both experiments", series=f"{loss_name} loss", figure=plt.gcf())
        plt.close()

        if np.sum(np.abs(ae_df[f"{loss}_standard"] - ae_df[f"{loss}_test"])) > 0:
            difference_df = pd.DataFrame.from_dict({"label": ae_df["label"], "difference": np.abs(ae_df[f"{loss}_standard"] - ae_df[f"{loss}_test"])})
            sns.kdeplot(difference_df[difference_df["label"]==0]["difference"], label="healthy", fill=True)
            sns.kdeplot(difference_df[difference_df["label"]==1]["difference"], label="pathological", fill=True)
            plt.legend()
            plt.title(f"Auto-Encoder - {loss_name} absolute difference on reconstruction ({cfg.dataset} dataset)", fontdict={"size":15})
            plt.xlabel("Reconstruction difference")
            task.get_logger().report_matplotlib_figure(title="Auto-Encoder - Absolute difference on reconstruction", series=f"{loss_name} loss", figure=plt.gcf())
            plt.close()
    
    # classifier results reporting
    for training_startegy in ["standard", "test"]:
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
    plt.scatter(clf_df[clf_df["label"]==0][f"probabilities_test"], ae_df[ae_df["label"]==0][f"mse_test"], marker="^", c="darkgreen")
    plt.scatter(clf_df[clf_df["label"]==1][f"probabilities_test"], ae_df[ae_df["label"]==1][f"mse_test"], marker="^", c="red")
    plt.title(f"Classifier - scatterplot ({cfg.dataset} dataset)", fontdict={"size":15})
    plt.legend(handles=[patches.Patch(color='green', label='healthy'),
                patches.Patch(color='red', label='pathological'),
                Line2D([0], [0], marker='o', markerfacecolor='grey', color='w', markersize=10, label='standard training'),
                Line2D([0], [0], marker='^', markerfacecolor='grey', color='w', markersize=10, label='test training')])
    plt.xlabel('Classifier squared error')
    plt.ylabel(f'Auto-Encoder MSE')
    task.get_logger().report_matplotlib_figure(title="Classifier - scatterplot", series="", figure=plt.gcf(), report_interactive=False)
    plt.close()
    
    # test results auto-encoder:
    ae_test_results = {"model": "Auto-Encoder", "score test": np.mean(ae_df["mse_test"]), "score standard": np.mean(ae_df["mse_standard"])}
    ae_test_results["result"], ae_test_results["distance from threshold"] = features_addition.MetamorphicTest(cfg.test).test(ae_test_results["score test"], ae_test_results["score standard"], model_arch="ae")

    # test results classifier:
    clf_test_results = {"model": "Classifier", "score test": roc_auc_score(clf_df["label"], clf_df["probabilities_test"]), "score standard": roc_auc_score(clf_df["label"], clf_df["probabilities_standard"])}
    clf_test_results["result"], clf_test_results["distance from threshold"] = features_addition.MetamorphicTest(cfg.test).test(clf_test_results["score test"], clf_test_results["score standard"], model_arch="clf")

    test_results_df = pd.DataFrame.from_records([ae_test_results, clf_test_results])
    
    task.get_logger().report_vector(title='Scores comparison', series='', values=test_results_df[["score test", "score standard"]].values.transpose(), labels=["test", "standard"], xlabels=test_results_df["model"], xaxis="Models", yaxis='Score')
    task.get_logger().report_table(title="Test results", series="", iteration=0, table_plot=test_results_df)


if __name__ == '__main__':
    main()
