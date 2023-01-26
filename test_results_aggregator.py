from clearml import Task
from matplotlib import patches, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve

from scripts.utils import connect_confiuration

from metamorphic_tests import *


def get_data(cfg: OmegaConf) -> list[pd.DataFrame, pd.DataFrame]:
    
    ae_standard_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    ae_test_df = pd.read_csv(Task.get_task(task_id=cfg.test_task_id.ae).artifacts["losses dataframe"].get_local_copy(), index_col=0)
    clf_standard_df = pd.read_csv(Task.get_task(task_id=cfg.standard_task_id.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_standard_df["absolute_error"] = clf_standard_df["label"] - clf_standard_df["probabilities"]
    clf_test_df = pd.read_csv(Task.get_task(task_id=cfg.test_task_id.clf).artifacts["predictions dataframe"].get_local_copy(), index_col=0)
    clf_test_df["absolute_error"] = np.abs(clf_test_df["label"] - clf_test_df["probabilities"])

    ae_df = pd.concat([ae_standard_df.drop(columns="label").add_suffix("_standard"), ae_test_df.drop(columns="label").add_suffix("_test")], axis=1)
    ae_df["label"] = ae_standard_df["label"]
    clf_df = pd.concat([clf_standard_df.drop(columns="label").add_suffix("_standard"), clf_test_df.drop(columns="label").add_suffix("_test")], axis=1)
    clf_df["label"] = clf_standard_df["label"]
    
    return ae_df, clf_df


@hydra.main(version_base=None, config_path="conf", config_name="test_results_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/metamorphic-testing',
                    task_name=cfg.task_name,
                    task_type=Task.TaskTypes.monitor)

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

    ae_df, clf_df = get_data(cfg)

    # test results auto-encoder:
    ae_test_results = {"model": "Auto-Encoder", "score test": np.mean(ae_df["mse_test"]), "score standard": np.mean(ae_df["mse_standard"])}
    ae_test_results["result"], ae_test_results["diference"] = features_addition.MetamorphicTest(cfg.test).test(np.mean(ae_df["mse_test"]), np.mean(ae_df["mse_standard"]), model_arch="ae")

    # test results classifier:
    clf_test_results = {"model": "Classifier", "score test": accuracy_score(clf_df["label"], clf_df["predictions_standard"]), "score standard": accuracy_score(clf_df["label"], clf_df["predictions_test"])}
    clf_test_results["result"], clf_test_results["diference"] = features_addition.MetamorphicTest(cfg.test).test(accuracy_score(clf_df["label"], clf_df["predictions_standard"]), accuracy_score(clf_df["label"], clf_df["predictions_test"]), model_arch="clf")

    test_results_df = pd.DataFrame.from_records([ae_test_results, clf_test_results])
    
    
    task.get_logger().report_vector(title='Scores comparison', series='Losses', values=test_results_df[["score test", "score standard"]].values.transpose(), labels=["standard", "test"], xlabels=test_results_df["model"], xaxis="Models", yaxis='Score')
    task.get_logger().report_table(title="Test results", series="", iteration=0, table_plot=test_results_df)


if __name__ == '__main__':
    main()
