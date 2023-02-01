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


@hydra.main(version_base=None, config_path="conf", config_name="test_results_config")
def main(cfg : DictConfig) -> None:
    task: Task = Task.init(project_name='e-muse/metamorphic-testing',
                    task_name=cfg.task_name,
                    task_type=Task.TaskTypes.qc,
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


    test_results_df = pd.DataFrame()
    for task_type in cfg.tasks:
        if task_type != "standard":
            test_task_cfg = cfg.tasks[task_type]

            ae_df, clf_df = get_data(cfg.tasks.standard, test_task_cfg)

            
            if test_task_cfg.test.name == "features addition":
                morphtest_object = features_addition.MetamorphicTest(test_task_cfg.test)
            elif test_task_cfg.test.name == "features removal":
                morphtest_object = features_removal.MetamorphicTest(test_task_cfg.test)
            elif test_task_cfg.test.name == "noise":
                morphtest_object = noise.MetamorphicTest(test_task_cfg.test)
            elif test_task_cfg.test.name in ["permutation", "permutation on evaluation"]:
                morphtest_object = permutation.MetamorphicTest(test_task_cfg.test)
            elif test_task_cfg.test.name in ["shifting", "shifting on evaluation"]:
                morphtest_object = shifting.MetamorphicTest(test_task_cfg.test)

            # test results auto-encoder:
            ae_test_results = {"test type": test_task_cfg.test.name, "model": "Auto-Encoder", "score test": np.mean(ae_df["mse_test"]), "score standard": np.mean(ae_df["mse_standard"])}
            ae_test_results["result"], ae_test_results["distance from threshold"] = morphtest_object.test(ae_test_results["score test"], ae_test_results["score standard"], model_arch="ae")

            # test results classifier:
            clf_test_results = {"test type": test_task_cfg.test.name, "model": "Classifier", 
                                "score test": roc_auc_score(clf_df["label"], clf_df["probabilities_test"]), 
                                "score standard": roc_auc_score(clf_df["label"], clf_df["probabilities_standard"])}
            clf_test_results["result"], clf_test_results["distance from threshold"] = morphtest_object.test(clf_test_results["score test"], clf_test_results["score standard"], model_arch="clf")
            
            test_results_df = pd.concat([test_results_df, pd.DataFrame.from_records([ae_test_results, clf_test_results])], ignore_index=True)
            
    task.get_logger().report_vector(title='Scores comparison Auto-Encoder', series='', values=test_results_df[test_results_df["model"]=="Auto-Encoder"][["score test", "score standard"]].values.transpose(), labels=["test", "standard"], xaxis="Metamorphic test", xlabels=test_results_df[test_results_df["model"]=="Auto-Encoder"]["test type"].tolist(), yaxis='MSE loss')
    task.get_logger().report_vector(title='Scores comparison Classifier', series='', values=test_results_df[test_results_df["model"]=="Classifier"][["score test", "score standard"]].values.transpose(), labels=["test", "standard"], xaxis="Metamorphic test", xlabels=test_results_df[test_results_df["model"]=="Classifier"]["test type"].tolist(), yaxis='AUC score')

    task.get_logger().report_vector(title='Distance from threshold', series='Auto-Encoder', values=test_results_df[test_results_df["model"]=="Auto-Encoder"]["distance from threshold"].values, xlabels=test_results_df[test_results_df["model"]=="Auto-Encoder"]["test type"].tolist(), xaxis="Metamorphic test", yaxis='Distance from threshold')
    task.get_logger().report_vector(title='Distance from threshold', series='Classifier', values=test_results_df[test_results_df["model"]=="Classifier"]["distance from threshold"].values, xlabels=test_results_df[test_results_df["model"]=="Classifier"]["test type"].tolist(), xaxis="Metamorphic test", yaxis='Distance from threshold')
    
    task.get_logger().report_table(title="Test results", series="", iteration=0, table_plot=test_results_df)


if __name__ == '__main__':
    main()
