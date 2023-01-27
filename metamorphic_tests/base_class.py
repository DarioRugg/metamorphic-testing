import numpy as np
from omegaconf import DictConfig
import pandas as pd

# class for adding uninforamtive features:
class BaseTestClass:
    def __init__(self, test_cfg: DictConfig, current_stage: str = "train") -> None:
        self.cfg: DictConfig = test_cfg

        self.current_stage = current_stage
        
    def mutation(self, x_data: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.stage == "both" or self.cfg.stage == self.current_stage:
            return self._transformation(x_data)
        else:
            return x_data
        
    def _transformation(self, data: pd.DataFrame) -> None:
        pass
        
    def test(self, model_arch: str) -> None:
        assert model_arch in ["ae", "clf"], "the architecture must be one of the following: ae, clf"

    def _equality_condition(self, result: float, reference: float, threshold: float) -> list[bool, float]:
        return np.abs(result - reference) <= threshold, np.abs(np.abs(result - reference) - threshold)

    def _difference_condition(self, result: float, reference: float, threshold: float) -> list[bool, float]:
        return np.abs(result - reference) > threshold, np.abs(np.abs(result - reference) - threshold)

    def update_current_stage(self, current_stage: str) -> None:
        assert current_stage in ["train", "test"] 
        self.current_stage = current_stage
