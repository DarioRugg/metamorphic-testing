from omegaconf import DictConfig
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest:
    def __init__(self, test_cfg: DictConfig, current_stage: str = "train") -> None:
        self.cfg: DictConfig = test_cfg
        self.current_stage = current_stage
        
    def mutation(self, x_data: pd.DataFrame) -> pd.DataFrame:
        if self.current_stage == "both" or self.cfg.stage == self.current_stage:
            return self._transformation(x_data)
        else:
            return x_data

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        uninformative_features: pd.DataFrame = pd.DataFrame(np.random.randint(0, 1, size=(data.shape[0], self.cfg.num_features))).add_prefix("uninformative_feature_")

        new_data = pd.concat([data, uninformative_features], axis=1)

        return new_data
    
    def test(self, result: float, reference: float, overwrite_round_decimal: int=None) -> list[bool, float]:
        overwrite_round_decimal = self.cfg.round_sensitivity if overwrite_round_decimal is None else overwrite_round_decimal
        return np.round(result, overwrite_round_decimal) == np.round(reference, overwrite_round_decimal), np.abs(result - reference)

    def update_current_stage(self, current_stage: str) -> None:
        assert current_stage in ["train", "test"] 
        self.current_stage = current_stage
