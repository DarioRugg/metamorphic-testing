from omegaconf import DictConfig
import pandas as pd

# class for adding uninforamtive features:
class BaseTestClass:
    def __init__(self, test_cfg: DictConfig, seed: int = None, current_stage: str = "train") -> None:
        test_cfg = test_cfg.merge_with({"seed": seed})
        self.cfg: DictConfig = test_cfg

        self.current_stage = current_stage
        
    def mutation(self, x_data: pd.DataFrame) -> pd.DataFrame:
        if self.current_stage == "both" or self.cfg.stage == self.current_stage:
            return self._transformation(x_data)
        else:
            return x_data
        
    def test(self, result: float, reference: float, model_arch: str) -> list[bool, float]:
        assert model_arch in ["ae", "clf"], "the architecture must be one of the following: ae, clf"
        return self._test_condition(result, reference, round_digits=self.cfg.ae_round_sensitivity if model_arch is "ae" else self.cfg.clf_round_sensitivity)

    def update_current_stage(self, current_stage: str) -> None:
        assert current_stage in ["train", "test"] 
        self.current_stage = current_stage
