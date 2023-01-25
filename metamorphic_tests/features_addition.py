import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest:
    def __init__(self, num_features: int, test_stage: str = "both", current_stage: str = "train") -> None:
        self.num_features: int = num_features
        self.test_stage = test_stage
        self.current_stage = current_stage
        
    def mutation(self, x_data: pd.DataFrame) -> pd.DataFrame:
        if self.current_stage == "both" or self.test_stage == self.current_stage:
            return self._transformation(x_data)
        else:
            return x_data

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        uninformative_features: pd.DataFrame = pd.DataFrame(np.random.randint(0, 1, size=(data.shape[0], self.num_features))).add_prefix("uninformative_feature_")

        new_data = pd.concat([data, uninformative_features], axis=1)

        return new_data
    
    def test(self, result: float, reference: float, round_decimal: int=5):
        return np.round(result, round_decimal) == np.round(reference, round_decimal)
    
    def update_current_stage(self, current_stage: str):
        assert current_stage in ["train", "test"] 
        self.current_stage = current_stage
