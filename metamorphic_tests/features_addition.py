import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest:
    def __init__(self, num_features: int) -> None:
        self.num_features: int = num_features
        
    def mutation(self, x_data: pd.DataFrame):
        uninformative_features: pd.DataFrame = pd.DataFrame(np.random.randint(0, 1, size=(x_data.shape[0], self.num_features))).add_prefix("uninformative_feature_")

        new_x_data = pd.concat([x_data, uninformative_features], axis=1)

        return new_x_data
    
    def test(self, result: float, reference: float, round_decimal: int=5):
        return np.round(result, round_decimal) == np.round(reference, round_decimal)
