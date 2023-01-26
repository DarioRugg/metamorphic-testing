from metamorphic_tests.base_class import BaseTestClass

import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        uninformative_features: pd.DataFrame = pd.DataFrame(np.random.randint(0, 1, size=(data.shape[0], self.cfg.num_features))).add_prefix("uninformative_feature_")

        new_data = pd.concat([data, uninformative_features], axis=1)

        return new_data
    
    def _test_condition(self, result: float, reference: float, round_digits: str) -> list[bool, float]:
        return np.round(result, round_digits) == np.round(reference, round_digits), np.abs(result - reference)
