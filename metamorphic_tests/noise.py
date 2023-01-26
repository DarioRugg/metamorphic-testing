from metamorphic_tests.base_class import BaseTestClass

import random
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        
        white_noise = np.random.normal(loc=0, scale=0.1, size=data.shape)
        
        new_data = data + white_noise

        return new_data
    
    def _test_condition(self, result: float, reference: float, round_digits: str) -> list[bool, float]:
        return np.round(result, round_digits) == np.round(reference, round_digits), np.abs(result - reference)