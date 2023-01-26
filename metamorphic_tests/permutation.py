from metamorphic_tests.base_class import BaseTestClass

import random
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        
        new_data = data.loc[:, random.shuffle(range(data.shape[1]))]

        return new_data
    
    def _test_condition(self, result: float, reference: float, round_digits: str) -> list[bool, float]:
        return np.round(result, round_digits) == np.round(reference, round_digits), np.abs(result - reference)