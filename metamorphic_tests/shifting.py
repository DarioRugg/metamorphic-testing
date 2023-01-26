from metamorphic_tests.base_class import BaseTestClass

import random
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:

        mean_perturbation = np.random.standard_normal(size=data.shape[1])
        var_perturbation = np.random.standard_normal(size=data.shape[1])
        
        new_data = (data * var_perturbation) + mean_perturbation

        return new_data
    
    def _test_condition(self, result: float, reference: float, round_digits: str) -> list[bool, float]:
        return np.round(result, round_digits) == np.round(reference, round_digits), np.abs(result - reference)