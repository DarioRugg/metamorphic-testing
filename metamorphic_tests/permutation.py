from metamorphic_tests.base_class import BaseTestClass

import random
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        print(" --> permuting!")
        
        shuffled_columns = data.columns.values
        indexes_shuffled = random.sample(list(range(shuffled_columns.shape[0])), self.cfg.param.num_features)

        data.iloc[:, indexes_shuffled] = data.iloc[:, sorted(indexes_shuffled)]
        shuffled_columns[sorted(indexes_shuffled)] = shuffled_columns[indexes_shuffled]
        data.columns = shuffled_columns
        return data
    
    def test(self, result: float, reference: float, model_arch: str) -> list[bool, float]:
        super().test(model_arch)
        
        if self.cfg.stage == "both":
            if model_arch == "ae":
                return self._equality_condition(result, reference, self.cfg.threshold.ae.equal)
            else:
                return self._equality_condition(result, reference, self.cfg.threshold.clf.equal)
        elif self.cfg.stage == "test":
            if model_arch == "ae":
                return self._difference_condition(result, reference, self.cfg.threshold.ae.different)
            else:
                return self._difference_condition(result, reference, self.cfg.threshold.clf.different)
