from metamorphic_tests.base_class import BaseTestClass

import random
import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        
        new_data = data.loc[:, random.shuffle(range(data.shape[1]))]

        return new_data
    
    def test(self, result: float, reference: float, model_arch: str) -> list[bool, float]:
        super().test(model_arch)
        
        if self.cfg.stage == "both":
            if model_arch == "ae":
                return self._equality_condition(result, reference, self.cfg.threshold.ae.equal)
            else:
                return self._equality_condition(result, reference, self.cfg.threshold.clf.equal)
        elif "test":
            if model_arch == "ae":
                return self._difference_condition(result, reference, self.cfg.threshold.ae.different)
            else:
                return self._difference_condition(result, reference, self.cfg.threshold.clf.different)
