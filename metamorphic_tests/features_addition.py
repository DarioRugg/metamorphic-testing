from metamorphic_tests.base_class import BaseTestClass

import pandas as pd
import numpy as np

# class for adding uninforamtive features:
class MetamorphicTest(BaseTestClass):

    def _transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        uninformative_features: pd.DataFrame = pd.DataFrame(np.random.randint(0, 1, size=(data.shape[0], self.cfg.param.num_features))).add_prefix("uninformative_feature_")

        new_data = pd.concat([data, uninformative_features], axis=1)

        return new_data
    
    def test(self, result: float, reference: float, model_arch: str) -> list[bool, float]:
        super().test(model_arch)
        
        if model_arch == "ae":
            return self._difference_condition(result, reference, self.cfg.threshold.ae.different)
        else:
            return self._equality_condition(result, reference, self.cfg.threshold.clf.equal)
