import pandas as pd
import os

from src.main.model import Model
from src.main.property import Property

data_path = os.path.abspath('../../src/main/datasets/Housing.csv')
data = pd.read_csv(filepath_or_buffer=data_path)

class TestHousePrices():
    def test_housePrices(self):
        """Tests for model prediction for high valued house"""
        linear_model = Model(model_type='normal').set_up_models(data=data)
        high_value_property = Property(7500, 3, 3, 2, 0, 1, 0, 1, 1, 1, 1, 0)
        med_value_property = Property(5400, 3, 2, 2, 1, 0, 0, 1, 1, 1, 1, 0.5)
        low_value_property = Property(3000, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1)

        high_pred = linear_model.predict(high_value_property.to_array())
        med_pred = linear_model.predict(med_value_property.to_array())
        low_pred = linear_model.predict(low_value_property.to_array())

        assert (high_pred, med_pred, low_pred) is not None
        assert int(high_pred) > 8000000
        assert int(med_pred) > 5000000
        assert int(low_pred) > 2500000





