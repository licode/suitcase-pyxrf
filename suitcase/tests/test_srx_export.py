import numpy as np
from suitcase.pyxrf.export import helper_scaler_value_2D


def test_scaler_value_2D():
    namelist = ['det1', 'det2']
    data = {}
    data['det1'] = np.zeros(50)
    data['det2'] = np.zeros(50)
    data1 = helper_scaler_value_2D(namelist, data, [10, 5])
    assert data1.shape == (10, 5, len(namelist))
