import numpy as np
from numpy.testing import assert_array_equal
from suitcase.pyxrf.export import helper_scaler_value_2D, flip_data


def test_scaler_value_2D():
    namelist = ['det1', 'det2']
    data = {}
    data['det1'] = np.zeros(50)
    data['det2'] = np.zeros(50)
    data1 = helper_scaler_value_2D(namelist, data, [10, 5])
    assert data1.shape == (10, 5, len(namelist))


def test_flip_data_2d():
    d = np.zeros([10, 10])
    d[1, :] = np.arange(10)
    d1 = flip_data(d)
    assert np.sum(d[0, :]) == 0
    # only flip even lines
    assert_array_equal(d1[1, :], np.arange(9, -1, -1))


def test_flip_data_3d():
    # think about 2D fluorescence scan
    # with shape (num_row, num_col, spectrum len)
    d = np.zeros([20, 10, 4096])
    d[1, 0, :] = 1     # this spectrum will move to last position on the line
    d1 = flip_data(d)
    assert np.sum(d[0, :, :]) == 0
    # only flip even lines
    assert_array_equal(d1[1, 9, :], np.ones(4096))
    assert_array_equal(d1[1, 0, :], np.zeros(4096))
