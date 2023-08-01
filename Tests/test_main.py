import ase
from ase.build import separate
from ase.io import write

import numpy as np
from numpy.testing import assert_allclose

import pytest

from Src.main import *


# def test_get_indices():
#     arr = np.array(list(range(10)))
#     values = [1, 5, 9]
#
#     expected = np.array([1, 5, 9])
#     result = get_indices(arr, values)
#
#     assert np.all(result == expected)
