from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_allclose
import pytest

from Src.stage3 import *

from Tests.common_fixtures import *


@pytest.mark.parametrize(['context', 'vector1', 'vector2', 'expected'],
                         [(does_not_raise(), [1, 0, 0], [0, 1, 0], np.array([[0., -1., 0.],
                                                                             [1., 0., 0.],
                                                                             [0., 0., 1.]])),
                          (does_not_raise(), [1, 0, 1], [0, 1, 0], np.array([[0.5, -0.70710678, -0.5],
                                                                             [0.70710678, 0., 0.70710678],
                                                                             [-0.5, -0.70710678, 0.5]])),
                          (does_not_raise(), [-3, 4, 0.2], [1, 1, 1], np.array([[0.30730152, 0.94983246, 0.05817281],
                                                                                [-0.66545658, 0.25819056, -0.70036075],
                                                                                [-0.68024504, 0.17651045,0.71141461]])),
                          (pytest.raises(ParallelVectorsError), [0, 5, 0], [0, 1, 0], np.array([])),
                          (pytest.raises(ParallelVectorsError), [0, -1, 0], [0, 1, 0], np.array([]))],
                         ids=['perpendicular_unit', 'non-unit_vectors', 'parallel', 'opposite'])
def test_rotation_matrix_from_vectors(context, vector1, vector2, expected):
    with context:
        result = rotation_matrix_from_vectors(np.array(vector1), np.array(vector2))
        assert_allclose(result, expected)
