from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from Src.stage3 import *

from Tests.common_fixtures import *


@pytest.mark.parametrize(['index', 'which', 'expected'],
                         [(0, 0, Rotation.from_quat([-0.45597479, -0.10370604, 0.17327517, 0.86678011])),
                          (1, 0, Rotation.from_quat([-0.55464471, 0.16174192, -0.27663454, 0.76790763])),
                          (0, 1, Rotation.from_quat([-0.0644361, -0.54430009, -0.33833897, 0.76492623])),
                          (1, 1, Rotation.from_quat([0.26427679, 0.66615146, 0.68880127, -0.10932899]))])
def test_compute_reactant_rotation(set_up_separate_molecules, index, which, expected):
    coordinates = (set_up_separate_molecules[0] if which == 0 else set_up_separate_molecules[1]).get_positions()
    molecules = set_up_separate_molecules[4] if which == 0 else set_up_separate_molecules[5]
    reactivity_matrix = set_up_separate_molecules[6]

    result = compute_reactant_rotation(coordinates, index, molecules, reactivity_matrix)
    print(repr(result.as_quat(canonical=False)))

    assert_allclose(result.as_quat(False), expected.as_quat(False))


@pytest.mark.parametrize(['context', 'vector1', 'vector2', 'expected'],
                         [(does_not_raise(), [1, 0, 0], [0, 1, 0], np.array([[0., -1., 0.],
                                                                             [1., 0., 0.],
                                                                             [0., 0., 1.]])),
                          (does_not_raise(), [1, 0, 1], [0, 1, 0], np.array([[0.5, -0.70710678, -0.5],
                                                                             [0.70710678, 0., 0.70710678],
                                                                             [-0.5, -0.70710678, 0.5]])),
                          (does_not_raise(), [-3, 4, 0.2], [1, 1, 1], np.array([[0.30730152, 0.94983246, 0.05817281],
                                                                                [-0.66545658, 0.25819056, -0.70036075],
                                                                                [-0.68024504, 0.17651045,
                                                                                 0.71141461]])),
                          (pytest.raises(ParallelVectorsError), [0, 5, 0], [0, 1, 0], np.array([])),
                          (pytest.raises(ParallelVectorsError), [0, -1, 0], [0, 1, 0], np.array([]))],
                         ids=['perpendicular_unit', 'non-unit_vectors', 'arbitrary', 'parallel', 'opposite'])
def test_rotation_matrix_from_vectors(context, vector1, vector2, expected):
    with context:
        result = rotation_matrix_from_vectors(np.array(vector1), np.array(vector2))
        assert_allclose(result, expected)
