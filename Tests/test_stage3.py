from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from Src.stage3 import *

from Tests.common_fixtures import *


@pytest.mark.parametrize(['index', 'which', 'expected'],
                         [(0, 0, Rotation.from_quat([-0.41616115, -0.08740613, 0.27850356, 0.86116539])),
                          (1, 0, Rotation.from_quat([0.22196907, 0.83375788, -0.43964306, 0.24958267])),
                          (0, 1, Rotation.from_quat([0.35479002, -0.37783375, -0.53250004, 0.66918563])),
                          (1, 1, Rotation.from_quat([0.26427679, 0.66615146, 0.68880127, -0.10932899]))])
def test_compute_reactant_rotation(set_up_separate_molecules, index, which, expected):
    coordinates = (set_up_separate_molecules[0] if which == 0 else set_up_separate_molecules[1]).get_positions()
    molecules = set_up_separate_molecules[4] if which == 0 else set_up_separate_molecules[5]
    reactivity_matrix = set_up_separate_molecules[6]

    result = compute_reactant_rotation(coordinates, index, molecules, reactivity_matrix)

    assert_allclose(result.as_quat(False), expected.as_quat(False))


def test_reorient_reactants(set_up_separate_molecules):
    reactant, product, _, _, reactant_molecules, product_molecules, reactivity_matrix = set_up_separate_molecules

    reorient_reactants(reactant, reactant_molecules, reactivity_matrix)
    reorient_reactants(product, product_molecules, reactivity_matrix)

    expected_reactants = np.array([[-0.98311, 3.24056, 3.04904],
                                   [-0.65912, 2.21241, 3.22234],
                                   [-0.1366, 3.89685, 1.97282],
                                   [-2.03143, 3.21908, 2.74231],
                                   [-0.91353, 3.79804, 3.9853],
                                   [-0.6342, 4.3413, 0.95941],
                                   [1.19922, 3.98387, 2.19841],
                                   [1.83754, 3.46194, 3.3907],
                                   [2.90943, 3.66713, 3.32774],
                                   [1.4297, 3.95553, 4.27688],
                                   [1.68243, 2.3814, 3.45031],
                                   [-0.92797, -0.55259, 0.37855],
                                   [-0.80599, -0.50696, 1.31958]])
    expected_products = np.array([[-4.01608, 0.23907, 0.06919],
                                  [-3.67274, -0.78719, 0.21706],
                                  [-3.13098, 0.94876, -0.93002],
                                  [-5.04885, 0.21402, -0.28706],
                                  [-3.9934, 0.75538, 1.03165],
                                  [-3.52226, 1.34984, -2.00601],
                                  [2.13267, 2.28495, 1.85001],
                                  [0.90251, 2.40252, 2.50429],
                                  [0.56829, 1.41042, 2.87807],
                                  [1.02182, 3.07934, 3.37522],
                                  [0.14375, 2.84389, 1.82226],
                                  [-1.89519, 1.08853, -0.52895],
                                  [-1.82089, 0.69183, 0.35564]])

    assert_allclose(reactant.get_positions(), expected_reactants)
    assert_allclose(product.get_positions(), expected_products)
