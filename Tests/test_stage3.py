from contextlib import nullcontext as does_not_raise

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from Src.stage3 import *

from Tests.common_fixtures import *


@pytest.mark.parametrize(['index', 'which', 'expected'],
                         [(0, 0, Rotation.from_quat([-0.41616115, -0.08740613,  0.27850356,  0.86116539])),
                          (1, 0, Rotation.from_quat([ 0.22196907,  0.83375788, -0.43964306,  0.24958267])),
                          (0, 1, Rotation.from_quat([ 0.35479002, -0.37783375, -0.53250004,  0.66918563])),
                          (1, 1, Rotation.from_quat([0.26427679, 0.66615146, 0.68880127, -0.10932899]))])
def test_compute_reactant_rotation(set_up_separate_molecules, index, which, expected):
    coordinates = (set_up_separate_molecules[0] if which == 0 else set_up_separate_molecules[1]).get_positions()
    molecules = set_up_separate_molecules[4] if which == 0 else set_up_separate_molecules[5]
    reactivity_matrix = set_up_separate_molecules[6]

    result = compute_reactant_rotation(coordinates, index, molecules, reactivity_matrix)

    assert_allclose(result.as_quat(False), expected.as_quat(False))
