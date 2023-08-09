import ase

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import dok_matrix

from Src.stage5 import *
from Src.common_functions import ConvergenceError

from Tests.common_fixtures import *


class TestHardSphereCalculator:
    # TODO: Add test where the values are non-zero
    def test_compute_forces_ester_hydrolysis(self, set_up_separate_molecules):
        reactant, product, reactant_indices, product_indices, reactant_molecules, _, reactivity_matrix = \
            set_up_separate_molecules

        calc = HardSphereCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces(reactant_molecules)

        expected = np.zeros((2, 3))

        assert_allclose(forces, expected, rtol=0, atol=1e-7)

    def test_compute_forces_one_molecule(self, one_molecule_breakdown):
        reactant, product, reactant_indices, product_indices, _, _, reactivity_matrix = one_molecule_breakdown

        calc = HardSphereCalculator(reactant_indices, product_indices, product, reactivity_matrix, 1.0)
        calc.atoms = reactant
        forces = calc.compute_forces([ase.Atoms()])

        expected = np.zeros((1, 3))

        assert_allclose(forces, expected, rtol=0, atol=1e-7)
